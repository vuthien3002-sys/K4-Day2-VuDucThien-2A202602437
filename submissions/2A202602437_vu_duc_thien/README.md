# Lab Day 2 — Vũ Đức Thiện (2A202602437)

Backbone, công thức huấn luyện và suy luận trên DeepWeeds (fold 0). Báo cáo: [report.md](report.md). Bảng: [results.xlsx](results.xlsx).

## Kết quả chính

| Cấu hình (test, 3 seed) | macro-F1 | top-1 | ECE | recall Chinee / Snake |
|---|---|---|---|---|
| **F02**: Swin-T + công thức nền T00 + TTA 10-crop gộp logit + temperature scaling | **0,9618 ± 0,0011** | **0,9707 ± 0,0003** | 0,0048 ± 0,0015 | 0,891 / 0,943 |
| Mốc T00 + I00 (1 view); cũng là cấu hình thời gian thực, p95 = 11,5 ms ở batch 1 | 0,9557 ± 0,0019 | 0,9659 ± 0,0021 | 0,0122 ± 0,0006 | 0,861 / 0,940 |

Chạy `eval.py grade` trên `predictions/` (kết quả ở `eval_results/grade_I.json`) cho đề xuất phần I = 19/20.

## Chạy lại

Notebook Colab (mở thẳng từ GitHub):
https://colab.research.google.com/github/vuthien3002-sys/K4-Day2-VuDucThien-2A202602437/blob/main/submissions/2A202602437_vu_duc_thien/code/lab_day2.ipynb

1. Colab: **Runtime → Change runtime type → GPU** (T4 trở lên). Bài này đã chạy trên T4 và A100.
2. Bấm **Run all**.
   - Ô đầu clone repo và gắn Google Drive. Các thư mục `runs/`, `predictions/`, `curves/`, `eval_out/`, `figures/`, `tables/` được trỏ sang `MyDrive/K4-Day2-2A202602437`.
   - Ô tải dữ liệu tải `images.zip` từ Zenodo, kiểm tra MD5 `b7b30f96d466fba86016aa5a26606e0f`, rồi tải 4 file nhãn fold 0 từ GitHub của tác giả.
3. Notebook tự dừng ở ô **⛔ trước test** để xem lại các quyết định trên val. Muốn chạy tiếp, chọn ô TEST bên dưới rồi **Run cell and below**.
4. Nếu phiên bị ngắt, chạy lại từ đầu:
   - Lần chạy đã xong (có `result.json`) được bỏ qua.
   - Quyết định đã chốt (`tables/decisions.json`) không bị tính lại.
   - Test không chạy lần thứ hai.

Chạy một thí nghiệm từ dòng lệnh (cùng hàm `train.run`):

```bash
cd code
python train.py --set exp_id=B01 backbone=resnet50.a1_in1k seed=0 images_dir=<...>/images labels_dir=<...>/labels
```

Kiểm tra tự viết (chạy được trên CPU, không cần GPU):

```bash
cd code
python -m unittest test_selfcheck -v
```

Thêm `DEEPWEEDS_DIR=<thư mục có images/ và labels/>` để chạy cả test đọc ảnh thật.

## Thứ tự thí nghiệm và seed

| Bước | exp_id | Nội dung | Seed |
|---|---|---|---|
| 0 | — | Kiểm tra split, EDA, ảnh sau augmentation, loss ban đầu, overfit 1 batch, test tự viết | 0 |
| 1 | B01–B06 | ResNet-50, ConvNeXt-T, DeiT-S, Swin-T, EfficientNet-B0, MobileNetV3-L với công thức nền T00 | 0 |
| 2 | T00 | Công thức nền trên Swin-T (backbone được chọn) | 0, 1, 2 |
| 2 | T01–T11 | Mỗi lần khác T00 một yếu tố: khởi tạo, augmentation, loss, sampler, EMA | 0 |
| 2 | T12 | Kết hợp T04 + T07 | 0 |
| 3 | I00–I08 | Suy luận trên val: TTA, multi-crop, gộp logit/xác suất, ensemble, EMA, temperature scaling, FP16 | — |
| 4 | F01 | Công thức T07 với 3 seed; bị loại trên val, không chạy test | 0, 1, 2 |
| 4 | **F02** | Chung kết: 3 mô hình T00 + 10-crop gộp logit + temperature scaling; test một lần mỗi seed | 0, 1, 2 |

Công thức nền T00 (GUIDE mục 1.4):
- Khởi tạo: trọng số ImageNet-1k, tinh chỉnh toàn bộ.
- Train: `RandomResizedCrop(224)` + lật ngang. Val/test: `CenterCrop(224)` từ ảnh 256.
- Tối ưu: AdamW, LR backbone 1e-4 và head 1e-3, weight decay 0,05 (không áp dụng cho norm/bias); warmup 1 epoch rồi cosine.
- Loss CE, batch 64, 12 epoch, AMP.
- Checkpoint theo macro-F1 val.

## Môi trường

- Google Colab: Tesla T4 cho B01–B06 và T00–T11; NVIDIA A100-SXM4-40GB cho T12, F01, Bước 3, đo độ trễ và test.
- Phần mềm: Python 3.13.15, torch 2.11.0+cu130, torchvision 0.26.0+cu130, timm 1.0.29, numpy 2.1.3, CUDA 13.0, cuDNN 9.27.
- Version và GPU của từng lần chạy ghi trong `logs/<exp_id>/seed<k>/config.json`.

## Cấu trúc

```
code/           dataset.py, model.py, losses.py, train.py, inference.py, benchmark.py (khung starter/ đã hoàn thiện)
                checks.py (EDA, kiểm tra pipeline), experiments.py (Bước 3, 4), results.py (results.xlsx)
                test_selfcheck.py (kiểm tra tự viết), lab_day2.ipynb
results.xlsx    Backbones, Training, Inference, Final, PerClass, Latency, Summary
report.md       báo cáo
curves/         đường cong training của mỗi lần chạy (<exp_id>_<mô tả>.png)
figures/        EDA, kiểm tra pipeline, đánh đổi độ chính xác/độ trễ, ma trận nhầm lẫn, ảnh bị đoán sai
predictions/    F02 và mốc T00 trên test (mỗi seed), F02uncal (chưa temperature scaling), F02 trên val
eval_results/   đầu ra của eval.py score / grade (tên gốc eval_out/)
logs/           config.json, history.csv, result.json của mọi lần chạy (tên gốc runs/)
tables/         EDA, kiểm tra pipeline, các quyết định đã chốt, kết quả Bước 3, độ trễ
```

`runs/` và `eval_out/` được đổi tên thành `logs/` và `eval_results/` vì `.gitignore` của repo chặn hai tên gốc.

Không commit dataset và checkpoint: `best.pt` của mọi lần chạy nằm trên Google Drive của tác giả.
