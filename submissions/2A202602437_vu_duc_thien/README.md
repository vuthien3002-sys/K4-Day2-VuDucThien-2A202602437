# Lab Day 2 — Vũ Đức Thiện (2A202602437)

Backbone, công thức huấn luyện và suy luận trên DeepWeeds (fold 0).

## Chạy lại

Notebook Colab (mở thẳng từ GitHub):
https://colab.research.google.com/github/vuthien3002-sys/K4-Day2-VuDucThien-2A202602437/blob/main/submissions/2A202602437_vu_duc_thien/code/lab_day2.ipynb

1. Colab: **Runtime → Change runtime type → T4 GPU**.
2. Chạy lần lượt từ trên xuống. Ô đầu clone repo và gắn Google Drive; `runs/`, `predictions/`, `curves/`, `eval_out/`,
   `figures/`, `tables/` được trỏ sang `MyDrive/K4-Day2-2A202602437` nên phiên bị ngắt không mất kết quả.
3. Ô tải dữ liệu tải `images.zip` từ Zenodo, kiểm tra MD5 `b7b30f96d466fba86016aa5a26606e0f`, tải 4 file nhãn fold 0
   từ GitHub của tác giả.
4. Phiên bị ngắt: chạy lại từ đầu. Lần chạy đã xong (có `runs/<exp_id>/seed<k>/result.json`) được bỏ qua; quyết định
   đã chốt trên val (`tables/decisions.json`) không bị tính lại; test không chạy lần thứ hai.

Chạy một thí nghiệm từ dòng lệnh (cùng hàm `train.run`):

```bash
cd code
python train.py --set exp_id=B01 backbone=resnet50.a1_in1k seed=0 images_dir=<...>/images labels_dir=<...>/labels
```

Kiểm tra tự viết (CPU, không cần GPU): `cd code && python -m unittest test_selfcheck -v`
(thêm `DEEPWEEDS_DIR=<thư mục có images/ và labels/>` để chạy cả test đọc ảnh thật).

## Thứ tự thí nghiệm và seed

| Bước | exp_id | Nội dung | Seed |
|---|---|---|---|
| 0 | — | Kiểm tra split, EDA, ảnh sau augmentation, loss ban đầu, overfit 1 batch, test tự viết | 0 |
| 1 | B01–B06 | ResNet-50, ConvNeXt-T, DeiT-S, Swin-T, EfficientNet-B0, MobileNetV3-L; công thức nền T00 | 0 |
| 2 | T00 | Công thức nền trên backbone đã chọn | 0, 1, 2 |
| 2 | T01–T11 | Mỗi lần khác T00 một yếu tố (khởi tạo, augmentation, loss, sampler, EMA) | 0 |
| 2 | T12 | Kết hợp các yếu tố tốt | 0 |
| 3 | I00–I08 | Suy luận trên val (TTA, multi-crop, độ phân giải, ensemble, EMA, temperature scaling, gộp BN/FP16) | — |
| 4 | F01 | Cấu hình chung kết, test một lần mỗi seed | 0, 1, 2 |

Công thức nền T00 (GUIDE mục 1.4): trọng số ImageNet-1k, tinh chỉnh toàn bộ; train `RandomResizedCrop(224)` + lật ngang;
val/test `CenterCrop(224)` từ ảnh 256; AdamW, LR backbone 1e-4 / head 1e-3, weight decay 0,05 (không áp dụng cho
norm/bias); warmup 1 epoch rồi cosine; CE; batch 64; 12 epoch; AMP; checkpoint theo macro-F1 val.

## Môi trường

Google Colab, GPU Tesla T4. Python 3.13.15, torch 2.11.0+cu130, timm 1.0.29. Version đầy đủ (torchvision, numpy,
CUDA, cuDNN, tên GPU) được ghi tự động trong `runs/<exp_id>/seed<k>/config.json` của từng lần chạy.

## Cấu trúc

```
code/          dataset.py, model.py, losses.py, train.py, inference.py, benchmark.py (khung starter/ đã hoàn thiện)
               checks.py (EDA, kiểm tra pipeline), experiments.py (Bước 3, 4), results.py (results.xlsx)
               test_selfcheck.py (kiểm tra tự viết), lab_day2.ipynb
results.xlsx   Backbones, Training, Inference, Final, PerClass, Latency, Summary
report.md      báo cáo
curves/        đường cong training của mỗi lần chạy (<exp_id>_<mô tả>.png)
figures/       EDA, kiểm tra pipeline, đánh đổi độ chính xác/độ trễ, ma trận nhầm lẫn, ảnh bị đoán sai
predictions/   F01 và T00 (mốc) trên test, mỗi seed; F01uncal (chưa temperature scaling); F01 trên val
```

Không commit dataset và checkpoint (`best.pt` nằm trên Google Drive của tác giả).
