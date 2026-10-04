# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

Vũ Đức Thiện · 2A202602437 · Track 4, Ngày 2

Mọi con số dưới đây lấy từ lần chạy thật, truy ngược được: `logs/<exp_id>/seed<k>/` (cấu hình, `history.csv`, `result.json`),
`curves/`, `tables/`, `predictions/`, `eval_results/` và `results.xlsx`. Số trên test được `eval.py` tính lại từ `predictions/`.

## 1. Tóm tắt

- Bài toán: phân loại 9 lớp ảnh cỏ dại DeepWeeds (fold 0 chia sẵn, 10.501 / 3.501 / 3.507 ảnh), chỉ số chính macro-F1.
- Đã làm: 6 backbone (B01–B06); 11 ablation một yếu tố trên 5 trục cùng 1 kết hợp (T01–T12) quanh công thức nền T00 (3 seed để đo
  nhiễu); 9 nhóm cách suy luận trên val (I00–I08) kèm độ trễ p50/p95/p99; chung kết 3 seed, test chạy một lần mỗi seed.
- **Cấu hình tốt nhất (F02):** Swin-T (ImageNet-1k) + công thức nền T00 + TTA 10-crop gộp logit + temperature scaling (T khớp trên val).
  **Test: macro-F1 0,9618 ± 0,0011; top-1 97,07 ± 0,03 %; ECE 0,0048 ± 0,0015** (3 seed).
- So với mốc T00 + 1 view (macro-F1 test 0,9557 ± 0,0019): **Δ = +0,0061, gấp 3,2 lần std**, nên đây là cải thiện vượt nhiễu.
- Kết luận chính: **backbone** quyết định nhiều nhất (macro-F1 val từ 0,625 đến 0,963). Không yếu tố công thức huấn luyện nào vượt T00
  quá nhiễu; phần cải thiện cuối cùng đến từ **suy luận** (TTA) và **hiệu chuẩn**.
- Thời gian thực: T00 + 1 view có p95 = 11,5 ms ở batch 1 (A100). F02 (p95 ≈ 113–117 ms) chỉ hợp xử lý ngoại tuyến.

## 2. Dữ liệu và thiết lập

**Chia dữ liệu (S1–S4).** Dùng nguyên `train/val/test_subset0.csv` từ GitHub của tác giả, kiểm tra trước khi train
(`tables/eda_summary.json`, `tables/eda_split_counts.csv`):

| | train | val | test | tổng |
|---|---|---|---|---|
| Số ảnh | 10.501 (59,97 %) | 3.501 (20,00 %) | 3.507 (20,03 %) | 17.509 |

Giao từng cặp tập (train&val, train&test, val&test) đều bằng 0. Hợp ba tập là 17.509 ảnh. Không thiếu file nào.
Tỉ lệ lệch khỏi 60/20/20 dưới 0,03 điểm %.

**EDA.** Phân bố lớp ở [figures/eda_class_distribution.png](figures/eda_class_distribution.png), ảnh mẫu ở
[figures/eda_samples.png](figures/eda_samples.png).

- `Negatives` có 9.106 ảnh (52,0 %). Các loài có từ 1.009 ảnh (Rubber Vine) đến 1.126 ảnh (Chinee Apple). Lớp lớn nhất gấp **9,0 lần** lớp nhỏ nhất.
- Đối chiếu với Table 1 của bài báo (số trích dẫn): khớp ở 7/9 lớp. Chinee Apple đếm được 1.126 (bài báo 1.125), Lantana 1.063 (bài báo 1.064). Tổng vẫn 17.509, nên có vẻ một ảnh mang nhãn khác với bảng của bài báo.
- Ảnh đều 256×256 RGB (500/500 ảnh mẫu). Trung bình kênh RGB là 0,378 / 0,388 / 0,380, tối hơn ImageNet (0,485 / 0,456 / 0,406). Mình vẫn chuẩn hoá theo `pretrained_cfg` của từng trọng số, tức mean/std của ImageNet, vì backbone được tiền huấn luyện với thống kê đó.
- Nhìn bằng mắt: Chinee Apple và Snake Weed đều là cây lá rộng, mọc lẫn trong thảm lá khô. `Negatives` gồm cỏ, lá rụng, đất, đá và nhiều cây không phải mục tiêu. Ánh sáng thay đổi mạnh, có nắng gắt, bóng đổ, đôi khi ảnh ám hồng.

**Kiểm tra pipeline** (`tables/check_pipeline.json`, `tables/check_overfit.json`):

| Kiểm tra | Kết quả |
|---|---|
| Loss ban đầu của head mới (ResNet-50, 640 ảnh val) | 2,216, xấp xỉ ln 9 = 2,197 |
| Overfit 1 batch 36 ảnh (4 ảnh/lớp, 150 bước, không augmentation) | loss 2,180 → 0,0023, accuracy 1,00 ([figures/check_overfit_one_batch.png](figures/check_overfit_one_batch.png)) |
| Ảnh sau augmentation, đã giải chuẩn hoá | nhãn khớp ảnh ([figures/check_aug_basic.png](figures/check_aug_basic.png), [figures/check_cutmix.png](figures/check_cutmix.png)) |
| Chế độ train/eval | `evaluate` đưa mọi lớp về eval; khi đóng băng backbone, BN giữ eval trong lúc train |
| Kiểm tra tự viết (`code/test_selfcheck.py`) | 28/28 đạt: focal γ = 0 ≡ CE (sai số < 1e-6), λ của CutMix bằng diện tích thật, gộp BN sai số < 1e-4, không weight decay cho norm/bias/pos_embed, công thức EMA, warmup + cosine |
| Test của repo (`tests/`) | 38/38 đạt, không sửa `eval.py` |

**Công thức nền T00** (GUIDE mục 1.4), dùng cho mọi backbone:

- Khởi tạo: trọng số ImageNet-1k, head 9 lớp mới, tinh chỉnh toàn bộ.
- Train: `RandomResizedCrop(224)` (scale 0,08–1) + lật ngang.
- Val/test: Resize 256 + CenterCrop 224 (crop_pct 0,875).
- Tối ưu: AdamW, LR backbone 1e-4 và head 1e-3, weight decay 0,05 (trừ norm, bias, pos_embed). Warmup 1 epoch rồi cosine theo từng bước.
- Loss CE, batch 64, 12 epoch, AMP fp16, channels_last.
- Checkpoint: epoch có macro-F1 val cao nhất, hoà thì lấy epoch sớm hơn. Macro-F1 tính bằng `eval.compute_metrics`.

**Phần cứng và phần mềm.**

- GPU: Google Colab, Tesla T4 cho B01–B06 và T00–T11; A100-SXM4-40GB cho T12, F01, Bước 3, đo độ trễ và mọi lần chạy test. Do đổi tài khoản Colab giữa chừng.
- Phần mềm: Python 3.13.15, torch 2.11.0+cu130, torchvision 0.26.0, timm 1.0.29, CUDA 13.0, cuDNN 9.27.
- Seed: 0 cho Bước 1 và ablation; 0/1/2 cho T00, F01 và F02.
- Mức tái lập: chạy lại T00 seed 0 trên cùng GPU T4 cho đúng con số của B04 (macro-F1 val 0,9630). Cùng cấu hình nhưng khác GPU thì lệch: T07 seed 0 trên T4 đạt 0,9633, F01 seed 0 trên A100 đạt 0,9590.

## 3. Bước 1 — So sánh backbone

Cùng công thức T00, cùng seed 0, val, GPU T4 (sheet `Backbones`, ảnh `curves/B0x_*.png`).

| exp | Backbone (tag) | #tham số | GMAC | macro-F1 val | top-1 val | F1 Chinee | F1 Snake | s/epoch | p50 b1 (ms) |
|---|---|---|---|---|---|---|---|---|---|
| B01 | resnet50.a1_in1k | 23,5M | 4,09 | 0,7936 | 0,8518 | 0,614 | 0,730 | 39 | 7,6 |
| B02 | convnext_tiny.fb_in1k | 27,8M | 4,45 | 0,9557 | 0,9666 | 0,916 | 0,907 | 49 | 5,8 |
| B03 | deit_small_patch16_224.fb_in1k | 21,7M | 4,24 | 0,9484 | 0,9643 | 0,890 | 0,898 | 38 | 5,0 |
| **B04** | **swin_tiny_patch4_window7_224.ms_in1k** | 27,5M | 4,49 | **0,9630** | **0,9732** | 0,921 | 0,929 | 59 | 13,2 |
| B05 | efficientnet_b0.ra_in1k | 4,0M | 0,38 | 0,7504 | 0,8175 | 0,631 | 0,717 | 41 | 8,5 |
| B06 | mobilenetv3_large_100.ra_in1k | 4,2M | 0,22 | 0,6252 | 0,7355 | 0,665 | 0,540 | 39 | 8,8 |

Biểu đồ F1 theo độ trễ và theo GMAC: [figures/backbones_tradeoff.png](figures/backbones_tradeoff.png).

**Nhận xét.**

- **Hai nhóm tách rõ.** Ba mạng dùng LayerNorm (Swin-T, ConvNeXt-T, DeiT-S) đạt 0,948–0,963. Ba mạng dùng BatchNorm, với trọng số timm theo công thức A1/RA (ResNet-50, EfficientNet-B0, MobileNetV3), chỉ đạt 0,625–0,794. Khoảng cách 0,15–0,34 lớn hơn nhiễu seed (std 0,0017) gấp hàng trăm lần.
- **Dấu hiệu từ đường cong.**
  - Ở MobileNetV3 ([curves/B06_mobilenetv3_large.png](curves/B06_mobilenetv3_large.png)), train acc (có augmentation) đạt 0,92 trong khi val top-1 chỉ 0,74, và val loss 0,78 so với train loss 0,22. Mô hình học được tập train nhưng **khái quát hoá kém**.
  - ResNet-50 có train loss 0,38 và val loss 0,44: học chưa tới.
  - Mình không đo thêm để tách nguyên nhân, nên chỉ nêu **giả thuyết**:
    - (a) Công thức chung (AdamW, LR 1e-4, 12 epoch) hợp với nhóm LayerNorm hơn. Trọng số A1/RA được huấn luyện bằng LAMB/RMSprop với LR lớn, có thể cần LR fine-tune cao hơn.
    - (b) Lệch thống kê BatchNorm giữa ảnh train (crop phóng to từ 8–100 % diện tích) và ảnh đánh giá (center crop đúng tỉ lệ), tức hiện tượng FixRes. LayerNorm chuẩn hoá theo từng ảnh nên không bị ảnh hưởng.
  - Kiểm tra (b) không cần train lại: tính lại thống kê BN trên ảnh train với transform của val. Đây là việc tiếp theo (mục 8).
- **Thứ hạng khác ImageNet.** Trên ImageNet, ResNet-50-A1, ConvNeXt-T và Swin-T đều quanh 80–82 % top-1 (số trích dẫn từ model card của timm, không phải kết quả đo của bài này). Trên DeepWeeds với cùng công thức, ResNet-50 kém xa. Thứ hạng trên ImageNet không chuyển nguyên sang bài toán này khi công thức fine-tune cố định.
- **FLOPs không dự đoán được độ trễ.** Swin-T (4,49 GMAC) chậm hơn ConvNeXt-T (4,45 GMAC) 2,3 lần ở batch 1 (13,2 so với 5,8 ms). MobileNetV3 chỉ 0,22 GMAC nhưng p50 8,8 ms, chậm hơn DeiT-S (4,24 GMAC, 5,0 ms): ở batch 1, độ trễ bị chi phối bởi số lớp tuần tự và overhead khởi chạy kernel, không phải số phép tính.
- **Chọn backbone đi tiếp: Swin-T.** Đây là backbone có macro-F1 val cao nhất và F1 hai lớp khó cao nhất. Độ trễ 13 ms vẫn thấp hơn nhiều so với ngân sách 100 ms của robot, nên đánh đổi tốc độ không phải ràng buộc.
  - Quy tắc chọn được chốt trước trong `tables/decisions.json`: ưu tiên backbone nhanh hơn ít nhất 2 lần nếu F1 kém không quá 0,005. ConvNeXt-T (5,8 ms, kém 0,007) không thoả điều kiện này.
  - Chênh 0,007 giữa Swin-T và ConvNeXt-T mới đo với 1 seed, chỉ gấp khoảng 4 lần std của T00. Nên coi hai backbone gần ngang nhau, Swin-T nhỉnh hơn.

## 4. Bước 2 — Công thức huấn luyện

Backbone Swin-T, seed 0, 12 epoch; mỗi lần chỉ khác T00 **một yếu tố** (sheet `Training`, ảnh `curves/T*.png`).
Nhiễu do seed: T00 với 3 seed đạt 0,9630 / 0,9614 / 0,9595, tức **0,9613 ± 0,0017**. Δ tính so với T00 cùng seed 0.

| exp | Trục | Khác T00 ở điểm nào | macro-F1 val | Δ vs T00 | Δ / std | Kết luận |
|---|---|---|---|---|---|---|
| T00 | nền | — | 0,9630 | — | — | mốc |
| T01 | A. khởi tạo | từ đầu, không tiền huấn luyện | 0,6738 | −0,2892 | −170 | kém hẳn |
| T02 | A. khởi tạo | đóng băng backbone, chỉ train head | 0,6928 | −0,2701 | −159 | kém hẳn |
| T03 | B. augmentation | + ColorJitter | 0,9569 | −0,0061 | −3,6 | kém hơn |
| T04 | B. augmentation | + TrivialAugmentWide | 0,9625 | −0,0004 | −0,2 | không phân biệt được |
| T05 | B. augmentation | + CutMix (α = 1) | 0,9478 | −0,0151 | −8,9 | kém hơn rõ |
| T06 | B. augmentation | + Mixup (α = 0,2) | 0,9608 | −0,0022 | −1,3 | gần nhiễu, chưa kết luận |
| T07 | C. loss | label smoothing 0,1 | 0,9633 | +0,0003 | +0,2 | không phân biệt được |
| T08 | C. loss | focal γ = 2 | 0,9564 | −0,0066 | −3,9 | kém hơn |
| T09 | C. loss | CE trọng số class-balanced (β = 0,999) | 0,9589 | −0,0041 | −2,4 | kém hơn |
| T10 | D. cân bằng mẫu | sampler cân bằng lớp | 0,9460 | −0,0169 | −9,9 | kém hơn rõ |
| T11 | F. chính quy hoá | EMA trọng số (decay 0,998) | 0,9599 | −0,0031 | −1,8 | gần nhiễu, chưa kết luận |
| T12 | kết hợp | T04 + T07 (A100) | 0,9616 | −0,0013 | −0,8 | không phân biệt được |

Thiết kế: 5 trục, mỗi trục 2–5 giá trị (gồm giá trị nền), có trục loss và trục augmentation; mỗi lần chạy khác nền một yếu tố.

Lưu ý về độ tin cậy:
- Std chỉ ước lượng từ 3 seed nên chính nó cũng không chắc chắn. Các Δ ở mức 1–2 lần std (T06, T11) mình coi là chưa đủ bằng chứng.
- Kết hợp T12 dùng cách tham lam theo trục. Chưa có trục nào vượt nhiễu theo chiều tốt, nên mình ghép 2 trục có Δ lớn nhất: augmentation (T04) và loss (T07). Việc này chỉ để kiểm tra hiệu ứng có cộng dồn hay không.

**Nhận xét.**

- **Khởi tạo là yếu tố lớn nhất.** Với khoảng 10k ảnh rất khác ImageNet, Swin-T học từ đầu trong 12 epoch chỉ đạt 0,674. Đóng băng backbone chỉ đạt 0,693, nghĩa là đặc trưng ImageNet không đủ tách các loài cỏ nếu không tinh chỉnh. Tinh chỉnh toàn bộ đạt 0,963.
  - Kết quả này khớp với slide: transformer có thiên kiến quy nạp yếu, thiếu dữ liệu thì học từ đầu kém, còn tiền huấn luyện thì bù lại được.
  - Đường cong T01 vẫn còn đi lên ở epoch 12 ([curves/T01_scratch.png](curves/T01_scratch.png)), nên với nhiều epoch hơn khoảng cách có thể thu hẹp.
- **Loss và sampler để cân bằng lớp không giúp được lớp hiếm** (số liệu val, seed 0):

  | exp | macro-F1 | recall TB 8 loài | recall Negatives | precision TB 8 loài | số ảnh đoán là Negatives (thật: 1.821) |
  |---|---|---|---|---|---|
  | T00 | 0,9630 | 0,9645 | 0,9819 | 0,957 | 1.807 |
  | T09 CB-CE | 0,9589 | 0,9663 | 0,9736 | 0,947 | 1.785 |
  | T10 sampler | 0,9460 | 0,9647 | 0,9473 | 0,925 | 1.743 |

  - Sampler cân bằng gần như không tăng recall các loài, nhưng làm recall của `Negatives` giảm 3,5 điểm, tức là đoán nhầm cây không mục tiêu thành cỏ dại. Với robot, đó là phun thuốc sai chỗ.
  - Với tỉ lệ 9:1 và macro-F1 nền đã 0,96, bài toán không thiếu tín hiệu lớp hiếm. Cân bằng lại chỉ đẩy ranh giới quyết định về phía các loài.
  - Focal loss cũng kém hơn (−0,0066): giảm trọng số mẫu dễ không giúp gì khi phần lớn lỗi nằm ở cặp lớp khó.
- **CutMix có hại rõ (−0,015)**, còn Mixup gần trung tính.
  - Cây cỏ dại thường chỉ chiếm một phần ảnh. Cắt dán một hình chữ nhật lớn dễ che mất cây, hoặc dán cây của ảnh khác vào, khiến nhãn trộn theo diện tích không còn khớp nội dung.
  - Đường cong T05 cho thấy train loss giữ ở khoảng 0,53 vì nhãn mềm, và val bão hoà sớm ở epoch 9 ([curves/T05_cutmix.png](curves/T05_cutmix.png)).
- **ColorJitter làm kém hơn (−0,006).** Màu lá có vẻ là tín hiệu phân biệt loài, và ảnh vốn đã đa dạng về ánh sáng.
- **Kết hợp không cộng dồn.** T12 (TrivialAugment + label smoothing) đạt 0,9616, không hơn T04 (0,9625), T07 (0,9633) hay T00. Cả ba đều nằm trong nhiễu, và T12 chạy trên A100 nên có thêm lệch do phần cứng.
- **EMA không "miễn phí" ở đây (−0,0031, chưa kết luận).** Đường cong T11 ([curves/T11_ema.png](curves/T11_ema.png)) cho thấy macro-F1 val của trọng số EMA vẫn tăng đến tận epoch 12 (0,9554 → 0,9599 trong 5 epoch cuối). Với decay 0,998 và khoảng 2.000 bước, trọng số EMA bị trễ so với trọng số thường và chưa kịp bắt kịp. Muốn EMA có ích, cần train lâu hơn hoặc dùng decay nhỏ hơn.

**Quyết định công thức chung kết và việc sửa quy tắc** (`tables/decisions.json`, test chưa chạy khi đổi):

- Quy tắc đầu tiên trong notebook chọn cấu hình có macro-F1 val cao nhất ở seed 0, tức T07 (+0,0003), **mà không so với nhiễu**.
- Huấn luyện lại T07 với 3 seed (F01, `logs/F01/`) cho **0,9542 ± 0,0042**, thấp hơn T00 (0,9613 ± 0,0017).
- Vì vậy, trước khi chạm vào test, mình sửa quy tắc: chỉ đổi khỏi T00 khi Δ vượt nhiễu (khoá `final_recipe_v2`).
- Kết quả: **công thức chung kết = T00**. F01 được giữ lại như một thử nghiệm bị loại trên val.
- Lưu ý: F01 chạy trên A100 còn T00 chạy trên T4, nên một phần chênh lệch có thể do phần cứng. Dù vậy, trên val không có bằng chứng label smoothing tốt hơn.

## 5. Bước 3 — Suy luận

Mô hình T00 seed 0, đánh giá trên val, không train lại (sheet `Inference`, [figures/inference_tradeoff.png](figures/inference_tradeoff.png)).
Độ trễ đo bằng `code/benchmark.py`: batch 1, fp32, A100, warmup 10 lần, `cuda.synchronize` trước và sau, 100 lần đo, không tính tiền xử lý.

| exp | Phương pháp | K | macro-F1 val | top-1 val | ECE val | p50 / p95 / p99 (ms) | chi phí so với I00 |
|---|---|---|---|---|---|---|---|
| I00 | 1 view (mốc) | 1 | 0,9630 | 0,9732 | 0,0091 | 11,2 / 11,5 / 12,0 | 1,0× |
| I01 | TTA lật ngang, gộp xác suất | 2 | 0,9639 | 0,9746 | 0,0078 | 22,4 / 23,5 / 24,3 | 2,0× |
| I02a | 5-crop, gộp xác suất | 5 | 0,9659 | 0,9746 | 0,0072 | 56,1 / 58,5 / 58,9 | 5,0× |
| I02b | 10-crop (5 crop + lật), gộp xác suất | 10 | 0,9669 | 0,9763 | 0,0059 | 110,9 / 116,8 / 126,3 | 9,9× |
| I03 | cùng các view, **gộp logit**: hflip / 5-crop / 10-crop | 2/5/10 | 0,9639 / 0,9662 / **0,9681** | 0,9746 / 0,9749 / 0,9771 | 0,0074 / 0,0096 / 0,0098 | như trên | |
| I05a | ensemble 3 seed T00 | 3 | 0,9690 | 0,9777 | 0,0087 | 32,9 / 33,5 / 39,6 | 2,9× |
| I05b | ensemble 3 backbone tốt nhất (B04 + B02 + B03) | 3 | 0,9680 | 0,9774 | 0,0147 | 22,9 / 23,8 / 25,5 | 2,0× |
| I06 | trọng số EMA (T11) so với không EMA (T00) | 1 | 0,9599 / 0,9630 | | | 11,2 | 1,0× |
| I07 | temperature scaling (T = 1,276) | 1 | 0,9630 | 0,9732 | 0,0091 → 0,0089* | 11,1 | 1,0× |
| I08 | FP16 (`model.half()`) | 1 | 0,9630 | 0,9732 | 0,0086 | 11,6 / 15,3 | 1,0× |

\* ECE sau temperature scaling theo cách khớp chéo hai nửa val (khớp T trên nửa này, đo trên nửa kia).

Không áp dụng được, ghi rõ:
- **Multi-scale và dò độ phân giải (I04):** Swin-T `window7_224` chỉ nhận ảnh 224, nên code tự bỏ qua.
- **Gộp BatchNorm:** Swin không có cặp Conv–BN. Code gộp BN đã được kiểm tra trên ResNet, EfficientNet và MobileNetV3 trong `test_selfcheck.py`.

Mình cũng giữ lượt Bước 3 trước đó trên T07 (cùng sheet, cột mô hình): xu hướng giống hệt, với 5-crop +0,002 và 10-crop +0,0015 so với 1 view.

Độ trễ theo dtype và batch (sheet `Latency`, A100, 1 view):

| dtype | batch 1 p50 / p95 (ms) | batch 32 p50 (ms) | ảnh/s ở batch 32 |
|---|---|---|---|
| fp32 | 11,2 / 11,3 | 29,5 | 1.085 |
| AMP | 14,2 / 14,6 | 14,2 | 2.251 |
| fp16 | 11,6 / 11,7 | 11,7 | 2.747 |

**Nhận xét.**

- **TTA giúp, nhưng tốn gần đúng K lần.** Lật ngang +0,0009 ở 2×. 5-crop +0,003 ở 5×. 10-crop gộp logit +0,0051 (khoảng 3 lần nhiễu) ở 9,9×. Độ trễ tăng gần tuyến tính theo K, khớp slide trang 63.
- **Gộp logit tốt hơn gộp xác suất một chút** ở 5-crop và 10-crop (+0,0003 / +0,0012), nhưng chênh lệch nằm trong nhiễu.
- **Ensemble 3 seed cho macro-F1 val cao nhất (0,9690)** với chi phí 2,9× (chạy tuần tự 3 mô hình), rẻ hơn 10-crop. Mình không chọn ensemble làm chung kết vì cấu hình chung kết phải báo cáo trên ≥ 3 seed, mỗi "seed" của ensemble là 3 mô hình, tức cần 9 lần train.
- **FP16 không đổi độ chính xác.** Ở batch 1 nó không nhanh hơn fp32 (11,6 so với 11,2 ms), còn AMP **chậm hơn** fp32 (14,2 ms) vì chi phí ép kiểu. Ở batch 32, fp16 nhanh gấp 2,5 lần. Kết quả khớp slide trang 73: không giả định FP16 luôn nhanh hơn.
- **Hiệu chuẩn.** T khớp trên val ≈ 1,25–1,29 (> 1), cho thấy mô hình hơi quá tự tin. Trên test, F02 giảm ECE từ 0,0100 xuống **0,0048** (mục 6).
- **Chọn cách suy luận cho chung kết** (khoá `final_inference_method_v2`): quy tắc là cách rẻ nhất trong các cách có macro-F1 val cách tốt nhất không quá nhiễu. Kết quả là **10-crop gộp logit**.
- **Đánh đổi theo ngữ cảnh** (theo slide):
  - Ngoại tuyến: 10-crop hoặc ensemble cho độ chính xác cao nhất.
  - Thời gian thực trên robot: 1 view (fp32 hoặc fp16) là hợp nhất. 5-crop (p95 58,5 ms) chỉ dùng được nếu phần cứng nhúng đủ mạnh.

## 6. Bước 4 — Cấu hình tốt nhất và kết quả test

**F02** = Swin-T `swin_tiny_patch4_window7_224.ms_in1k` + công thức T00, dùng lại 3 lần chạy T00 seed 0/1/2 (cùng cấu hình) + TTA 10-crop gộp logit + temperature scaling. T khớp trên val, theo từng seed: 1,253 / 1,244 / 1,295.

**Mốc** = T00 + I00 (1 view), cùng 3 mô hình.

Test chạy **một lần cho mỗi seed** của mỗi cấu hình (`predictions/`), và `eval.py` tính lại được kết quả trùng khớp (`eval_results/`).

| Cấu hình (3 seed) | macro-F1 val | **macro-F1 test** | top-1 test | balanced acc | ECE test | recall Chinee | recall Snake |
|---|---|---|---|---|---|---|---|
| **F02** | 0,9650 ± 0,0030 | **0,9618 ± 0,0011** | **0,9707 ± 0,0003** | 0,9640 ± 0,0013 | **0,0048 ± 0,0015** | 0,891 ± 0,017 | 0,943 ± 0,015 |
| F02 chưa TS (F02uncal) | — | 0,9618 ± 0,0011 | 0,9707 ± 0,0003 | 0,9640 ± 0,0013 | 0,0100 ± 0,0007 | 0,891 ± 0,017 | 0,943 ± 0,015 |
| Mốc T00 + I00 | 0,9613 ± 0,0017 | 0,9557 ± 0,0019 | 0,9659 ± 0,0021 | 0,9589 ± 0,0016 | 0,0122 ± 0,0006 | 0,861 ± 0,007 | 0,940 ± 0,007 |

Theo từng seed, macro-F1 test của F02 là 0,9620 / 0,9606 / 0,9629; của mốc là 0,9536 / 0,9575 / 0,9559.

- **Cải thiện so với mốc:** Δ macro-F1 = +0,0061, lớn hơn std lớn nhất của hai nhóm (0,0019) gấp 3,2 lần. Toàn bộ cải thiện đến từ suy luận, vì cùng mô hình.
- **Hiệu chuẩn:** temperature scaling không đổi dự đoán (argmax giữ nguyên) nhưng giảm ECE test từ 0,0100 xuống 0,0048.
- **Chênh val/test:** 0,0032, cho thấy không có dấu hiệu chọn cấu hình quá khớp với val.
- **So với bài báo** (số trích dẫn: ResNet-50 95,7 %, Inception-v3 95,1 % weighted accuracy, 5 fold, khoảng 100 epoch, augmentation mạnh): F02 đạt top-1 97,07 % trên fold 0 với 12 epoch. Định nghĩa chỉ số và số fold khác nhau, nên đây chỉ là tham khảo.
- **Tự chấm phần I bằng `eval.py grade`** (`eval_results/grade_I.json`, giảng viên xác nhận) được 19/20:
  - I1 = 7 (top-1 97,07 %);
  - I2 = 4 (Δ > s nhưng < 0,01);
  - I3 = 4 (recall Chinee 89,1 % ≥ 88,5 %, Snake 94,3 % ≥ 88,8 %);
  - I4a = 1; I4b = 1;
  - I5 = 2 (cấu hình thời gian thực T00 + I00, p95 = 11,6 ms).

**Theo lớp** (sheet `PerClass`):
- F1 thấp nhất là **Chinee Apple** (0,931; recall 0,891 nhưng precision 0,976) và **Snake Weed** (0,940).
- Lantana có precision thấp nhất (0,914), vì nhiều ảnh `Negatives` bị đoán là Lantana.
- So với mốc, 10-crop tăng recall Chinee Apple từ 0,861 lên 0,891.

**Ma trận nhầm lẫn** (tổng 3 seed, [figures/final_confusion_matrix.png](figures/final_confusion_matrix.png)):

| Nhầm chính (thật → đoán) | F02 | Mốc T00 | Bài báo (trích dẫn) |
|---|---|---|---|
| Chinee Apple → Snake Weed | 26 / 678 (3,8 %) | 32 / 678 (4,7 %) | 3,4 % |
| Chinee Apple → Negatives | 31 / 678 (4,6 %) | 38 / 678 | |
| Chinee Apple → Lantana | 13 / 678 | 21 / 678 | |
| Snake Weed → Chinee Apple | 7 / 612 (1,1 %) | 7 / 612 | 4,1 % |
| Snake Weed → Negatives | 20 / 612 (3,3 %) | 22 / 612 | |
| Negatives → Lantana / Prickly Acacia | 38 / 27 trên 5.466 | 46 / 27 | |
| Parthenium ↔ Prickly Acacia | 12 / 12 | 14 / 14 | |
| Parkinsonia → Prickly Acacia | 2 / 621 (0,3 %) | 1 / 621 | 1,3 % |

**Phân tích lỗi bằng ảnh** (seed 0: [figures/errors_chinee_as_snake.png](figures/errors_chinee_as_snake.png), [figures/errors_snake_as_chinee.png](figures/errors_snake_as_chinee.png)):

- 8 ảnh Chinee Apple bị đoán thành Snake Weed có chung các đặc điểm:
  - tán lá dày chồng lấp, nắng gắt và bóng đổ mạnh, đôi khi ảnh ám hồng do cân bằng trắng;
  - cây chỉ chiếm một phần ảnh, lẫn trong lá khô.
- Mô hình cũng không chắc chắn ở các ảnh này: xác suất lớp bị đoán chỉ 0,36–0,84.
- 3 ảnh Snake Weed bị đoán thành Chinee Apple đều tối hoặc ám màu.

Giả thuyết:
- (1) Ở 224 px, hai loài cùng là cây lá rộng, hình lá tương tự; khác biệt nằm ở chi tiết nhỏ (gai, gân lá) dễ mất khi ảnh bị thu nhỏ hoặc thiếu sáng.
- (2) Ánh sáng và màu thay đổi theo địa điểm và thời điểm chụp.
- (3) Cây bị che khuất một phần, nên ảnh trông giống `Negatives`. Đây cũng là lý do lỗi nhầm sang `Negatives` lớn nhất ở cả hai lớp.

Hướng cải thiện: độ phân giải đầu vào cao hơn (dùng biến thể Swin 256/384), augmentation độ sáng có kiểm soát, xem lại nhãn của các ảnh có xác suất cao mà vẫn sai.

## 7. Kết luận và khuyến nghị

- **Cấu hình nào tốt nhất? Tốt hơn mốc bao nhiêu?**
  - F02 (Swin-T + công thức nền + 10-crop gộp logit + temperature scaling) đạt macro-F1 test 0,9618 ± 0,0011 và top-1 97,07 %.
  - So với mốc T00 + 1 view, Δ = +0,0061, gấp 3,2 lần std: cải thiện thật nhưng nhỏ.
- **Yếu tố nào đóng góp nhiều nhất?**
  1. **Backbone và khởi tạo:**
     - Đổi backbone thay đổi macro-F1 val tới 0,34 (0,625 → 0,963).
     - Bỏ tiền huấn luyện mất 0,29.
  2. **Suy luận:** TTA thêm +0,005–0,006 macro-F1 và temperature scaling giảm ECE còn một nửa.
  3. **Công thức huấn luyện:** trong phạm vi đã thử, không thay đổi nào vượt T00 quá nhiễu. Nhiều thay đổi làm kém đi rõ: CutMix, sampler cân bằng, focal loss, ColorJitter.

  Công thức mặc định đã gần tối ưu cho Swin-T trong 12 epoch. Khẳng định "công thức quan trọng ngang kiến trúc" của slide chủ yếu thể hiện qua khởi tạo (finetune so với scratch/frozen).
- **Robot với ngân sách 30–100 ms/khung chọn gì?**
  - **Swin-T + T00 + 1 view, fp32 hoặc fp16:** macro-F1 test 0,9557 ± 0,0019, p95 11,5 ms trên A100 và khoảng 13 ms trên T4 (đo nhanh ở Bước 1). Thêm temperature scaling miễn phí để ngưỡng tin cậy có nghĩa.
  - Nếu phần cứng nhúng còn dư ngân sách, 5-crop (khoảng 5 lần chi phí) thêm khoảng +0,003 macro-F1 trên val.
  - Cần đo lại trên chính thiết bị nhúng, ví dụ Jetson. Bài báo báo ResNet-50 mất 53,4 ms với TensorRT trên TX2 (số trích dẫn), còn Swin-T có GMAC tương đương.
  - 10-crop và ensemble chỉ hợp xử lý ngoại tuyến, ví dụ lập bản đồ cỏ dại sau mỗi lượt bay hoặc chạy.

## 8. Hạn chế và việc tiếp theo

- **Một fold, chia ngẫu nhiên.** Fold 0 chia ngẫu nhiên, không theo địa điểm chụp. Ảnh cùng địa điểm hoặc cùng buổi chụp có thể nằm ở cả train và test, nên điểm test có thể **lạc quan** so với khi gặp địa điểm hoặc mùa mới. Rủi ro lệch phân phối khi triển khai (ánh sáng, mùa, camera khác) chưa được đo.
- **Số seed.** Ablation chỉ 1 seed. Std nhiễu ước lượng từ 3 seed nên kém chắc chắn, và các Δ cỡ 1–2 lần std (Mixup, EMA) chưa đủ để kết luận.
- **Phần cứng không đồng nhất.** B01–B06 và T00–T11 chạy trên T4; T12, F01, Bước 3 và test chạy trên A100. Cùng cấu hình, cùng seed nhưng khác GPU lệch tới 0,004 macro-F1 val (T07 so với F01 seed 0). Độ trễ ở Bước 1 (T4) và Bước 3 (A100) không so trực tiếp được.
- **Thay đổi quy tắc chọn công thức giữa chừng.** Quy tắc đầu chọn T07 mà không so với nhiễu. Mình đã sửa trên val trước khi chạy test, giữ F01 làm bằng chứng và ghi lại toàn bộ trong `tables/decisions.json` kèm thời điểm.
- **Ngân sách GPU.**
  - Tổng thời gian train khoảng 4,6 giờ trên T4 và 0,3 giờ trên A100 (`logs/*/seed*/result.json`).
  - Đã giảm theo GUIDE mục 7: ablation chỉ trên 1 backbone, 1 seed; dành 3 seed cho T00 và chung kết; giữ 12 epoch.
- **Chưa làm.**
  - Tính lại thống kê BN để kiểm tra giả thuyết nhóm BatchNorm kém (mục 3).
  - Dò độ phân giải kiểm tra (Swin-T 224 cố định).
  - Ensemble làm cấu hình chung kết.
- **Việc tiếp theo nếu có thêm một ngày:**
  - (1) Chạy 5 fold, hoặc chia theo địa điểm, để có ước lượng thực tế hơn.
  - (2) Kiểm tra giả thuyết BN (không cần train) và thử LR fine-tune cao hơn cho nhóm CNN BatchNorm.
  - (3) Chưng cất Swin-T sang MobileNetV3 cho robot.
  - (4) Xuất ONNX/TensorRT và đo trên thiết bị nhúng.
  - (5) Đánh giá trên ảnh bị làm tối, nhiễu hoặc mờ để đo độ bền khi lệch miền.

## 9. Phụ lục

**Danh sách thí nghiệm.** Cấu hình đầy đủ của mỗi lần chạy ở `logs/<exp_id>/seed<k>/config.json`. Các thông số không đổi giữa mọi lần chạy: 12 epoch, batch 64, ảnh 224.

| exp_id | Backbone | Khác T00 | Seed | GPU |
|---|---|---|---|---|
| B01–B06 | 6 backbone ở mục 3 | — | 0 | T4 |
| T00 | Swin-T | — | 0, 1, 2 | T4 |
| T01–T11 | Swin-T | một yếu tố (mục 4) | 0 | T4 |
| T12 | Swin-T | TrivialAugment + label smoothing 0,1 | 0 | A100 |
| F01 | Swin-T | label smoothing 0,1 (bị loại trên val, không chạy test) | 0, 1, 2 | A100 |
| F02 | = T00 seed 0/1/2 | suy luận 10-crop gộp logit + temperature scaling | 0, 1, 2 | A100 (suy luận) |

**Quyết định đã chốt** (`tables/decisions.json`, giờ UTC):

| Khoá | Giá trị | Ghi chú |
|---|---|---|
| `backbone_for_steps_2_to_4` | B04 Swin-T | |
| `combo_factors_T12` | T04, T07 | |
| `final_recipe` | T07 | lượt đầu, đã thay |
| `final_recipe_v2` | T00 | quy tắc có nhiễu |
| `final_inference_method_v2` | 10-crop, gộp logit | |

**Tái lập.**
- Notebook Colab: https://colab.research.google.com/github/vuthien3002-sys/K4-Day2-VuDucThien-2A202602437/blob/main/submissions/2A202602437_vu_duc_thien/code/lab_day2.ipynb
- Code nằm trong `code/`. Cách chạy xem `README.md`.
