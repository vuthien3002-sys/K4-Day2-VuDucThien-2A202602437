"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).
Kiểm tra tự viết cho các phần dễ sai nằm ở test_selfcheck.py (focal gamma=0 == CE, CutMix...).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSSES = ("ce", "ls", "focal", "ce_weighted")


def build_criterion(kind: str = "ce", smoothing: float = 0.1, gamma: float = 2.0, alpha=None,
                    weight=None):
    """Trả về hàm loss theo `kind`:
      - "ce"          : cross-entropy
      - "ls"          : CE + label smoothing (`smoothing`, tự cài đặt, xem LabelSmoothingCE)
      - "focal"       : focal loss (`gamma`, `alpha` tuỳ chọn)
      - "ce_weighted" : CE có trọng số theo lớp (`weight`, tính bằng class_weights từ tập train)
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing)
    if kind == "focal":
        return FocalLoss(gamma, alpha)
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted cần `weight` (dùng class_weights trên số ảnh train)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(weight, dtype=torch.float32))
    raise ValueError(f"loss={kind!r} không hợp lệ, chọn một trong {LOSSES}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt: loss = (1 - eps) * (-log p_y) + eps * mean_k(-log p_k). Với eps = 0 cho đúng CE
    (kiểm tra ở test_selfcheck.py), và bằng nn.CrossEntropyLoss(label_smoothing=eps).
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = -logp.gather(1, target[:, None]).squeeze(1)
        uniform = -logp.mean(dim=-1)
        return ((1.0 - self.smoothing) * nll + self.smoothing * uniform).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    Tính trên log_softmax (ổn định số học), lấy trung bình theo batch. `alpha`: None hoặc vector
    trọng số theo lớp. Với gamma = 0 và alpha = None cho đúng cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN (không dùng val hay test).

    - beta = 0: tỉ lệ nghịch với số ảnh (1 / n_c), chuẩn hoá về trung bình 1
    - beta > 0: class-balanced theo "số mẫu hiệu dụng": w_c = (1 - beta) / (1 - beta ** n_c)
      (slide trang 57, Cui et al. arXiv:1901.05555), chuẩn hoá tổng trọng số về số lớp
    Trả về tensor float32 độ dài bằng số lớp.
    """
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("mọi lớp phải có ít nhất 1 ảnh")
    if beta == 0:
        w = 1.0 / n
        w = w / w.mean()
    else:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w / w.sum() * len(n)
    return torch.as_tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn.

    - lam ~ Beta(alpha, alpha); perm là hoán vị ngẫu nhiên của batch
    - mode="mixup": x_mix = lam * x + (1 - lam) * x[perm]
    - mode="cutmix": cắt hộp có diện tích ~ (1 - lam) từ x[perm] dán vào x; hộp bị cắt ở biên ảnh
      nên lam được tính lại theo DIỆN TÍCH THỰC: lam = 1 - (diện tích hộp) / (H * W)  (slide trang 48)
    Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].
    """
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        cut = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(h * cut), int(w * cut)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = np.clip(cy - cut_h // 2, 0, h), np.clip(cy + cut_h // 2, 0, h)
        x1, x2 = np.clip(cx - cut_w // 2, 0, w), np.clip(cx + cut_w // 2, 0, w)
        x_mix = x.clone()
        x_mix[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1.0 - float((y2 - y1) * (x2 - x1)) / float(h * w)
    else:
        raise ValueError(f"mode={mode!r} không hợp lệ (mixup | cutmix)")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b).

    Accuracy trên batch đã trộn không còn nghĩa bình thường; đánh giá bằng val.
    """
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
