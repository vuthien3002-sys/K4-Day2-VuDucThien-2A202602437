"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val; nhiệt độ T khớp trên
VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9] hoặc list K view)
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def predict_logits(model, loader, device, view=None, amp: bool = True, channels_last: bool = True):
    """Chạy model trên loader và gom logit theo đúng thứ tự file.

    `view` biến đổi batch ảnh trước khi đưa vào model: None (giữ nguyên), một hàm trả về một batch
    (ví dụ view_hflip), hoặc một hàm trả về LIST các batch (multi-crop, multi-scale). Ảnh chỉ được
    đọc một lần cho mọi view. Trả về (filenames, y_true, logits) với logits là ndarray (N, 9), hoặc
    list K ndarray (N, 9) nếu `view` trả về list.
    """
    model.eval()
    use_amp = amp and torch.device(device).type == "cuda"
    names, ys, outs = [], [], None
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            views = view(x) if view is not None else x
            single = not isinstance(views, (list, tuple))
            views = [views] if single else list(views)
            if outs is None:
                outs = [[] for _ in views]
            for k, v in enumerate(views):
                if channels_last:
                    v = v.contiguous(memory_format=torch.channels_last)
                with torch.autocast(torch.device(device).type, dtype=torch.float16, enabled=use_amp):
                    outs[k].append(model(v).float().cpu())
            names.extend(f)
            ys.append(y)
    logits = [torch.cat(o).numpy() for o in outs]
    return names, torch.cat(ys).numpy(), logits[0] if single else logits


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) trên chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_hflip_pair(x):
    """TTA lật ngang K = 2: [ảnh gốc, ảnh lật]."""
    return [x, view_hflip(x)]


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop kích thước `crop` (4 góc + giữa) từ batch ảnh lớn hơn (ví dụ ảnh gốc 256), tuỳ chọn
    thêm bản lật ngang của mỗi crop (K = 10). Trả về list các batch."""
    h, w = x.shape[-2:]
    if crop > min(h, w):
        raise ValueError(f"crop {crop} lớn hơn ảnh {h}x{w}")
    top, left = (h - crop) // 2, (w - crop) // 2
    origins = [(0, 0), (0, w - crop), (h - crop, 0), (h - crop, w - crop), (top, left)]
    crops = [x[..., i:i + crop, j:j + crop] for i, j in origins]
    return crops + [view_hflip(c) for c in crops] if flip else crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bilinear, antialias khi thu nhỏ), trả về list batch.

    CNN có global pooling nhận được mọi kích thước. ViT/Swin cần vị trí/cửa sổ cố định theo kích thước
    lúc train (Swin window7_224 báo lỗi với kích thước khác), nên chỉ dùng multi-scale cho CNN.
    """
    out = []
    for s in sizes:
        if x.shape[-1] == s and x.shape[-2] == s:
            out.append(x)
        else:
            out.append(F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False, antialias=True))
    return out


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    Trả về xác suất (N, 9) đã chuẩn hoá. Bước 3 so sánh cả hai cách (I03).
    """
    z = np.stack([np.asarray(v, dtype=np.float64) for v in logits_per_view])  # (K, N, C)
    if space == "prob":
        p = _softmax(z).mean(axis=0)
    elif space == "logit":
        p = _softmax(z.mean(axis=0))
    else:
        raise ValueError(f"space={space!r} không hợp lệ (prob | logit)")
    return p / p.sum(axis=1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed).

    Chi phí suy luận = số mô hình. Chỉ ghép các mô hình trên CÙNG tập ảnh và cùng thứ tự file.
    """
    p = np.mean(np.stack([np.asarray(q, dtype=np.float64) for q in list_of_probs]), axis=0)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69).

    Tìm lưới thô trên log T trong [1/20, 20] rồi tinh bằng LBFGS trên log T (luôn dương).
    Accuracy không đổi vì thứ tự lớp không đổi. KHÔNG khớp T trên test.
    """
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)

    def nll(log_t):
        return F.cross_entropy(z / log_t.exp(), y)

    grid = torch.linspace(np.log(1 / 20), np.log(20), 81, dtype=torch.float64)
    start = grid[int(torch.stack([nll(g) for g in grid]).argmin())]
    log_t = start.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(log_t)
        loss.backward()
        return loss

    opt.step(closure)
    best = log_t.detach() if nll(log_t.detach()) <= nll(start) else start
    T = float(best.exp())
    if not 0.05 <= T <= 50:
        print(f"CẢNH BÁO: T = {T:.3g} ngoài [0.05, 50] (mô hình gần như đoán hằng?), cắt về biên")
        T = min(max(T, 0.05), 50.0)
    return T


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma * w / sqrt(var + eps);  b' = beta + gamma * (b - mean) / sqrt(var + eps)."""
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    fused = fused.to(device=conv.weight.device, dtype=conv.weight.dtype)
    gamma = bn.weight if bn.affine else torch.ones_like(bn.running_var)
    beta = bn.bias if bn.affine else torch.zeros_like(bn.running_var)
    scale = gamma / torch.sqrt(bn.running_var + bn.eps)
    bias = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        fused.bias.copy_(beta + (bias - bn.running_mean) * scale)
    return fused


def _bn_replacement(bn: nn.Module) -> nn.Module:
    """BN thường -> Identity. BatchNormAct2d của timm (BN + dropout + activation trong một module)
    -> giữ lại phần dropout + activation, nếu không sẽ mất hàm kích hoạt."""
    if hasattr(bn, "act") and hasattr(bn, "drop"):
        return nn.Sequential(bn.drop, bn.act)
    return nn.Identity()


def _fuse_children(module: nn.Module) -> int:
    count = 0
    keys = list(module._modules.keys())
    i = 0
    while i < len(keys):
        child = module._modules[keys[i]]
        nxt = module._modules[keys[i + 1]] if i + 1 < len(keys) else None
        if (type(child) is nn.Conv2d and isinstance(nxt, nn.BatchNorm2d)
                and nxt.num_features == child.out_channels and nxt.track_running_stats):
            module._modules[keys[i]] = _fuse_pair(child, nxt)
            module._modules[keys[i + 1]] = _bn_replacement(nxt)
            count += 1
            i += 2
            continue
        if child is not None:
            count += _fuse_children(child)
        i += 1
    return count


def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).

    Trả về BẢN SAO đã gộp (model gốc không đổi), ở chế độ eval; `fused.fused_pairs` là số cặp đã gộp.
    Chỉ gộp cặp (nn.Conv2d, BatchNorm2d) đứng liền nhau trong cùng module cha, đúng thứ tự đăng ký
    (ResNet, EfficientNet, MobileNetV3 của timm đều theo mẫu này). Luôn kiểm tra bằng check_fusion.
    Kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm) thì fused_pairs = 0: không áp dụng.
    """
    fused = copy.deepcopy(model).eval()
    fused.fused_pairs = _fuse_children(fused)
    return fused


@torch.no_grad()
def check_fusion(model, fused, img_size: int = 224, device="cpu", n: int = 4) -> float:
    """Sai số tuyệt đối lớn nhất giữa đầu ra trước và sau khi gộp BN (FP32, ảnh ngẫu nhiên)."""
    model.eval()
    fused.eval()
    x = torch.randn(n, 3, img_size, img_size, device=device)
    return float((model(x).float() - fused(x).float()).abs().max())
