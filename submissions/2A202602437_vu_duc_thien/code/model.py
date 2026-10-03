"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    set_train_mode(model)                                         -> None (train() nhưng giữ BN đóng băng ở eval)
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import timm
import torch
from torch.nn.modules.batchnorm import _BatchNorm

# Backbone dùng ở Bước 1 (GUIDE.md mục 2.1), ghi rõ tag trọng số. Chọn tag ImageNet-1k cho mọi
# backbone để so sánh công bằng (không trộn trọng số in1k với in12k/in22k).
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50.a1_in1k",
    "resnext50": "resnext50_32x4d.a1_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "deit_small": "deit_small_patch16_224.fb_in1k",
    "swin_tiny": "swin_tiny_patch4_window7_224.ms_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",          # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100.ra_in1k",        # mạng nhẹ
}
INITS = ("scratch", "frozen", "finetune")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", drop_path_rate: float | None = None):
    """Tạo model phân loại 9 lớp bằng timm (timm tự thay head mới, head khởi tạo ngẫu nhiên).

    `init` (trục A của GUIDE.md mục 3):
      - "scratch"  : không tải trọng số, huấn luyện toàn bộ
      - "frozen"   : tải trọng số, đóng băng backbone, chỉ train head
      - "finetune" : tải trọng số, train toàn bộ
    Tag trọng số thực sự được tải lưu ở `model.weight_tag` (ghi vào results.xlsx).
    """
    if init not in INITS:
        raise ValueError(f"init={init!r} không hợp lệ, chọn một trong {INITS}")
    load = pretrained and init != "scratch"
    kwargs = {"pretrained": load, "num_classes": num_classes, "drop_rate": drop_rate}
    if drop_path_rate is not None:
        kwargs["drop_path_rate"] = drop_path_rate
    model = timm.create_model(name, **kwargs)

    cfg = model.pretrained_cfg
    arch, tag = cfg.get("architecture", name), cfg.get("tag")
    model.weight_tag = (f"{arch}.{tag}" if tag else arch) if load else f"{arch} (scratch)"
    model.frozen_backbone = False
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_param_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head (model.get_classifier()).

    BatchNorm của backbone đóng băng phải ở chế độ eval, nếu không running_mean/var vẫn bị cập nhật
    theo batch dù trọng số không đổi. Vì model.train() bật lại mọi BN, vòng huấn luyện gọi
    `set_train_mode(model)` thay cho model.train().
    """
    head = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.frozen_backbone = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone bị đóng băng thì đưa các BatchNorm đóng băng về eval."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        for m in model.modules():
            if isinstance(m, _BatchNorm) and not any(p.requires_grad for p in m.parameters()):
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số theo slide Day 2, trang 52.

    - backbone, trọng số ndim > 1               : lr_backbone, weight_decay
    - backbone, norm/bias (ndim <= 1) và các tham số timm khai báo no_weight_decay (pos_embed,
      cls_token, bảng bias vị trí...)            : lr_backbone, weight_decay = 0
    - head mới                                   : lr_head (gấp 10 lần), weight_decay cho trọng số,
                                                   bias của head cũng không decay
    Bỏ qua tham số requires_grad == False. Mỗi nhóm có khoá "name" để dễ kiểm tra.
    """
    head = _head_param_ids(model)
    no_wd_names = set(model.no_weight_decay()) if hasattr(model, "no_weight_decay") else set()
    groups = {
        "backbone_decay": ([], lr_backbone, weight_decay),
        "backbone_no_decay": ([], lr_backbone, 0.0),
        "head_decay": ([], lr_head, weight_decay),
        "head_no_decay": ([], lr_head, 0.0),
    }
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head else "backbone"
        decay = p.ndim > 1 and name not in no_wd_names
        groups[f"{part}_{'decay' if decay else 'no_decay'}"][0].append(p)
    return [{"name": k, "params": ps, "lr": lr, "weight_decay": wd}
            for k, (ps, lr, wd) in groups.items() if ps]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


@torch.no_grad()
def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size.

    Công cụ: torch.utils.flop_counter.FlopCounterMode (có sẵn trong PyTorch), đếm conv/matmul/attention
    theo FLOPs = 2 x MAC, nên GMAC = FLOPs / 2 / 1e9. Bỏ qua phép theo phần tử (activation, norm),
    nên số có thể lệch vài phần trăm so với fvcore/ptflops.
    """
    from torch.utils.flop_counter import FlopCounterMode

    was_training = model.training
    model.eval()
    p = next(model.parameters())
    x = torch.zeros(1, 3, img_size, img_size, device=p.device, dtype=p.dtype)
    with FlopCounterMode(display=False) as fc:
        model(x)
    model.train(was_training)
    return fc.get_total_flops() / 2 / 1e9
