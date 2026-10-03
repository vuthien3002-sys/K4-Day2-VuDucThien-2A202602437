"""checks.py - Bước 0: EDA và kiểm tra pipeline (GUIDE.md mục 1.2 và 1.3).

Mọi hình lưu vào `fig_dir` (mặc định figures/) để đưa vào báo cáo. Hàm overfit_one_batch có huấn luyện
(vài chục bước trên một batch) nên chỉ chạy trên GPU của Colab.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from PIL import Image

import dataset as D
import losses as L
import model as M

# Table 1 của bài báo (Olsen et al., 2019): số TRÍCH DẪN để đối chiếu, không phải kết quả đo.
PAPER_TABLE1 = {"Chinee Apple": 1125, "Lantana": 1064, "Parkinsonia": 1031, "Parthenium": 1022,
                "Prickly Acacia": 1062, "Rubber Vine": 1009, "Siam Weed": 1074, "Snake Weed": 1016,
                "Negatives": 9106}


def _save(fig: Figure, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def eda_table(per_class: pd.DataFrame) -> pd.DataFrame:
    """Bảng số ảnh theo lớp (train/val/test/tổng) đối chiếu Table 1 của bài báo, kèm tỉ trọng."""
    t = per_class.copy()
    t["paper_table1"] = [PAPER_TABLE1[c] for c in t.index]
    t["diff_vs_paper"] = t["total"] - t["paper_table1"]
    t["share_%"] = 100 * t["total"] / t["total"].sum()
    for s in D.SPLITS:
        t[f"{s}_%_of_class"] = 100 * t[s] / t["total"]
    return t


def plot_class_distribution(per_class: pd.DataFrame, path) -> dict:
    """Biểu đồ cột số ảnh mỗi lớp theo tập (train/val/test). Trả về tỉ lệ lớp lớn nhất / nhỏ nhất."""
    fig = Figure(figsize=(11, 4.5), dpi=120)
    ax = fig.subplots()
    x = np.arange(len(per_class))
    for k, s in enumerate(D.SPLITS):
        bars = ax.bar(x + (k - 1) * 0.27, per_class[s], 0.27, label=s)
        ax.bar_label(bars, fontsize=6, rotation=90, padding=2)
    ax.set_xticks(x, per_class.index, rotation=30, ha="right")
    ax.set(ylabel="số ảnh", yscale="log")
    tot = per_class["total"]
    ratio = tot.max() / tot.min()
    ax.set_title(f"DeepWeeds fold 0: phân bố lớp (thang log). Lớn nhất/nhỏ nhất = "
                 f"{tot.idxmax()} {tot.max()} / {tot.idxmin()} {tot.min()} = {ratio:.1f} lần")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)
    return {"largest": tot.idxmax(), "smallest": tot.idxmin(), "ratio_max_min": float(ratio)}


def plot_samples(df: pd.DataFrame, images_dir, path, n_per_class: int = 4, seed: int = 0) -> Path:
    """Lưới ảnh mẫu: mỗi hàng một lớp, `n_per_class` ảnh ngẫu nhiên (cố định theo seed)."""
    fig = Figure(figsize=(2.1 * n_per_class, 2.1 * D.NUM_CLASSES), dpi=100)
    axes = fig.subplots(D.NUM_CLASSES, n_per_class)
    for c, name in enumerate(D.CLASS_NAMES):
        rows = df[df["Label"] == c].sample(n_per_class, random_state=seed)
        for j, fn in enumerate(rows["Filename"]):
            ax = axes[c, j]
            ax.imshow(Image.open(Path(images_dir) / fn).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.suptitle("Ảnh mẫu theo lớp (train)")
    fig.tight_layout()
    return _save(fig, path)


def image_stats(df: pd.DataFrame, images_dir, n: int = 500, seed: int = 0) -> dict:
    """Kích thước, chế độ màu và mean/std từng kênh (trên [0, 1]) của n ảnh ngẫu nhiên."""
    rows = df.sample(min(n, len(df)), random_state=seed)
    sizes, modes, s1, s2, cnt = {}, {}, np.zeros(3), np.zeros(3), 0
    for fn in rows["Filename"]:
        with Image.open(Path(images_dir) / fn) as im:
            sizes[im.size] = sizes.get(im.size, 0) + 1
            modes[im.mode] = modes.get(im.mode, 0) + 1
            a = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        s1 += a.reshape(-1, 3).sum(0)
        s2 += (a.reshape(-1, 3) ** 2).sum(0)
        cnt += a.shape[0] * a.shape[1]
    mean = s1 / cnt
    std = np.sqrt(s2 / cnt - mean ** 2)
    return {"n_sampled": len(rows), "sizes": {f"{w}x{h}": k for (w, h), k in sizes.items()}, "modes": modes,
            "mean_rgb": mean.round(4).tolist(), "std_rgb": std.round(4).tolist(),
            "imagenet_mean": list(D.IMAGENET_MEAN), "imagenet_std": list(D.IMAGENET_STD)}


def plot_augmented(df: pd.DataFrame, images_dir, aug: str, path, n: int = 6, seed: int = 0,
                   mean=D.IMAGENET_MEAN, std=D.IMAGENET_STD, img_size: int = 224) -> Path:
    """Hàng trên: ảnh gốc; hàng dưới: ảnh sau augmentation `aug` (đã giải chuẩn hoá), kèm nhãn.
    Mỗi cột lấy một ảnh của một lớp khác nhau để thấy rõ nhãn đi theo đúng ảnh."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    classes = rng.permutation(D.NUM_CLASSES)[:n]
    rows = pd.concat([df[df["Label"] == c].sample(1, random_state=seed) for c in classes])
    tf = D.build_transforms(True, img_size, aug, mean, std)
    fig = Figure(figsize=(2.3 * n, 5.4), dpi=100)
    axes = fig.subplots(2, n)
    for j, (fn, lab) in enumerate(zip(rows["Filename"], rows["Label"])):
        img = Image.open(Path(images_dir) / fn).convert("RGB")
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"{lab}: {D.CLASS_NAMES[lab]}", fontsize=8)
        x = D.denormalize(tf(img), mean, std).permute(1, 2, 0).numpy()
        axes[1, j].imshow(x)
        axes[1, j].set_xlabel(f"sau '{aug}', nhãn {lab}", fontsize=8)
        for ax in axes[:, j]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Kiểm tra ảnh sau augmentation '{aug}' (đã giải chuẩn hoá): ảnh và nhãn phải khớp")
    fig.tight_layout()
    return _save(fig, path)


def plot_mix(df: pd.DataFrame, images_dir, mode: str, path, n: int = 6, seed: int = 0,
             mean=D.IMAGENET_MEAN, std=D.IMAGENET_STD) -> Path:
    """Ảnh sau Mixup/CutMix kèm hai nhãn và lam (phần của nhãn thứ nhất)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    rows = df.sample(n, random_state=seed)
    tf = D.build_transforms(False, 224, mean=mean, std=std)
    x = torch.stack([tf(Image.open(Path(images_dir) / fn).convert("RGB")) for fn in rows["Filename"]])
    y = torch.as_tensor(rows["Label"].to_numpy().copy())
    xm, (ya, yb, lam) = L.mix_batch(x, y, 1.0, mode)
    fig = Figure(figsize=(2.3 * n, 2.8), dpi=100)
    axes = fig.subplots(1, n)
    for j in range(n):
        axes[j].imshow(D.denormalize(xm[j], mean, std).permute(1, 2, 0).numpy())
        axes[j].set_title(f"{lam:.2f}·{D.CLASS_NAMES[ya[j]][:12]}\n+{1 - lam:.2f}·{D.CLASS_NAMES[yb[j]][:12]}",
                          fontsize=7)
        axes[j].set_xticks([])
        axes[j].set_yticks([])
    fig.suptitle(f"{mode}: lam = {lam:.3f} (tính theo diện tích thật của hộp với CutMix)")
    fig.tight_layout()
    return _save(fig, path)


@torch.no_grad()
def initial_loss(model, loader, device, n_batches: int = 10) -> dict:
    """Loss CE ban đầu (head mới, chưa train) trên vài batch val, ở chế độ eval. Kỳ vọng ~ ln 9 = 2,197."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    tot, n = 0.0, 0
    for i, (x, y, _) in enumerate(loader):
        if i >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        tot += ce(model(x).float(), y).item()
        n += y.size(0)
    return {"initial_loss": tot / n, "ln9": math.log(9), "n_images": n}


def overfit_one_batch(model, x, y, device, steps: int = 100, lr: float = 1e-3, path=None) -> dict:
    """Huấn luyện liên tục trên MỘT batch nhỏ cố định (không augmentation) tới loss gần 0.

    Không giảm được loss thì lỗi nằm ở code/model (Karpathy; slide trang 59). CÓ HUẤN LUYỆN: chạy trên GPU.
    """
    model.to(device)
    M.set_train_mode(model)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ce = nn.CrossEntropyLoss()
    losses = []
    for _ in range(steps):
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            loss = ce(model(x).float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        acc = (model(x).argmax(1) == y).float().mean().item()
    if path:
        fig = Figure(figsize=(6, 3.5), dpi=120)
        ax = fig.subplots()
        ax.plot(np.arange(1, steps + 1), losses)
        ax.axhline(math.log(9), ls=":", color="gray", label="ln 9 = 2,197")
        ax.set(yscale="log", xlabel="bước", ylabel="loss CE (log)",
               title=f"Overfit 1 batch ({len(y)} ảnh): loss cuối {losses[-1]:.4f}, acc {acc:.2f}")
        ax.legend()
        ax.grid(alpha=0.3)
        _save(fig, path)
    return {"first_loss": losses[0], "final_loss": losses[-1], "batch_acc_eval": acc, "steps": steps}


def check_modes(backbone: str) -> dict:
    """Kiểm tra chế độ train/eval: evaluate để model ở eval; backbone đóng băng giữ BN ở eval khi train."""
    import train as TR
    net = M.build_model(backbone, pretrained=False, init="frozen")
    M.set_train_mode(net)
    bn = [m for m in net.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    res = {"bn_layers": len(bn),
           "frozen_bn_eval_during_train": all(not m.training for m in bn) if bn else None,
           "head_train_during_train": net.get_classifier().training}
    loader = [(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]), ["a", "b"])]
    TR.evaluate(net, loader, None, torch.device("cpu"), amp=False, channels_last=False)
    res["all_eval_after_evaluate"] = not any(m.training for m in net.modules())
    return res
