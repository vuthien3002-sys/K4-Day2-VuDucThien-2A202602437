"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50.a1_in1k seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc, để cùng
định nghĩa với lúc chấm. File dự đoán ghi bằng eval.save_predictions.

Mỗi lần chạy lưu ở <out_dir>/<exp_id>/seed<k>/: config.json (kèm version thư viện, GPU), history.csv,
best.pt (trọng số tốt nhất theo macro-F1 val), val_logits.npy, result.json. Ảnh đường cong ở
<curves_dir>/<exp_id>_<mota>.png, file dự đoán ở <pred_dir>/<exp_id>_seed<k>_<split>.csv.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import dataset as D
import losses as L
import model as M

# eval.py nằm ở gốc repo (cha của submissions/...). Thêm vào sys.path nếu notebook chưa thêm.
_ROOT = next((p for p in Path(__file__).resolve().parents if (p / "eval.py").exists()), None)
if _ROOT is not None and str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))
from eval import compute_metrics, save_predictions  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves/<exp_id>_<desc>.png (rỗng: tên backbone)
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    pretrained: bool = True
    drop_rate: float = 0.0
    drop_path_rate: float | None = None
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    eval_crop_pct: float = 0.875      # val/test: Resize(img_size / crop_pct) + CenterCrop(img_size)
    aug: str = "basic"                # basic | color | trivial | randaug | vflip
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    optimizer: str = "adamw"          # adamw | sgd
    momentum: float = 0.9             # chỉ dùng cho sgd
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    grad_clip: float | None = None
    ema_decay: float | None = None
    amp: bool = True
    channels_last: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"        # ảnh đường cong training
    ckpt_dir: str | None = None       # nơi ghi checkpoint TRONG lúc train (đĩa cục bộ); None: dùng run_dir
    resume: bool = True               # bỏ qua lần chạy đã xong (có result.json), tiếp tục từ last.pt nếu có
    debug_subset: int | None = None   # CHỈ để chạy thử code: lấy N ảnh đầu mỗi tập. Thí nghiệm thật: None
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    """<curves_dir>/<exp_id>_<desc>.png, thêm _seed<k> khi seed khác 0 để mỗi lần chạy một ảnh."""
    desc = cfg.desc or cfg.backbone.split(".")[0]
    suffix = f"_seed{cfg.seed}" if cfg.seed != 0 else ""
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{desc}{suffix}.png"


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU và CUDA). Seed cho worker DataLoader nằm ở dataset.make_loader.

    Mức tái lập: cùng seed cho cùng khởi tạo head, cùng thứ tự batch và cùng augmentation. Để nhanh,
    cudnn.benchmark = True và không bật thuật toán tất định, nên hai lần chạy cùng seed trên GPU có thể
    lệch rất nhỏ (cỡ nhiễu số học), không lệch về cách chia dữ liệu (S5).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def env_info() -> dict:
    """Version thư viện và phần cứng, ghi vào config.json của mỗi lần chạy."""
    import timm
    import torchvision
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torchvision": torchvision.__version__, "timm": timm.__version__, "numpy": np.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


def build_optimizer(model, cfg: Config):
    """AdamW (hoặc SGD + Nesterov momentum) với các nhóm tham số của model.param_groups."""
    groups = M.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr_backbone)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr_backbone, momentum=cfg.momentum, nesterov=True)
    raise ValueError(f"optimizer={cfg.optimizer!r} không hợp lệ (adamw | sgd)")


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0 (slide trang 55), cập nhật theo BƯỚC (iteration).

    Hệ số nhân LR: (step + 1) / warmup trong warmup, sau đó 0.5 * (1 + cos(pi * tiến độ)). Mỗi nhóm
    tham số giữ tỉ lệ LR riêng (backbone / head).
    """
    total = max(1, cfg.epochs * steps_per_epoch)
    warmup = min(total, int(round(cfg.warmup_epochs * steps_per_epoch)))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    - Giữ một bản sao riêng (`self.module`, luôn ở eval) để đánh giá bằng trọng số EMA.
    - Decay có warmup: d_t = min(decay, (1 + t) / (10 + t)), để các bước đầu không bị trọng số
      ngẫu nhiên của head mới kéo lại quá lâu.
    - Buffer BatchNorm (running_mean/var) cũng được lấy trung bình động như trọng số; buffer số nguyên
      (num_batches_tracked) được chép thẳng. Nhờ vậy thống kê BN khớp với trọng số EMA.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        src = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src[k].detach(), alpha=1.0 - d)
            else:
                v.copy_(src[k])


def _to_device(x, y, device, channels_last: bool):
    x = x.to(device, non_blocking=True)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return x, y.to(device, non_blocking=True)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc", "lr_steps"}.

    - set_train_mode: model.train() nhưng giữ BN của backbone đóng băng ở eval
    - Mixup/CutMix: mix_batch rồi mixed_loss; train_acc khi đó không có nghĩa nên để NaN
    - AMP (autocast fp16 + GradScaler), clip gradient nếu cfg.grad_clip, scheduler theo bước, EMA
    """
    M.set_train_mode(model)
    use_amp = cfg.amp and device.type == "cuda"
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), device=device)
    n, lr_steps = 0, []
    for x, y, _ in loader:
        x, y = _to_device(x, y, device, cfg.channels_last)
        targets = None
        if cfg.mix:
            x, targets = L.mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = L.mixed_loss(criterion, logits, targets) if targets else criterion(logits, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        loss_sum += loss.detach().float() * y.size(0)
        if targets is None:
            correct += (logits.argmax(1) == y).sum()
        n += y.size(0)
        lr_steps.append(optimizer.param_groups[0]["lr"])
    return {"train_loss": loss_sum.item() / n,
            "train_acc": correct.item() / n if not cfg.mix else float("nan"),
            "lr_steps": lr_steps}


def evaluate(model, loader, criterion, device, amp: bool = True, channels_last: bool = True):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient.

    Trả về (filenames: list[str], y_true: ndarray[N], logits: ndarray[N, 9] float32, loss: float),
    giữ đúng thứ tự của loader. `criterion` None thì dùng CE thường (để val loss so sánh được giữa
    các lần chạy dùng loss huấn luyện khác nhau).
    """
    criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    use_amp = amp and device.type == "cuda"
    names, ys, outs = [], [], []
    loss_sum, n = 0.0, 0
    with torch.inference_mode():
        for x, y, f in loader:
            x, y = _to_device(x, y, device, channels_last)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            logits = logits.float()
            loss_sum += criterion(logits, y).item() * y.size(0)
            n += y.size(0)
            names.extend(f)
            ys.append(y.cpu())
            outs.append(logits.cpu())
    return names, torch.cat(ys).numpy(), torch.cat(outs).numpy(), loss_sum / n


def softmax(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def metrics_from_logits(y_true, logits) -> dict:
    """Chỉ số theo đúng định nghĩa của eval.py (macro-F1, top-1, ECE, theo lớp...)."""
    probs = softmax(logits)
    return compute_metrics(np.asarray(y_true), probs.argmax(1), probs)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps=None) -> None:
    """Vẽ đường cong training -> curves/<exp_id>_<mota>.png (GUIDE.md mục 6.2).

    Ba ô: loss train/val theo epoch; macro-F1 val, top-1 val (và train acc nếu có) theo epoch, đánh
    dấu epoch tốt nhất; LR theo bước (thấy warmup + cosine).
    """
    from matplotlib.figure import Figure  # không dùng pyplot để không đổi backend của notebook

    h = pd.DataFrame(history)
    best = h.loc[h["val_macro_f1"].idxmax()]
    ncol = 3 if lr_steps else 2
    fig = Figure(figsize=(5 * ncol, 4), dpi=120)
    axes = fig.subplots(1, ncol)
    ax = axes[0]
    ax.plot(h["epoch"], h["train_loss"], "o-", label="train loss")
    ax.plot(h["epoch"], h["val_loss"], "s-", label="val loss (CE)")
    ax.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(h["epoch"], h["val_macro_f1"], "o-", label="val macro-F1")
    ax.plot(h["epoch"], h["val_top1"], "s-", label="val top-1")
    if h["train_acc"].notna().any():
        ax.plot(h["epoch"], h["train_acc"], "^--", label="train acc (có augmentation)")
    ax.axvline(best["epoch"], color="gray", ls=":",
               label=f"best epoch {int(best['epoch'])}: F1 {best['val_macro_f1']:.4f}")
    ax.set(xlabel="epoch", ylabel="điểm", title="Chỉ số trên val")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    if lr_steps:
        ax = axes[2]
        ax.plot(np.arange(1, len(lr_steps) + 1), lr_steps)
        ax.set(xlabel="bước (iteration)", ylabel="LR backbone", title="Lịch LR")
        ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)


def load_trained(cfg: Config, device=None):
    """Dựng lại model của một lần chạy đã xong và nạp best.pt (trọng số EMA nếu run dùng EMA)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = M.build_model(cfg.backbone, pretrained=False, num_classes=D.NUM_CLASSES,
                          drop_rate=cfg.drop_rate, init="finetune", drop_path_rate=cfg.drop_path_rate)
    state = torch.load(run_dir(cfg) / "best.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)
    return model


def model_norm(model) -> tuple[tuple, tuple]:
    """mean/std chuẩn hoá theo pretrained_cfg của timm (đúng cấu hình của trọng số đang dùng)."""
    pc = model.pretrained_cfg
    return tuple(pc.get("mean", D.IMAGENET_MEAN)), tuple(pc.get("std", D.IMAGENET_STD))


def predict_test_once(cfg: Config, device=None) -> dict:
    """Bước 4: chạy TEST đúng MỘT lần cho lần chạy này (1 view), lưu logit và file dự đoán.

    Từ chối chạy lại nếu file dự đoán test đã tồn tại (quy tắc: test một lần mỗi seed).
    """
    out_csv = pred_path(cfg, "test")
    if out_csv.exists():
        raise RuntimeError(f"{out_csv} đã có: test chỉ được chạy một lần cho mỗi seed")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    if cfg.debug_subset:
        test_df = test_df.iloc[:cfg.debug_subset]
    model = load_trained(cfg, device)
    mean, std = model_norm(model)
    tf = D.build_transforms(False, cfg.img_size, mean=mean, std=std, crop_pct=cfg.eval_crop_pct)
    loader = D.make_loader(test_df, cfg.images_dir, tf, cfg.batch_size * 2, False,
                           num_workers=cfg.num_workers, seed=cfg.seed)
    names, y, logits, _ = evaluate(model, loader, None, device, cfg.amp, cfg.channels_last)
    np.save(run_dir(cfg) / "test_logits.npy", logits)
    save_predictions(out_csv, names, y, softmax(logits))
    print(f"[{cfg.exp_id} seed{cfg.seed}] đã ghi {out_csv} (test chạy 1 lần)")
    return {"test_predictions": str(out_csv), "test_predicted_at": time.strftime("%Y-%m-%d %H:%M:%S")}


def _quick_latency(model, cfg: Config, device) -> float | None:
    """Độ trễ sơ bộ ở Bước 1 (batch 1, fp32, p50 ms). Đo kỹ p50/p95/p99 ở Bước 3 bằng benchmark.py."""
    if device.type != "cuda":
        return None
    from benchmark import latency_report
    return latency_report(model, 1, cfg.img_size, "fp32", "cuda", warmup=10, iters=50)["p50"]


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

      1. set_seed; tạo run_dir; ghi config.json (cấu hình + version thư viện + GPU)
      2. load_split + check_split (dừng nếu vi phạm S1-S4)
      3. model (tag trọng số, #params, GMAC), transform theo mean/std của trọng số, loader train/val
      4. criterion, optimizer (nhóm tham số), scheduler, GradScaler, EMA
      5. mỗi epoch: train_one_epoch -> evaluate(val) -> history; lưu best.pt theo MACRO-F1 VAL
         (chỉ cập nhật khi tốt hơn hẳn, nên hoà thì giữ epoch sớm hơn); lưu last.pt để tiếp tục
      6. nạp best.pt, lưu val_logits.npy và predictions/<exp_id>_seed<k>_val.csv
      7. NẾU cfg.save_test_predictions (chỉ ở Bước 4): predict_test_once
      8. history.csv, ảnh đường cong, result.json (best_epoch, chỉ số val, thời gian/epoch, params, GMAC)
    Quy tắc: KHÔNG dùng test để chọn checkpoint hay bất kỳ quyết định nào (README.md, S4).
    Lần chạy đã có result.json thì bỏ qua (chỉ chạy thêm test nếu được yêu cầu và chưa có).
    """
    out = run_dir(cfg)
    result_file = out / "result.json"
    if cfg.resume and result_file.exists():
        res = json.loads(result_file.read_text(encoding="utf-8"))
        if cfg.save_test_predictions and not pred_path(cfg, "test").exists():
            res.update(predict_test_once(cfg))
            result_file.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{cfg.exp_id} seed{cfg.seed}] đã xong trước đó: val macro-F1 {res['val_macro_f1']:.4f}, bỏ qua")
        return res

    # 1. seed, thư mục, cấu hình
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Path(cfg.ckpt_dir) / cfg.exp_id / f"seed{cfg.seed}" if cfg.ckpt_dir else out
    ckpt.mkdir(parents=True, exist_ok=True)
    env = env_info()
    (out / "config.json").write_text(json.dumps({**asdict(cfg), "env": env}, indent=2, ensure_ascii=False),
                                     encoding="utf-8")

    # 2. dữ liệu: đọc và kiểm tra split (trên toàn bộ fold, trước khi lấy tập con debug)
    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    D.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)
    if cfg.debug_subset:
        train_df, val_df = train_df.iloc[:cfg.debug_subset], val_df.iloc[:cfg.debug_subset]

    # 3. model, transform, loader
    model = M.build_model(cfg.backbone, cfg.pretrained, D.NUM_CLASSES, cfg.drop_rate, cfg.init,
                          cfg.drop_path_rate)
    params_m, gmac = M.count_params(model), M.count_gmacs(model, cfg.img_size)
    mean, std = model_norm(model)
    model.to(device)
    if cfg.channels_last:
        model.to(memory_format=torch.channels_last)
    train_tf = D.build_transforms(True, cfg.img_size, cfg.aug, mean, std)
    eval_tf = D.build_transforms(False, cfg.img_size, mean=mean, std=std, crop_pct=cfg.eval_crop_pct)
    train_loader = D.make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, True,
                                 cfg.sampler, cfg.num_workers, cfg.seed)
    val_loader = D.make_loader(val_df, cfg.images_dir, eval_tf, cfg.batch_size * 2, False,
                               num_workers=cfg.num_workers, seed=cfg.seed)

    # 4. loss, optimizer, scheduler, AMP, EMA
    weight = None
    if cfg.loss == "ce_weighted":
        counts = np.bincount(train_df["Label"], minlength=D.NUM_CLASSES)  # chỉ số liệu của TRAIN
        weight = L.class_weights(counts, cfg.class_weight_beta or 0.0)
    if cfg.loss == "ls" and cfg.label_smoothing <= 0:
        raise ValueError("loss='ls' cần label_smoothing > 0 (ví dụ 0.1)")
    criterion = L.build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma,
                                  weight=weight).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None

    history, lr_steps = [], []
    best = {"val_macro_f1": -1.0, "epoch": 0}
    start = 1
    last_file = ckpt / "last.pt"
    if cfg.resume and last_file.exists():
        st = torch.load(last_file, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scheduler.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        if ema is not None:
            ema.module.load_state_dict(st["ema"])
            ema.updates = st["ema_updates"]
        history, lr_steps, best, start = st["history"], st["lr_steps"], st["best"], st["epoch"] + 1
        print(f"[{cfg.exp_id} seed{cfg.seed}] tiếp tục từ epoch {start}")

    print(f"[{cfg.exp_id} seed{cfg.seed}] {model.weight_tag} | {params_m:.1f}M tham số | {gmac:.2f} GMAC | "
          f"{len(train_df)} ảnh train, {len(val_df)} ảnh val | {env['gpu']}")

    # 5. vòng epoch
    for epoch in range(start, cfg.epochs + 1):
        t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_s = time.perf_counter() - t0
        eval_model = ema.module if ema is not None else model
        _, y_val, logits, val_loss = evaluate(eval_model, val_loader, None, device, cfg.amp, cfg.channels_last)
        m = metrics_from_logits(y_val, logits)
        row = {"epoch": epoch, "train_loss": tr["train_loss"], "train_acc": tr["train_acc"],
               "val_loss": val_loss, "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
               "val_balanced_acc": m["balanced_acc"], "lr": tr["lr_steps"][-1],
               "train_time_s": train_s, "epoch_time_s": time.perf_counter() - t0}
        history.append(row)
        lr_steps += tr["lr_steps"]
        if m["macro_f1"] > best["val_macro_f1"]:
            best = {"val_macro_f1": m["macro_f1"], "epoch": epoch}
            torch.save(eval_model.state_dict(), ckpt / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "ema": ema.module.state_dict() if ema is not None else None,
                    "ema_updates": ema.updates if ema is not None else 0,
                    "history": history, "lr_steps": lr_steps, "best": best, "epoch": epoch}, last_file)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(f"  epoch {epoch:2d}/{cfg.epochs} | train loss {row['train_loss']:.4f} | val loss {val_loss:.4f} | "
              f"val F1 {m['macro_f1']:.4f} | val top-1 {m['top1']:.4f} | {train_s:.0f}s")

    # 6. nạp checkpoint tốt nhất, lưu logit và dự đoán val
    if ckpt != out:
        shutil.copy2(ckpt / "best.pt", out / "best.pt")
    del optimizer, scheduler, ema
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    names, y_val, logits, val_loss = evaluate(model, val_loader, None, device, cfg.amp, cfg.channels_last)
    np.save(out / "val_logits.npy", logits)
    save_predictions(pred_path(cfg, "val"), names, y_val, softmax(logits))
    m = metrics_from_logits(y_val, logits)

    # 8. tóm tắt
    h = pd.DataFrame(history)
    result = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weight_tag": model.weight_tag,
        "init": cfg.init, "aug": cfg.aug, "mix": cfg.mix, "loss": cfg.loss, "sampler": cfg.sampler,
        "optimizer": cfg.optimizer, "ema_decay": cfg.ema_decay, "img_size": cfg.img_size,
        "epochs": cfg.epochs, "params_m": params_m, "gmac": gmac,
        "best_epoch": int(best["epoch"]), "val_loss": val_loss,
        "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_balanced_acc": m["balanced_acc"],
        "val_ece": m["ece"], "val_nll": m["nll"],
        "val_f1_per_class": m["f1"].tolist(), "val_recall_per_class": m["recall"].tolist(),
        "train_s_per_epoch": float(h["train_time_s"].mean()),
        "total_train_min": float(h["epoch_time_s"].sum() / 60),
        "latency_b1_fp32_p50_ms": _quick_latency(model, cfg, device),
        "curve": str(curve_path(cfg)), "env": env,
    }
    plot_curves(history, curve_path(cfg),
                f"{cfg.exp_id} | {model.weight_tag} | seed {cfg.seed} | best epoch {result['best_epoch']}",
                lr_steps)

    # 7. test chỉ ở Bước 4
    if cfg.save_test_predictions:
        result.update(predict_test_once(cfg, device))
    result_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    last_file.unlink(missing_ok=True)
    if ckpt != out:
        (ckpt / "best.pt").unlink(missing_ok=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[{cfg.exp_id} seed{cfg.seed}] xong: best epoch {result['best_epoch']}, val macro-F1 "
          f"{result['val_macro_f1']:.4f}, val top-1 {result['val_top1']:.4f}, "
          f"{result['train_s_per_epoch']:.0f}s/epoch")
    return result


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    types = {f.name: str(f.type) for f in fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'{pair}' không có dạng KEY=VALUE")
        key, value = pair.split("=", 1)
        if key not in types:
            raise ValueError(f"Config không có trường '{key}'. Các trường: {sorted(types)}")
        ftype = types[key]
        if value.lower() in ("none", "null"):
            if "None" not in ftype:
                raise ValueError(f"'{key}' không nhận None")
            out[key] = None
            continue
        base = [t.strip() for t in ftype.split("|") if t.strip() != "None"][0]
        if base == "bool":
            if value.lower() not in ("true", "false", "1", "0", "yes", "no"):
                raise ValueError(f"'{key}' cần true/false, nhận '{value}'")
            out[key] = value.lower() in ("true", "1", "yes")
        elif base == "int":
            out[key] = int(value)
        elif base == "float":
            out[key] = float(value)
        else:
            out[key] = value
    return out


def main(argv=None) -> None:
    """Điểm vào dòng lệnh: `python train.py --set exp_id=B01 backbone=resnet50.a1_in1k seed=0`."""
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình Lab Day 2")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="ghi đè trường của Config")
    args = ap.parse_args(argv)
    res = run(Config(**parse_overrides(args.set)))
    keys = ("exp_id", "seed", "weight_tag", "best_epoch", "val_macro_f1", "val_top1", "train_s_per_epoch")
    print(json.dumps({k: res[k] for k in keys}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
