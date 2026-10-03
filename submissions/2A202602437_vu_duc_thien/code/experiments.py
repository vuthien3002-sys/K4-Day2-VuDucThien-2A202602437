"""experiments.py - Bước 3 (phương pháp suy luận, trên VAL) và Bước 4 (chung kết, TEST một lần mỗi seed).

Bước 3 không huấn luyện lại: nạp best.pt của các lần chạy đã xong, chạy các cách suy luận trên VAL,
đo độ trễ bằng benchmark.py. Bước 4 ghi file dự đoán test đúng định dạng eval.py, từ chối ghi đè.

Tên phương pháp (`method`) dùng chung cho Bước 3 và Bước 4:
    "1view" · "hflip" · "5crop" · "10crop" · "3scale" · "res<s>" (ví dụ "res288": Resize(s/0.875) + CenterCrop(s))
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib.figure import Figure
from PIL import Image

import benchmark as B
import dataset as D
import inference as I
from train import Config, load_trained, model_norm, run_dir

from eval import compute_metrics, save_predictions  # noqa: E402  (train.py đã thêm gốc repo vào sys.path)

HARD = {"Chinee Apple": 0, "Snake Weed": 7}

# method -> (loader, hàm tạo view, kích thước ảnh của từng view để đo độ trễ)
#   loader "crop": Resize(256) + CenterCrop(224) như lúc val;  "full": giữ nguyên ảnh 256x256
VIEW_METHODS = {
    "1view": ("crop", None, [224]),
    "hflip": ("crop", I.views_hflip_pair, [224, 224]),
    "5crop": ("full", lambda x: I.views_multicrop(x, 224), [224] * 5),
    "10crop": ("full", lambda x: I.views_multicrop(x, 224, flip=True), [224] * 10),
    "3scale": ("full", lambda x: I.views_multiscale(x, [224, 256, 288]), [224, 256, 288]),
}


def method_sizes(method: str) -> list[int]:
    return [int(method[3:])] if method.startswith("res") else VIEW_METHODS[method][2]


def summarize(y, probs) -> dict:
    """Chỉ số theo eval.compute_metrics, kèm F1/recall hai lớp khó."""
    m = compute_metrics(np.asarray(y), probs.argmax(1), probs)
    out = {"macro_f1": m["macro_f1"], "top1": m["top1"], "balanced_acc": m["balanced_acc"],
           "ece": m["ece"], "nll": m["nll"]}
    for name, i in HARD.items():
        key = name.split()[0].lower()
        out[f"f1_{key}"], out[f"recall_{key}"] = float(m["f1"][i]), float(m["recall"][i])
    return out


def _split_df(cfg: Config, split: str) -> pd.DataFrame:
    _, va, te = D.load_split(cfg.labels_dir, cfg.fold)
    df = {"val": va, "test": te}[split]
    return df.iloc[:cfg.debug_subset] if cfg.debug_subset else df


def _loader(cfg: Config, df, mean, std, kind: str = "crop", img_size: int = 224):
    if kind == "crop":
        tf = D.build_transforms(False, img_size, mean=mean, std=std, crop_pct=cfg.eval_crop_pct)
    else:  # "full": cả ảnh 256x256, nguồn cho multi-crop / multi-scale
        tf = D.build_transforms(False, 256, mean=mean, std=std, crop_pct=1.0)
    return D.make_loader(df, cfg.images_dir, tf, cfg.batch_size, False, num_workers=cfg.num_workers, seed=cfg.seed)


def view_logits(model, cfg: Config, split: str, method: str, device):
    """Logit từng view của `method` trên `split`. Trả về (filenames, y_true, [K mảng (N, 9)])."""
    mean, std = model_norm(model)
    df = _split_df(cfg, split)
    if method.startswith("res"):
        loader, fn = _loader(cfg, df, mean, std, "crop", int(method[3:])), None
    else:
        kind, fn, _ = VIEW_METHODS[method]
        loader = _loader(cfg, df, mean, std, kind)
    names, y, logits = I.predict_logits(model, loader, device, view=fn, amp=cfg.amp,
                                        channels_last=cfg.channels_last)
    return names, y, logits if isinstance(logits, list) else [logits]


def latency_views(model, sizes, batch: int = 1, dtype: str = "fp32", iters: int = 100) -> dict:
    """p50/p95/p99 (ms) cho một lượt suy luận gồm các view có kích thước `sizes`, chạy tuần tự.

    Đo đúng cách: warmup 10, synchronize trước/sau, >= 50 lần (benchmark.bench). Không tính tiền xử lý.
    """
    dev = next(model.parameters()).device
    m = B._prepare(model, dtype, str(dev), True)
    xs = [torch.randn(batch, 3, s, s, device=dev, dtype=torch.float16 if dtype == "fp16" else torch.float32)
          .contiguous(memory_format=torch.channels_last) for s in sizes]

    def fn():
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=dtype == "amp"):
            for x in xs:
                m(x)

    r = B.bench(fn, 10, iters, torch.cuda.synchronize if dev.type == "cuda" else None)
    return {"p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "images_per_s": batch / (r["p50"] / 1000)}


def crossfit_ece(logits, y, seed: int = 0) -> float:
    """ECE sau temperature scaling khi khớp T trên một nửa VAL và đo trên nửa còn lại (đổi vai, lấy trung
    bình): tránh lạc quan khi vừa khớp vừa đo trên cùng ảnh. Vẫn chỉ dùng VAL."""
    idx = np.random.default_rng(seed).permutation(len(y))
    halves = (idx[: len(y) // 2], idx[len(y) // 2:])
    eces = []
    for a, b in (halves, halves[::-1]):
        T = I.fit_temperature(logits[a], y[a])
        eces.append(summarize(y[b], I.apply_temperature(logits[b], T))["ece"])
    return float(np.mean(eces))


def inference_suite(cfg: Config, device=None, ensembles: dict | None = None,
                    ema_pair: tuple[Config, Config] | None = None, resolutions=(256, 288, 320),
                    lat_iters: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bước 3 trên VAL cho mô hình của `cfg` (best.pt). Trả về (bảng Inference, bảng Latency).

    I00 1 view · I01 TTA lật ngang · I02 multi-crop / multi-scale · I03 gộp logit (so với gộp xác suất) ·
    I04 dò độ phân giải · I05 ensemble (từ val_logits.npy đã lưu) · I06 trọng số EMA ·
    I07 temperature scaling + ECE · I08 gộp BN / FP16. Cột `selectable` đánh dấu các cách suy luận áp
    dụng được cho chính mô hình này (dùng để chọn phương pháp cho Bước 4).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.cuda.synchronize if device.type == "cuda" else None
    model = load_trained(cfg, device)
    name = f"{cfg.exp_id}/seed{cfg.seed}"
    rows, lat_rows = [], []

    def add(exp_id, label, probs, y, k, sizes=None, lat=None, model_used=name, method=None, space=None, **extra):
        lat = lat or (latency_views(model, sizes, iters=lat_iters) if sizes else {})
        rows.append({"exp_id": exp_id, "method_label": label, "method": method, "space": space,
                     "selectable": method is not None, "model": model_used, "K": k, **summarize(y, probs),
                     **{f"lat_b1_{q}_ms": lat.get(q) for q in ("p50", "p95", "p99")}, **extra})

    # I00, I01, I02 (gộp xác suất) và I03 (gộp logit) của cùng các view
    cache = {}
    for method, code in [("1view", "I00"), ("hflip", "I01"), ("5crop", "I02a"), ("10crop", "I02b"),
                         ("3scale", "I02c")]:
        try:
            _, y, views = view_logits(model, cfg, "val", method, device)
        except Exception as e:  # ví dụ Swin không nhận ảnh khác kích thước lúc train
            print(f"{code} {method}: bỏ qua ({type(e).__name__}: {e})")
            continue
        cache[method] = (y, views)
        label = "1 view (mốc)" if method == "1view" else f"{method}, gộp xác suất"
        add(code, label, I.aggregate_views(views, "prob"), y, len(views), method_sizes(method),
            method=method, space="prob")
        if len(views) > 1:
            lat = {q: rows[-1][f"lat_b1_{q}_ms"] for q in ("p50", "p95", "p99")}
            add(f"I03_{method}", f"{method}, gộp logit", I.aggregate_views(views, "logit"), y, len(views),
                lat=lat, method=method, space="logit")

    # I04: dò độ phân giải kiểm tra
    for s in resolutions:
        try:
            _, y, views = view_logits(model, cfg, "val", f"res{s}", device)
            add(f"I04_{s}", f"độ phân giải kiểm tra {s}", I.aggregate_views(views, "prob"), y, 1, [s],
                method=f"res{s}", space="prob")
        except Exception as e:
            print(f"I04 {s}: bỏ qua ({type(e).__name__}: {e})")

    y_val, views_1 = cache["1view"]

    # I05: ensemble (xác suất trung bình, 1 view mỗi mô hình; logit val đã lưu khi train)
    for code, cfgs in (ensembles or {}).items():
        probs = [I.apply_temperature(np.load(run_dir(c) / "val_logits.npy"), 1.0) for c in cfgs]
        members = [load_trained(c, device) for c in cfgs]
        x = torch.randn(1, 3, 224, 224, device=device).contiguous(memory_format=torch.channels_last)

        def fn(members=members, x=x):
            with torch.inference_mode():
                for mm in members:
                    mm(x)

        lat = B.bench(fn, 10, lat_iters, sync)
        add(code, f"ensemble {len(cfgs)} mô hình", I.ensemble_probs(probs), y_val, len(cfgs), lat=lat,
            model_used=" + ".join(f"{c.exp_id}/seed{c.seed}" for c in cfgs))
        del members
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # I06: cùng công thức, chỉ khác có/không EMA (không tốn thêm khi suy luận)
    if ema_pair is not None:
        for c in ema_pair:
            logits = np.load(run_dir(c) / "val_logits.npy")
            tag = "ema" if c.ema_decay else "noema"
            add(f"I06_{tag}", f"1 view, {'trọng số EMA' if c.ema_decay else 'không EMA'}",
                I.apply_temperature(logits, 1.0), y_val, 1, [224], model_used=f"{c.exp_id}/seed{c.seed}")

    # I07: temperature scaling, T khớp trên VAL
    T = I.fit_temperature(views_1[0], y_val)
    add("I07", "1 view + temperature scaling", I.apply_temperature(views_1[0], T), y_val, 1, [224],
        temperature=T, ece_before=summarize(y_val, I.apply_temperature(views_1[0], 1.0))["ece"],
        ece_crossfit_after=crossfit_ece(views_1[0], y_val))

    # I08: gộp BN (FP32) và FP16: độ chính xác trên val và độ trễ
    mean, std = model_norm(model)
    fused = I.fuse_conv_bn(model)
    if fused.fused_pairs:
        err = I.check_fusion(model, fused, 224, device)
        _, y, logits = I.predict_logits(fused, _loader(cfg, _split_df(cfg, "val"), mean, std), device,
                                        amp=False, channels_last=cfg.channels_last)
        add("I08_fuse", "gộp BN vào conv (FP32)", I.apply_temperature(logits, 1.0), y, 1,
            lat=latency_views(fused, [224], iters=lat_iters), fused_pairs=fused.fused_pairs, max_abs_err=err)
    else:
        print("I08 gộp BN: không áp dụng (kiến trúc không có cặp Conv-BN, ví dụ dùng LayerNorm)")
    half = load_trained(cfg, device).half()
    ys, outs = [], []
    with torch.inference_mode():
        for x, yy, _ in _loader(cfg, _split_df(cfg, "val"), mean, std):
            outs.append(half(x.to(device).half().contiguous(memory_format=torch.channels_last)).float().cpu())
            ys.append(yy)
    add("I08_fp16", "FP16 (model.half())", I.apply_temperature(torch.cat(outs).numpy(), 1.0),
        torch.cat(ys).numpy(), 1, lat=latency_views(model, [224], dtype="fp16", iters=lat_iters))
    del half

    # Bảng Latency: 1 view, các dtype, batch 1 và 32, có/không gộp BN
    for dtype in B.DTYPES:
        for bs in (1, 32):
            lat_rows.append({"config": f"{name} 1 view",
                             **B.latency_report(model, bs, 224, dtype, str(device), iters=lat_iters)})
            if fused.fused_pairs and dtype != "amp":
                lat_rows.append({"config": f"{name} 1 view, gộp BN",
                                 **B.latency_report(fused, bs, 224, dtype, str(device), iters=lat_iters,
                                                    fused_bn=True)})

    df = pd.DataFrame(rows)
    base = df.loc[df["exp_id"] == "I00", "lat_b1_p50_ms"].iloc[0]
    df["rel_cost_vs_I00"] = df["lat_b1_p50_ms"] / base
    return df, pd.DataFrame(lat_rows)


def choose_method(df: pd.DataFrame, noise: float) -> pd.Series:
    """Chọn cách suy luận cho Bước 4, CHỈ dựa trên VAL: trong các cách áp dụng được cho mô hình này
    (cột selectable) có macro-F1 val cách tốt nhất không quá `noise` (std giữa các seed), chọn cách có
    độ trễ p50 thấp nhất. Chênh lệch nhỏ hơn nhiễu thì ưu tiên cách rẻ hơn."""
    cand = df[df["selectable"]]
    near = cand[cand["macro_f1"] >= cand["macro_f1"].max() - noise]
    return near.sort_values(["lat_b1_p50_ms", "macro_f1"], ascending=[True, False]).iloc[0]


def plot_tradeoff(df: pd.DataFrame, path, title: str) -> Path:
    """Đường đánh đổi macro-F1 val và độ trễ p50 batch 1 (thang log)."""
    fig = Figure(figsize=(8, 5), dpi=120)
    ax = fig.subplots()
    d = df[df["lat_b1_p50_ms"].notna()]
    ax.scatter(d["lat_b1_p50_ms"], d["macro_f1"])
    for _, r in d.iterrows():
        ax.annotate(r["exp_id"], (r["lat_b1_p50_ms"], r["macro_f1"]), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.set(xscale="log", xlabel="độ trễ p50, batch 1 (ms, thang log)", ylabel="macro-F1 val", title=title)
    ax.grid(alpha=0.3)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path


def final_predictions(cfg: Config, method: str, space: str, out_exp: str, device=None) -> dict:
    """Bước 4 cho MỘT seed: suy luận theo `method`/`space` đã chọn trên val, kèm temperature scaling.

    1. VAL: logit các view -> gộp -> khớp T trên VAL -> ghi <out_exp>_seed<k>_val.csv (đã hiệu chuẩn)
    2. TEST đúng MỘT lần: logit các view -> gộp -> ghi <out_exp>uncal_seed<k>_test.csv (chưa hiệu chuẩn)
       và <out_exp>_seed<k>_test.csv (T lấy từ VAL)
    Từ chối chạy nếu file test đã tồn tại. Gộp xác suất được viết lại thành logit log(p) để áp dụng T.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pdir = Path(cfg.pred_dir)
    test_csv = pdir / f"{out_exp}_seed{cfg.seed}_test.csv"
    uncal_csv = pdir / f"{out_exp}uncal_seed{cfg.seed}_test.csv"
    if test_csv.exists() or uncal_csv.exists():
        raise RuntimeError(f"{test_csv} đã có: test chỉ chạy một lần cho mỗi seed")
    model = load_trained(cfg, device)

    def agg_logits(views):
        return np.log(np.clip(I.aggregate_views(views, space), 1e-12, None))

    names, y_val, views = view_logits(model, cfg, "val", method, device)
    z_val = agg_logits(views)
    T = I.fit_temperature(z_val, y_val)
    save_predictions(pdir / f"{out_exp}_seed{cfg.seed}_val.csv", names, y_val, I.apply_temperature(z_val, T))

    names, y_test, views = view_logits(model, cfg, "test", method, device)  # lần duy nhất chạm vào test
    z_test = agg_logits(views)
    save_predictions(uncal_csv, names, y_test, I.apply_temperature(z_test, 1.0))
    save_predictions(test_csv, names, y_test, I.apply_temperature(z_test, T))
    np.save(run_dir(cfg) / f"{out_exp}_test_agg_logits.npy", z_test)
    info = {"exp": out_exp, "seed": cfg.seed, "source_run": f"{cfg.exp_id}/seed{cfg.seed}", "method": method,
            "space": space, "temperature": T, "val": summarize(y_val, I.apply_temperature(z_val, T)),
            "latency_b1_fp32": latency_views(model, method_sizes(method))}
    (run_dir(cfg) / f"{out_exp}_final.json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    print(f"[{out_exp} seed{cfg.seed}] T = {T:.3f}; đã ghi {test_csv.name}, {uncal_csv.name} (test chạy 1 lần)")
    return info


def plot_errors(pred_csv, images_dir, path, true_cls: int, pred_cls: int, n: int = 8) -> Path:
    """Ảnh test bị đoán sai từ lớp `true_cls` sang `pred_cls` (phân tích lỗi SAU Bước 4, không để chọn gì)."""
    df = pd.read_csv(pred_csv)
    wrong = df[(df["y_true"] == true_cls) & (df["y_pred"] == pred_cls)].head(n)
    k = max(1, len(wrong))
    fig = Figure(figsize=(2.3 * k, 2.8), dpi=100)
    axes = np.atleast_1d(fig.subplots(1, k))
    for ax, (_, r) in zip(axes, wrong.iterrows()):
        ax.imshow(Image.open(Path(images_dir) / r["Filename"]).convert("RGB"))
        ax.set_title(f"p(đoán)={r[f'p{pred_cls}']:.2f}", fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Thật: {D.CLASS_NAMES[true_cls]} -> đoán: {D.CLASS_NAMES[pred_cls]} ({len(wrong)} ảnh)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path
