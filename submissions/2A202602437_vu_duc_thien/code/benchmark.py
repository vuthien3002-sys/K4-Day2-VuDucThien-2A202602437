"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý: đầu vào là tensor đã nằm sẵn trên GPU (chỉ đo forward của model)
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

DTYPES = ("fp32", "amp", "fp16")


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (torch.cuda.synchronize) hoặc None trên CPU. Mỗi lần đo:
    sync(); t0 = perf_counter(); fn(); sync(); lấy hiệu.
    """
    for _ in range(warmup):
        fn()
    if sync:
        sync()
    times = np.empty(iters)
    for i in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times[i] = (time.perf_counter() - t0) * 1000.0
    p50, p95, p99 = np.percentile(times, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99),
            "mean": float(times.mean()), "std": float(times.std(ddof=1)), "n": iters}


def _prepare(model, dtype: str, device: str, channels_last: bool):
    if dtype not in DTYPES:
        raise ValueError(f"dtype={dtype!r} không hợp lệ, chọn một trong {DTYPES}")
    m = copy.deepcopy(model).half() if dtype == "fp16" else model
    m = m.to(device).eval()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    return m


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, fused_bn: bool = False, k_views: int = 1,
                   channels_last: bool = True) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    - dtype: "fp32" | "amp" (autocast fp16) | "fp16" (bản sao model.half())
    - k_views > 1: chạy model K lần liên tiếp mỗi lượt (độ trễ TTA K view, chạy tuần tự)
    Trả về dict ghi thẳng được vào sheet `Latency` của results.xlsx. Ở batch 1, AMP có thể CHẬM hơn
    FP32 (slide trang 73): đây là số đo thật, không giả định.
    """
    dev = torch.device(device)
    m = _prepare(model, dtype, device, channels_last)
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev,
                    dtype=torch.float16 if dtype == "fp16" else torch.float32)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)

    def fn():
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.float16, enabled=dtype == "amp"):
            for _ in range(k_views):
                m(x)

    r = bench(fn, warmup, iters, torch.cuda.synchronize if dev.type == "cuda" else None)
    return {
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "dtype": dtype, "batch": batch_size, "img_size": img_size, "fused_bn": fused_bn,
        "k_views": k_views, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"],
        "images_per_s": batch_size / (r["p50"] / 1000.0), "n_iters": iters, "warmup": warmup,
        "torch": torch.__version__, "preprocessing": "không tính",
    }


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ của TTA K view (K lượt forward tuần tự), kèm tỉ lệ so với 1 view đo cùng điều kiện.

    Slide trang 63: chi phí xấp xỉ K lần một lượt chạy; hàm này đo thật để kiểm tra.
    """
    one = latency_report(model, k_views=1, **kw)
    many = latency_report(model, k_views=k_views, **kw)
    many["ratio_vs_1view"] = many["p50"] / one["p50"]
    return many
