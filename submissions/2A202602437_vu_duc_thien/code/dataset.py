"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # mặc định; train.py lấy mean/std đúng theo pretrained_cfg của timm
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509
SPLITS = ("train", "val", "test")
AUGS = ("basic", "color", "trivial", "randaug", "vflip")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1).

    Mỗi file có cột `Filename, Label, Species`. Trả về ba DataFrame, giữ nguyên thứ tự dòng.
    KHÔNG sửa, lọc hay chia lại dữ liệu.
    """
    labels_dir = Path(labels_dir)
    dfs = []
    for split in SPLITS:
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{split}_subset{fold}.csv thiếu cột {missing}")
        df["Label"] = df["Label"].astype(int)
        dfs.append(df)
    return tuple(dfs)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu.

    Dừng ngay (AssertionError) nếu: có giao khác rỗng giữa hai tập, hợp ba tập khác 17.509 ảnh,
    có file trong CSV không tồn tại trong `images_dir`, hoặc một tập có tên file trùng lặp.
    Lệch tỉ lệ 60/20/20 quá 1 điểm phần trăm chỉ cảnh báo (README: báo giảng viên).
    """
    dfs = dict(zip(SPLITS, (train_df, val_df, test_df)))
    names = {s: set(df["Filename"]) for s, df in dfs.items()}

    for s, df in dfs.items():
        assert df["Filename"].is_unique, f"{s}: có Filename trùng lặp"
        assert df["Label"].between(0, NUM_CLASSES - 1).all(), f"{s}: Label ngoài 0..{NUM_CLASSES - 1}"

    overlap = {
        "train∩val": len(names["train"] & names["val"]),
        "train∩test": len(names["train"] & names["test"]),
        "val∩test": len(names["val"] & names["test"]),
    }
    union = len(names["train"] | names["val"] | names["test"])
    on_disk = set(os.listdir(images_dir))
    missing = sorted((names["train"] | names["val"] | names["test"]) - on_disk)

    n = {s: len(df) for s, df in dfs.items()}
    total = sum(n.values())
    ratio = {s: n[s] / total for s in SPLITS}
    ratio_warn = [s for s, r in zip(SPLITS, (0.6, 0.2, 0.2)) if abs(ratio[s] - r) > 0.01]

    per_class = pd.DataFrame({s: df["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
                              for s, df in dfs.items()})
    per_class.index = CLASS_NAMES
    per_class["total"] = per_class.sum(axis=1)

    if verbose:
        print("Số ảnh:", n, "| tổng", total)
        print("Tỉ lệ:", {s: f"{100 * r:.2f}%" for s, r in ratio.items()})
        print("Giao từng cặp:", overlap, "| hợp ba tập:", union, "| thiếu file:", len(missing))
        print(per_class.to_string())
        if ratio_warn:
            print(f"CẢNH BÁO: tỉ lệ của {ratio_warn} lệch > 1 điểm % so với 60/20/20, báo giảng viên")

    assert all(v == 0 for v in overlap.values()), f"giao khác rỗng: {overlap}"
    assert union == TOTAL_IMAGES, f"hợp ba tập = {union}, kỳ vọng {TOTAL_IMAGES}"
    assert not missing, f"{len(missing)} file không có trong {images_dir}, ví dụ {missing[:5]}"

    return {"n": n, "total": total, "ratio": ratio, "overlap": overlap, "union": union,
            "missing": len(missing), "ratio_warning": ratio_warn, "per_class": per_class}


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD, crop_pct: float = 0.875):
    """Tạo transform.

    Train (mọi `aug` đều bắt đầu bằng RandomResizedCrop(img_size) + lật ngang):
      - "basic"  : chỉ crop + lật ngang (công thức nền T00)
      - "color"  : + ColorJitter (độ sáng, tương phản, bão hoà, sắc độ)
      - "trivial": + TrivialAugmentWide
      - "randaug": + RandAugment(2 phép, cường độ 9)
      - "vflip"  : + lật dọc (ảnh chụp từ trên xuống nên hướng "lên/xuống" không mang nghĩa)
    Mixup/CutMix trộn theo batch nên nằm ở losses.py, không ở đây.

    Val/test: Resize(img_size / crop_pct) rồi CenterCrop(img_size). Với 224 và crop_pct = 0.875 thì
    Resize(256) không đổi ảnh gốc 256x256, chỉ còn CenterCrop(224). crop_pct = 1.0 giữ nguyên cả ảnh.
    Không có augmentation ngẫu nhiên khi đánh giá.
    """
    normalize = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        resize = int(round(img_size / crop_pct))
        return T.Compose([T.Resize(resize, antialias=True), T.CenterCrop(img_size), *normalize])

    if aug not in AUGS:
        raise ValueError(f"aug={aug!r} không hợp lệ, chọn một trong {AUGS}")
    ops = [T.RandomResizedCrop(img_size, antialias=True), T.RandomHorizontalFlip()]
    if aug == "color":
        ops.append(T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    elif aug == "vflip":
        ops.append(T.RandomVerticalFlip())
    return T.Compose([*ops, *normalize])


def denormalize(x: torch.Tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> torch.Tensor:
    """Giải chuẩn hoá (C, H, W) hoặc (N, C, H, W) về [0, 1] để vẽ ảnh."""
    m = torch.tensor(mean, dtype=x.dtype).view(-1, 1, 1)
    s = torch.tensor(std, dtype=x.dtype).view(-1, 1, 1)
    return (x * s + m).clamp(0, 1)


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label).

    __getitem__(i) trả về (ảnh đã transform, nhãn int, tên file str). Tên file cần có để ghi
    `predictions/*.csv` đúng định dạng của eval.py.
    """

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        # Giữ mảng numpy thay vì DataFrame: rẻ hơn khi DataLoader fork sang worker.
        self.filenames = df["Filename"].to_numpy()
        self.labels = df["Label"].to_numpy(dtype=np.int64)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        name = self.filenames[i]
        with Image.open(self.images_dir / name) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), str(name)


def seed_worker(worker_id: int) -> None:
    """Seed cho random/numpy trong mỗi worker, suy ra từ seed torch của worker (để tái lập augmentation)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2, seed: int = 0):
    """Tạo DataLoader.

    - train=True: xáo trộn (hoặc sampler cân bằng), drop_last=True để batch cuối nhỏ không làm
      BatchNorm mất ổn định. train=False: không xáo, giữ đúng thứ tự df để ghép logit với Filename.
    - sampler="balanced": WeightedRandomSampler với trọng số 1/(số ảnh của lớp), lấy có hoàn lại,
      số mẫu mỗi epoch = số ảnh train (trục D của GUIDE.md mục 3).
    - Thứ tự batch và augmentation cố định theo `seed` (generator + worker_init_fn).
    """
    ds = DeepWeedsDataset(df, images_dir, transform)
    g = torch.Generator().manual_seed(seed)
    shuffle, smp = False, None
    if train and sampler == "balanced":
        counts = np.bincount(ds.labels, minlength=NUM_CLASSES)
        weights = torch.as_tensor(1.0 / counts[ds.labels], dtype=torch.double)
        smp = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True, generator=g)
    elif train and sampler is None:
        shuffle = True
    elif sampler is not None:
        raise ValueError(f"sampler={sampler!r} không hợp lệ (None | 'balanced'; chỉ dùng khi train)")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, sampler=smp, drop_last=train,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0, worker_init_fn=seed_worker, generator=g,
    )
