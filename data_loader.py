"""
data_loader.py
==============
Dataset loading, preprocessing, and augmentation for the ECSSD Salient
Object Detection dataset.

Public API:
    ECSSDDataset       -- PyTorch Dataset for ECSSD image / mask pairs.
    get_loaders(...)   -- Build train / val / test DataLoaders with a 70/15/15 split.

Expected directory layout:
    <data_root>/images/   -> 0001.jpg, 0002.jpg, ...
    <data_root>/masks/    -> 0001.png, 0002.png, ...

Example:
    >>> train_loader, val_loader, test_loader = get_loaders(
    ...     img_dir="data/ECSSD/test/image",
    ...     mask_dir="data/ECSSD/test/mask",
    ...     img_size=128, batch_size=16,
    ... )
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class ECSSDDataset(Dataset):
    """
    PyTorch Dataset that loads (image, mask) pairs from disk, resizes them to
    a square ``img_size``, optionally augments them, and returns float tensors
    in the layouts expected by the model:

        image -> torch.float32 tensor of shape (3, H, W), values in [0, 1]
        mask  -> torch.float32 tensor of shape (1, H, W), values in {0, 1}

    Augmentations (only applied when ``augment=True``):
        * horizontal flip (image + mask, kept aligned)
        * brightness jitter (image only)
        * random crop + resize back (image + mask, kept aligned)
    """

    def __init__(self, image_paths, mask_paths, img_size=128, augment=False):
        assert len(image_paths) == len(mask_paths), "image/mask count mismatch"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ---- load ----
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)

        # ---- resize ----
        img = cv2.resize(img, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        # ---- augmentation (training only) ----
        if self.augment:
            # horizontal flip - apply to BOTH so they stay aligned
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

            # brightness jitter - image only
            if np.random.rand() < 0.5:
                factor = np.random.uniform(0.8, 1.2)
                img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

            # random crop + resize back - image and mask kept aligned
            if np.random.rand() < 0.5:
                h, w = img.shape[:2]
                ch, cw = int(h * 0.85), int(w * 0.85)
                y = np.random.randint(0, h - ch)
                x = np.random.randint(0, w - cw)
                img = cv2.resize(img[y:y + ch, x:x + cw], (self.img_size, self.img_size))
                mask = cv2.resize(mask[y:y + ch, x:x + cw], (self.img_size, self.img_size))

        # ---- normalize ----
        img = img.astype(np.float32) / 255.0          # values in [0, 1]
        mask = (mask > 127).astype(np.float32)        # binarize -> {0, 1}

        # ---- to tensor (C, H, W) ----
        img = torch.from_numpy(img).permute(2, 0, 1)  # 3 x H x W
        mask = torch.from_numpy(mask).unsqueeze(0)    # 1 x H x W
        return img, mask


def get_loaders(img_dir, mask_dir, img_size=128, batch_size=16,
                seed=42, num_workers=2):
    """
    Build train / val / test DataLoaders with a 70/15/15 split.

    Args:
        img_dir (str): folder containing the .jpg images.
        mask_dir (str): folder containing the corresponding .png masks
            (same filename stem as images).
        img_size (int): square resize size.
        batch_size (int): batch size for all loaders.
        seed (int): random seed for the train/val/test split (reproducible).
        num_workers (int): number of DataLoader worker processes.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    img_files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    images = [os.path.join(img_dir, f) for f in img_files]
    masks = [os.path.join(mask_dir, f.replace(".jpg", ".png")) for f in img_files]

    # 70 / 15 / 15
    tr_i, tmp_i, tr_m, tmp_m = train_test_split(
        images, masks, test_size=0.30, random_state=seed
    )
    va_i, te_i, va_m, te_m = train_test_split(
        tmp_i, tmp_m, test_size=0.50, random_state=seed
    )

    train_ds = ECSSDDataset(tr_i, tr_m, img_size, augment=True)
    val_ds = ECSSDDataset(va_i, va_m, img_size, augment=False)
    test_ds = ECSSDDataset(te_i, te_m, img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"Splits  ->  train: {len(train_ds)}  |  val: {len(val_ds)}  |  test: {len(test_ds)}")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick smoke test - run as: python data_loader.py
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    args = parser.parse_args()

    train_loader, val_loader, test_loader = get_loaders(args.img_dir, args.mask_dir)
    imgs, masks = next(iter(train_loader))
    print("Batch images:", imgs.shape, "range:", imgs.min().item(), "->", imgs.max().item())
    print("Batch masks: ", masks.shape, "unique:", torch.unique(masks).tolist())
