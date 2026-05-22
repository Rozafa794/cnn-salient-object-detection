"""
data_loader.py
--------------
Handles dataset loading for the Salient Object Detection (SOD) project.

Responsibilities:
- Load image-mask pairs from disk
- Resize images and masks to 128x128
- Normalize pixel values to [0, 1]
- Apply data augmentation (random horizontal flip, brightness, random crop)
- Split the dataset into Train (70%) / Validation (15%) / Test (15%)
- Provide PyTorch DataLoaders for training, validation, and testing

Expected directory structure:
    dataset/
        images/      <- input RGB images (.jpg / .png)
        masks/       <- corresponding ground-truth saliency masks (.png)

Image and mask filenames must match (e.g. img_001.jpg <-> img_001.png).
"""

import os
import random
from PIL import Image, ImageEnhance

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF


# ----------------------------- Dataset class ----------------------------- #
class SODDataset(Dataset):
    """
    Custom PyTorch Dataset for Salient Object Detection.

    Args:
        images_dir (str): path to the folder containing input RGB images.
        masks_dir  (str): path to the folder containing ground-truth masks.
        image_size (int): final spatial size (height = width = image_size).
        augment    (bool): whether to apply data augmentation (training only).
    """

    def __init__(self, images_dir, masks_dir, image_size=128, augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.augment = augment

        # Collect all image filenames (sorted for reproducibility)
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

    def __len__(self):
        return len(self.image_files)

    # ---- Helper: build the matching mask filename ---- #
    def _get_mask_path(self, image_filename):
        """
        Try common mask extensions (.png first, then .jpg) since SOD datasets
        usually store masks as PNG, but some use JPG.
        """
        name, _ = os.path.splitext(image_filename)
        for ext in (".png", ".jpg", ".jpeg", ".bmp"):
            candidate = os.path.join(self.masks_dir, name + ext)
            if os.path.exists(candidate):
                return candidate
        # Fallback (will raise FileNotFoundError later if not valid)
        return os.path.join(self.masks_dir, name + ".png")

    # ---- Data augmentation applied jointly to image and mask ---- #
    def _augment(self, image, mask):
        # 1. Random horizontal flip (50% probability)
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # 2. Random brightness change (image only, mask unchanged)
        if random.random() < 0.5:
            factor = random.uniform(0.7, 1.3)  # 70%–130% brightness
            image = ImageEnhance.Brightness(image).enhance(factor)

        # 3. Random crop then resize back to image_size (a "random zoom" effect)
        if random.random() < 0.5:
            crop_size = int(self.image_size * random.uniform(0.8, 1.0))
            i = random.randint(0, self.image_size - crop_size)
            j = random.randint(0, self.image_size - crop_size)
            image = TF.crop(image, i, j, crop_size, crop_size)
            mask = TF.crop(mask, i, j, crop_size, crop_size)
            image = TF.resize(image, [self.image_size, self.image_size])
            mask = TF.resize(mask, [self.image_size, self.image_size])

        return image, mask

    def __getitem__(self, idx):
        # ---- Load image and mask ---- #
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = self._get_mask_path(img_name)

        image = Image.open(img_path).convert("RGB")          # 3-channel RGB
        mask = Image.open(mask_path).convert("L")            # 1-channel grayscale

        # ---- Resize both to a fixed size ---- #
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # ---- Apply augmentation only on training data ---- #
        if self.augment:
            image, mask = self._augment(image, mask)

        # ---- Convert PIL -> Tensor and normalize to [0, 1] ---- #
        image = TF.to_tensor(image)                          # shape: (3, H, W)
        mask = TF.to_tensor(mask)                            # shape: (1, H, W)

        # Binarize the mask: any pixel > 0.5 is foreground (salient)
        mask = (mask > 0.5).float()

        return image, mask


# ----------------------------- Loader factory ----------------------------- #
def get_dataloaders(
    images_dir,
    masks_dir,
    image_size=128,
    batch_size=16,
    num_workers=2,
    seed=42,
):
    """
    Build train / val / test DataLoaders with a 70 / 15 / 15 split.

    Returns:
        train_loader, val_loader, test_loader
    """
    # Two dataset views: one with augmentation (train), one without (val/test).
    # We use the same indices across both, then assign them appropriately.
    full_dataset_aug = SODDataset(images_dir, masks_dir, image_size, augment=True)
    full_dataset_clean = SODDataset(images_dir, masks_dir, image_size, augment=False)

    total_len = len(full_dataset_aug)
    train_len = int(0.70 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    # Reproducible random split
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        full_dataset_aug,
        [train_len, val_len, test_len],
        generator=generator,
    )

    # Re-route val/test to the non-augmented view so they evaluate fairly.
    val_set.dataset = full_dataset_clean
    test_set.dataset = full_dataset_clean

    # Build PyTorch DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(f"Dataset split -> Train: {train_len} | Val: {val_len} | Test: {test_len}")
    return train_loader, val_loader, test_loader


# ----------------------------- Quick self-test ----------------------------- #
if __name__ == "__main__":
    # Example usage (update the paths to your dataset before running)
    images_dir = "dataset/images"
    masks_dir = "dataset/masks"

    if os.path.isdir(images_dir) and os.path.isdir(masks_dir):
        train_loader, val_loader, test_loader = get_dataloaders(
            images_dir, masks_dir, batch_size=4, num_workers=0
        )
        for imgs, masks in train_loader:
            print("Image batch:", imgs.shape, "Mask batch:", masks.shape)
            print("Image range:", imgs.min().item(), imgs.max().item())
            print("Mask  range:", masks.min().item(), masks.max().item())
            break
    else:
        print("Update images_dir / masks_dir to point to your dataset before running.")