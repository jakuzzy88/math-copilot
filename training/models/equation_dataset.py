"""
equation_dataset.py – PyTorch Dataset for loading synthetic equation images.

Loads image/label pairs from the dataset directory structure produced by
``build_synthetic_dataset.py``:

    <root>/images/<split>/000001.png
    <root>/labels/<split>/000001.txt

Images are converted to greyscale, resized, and normalised to [0, 1].
Labels are encoded using the CTC vocabulary.

Sprint 4A: Added optional online image augmentation for training.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFilter, ImageEnhance

from training.models.vocabulary import encode_label


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _apply_augmentation(img: Image.Image, rng: random.Random) -> Image.Image:
    """Apply small, training-only augmentations to a greyscale PIL image.

    Augmentations:
    - Small brightness variation (factor 0.85 – 1.15)
    - Small contrast variation (factor 0.85 – 1.15)
    - Small Gaussian blur (10% probability, radius 0.5–1.0)
    - Small Gaussian noise (10% probability, σ = 2–8)

    All augmentations use subtle parameters to avoid destroying handwriting
    legibility.
    """
    # Brightness variation
    brightness_factor = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # Contrast variation
    contrast_factor = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # Gaussian blur (10% chance)
    if rng.random() < 0.10:
        radius = rng.uniform(0.5, 1.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # Gaussian noise (10% chance)
    if rng.random() < 0.10:
        arr = np.array(img, dtype=np.float32)
        sigma = rng.uniform(2.0, 8.0)
        noise = np.random.RandomState(rng.randint(0, 2**31)).normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

    return img


class EquationDataset(Dataset):
    """Dataset of synthetic equation images and their CTC-encoded labels.

    Args:
        root: Path to the dataset root (e.g. ``training/datasets/synthetic_v1``).
        split: One of ``'train'``, ``'valid'``, ``'test'``.
        height: Target image height in pixels (default 128).
        width: Target image width in pixels (default 512).
        max_samples: If set, limit to the first *max_samples* items (for
            smoke testing).
        augment: If True **and** split is ``'train'``, apply online image
            augmentation.  Never applied to valid/test splits.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        height: int = 128,
        width: int = 512,
        max_samples: int | None = None,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.height = height
        self.width = width
        # Only augment training data.
        self.augment = augment and (split == "train")
        self._rng = random.Random()

        img_dir = self.root / "images" / split
        lbl_dir = self.root / "labels" / split

        if not img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not lbl_dir.is_dir():
            raise FileNotFoundError(f"Label directory not found: {lbl_dir}")

        # Collect sorted (image_path, label_path) pairs by matching basenames.
        img_files = sorted(img_dir.glob("*.png"))
        self.samples: list[tuple[Path, Path]] = []
        for img_path in img_files:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                self.samples.append((img_path, lbl_path))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No image/label pairs found in {img_dir} / {lbl_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """Return a single sample.

        Returns:
            image: ``(1, H, W)`` float32 tensor normalised to [0, 1].
            label: 1-D int32 tensor of encoded character indices.
            label_length: Number of characters in the label.
            raw_label: Original label string (for debugging / metrics).
        """
        img_path, lbl_path = self.samples[idx]

        # Load image as greyscale, resize.
        img = Image.open(img_path).convert("L")
        img = img.resize((self.width, self.height), Image.BILINEAR)

        # Apply augmentation if enabled (training split only).
        if self.augment:
            img = _apply_augmentation(img, self._rng)

        # Convert to float tensor and normalise to [0, 1].
        img_tensor = torch.from_numpy(
            np.array(img, dtype="float32")
        ) / 255.0
        # Add channel dimension: (H, W) → (1, H, W)
        img_tensor = img_tensor.unsqueeze(0)

        # Load label string.
        raw_label = lbl_path.read_text(encoding="utf-8").strip()
        encoded = encode_label(raw_label)
        label_tensor = torch.tensor(encoded, dtype=torch.int32)

        return img_tensor, label_tensor, len(encoded), raw_label


# ---------------------------------------------------------------------------
# Collate function for CTC training
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, str]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Custom collate for CTC batches.

    CTC loss expects:
    * ``images``:  ``(B, 1, H, W)`` float32 tensor.
    * ``labels``:  1-D concatenation of all label tensors.
    * ``label_lengths``: ``(B,)`` int32 tensor with each label's length.
    * ``raw_labels``: list of original label strings.

    The images are already the same size, so we simply stack them.
    Labels are concatenated into a single 1-D tensor as required by
    ``torch.nn.CTCLoss``.
    """
    images, labels, lengths, raw_labels = zip(*batch)

    images = torch.stack(images, dim=0)                 # (B, 1, H, W)
    labels = torch.cat(labels, dim=0)                   # (sum_of_lengths,)
    label_lengths = torch.tensor(lengths, dtype=torch.int32)  # (B,)

    return images, labels, label_lengths, list(raw_labels)
