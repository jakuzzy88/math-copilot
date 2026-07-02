"""
Tests for training.models.equation_dataset
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.models.equation_dataset import EquationDataset, collate_fn

# Path to the synthetic dataset (must exist from Sprint 2).
_DATASET_ROOT = Path(__file__).resolve().parent.parent / "datasets" / "synthetic_v1"


@pytest.fixture
def dataset():
    """Small train dataset for testing."""
    return EquationDataset(
        root=_DATASET_ROOT, split="train",
        height=128, width=512,
        max_samples=8,
    )


class TestEquationDataset:
    """Tests for EquationDataset loading."""

    def test_loads_samples(self, dataset):
        """Dataset should contain samples."""
        assert len(dataset) == 8

    def test_getitem_returns_correct_types(self, dataset):
        """Each sample should return (image, label, length, raw_label)."""
        img, label, length, raw_label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert isinstance(length, int)
        assert isinstance(raw_label, str)

    def test_image_shape(self, dataset):
        """Image tensor should be (1, H, W)."""
        img, _, _, _ = dataset[0]
        assert img.shape == (1, 128, 512)

    def test_image_dtype_and_range(self, dataset):
        """Image should be float32 in [0, 1]."""
        img, _, _, _ = dataset[0]
        assert img.dtype == torch.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_label_dtype(self, dataset):
        """Label tensor should be int32."""
        _, label, _, _ = dataset[0]
        assert label.dtype == torch.int32

    def test_label_length_matches(self, dataset):
        """label_length should match len(label_tensor)."""
        _, label, length, raw_label = dataset[0]
        assert length == len(label)
        assert length == len(raw_label)

    def test_raw_label_not_empty(self, dataset):
        """Raw label should be a non-empty string."""
        _, _, _, raw_label = dataset[0]
        assert len(raw_label) > 0

    def test_missing_split_raises(self):
        """Non-existent split should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            EquationDataset(root=_DATASET_ROOT, split="nonexistent")


class TestCollateFn:
    """Tests for the CTC collate function."""

    def test_collate_returns_correct_types(self, dataset):
        """collate_fn should return stacked images, concatenated labels,
        label lengths, and raw label strings."""
        batch = [dataset[i] for i in range(4)]
        images, labels, label_lengths, raw_labels = collate_fn(batch)

        assert isinstance(images, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert isinstance(label_lengths, torch.Tensor)
        assert isinstance(raw_labels, list)

    def test_collate_image_shape(self, dataset):
        """Collated images should be (B, 1, H, W)."""
        batch = [dataset[i] for i in range(4)]
        images, _, _, _ = collate_fn(batch)
        assert images.shape == (4, 1, 128, 512)

    def test_collate_labels_1d(self, dataset):
        """Labels should be concatenated into a single 1-D tensor."""
        batch = [dataset[i] for i in range(4)]
        _, labels, label_lengths, _ = collate_fn(batch)
        assert labels.dim() == 1
        # Total length should equal sum of individual lengths.
        assert labels.shape[0] == label_lengths.sum().item()

    def test_collate_label_lengths_shape(self, dataset):
        """label_lengths should be (B,)."""
        batch = [dataset[i] for i in range(4)]
        _, _, label_lengths, _ = collate_fn(batch)
        assert label_lengths.shape == (4,)

    def test_collate_raw_labels(self, dataset):
        """raw_labels should be a list of strings."""
        batch = [dataset[i] for i in range(4)]
        _, _, _, raw_labels = collate_fn(batch)
        assert len(raw_labels) == 4
        assert all(isinstance(r, str) for r in raw_labels)
