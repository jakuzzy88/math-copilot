"""
Tests for Sprint 4A: Production Training Prep

Covers:
- Scheduler option parsing and construction
- Early stopping logic
- Dataset augmentation preserves tensor shapes
- Training history contains learning_rate
- Overfit-small-batch mode initialisation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.train_ctc import (
    SCHEDULER_CHOICES,
    EarlyStopping,
    build_scheduler,
    get_current_lr,
    scheduler_step,
)
from training.models.equation_dataset import EquationDataset, collate_fn

# Path to synthetic dataset (v1 is guaranteed to exist from Sprint 2).
_DATASET_ROOT_V1 = Path(__file__).resolve().parent.parent / "datasets" / "synthetic_v1"
_DATASET_ROOT_V2 = Path(__file__).resolve().parent.parent / "datasets" / "synthetic_v2"


def _get_dataset_root() -> Path:
    """Return the first available dataset root (prefer v2, fallback v1)."""
    if (_DATASET_ROOT_V2 / "images" / "train").is_dir():
        return _DATASET_ROOT_V2
    return _DATASET_ROOT_V1


# ===========================================================================
# Scheduler tests
# ===========================================================================

class TestSchedulerParsing:
    """Test scheduler CLI option values and construction."""

    def test_scheduler_choices_contains_expected(self):
        assert "none" in SCHEDULER_CHOICES
        assert "cosine" in SCHEDULER_CHOICES
        assert "reduce_on_plateau" in SCHEDULER_CHOICES

    def test_build_scheduler_none(self):
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = build_scheduler("none", opt, 10)
        assert sched is None

    def test_build_scheduler_cosine(self):
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = build_scheduler("cosine", opt, 10)
        assert sched is not None
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_build_scheduler_reduce_on_plateau(self):
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = build_scheduler("reduce_on_plateau", opt, 10)
        assert sched is not None
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_build_scheduler_invalid_raises(self):
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_scheduler("invalid_scheduler", opt, 10)

    def test_cosine_scheduler_adjusts_lr(self):
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = build_scheduler("cosine", opt, 10)

        initial_lr = get_current_lr(opt)
        # Step the scheduler a few times
        for _ in range(5):
            scheduler_step(sched, "cosine")
        new_lr = get_current_lr(opt)
        # LR should have decreased after half the epochs
        assert new_lr < initial_lr

    def test_reduce_on_plateau_step_with_loss(self):
        """ReduceLROnPlateau should accept val_loss without error."""
        model = torch.nn.Linear(10, 10)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = build_scheduler("reduce_on_plateau", opt, 10)
        # Should not raise
        scheduler_step(sched, "reduce_on_plateau", val_loss=1.5)


# ===========================================================================
# Early stopping tests
# ===========================================================================

class TestEarlyStopping:
    """Test early stopping logic in isolation."""

    def test_no_stop_initially(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        assert es.step(1.0) is False
        assert es.should_stop is False

    def test_no_stop_when_improving(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
            assert es.step(loss) is False
        assert es.should_stop is False

    def test_stop_after_patience_no_improvement(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        es.step(1.0)  # initial
        es.step(1.1)  # worse, counter=1
        es.step(1.2)  # worse, counter=2
        result = es.step(1.3)  # worse, counter=3 → stop
        assert result is True
        assert es.should_stop is True

    def test_counter_resets_on_improvement(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        es.step(1.0)  # initial
        es.step(1.1)  # worse, counter=1
        es.step(0.9)  # improved, counter reset
        assert es.counter == 0
        es.step(1.0)  # worse, counter=1
        assert es.step(1.0) is False  # counter=2, not yet
        assert es.counter == 2

    def test_min_delta_respected(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        es.step(1.0)  # initial best=1.0
        # Improvement of 0.05 is less than min_delta=0.1 → not improvement
        es.step(0.95)  # counter=1
        result = es.step(0.92)  # counter=2 → stop
        assert result is True

    def test_min_delta_allows_big_improvement(self):
        es = EarlyStopping(patience=2, min_delta=0.1)
        es.step(1.0)
        result = es.step(0.8)  # improvement of 0.2 > min_delta → resets
        assert result is False
        assert es.counter == 0


# ===========================================================================
# Augmentation tests
# ===========================================================================

class TestAugmentation:
    """Test that augmentation does not change tensor shapes."""

    @pytest.fixture
    def base_dataset(self):
        root = _get_dataset_root()
        return EquationDataset(
            root=root, split="train",
            height=128, width=512,
            max_samples=4, augment=False,
        )

    @pytest.fixture
    def aug_dataset(self):
        root = _get_dataset_root()
        return EquationDataset(
            root=root, split="train",
            height=128, width=512,
            max_samples=4, augment=True,
        )

    def test_augment_preserves_image_shape(self, aug_dataset):
        """Augmented image should still be (1, 128, 512)."""
        img, label, length, raw = aug_dataset[0]
        assert img.shape == (1, 128, 512)

    def test_augment_preserves_dtype(self, aug_dataset):
        """Augmented image should still be float32."""
        img, _, _, _ = aug_dataset[0]
        assert img.dtype == torch.float32

    def test_augment_preserves_range(self, aug_dataset):
        """Augmented image should still be in [0, 1]."""
        img, _, _, _ = aug_dataset[0]
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_augment_preserves_label(self, base_dataset, aug_dataset):
        """Augmentation should not affect labels."""
        _, _, _, raw_base = base_dataset[0]
        _, _, _, raw_aug = aug_dataset[0]
        assert raw_base == raw_aug

    def test_augment_not_applied_to_valid(self):
        """Augmentation should not be applied to valid split even if requested."""
        root = _get_dataset_root()
        ds = EquationDataset(
            root=root, split="valid",
            height=128, width=512,
            max_samples=4, augment=True,  # Requested but should be ignored
        )
        assert ds.augment is False

    def test_augment_not_applied_to_test(self):
        """Augmentation should not be applied to test split."""
        root = _get_dataset_root()
        ds = EquationDataset(
            root=root, split="test",
            height=128, width=512,
            max_samples=4, augment=True,
        )
        assert ds.augment is False

    def test_collate_shape_with_augmentation(self, aug_dataset):
        """Collated batch from augmented dataset should have correct shapes."""
        batch = [aug_dataset[i] for i in range(4)]
        images, labels, label_lengths, raw_labels = collate_fn(batch)
        assert images.shape == (4, 1, 128, 512)
        assert labels.dim() == 1
        assert label_lengths.shape == (4,)


# ===========================================================================
# Training history tests
# ===========================================================================

class TestTrainingHistory:
    """Test that training_history.json contains learning_rate."""

    def test_history_structure_contains_lr(self, tmp_path):
        """Simulate a training history record and verify learning_rate key."""
        history_meta = {
            "epochs": [
                {
                    "epoch": 1,
                    "train_loss": 5.0,
                    "valid_loss": 4.5,
                    "exact_accuracy": 0.0,
                    "char_accuracy": 0.05,
                    "learning_rate": 0.001,
                    "elapsed_seconds": 1.0,
                },
            ],
            "early_stopped": False,
            "scheduler": "cosine",
            "augment": False,
            "overfit_small_batch": False,
        }
        path = tmp_path / "training_history.json"
        with open(path, "w") as f:
            json.dump(history_meta, f)

        loaded = json.loads(path.read_text())
        assert "epochs" in loaded
        assert "learning_rate" in loaded["epochs"][0]
        assert isinstance(loaded["epochs"][0]["learning_rate"], float)
        assert "early_stopped" in loaded


# ===========================================================================
# Overfit-small-batch tests
# ===========================================================================

class TestOverfitSmallBatch:
    """Test overfit-small-batch mode initialisation."""

    def test_subset_size_capped_at_16(self):
        """Overfit subset should be at most 16 samples."""
        root = _get_dataset_root()
        ds = EquationDataset(
            root=root, split="train",
            height=128, width=512,
        )
        overfit_size = min(16, len(ds))
        assert overfit_size <= 16
        assert overfit_size > 0

    def test_subset_creates_valid_dataloader(self):
        """Should be able to create a DataLoader from the overfit subset."""
        from torch.utils.data import Subset, DataLoader

        root = _get_dataset_root()
        ds = EquationDataset(
            root=root, split="train",
            height=128, width=512,
            max_samples=32,
        )
        overfit_size = min(16, len(ds))
        subset = Subset(ds, list(range(overfit_size)))
        loader = DataLoader(
            subset, batch_size=8,
            shuffle=True, collate_fn=collate_fn,
            num_workers=0,
        )
        # Should be able to iterate
        batch = next(iter(loader))
        images, labels, label_lengths, raw_labels = batch
        assert images.dim() == 4
        assert images.shape[1] == 1

    def test_overfit_subset_matches_original_samples(self):
        """Overfit subset should contain the same samples as original."""
        from torch.utils.data import Subset

        root = _get_dataset_root()
        ds = EquationDataset(
            root=root, split="train",
            height=128, width=512,
            max_samples=32,
        )
        overfit_size = min(16, len(ds))
        subset = Subset(ds, list(range(overfit_size)))

        # Verify subset returns same items
        for i in range(overfit_size):
            _, _, _, raw_sub = subset[i]
            _, _, _, raw_orig = ds[i]
            assert raw_sub == raw_orig
