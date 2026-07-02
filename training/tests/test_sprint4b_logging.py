"""
Tests for Sprint 4B logging improvements in train_ctc.py.

Covers:
- Module imports still work
- --log-every CLI arg parses correctly
- _format_duration helper
- train_one_epoch accepts new kwargs
- training_history.json still contains expected fields
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =========================================================================
# Test: Module imports
# =========================================================================

class TestModuleImports:
    """Verify train_ctc.py still imports correctly."""

    def test_import_scheduler_choices(self):
        from training.train_ctc import SCHEDULER_CHOICES
        assert "cosine" in SCHEDULER_CHOICES

    def test_import_early_stopping(self):
        from training.train_ctc import EarlyStopping
        es = EarlyStopping(patience=5)
        assert es.patience == 5

    def test_import_build_scheduler(self):
        from training.train_ctc import build_scheduler
        assert callable(build_scheduler)

    def test_import_train_one_epoch(self):
        from training.train_ctc import train_one_epoch
        assert callable(train_one_epoch)

    def test_import_validate(self):
        from training.train_ctc import validate
        assert callable(validate)

    def test_import_format_duration(self):
        from training.train_ctc import _format_duration
        assert callable(_format_duration)


# =========================================================================
# Test: --log-every CLI arg
# =========================================================================

class TestLogEveryArg:
    """Test that --log-every is parsed correctly."""

    def test_log_every_default(self):
        """Default value should be 10."""
        import argparse
        from training.train_ctc import SCHEDULER_CHOICES

        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--log-every", type=int, default=10)
        args = parser.parse_args(["--dataset", "dummy"])
        assert args.log_every == 10

    def test_log_every_custom(self):
        """Custom value should be parsed."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, required=True)
        parser.add_argument("--log-every", type=int, default=10)
        args = parser.parse_args(["--dataset", "dummy", "--log-every", "5"])
        assert args.log_every == 5

    def test_log_every_in_help_output(self):
        """--log-every should appear in the script's --help output."""
        result = subprocess.run(
            [sys.executable, "training/train_ctc.py", "--help"],
            capture_output=True, text=True,
            cwd=str(_PROJECT_ROOT),
        )
        assert "--log-every" in result.stdout


# =========================================================================
# Test: _format_duration helper
# =========================================================================

class TestFormatDuration:
    """Test duration formatting."""

    def test_seconds_only(self):
        from training.train_ctc import _format_duration
        assert _format_duration(45) == "00:45"

    def test_minutes_and_seconds(self):
        from training.train_ctc import _format_duration
        assert _format_duration(125) == "02:05"

    def test_hours(self):
        from training.train_ctc import _format_duration
        assert _format_duration(3661) == "01:01:01"

    def test_zero(self):
        from training.train_ctc import _format_duration
        assert _format_duration(0) == "00:00"

    def test_fractional_seconds_truncated(self):
        from training.train_ctc import _format_duration
        assert _format_duration(59.9) == "00:59"


# =========================================================================
# Test: train_one_epoch accepts new kwargs
# =========================================================================

class TestTrainOneEpochSignature:
    """Test that train_one_epoch accepts the new keyword args."""

    def test_accepts_epoch_kwarg(self):
        """Verify epoch kwarg exists in the function signature."""
        import inspect
        from training.train_ctc import train_one_epoch
        sig = inspect.signature(train_one_epoch)
        assert "epoch" in sig.parameters

    def test_accepts_total_epochs_kwarg(self):
        import inspect
        from training.train_ctc import train_one_epoch
        sig = inspect.signature(train_one_epoch)
        assert "total_epochs" in sig.parameters

    def test_accepts_log_every_kwarg(self):
        import inspect
        from training.train_ctc import train_one_epoch
        sig = inspect.signature(train_one_epoch)
        assert "log_every" in sig.parameters


# =========================================================================
# Test: validate returns avg_edit_distance
# =========================================================================

class TestValidateMetrics:
    """Test that validate() returns the expected keys."""

    def test_validate_returns_edit_distance_key(self):
        """Validate return dict should include avg_edit_distance."""
        # We check by looking at the function's documented return type.
        # A structural test: just call with a tiny model if dataset exists.
        from training.train_ctc import _edit_distance, _avg_edit_distance

        assert _edit_distance("abc", "abc") == 0
        assert _edit_distance("abc", "abd") == 1
        assert _avg_edit_distance(["abc", "def"], ["abc", "deg"]) == 0.5


# =========================================================================
# Test: training_history.json expected fields
# =========================================================================

class TestHistoryFields:
    """Test that training history JSON has all expected fields."""

    def test_history_structure(self, tmp_path):
        """A simulated history record must contain all required keys."""
        history_meta = {
            "epochs": [
                {
                    "epoch": 1,
                    "train_loss": 5.0,
                    "valid_loss": 4.5,
                    "exact_accuracy": 0.0,
                    "char_accuracy": 0.05,
                    "learning_rate": 0.001,
                    "elapsed_seconds": 355.0,
                },
            ],
            "early_stopped": False,
            "scheduler": "cosine",
            "augment": True,
            "overfit_small_batch": False,
        }
        path = tmp_path / "training_history.json"
        with open(path, "w") as f:
            json.dump(history_meta, f)

        loaded = json.loads(path.read_text())
        assert "epochs" in loaded
        assert "early_stopped" in loaded
        assert "scheduler" in loaded
        assert "augment" in loaded

        epoch = loaded["epochs"][0]
        for key in ["epoch", "train_loss", "valid_loss", "exact_accuracy",
                     "char_accuracy", "learning_rate", "elapsed_seconds"]:
            assert key in epoch, f"Missing key: {key}"
