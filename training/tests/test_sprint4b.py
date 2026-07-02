"""
test_sprint4b.py – Tests for Sprint 4B features.

Covers:
- Equation form classifier
- Per-form metrics aggregation
- Training summary parsing
- Prediction samples export format
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.analyze_predictions import (
    classify_equation_form,
    compute_per_form_metrics,
    build_prediction_samples,
    _edit_distance,
    _exact_match,
    _char_accuracy,
    _avg_edit_distance,
)
from training.summarize_training_run import (
    load_training_history,
    summarize,
)


# =========================================================================
# Test: Equation form classifier
# =========================================================================

class TestEquationFormClassifier:
    """Test classify_equation_form for all known forms."""

    def test_x_plus_a_eq_b(self):
        assert classify_equation_form("x+3=7") == "x+a=b"
        assert classify_equation_form("x+15=20") == "x+a=b"

    def test_x_minus_a_eq_b(self):
        assert classify_equation_form("x-4=10") == "x-a=b"
        assert classify_equation_form("x-12=-3") == "x-a=b"

    def test_ax_eq_b(self):
        assert classify_equation_form("3x=9") == "ax=b"
        assert classify_equation_form("12x=-36") == "ax=b"

    def test_ax_plus_b_eq_c(self):
        assert classify_equation_form("2x+5=11") == "ax+b=c"
        assert classify_equation_form("9x+1=10") == "ax+b=c"

    def test_ax_minus_b_eq_c(self):
        assert classify_equation_form("3x-2=7") == "ax-b=c"
        assert classify_equation_form("5x-10=-5") == "ax-b=c"

    def test_x_div_a_eq_b(self):
        assert classify_equation_form("x/3=4") == "x/a=b"
        assert classify_equation_form("x/10=-2") == "x/a=b"

    def test_ax_div_b_eq_c(self):
        assert classify_equation_form("4x/2=8") == "ax/b=c"
        assert classify_equation_form("6x/3=-6") == "ax/b=c"

    def test_a_paren_x_plus_b_eq_c(self):
        assert classify_equation_form("3(x+2)=15") == "a(x+b)=c"
        assert classify_equation_form("5(x+1)=30") == "a(x+b)=c"

    def test_a_paren_x_minus_b_eq_c(self):
        assert classify_equation_form("2(x-4)=10") == "a(x-b)=c"
        assert classify_equation_form("7(x-3)=-7") == "a(x-b)=c"

    def test_unknown_form(self):
        assert classify_equation_form("hello") == "unknown"
        assert classify_equation_form("x^2+1=5") == "unknown"
        assert classify_equation_form("") == "unknown"

    def test_negative_coefficients(self):
        assert classify_equation_form("x+-3=7") == "x+a=b"
        assert classify_equation_form("3x=-9") == "ax=b"

    def test_whitespace_handling(self):
        """Whitespace should be stripped before matching."""
        assert classify_equation_form("  x+3=7  ") == "x+a=b"
        # Spaces are stripped, so "3 x = 9" → "3x=9" → ax=b
        assert classify_equation_form("3 x = 9") == "ax=b"
        assert classify_equation_form("abc def") == "unknown"


# =========================================================================
# Test: Per-form metrics aggregation
# =========================================================================

class TestPerFormMetrics:
    """Test compute_per_form_metrics aggregation logic."""

    def test_perfect_predictions(self):
        targets = ["x+1=2", "3x=6", "2x+1=5"]
        preds = ["x+1=2", "3x=6", "2x+1=5"]
        result = compute_per_form_metrics(preds, targets)
        for form_metrics in result.values():
            assert form_metrics["exact_accuracy"] == 1.0
            assert form_metrics["char_accuracy"] == 1.0
            assert form_metrics["avg_edit_distance"] == 0.0

    def test_wrong_predictions(self):
        targets = ["x+1=2", "x+3=5"]
        preds = ["x+2=3", "x+4=6"]
        result = compute_per_form_metrics(preds, targets)
        # Both are x+a=b form
        assert "x+a=b" in result
        assert result["x+a=b"]["sample_count"] == 2
        assert result["x+a=b"]["exact_accuracy"] == 0.0

    def test_mixed_forms(self):
        targets = ["x+1=2", "3x=9", "2(x+1)=6"]
        preds = ["x+1=2", "3x=9", "2(x+1)=6"]
        result = compute_per_form_metrics(preds, targets)
        assert len(result) == 3
        assert result["x+a=b"]["sample_count"] == 1
        assert result["ax=b"]["sample_count"] == 1
        assert result["a(x+b)=c"]["sample_count"] == 1

    def test_empty_input(self):
        result = compute_per_form_metrics([], [])
        assert result == {}

    def test_sample_count_aggregation(self):
        targets = ["x+1=2", "x+3=5", "x+5=8"]
        preds = ["x+1=2", "x+3=5", "x+5=8"]
        result = compute_per_form_metrics(preds, targets)
        assert result["x+a=b"]["sample_count"] == 3


# =========================================================================
# Test: Edit distance
# =========================================================================

class TestEditDistance:
    """Test Levenshtein edit distance implementation."""

    def test_identical_strings(self):
        assert _edit_distance("abc", "abc") == 0

    def test_empty_strings(self):
        assert _edit_distance("", "") == 0
        assert _edit_distance("abc", "") == 3
        assert _edit_distance("", "abc") == 3

    def test_single_edit(self):
        assert _edit_distance("abc", "adc") == 1  # substitution
        assert _edit_distance("abc", "abcd") == 1  # insertion
        assert _edit_distance("abc", "ab") == 1  # deletion

    def test_equation_edit(self):
        assert _edit_distance("3x+1=7", "3x+1=8") == 1


# =========================================================================
# Test: Prediction samples export
# =========================================================================

class TestPredictionSamples:
    """Test build_prediction_samples output format."""

    def test_sample_format(self):
        preds = ["x+1=2", "3x=7"]
        targets = ["x+1=2", "3x=6"]
        samples = build_prediction_samples(preds, targets)
        assert len(samples) == 2
        assert samples[0]["label"] == "x+1=2"
        assert samples[0]["prediction"] == "x+1=2"
        assert samples[0]["exact"] is True
        assert samples[0]["edit_distance"] == 0
        assert samples[1]["exact"] is False
        assert samples[1]["edit_distance"] == 1

    def test_max_samples_limit(self):
        preds = [f"x+{i}={i+1}" for i in range(200)]
        targets = preds.copy()
        samples = build_prediction_samples(preds, targets, max_samples=50)
        assert len(samples) == 50

    def test_keys_present(self):
        samples = build_prediction_samples(["a"], ["b"])
        required_keys = {"label", "prediction", "exact", "edit_distance"}
        assert set(samples[0].keys()) == required_keys


# =========================================================================
# Test: Training summary parsing
# =========================================================================

class TestTrainingSummary:
    """Test summarize_training_run.py logic."""

    def _make_history(self, epochs: list[dict], **kwargs) -> dict:
        """Helper to create a training history dict."""
        return {
            "epochs": epochs,
            "early_stopped": kwargs.get("early_stopped", False),
            "scheduler": kwargs.get("scheduler", "cosine"),
            "augment": kwargs.get("augment", True),
        }

    def test_basic_summary(self):
        epochs = [
            {"epoch": 1, "train_loss": 5.0, "valid_loss": 6.0,
             "exact_accuracy": 0.0, "char_accuracy": 0.1, "learning_rate": 1e-3},
            {"epoch": 2, "train_loss": 3.0, "valid_loss": 4.0,
             "exact_accuracy": 0.1, "char_accuracy": 0.3, "learning_rate": 5e-4},
            {"epoch": 3, "train_loss": 2.5, "valid_loss": 3.5,
             "exact_accuracy": 0.2, "char_accuracy": 0.5, "learning_rate": 1e-4},
        ]
        history = self._make_history(epochs, early_stopped=False)
        summary = summarize(history)
        assert summary["total_epochs"] == 3
        assert summary["best_epoch"] == 3
        assert summary["best_valid_loss"] == 3.5
        assert summary["final_train_loss"] == 2.5
        assert summary["final_exact_accuracy"] == 0.2
        assert summary["early_stopped"] is False

    def test_best_epoch_not_last(self):
        epochs = [
            {"epoch": 1, "train_loss": 5.0, "valid_loss": 3.0,
             "exact_accuracy": 0.1, "char_accuracy": 0.2, "learning_rate": 1e-3},
            {"epoch": 2, "train_loss": 3.0, "valid_loss": 5.0,
             "exact_accuracy": 0.15, "char_accuracy": 0.3, "learning_rate": 5e-4},
        ]
        history = self._make_history(epochs)
        summary = summarize(history)
        assert summary["best_epoch"] == 1
        assert summary["best_valid_loss"] == 3.0
        assert summary["final_valid_loss"] == 5.0

    def test_early_stopped_flag(self):
        epochs = [
            {"epoch": 1, "train_loss": 5.0, "valid_loss": 6.0,
             "exact_accuracy": 0.0, "char_accuracy": 0.1, "learning_rate": 1e-3},
        ]
        history = self._make_history(epochs, early_stopped=True)
        summary = summarize(history)
        assert summary["early_stopped"] is True

    def test_scheduler_and_augment_fields(self):
        epochs = [
            {"epoch": 1, "train_loss": 5.0, "valid_loss": 6.0,
             "exact_accuracy": 0.0, "char_accuracy": 0.1, "learning_rate": 1e-3},
        ]
        history = self._make_history(
            epochs, scheduler="reduce_on_plateau", augment=False,
        )
        summary = summarize(history)
        assert summary["scheduler"] == "reduce_on_plateau"
        assert summary["augment"] is False

    def test_empty_epochs_raises(self):
        with pytest.raises(ValueError):
            summarize({"epochs": []})

    def test_load_and_summarize_roundtrip(self):
        """Write a history file, load it, and summarize."""
        epochs = [
            {"epoch": 1, "train_loss": 4.0, "valid_loss": 5.0,
             "exact_accuracy": 0.05, "char_accuracy": 0.2, "learning_rate": 1e-3},
        ]
        history = self._make_history(epochs)

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "training_history.json"
            with open(history_path, "w") as f:
                json.dump(history, f)

            loaded = load_training_history(tmpdir)
            summary = summarize(loaded)
            assert summary["total_epochs"] == 1

    def test_load_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_training_history(tmpdir)
