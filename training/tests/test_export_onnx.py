"""
test_export_onnx.py -- Sprint C: Export smoke tests.

Tests the ONNX export pipeline with dummy tensors including:
  - Model shape inspection
  - Forward pass verification
  - ONNX export with embedded weights (single-file)
  - Rejection of suspiciously tiny ONNX files
  - Numerical equivalence between PyTorch and ONNX outputs
  - Validation function coverage
"""

from __future__ import annotations

import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.models.vocabulary import VOCAB_SIZE
from training.models.cnn_ctc import build_model
from training.export_onnx import (
    inspect_model_shapes,
    export_to_onnx,
    validate_onnx,
    MIN_ONNX_FILE_SIZE_BYTES,
    MAX_NUMERICAL_DIFF,
    DEFAULT_OPSET_VERSION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_model():
    """Build a default model with random weights."""
    m = build_model(img_height=128, img_width=512)
    m.eval()
    return m


@pytest.fixture
def small_model():
    """Build a smaller model for faster tests."""
    m = build_model(img_height=64, img_width=256)
    m.eval()
    return m


# ---------------------------------------------------------------------------
# Test: Model Shape Inspection
# ---------------------------------------------------------------------------

class TestModelShapeInspection:
    """Test model input/output shape inspection."""

    def test_default_shapes(self):
        info = inspect_model_shapes()
        assert info["input_shape"] == [1, 1, 128, 512]
        assert info["output_shape"][0] == 32    # T = 512 // 16
        assert info["output_shape"][1] == 1     # B
        assert info["output_shape"][2] == VOCAB_SIZE

    def test_custom_image_size(self):
        info = inspect_model_shapes(img_height=64, img_width=256)
        assert info["input_shape"] == [1, 1, 64, 256]
        assert info["output_shape"][0] == 16    # T = 256 // 16
        assert info["time_steps"] == 16

    def test_param_count_positive(self):
        info = inspect_model_shapes()
        assert info["total_params"] > 0
        assert info["trainable_params"] > 0
        assert info["trainable_params"] <= info["total_params"]

    def test_without_bilstm(self):
        info = inspect_model_shapes(use_bilstm=False)
        assert info["use_bilstm"] is False
        assert info["output_shape"][2] == VOCAB_SIZE

    def test_feature_dimensions(self):
        info = inspect_model_shapes(img_height=128, img_width=512)
        assert info["feature_h"] == 8    # 128 // 16
        assert info["feature_w"] == 32   # 512 // 16

    def test_inspect_returns_all_keys(self):
        info = inspect_model_shapes()
        expected_keys = [
            "input_shape", "output_shape", "total_params",
            "trainable_params", "feature_h", "feature_w",
            "time_steps", "num_classes", "use_bilstm",
            "lstm_hidden", "img_height", "img_width",
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_time_steps_equals_width_div_16(self):
        for width in [256, 512, 1024]:
            info = inspect_model_shapes(img_width=width)
            assert info["time_steps"] == width // 16


# ---------------------------------------------------------------------------
# Test: Dummy Tensor Forward Pass
# ---------------------------------------------------------------------------

class TestDummyTensorForwardPass:
    """Test forward pass with dummy tensors at various batch sizes."""

    def test_single_sample(self, default_model):
        x = torch.randn(1, 1, 128, 512)
        with torch.no_grad():
            out = default_model(x)
        assert out.shape == (32, 1, VOCAB_SIZE)

    def test_batch_of_4(self, default_model):
        x = torch.randn(4, 1, 128, 512)
        with torch.no_grad():
            out = default_model(x)
        assert out.shape == (32, 4, VOCAB_SIZE)

    def test_batch_of_16(self, default_model):
        x = torch.randn(16, 1, 128, 512)
        with torch.no_grad():
            out = default_model(x)
        assert out.shape == (32, 16, VOCAB_SIZE)

    def test_output_is_log_probabilities(self, default_model):
        """Output should be log-softmax (values <= 0, exp sums to ~1)."""
        x = torch.randn(2, 1, 128, 512)
        with torch.no_grad():
            out = default_model(x)
        assert (out <= 0).all(), "Log probabilities should be <= 0"
        # Check that exp of log-probs sums to ~1 along class dim.
        probs = torch.exp(out)
        sums = probs.sum(dim=2)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_different_width(self):
        """Model with different width should produce different T."""
        model = build_model(img_height=128, img_width=256)
        model.eval()
        x = torch.randn(1, 1, 128, 256)
        with torch.no_grad():
            out = model(x)
        assert out.shape[0] == 16   # T = 256 // 16


# ---------------------------------------------------------------------------
# Test: ONNX Export (single-file, no external data)
# ---------------------------------------------------------------------------

class TestOnnxExport:
    """Test ONNX export produces a valid single-file model."""

    def test_export_creates_file(self, default_model):
        """Export should create a .onnx file that exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "test_model.onnx"
            result = export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )
            assert result == onnx_path
            assert onnx_path.exists()

    def test_export_file_size_above_minimum(self, default_model):
        """Exported model must be at least 1 MB (all weights embedded)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "test_model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )
            file_size = onnx_path.stat().st_size
            assert file_size >= MIN_ONNX_FILE_SIZE_BYTES, (
                f"ONNX file too small: {file_size:,} bytes "
                f"(minimum: {MIN_ONNX_FILE_SIZE_BYTES:,})"
            )

    def test_no_external_data_file(self, default_model):
        """Export must NOT create a .onnx.data sidecar file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "test_model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )
            data_file = Path(str(onnx_path) + ".data")
            assert not data_file.exists(), (
                f"External data file was created: {data_file}.  "
                f"Weights should be embedded in the .onnx file."
            )

    def test_export_uses_opset_17(self, default_model):
        """Default opset version should be 17."""
        assert DEFAULT_OPSET_VERSION == 17
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "test_model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
                opset_version=17,
            )
            import onnx
            model = onnx.load(str(onnx_path))
            opset = model.opset_import[0].version
            assert opset == 17, f"Expected opset 17, got {opset}"

    def test_export_file_size_under_10mb(self, default_model):
        """Exported model should be < 10 MB for our small architecture."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "test_model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )
            size_mb = onnx_path.stat().st_size / 1024 / 1024
            assert size_mb < 10, f"ONNX file too large: {size_mb:.1f} MB"

    def test_export_creates_parent_dirs(self, default_model):
        """Export should create parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "nested" / "deep" / "model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )
            assert onnx_path.exists()


# ---------------------------------------------------------------------------
# Test: Rejecting Suspiciously Tiny ONNX Files
# ---------------------------------------------------------------------------

class TestRejectTinyOnnxFiles:
    """Test that the pipeline rejects suspiciously small ONNX files."""

    def test_validate_rejects_tiny_file(self):
        """validate_onnx must fail for files smaller than MIN_ONNX_FILE_SIZE_BYTES."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_path = Path(tmpdir) / "tiny.onnx"
            # Write a small file (just a few KB of garbage).
            tiny_path.write_bytes(b"\x00" * 1024)
            result = validate_onnx(tiny_path)
            assert result is False

    def test_validate_rejects_12kb_onnx(self):
        """The original 11.9 KB export should be rejected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = Path(tmpdir) / "model.onnx"
            # Simulate the bad export: ~12 KB file.
            bad_path.write_bytes(b"\x08\x07" * 6000)  # 12000 bytes
            result = validate_onnx(bad_path)
            assert result is False

    def test_validate_rejects_nonexistent_file(self):
        """validate_onnx must fail for non-existent files."""
        result = validate_onnx("/nonexistent/path/model.onnx")
        assert result is False

    def test_export_raises_on_tiny_file(self, default_model):
        """export_to_onnx should raise RuntimeError if output is tiny.

        We mock torch.onnx.export to produce a tiny file, verifying
        that the size check catches it.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "bad_model.onnx"

            def fake_export(*args, **kwargs):
                # Write a tiny file simulating the external-data bug.
                target = args[2] if len(args) > 2 else kwargs.get("f", str(onnx_path))
                Path(target).write_bytes(b"\x00" * 500)

            with patch("torch.onnx.export", side_effect=fake_export):
                with pytest.raises(RuntimeError, match="suspiciously small"):
                    export_to_onnx(
                        default_model, onnx_path,
                        img_height=128, img_width=512,
                    )

    def test_min_size_constant_is_1mb(self):
        """The minimum file size threshold should be 1 MB."""
        assert MIN_ONNX_FILE_SIZE_BYTES == 1 * 1024 * 1024


# ---------------------------------------------------------------------------
# Test: ONNX Validation (full pipeline)
# ---------------------------------------------------------------------------

class TestOnnxValidation:
    """Test the comprehensive ONNX validation function."""

    def test_valid_export_passes_all_checks(self, default_model):
        """A proper export should pass all validation checks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )
            result = validate_onnx(
                onnx_path, model=default_model,
                img_height=128, img_width=512,
            )
            assert result is True

    def test_validation_without_model_skips_numerical(self, default_model):
        """Validation without a PyTorch model should still check structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )
            result = validate_onnx(
                onnx_path, model=None,
                img_height=128, img_width=512,
            )
            assert result is True

    def test_numerical_diff_is_small(self, default_model):
        """Max numerical difference between PyTorch and ONNX must be small."""
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )

            session = ort.InferenceSession(str(onnx_path))
            rng = np.random.RandomState(42)
            dummy = rng.randn(1, 1, 128, 512).astype(np.float32)

            ort_out = session.run(None, {"image": dummy})[0]

            pt_out = default_model(torch.from_numpy(dummy))
            pt_out = pt_out.detach().numpy()

            max_diff = float(np.max(np.abs(pt_out - ort_out)))
            assert max_diff < MAX_NUMERICAL_DIFF, (
                f"Numerical diff too large: {max_diff:.6e}"
            )

    def test_onnx_output_shape_matches_pytorch(self, default_model):
        """ONNX output shape must exactly match PyTorch output shape."""
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )

            session = ort.InferenceSession(str(onnx_path))
            dummy = np.random.randn(1, 1, 128, 512).astype(np.float32)

            ort_shape = session.run(None, {"image": dummy})[0].shape
            pt_shape = default_model(torch.from_numpy(dummy)).shape

            assert ort_shape == tuple(pt_shape), (
                f"Shape mismatch: ONNX={ort_shape}, PyTorch={tuple(pt_shape)}"
            )

    def test_onnx_checker_passes(self, default_model):
        """onnx.checker.check_model should pass on a valid export."""
        import onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )
            onnx_model = onnx.load(str(onnx_path))
            # Should not raise.
            onnx.checker.check_model(onnx_model)


# ---------------------------------------------------------------------------
# Test: Validation dependency handling
# ---------------------------------------------------------------------------

class TestValidationDependencyHandling:
    """Test that missing validation dependencies cause clear failures."""

    def test_missing_onnx_returns_false(self, default_model):
        """If 'onnx' is not importable, validation must fail (not skip)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )

            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "onnx":
                    raise ImportError("Mocked: onnx not installed")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = validate_onnx(onnx_path)
            assert result is False

    def test_missing_onnxruntime_returns_false(self, default_model):
        """If 'onnxruntime' is not importable, validation must fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "model.onnx"
            export_to_onnx(
                default_model, onnx_path,
                img_height=128, img_width=512,
            )

            import builtins
            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if name == "onnxruntime":
                    raise ImportError("Mocked: onnxruntime not installed")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = validate_onnx(onnx_path)
            assert result is False
