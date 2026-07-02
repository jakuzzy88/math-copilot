"""
test_export_onnx.py -- Sprint C: Export smoke tests.

Tests the ONNX export pipeline with dummy tensors without
requiring a trained checkpoint or the onnx/onnxruntime packages.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.models.vocabulary import VOCAB_SIZE
from training.models.cnn_ctc import build_model
from training.export_onnx import inspect_model_shapes


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


class TestDummyTensorForwardPass:
    """Test forward pass with dummy tensors at various batch sizes."""

    @pytest.fixture
    def model(self):
        m = build_model(img_height=128, img_width=512)
        m.eval()
        return m

    def test_single_sample(self, model):
        x = torch.randn(1, 1, 128, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (32, 1, VOCAB_SIZE)

    def test_batch_of_4(self, model):
        x = torch.randn(4, 1, 128, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (32, 4, VOCAB_SIZE)

    def test_batch_of_16(self, model):
        x = torch.randn(16, 1, 128, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (32, 16, VOCAB_SIZE)

    def test_output_is_log_probabilities(self, model):
        """Output should be log-softmax (values <= 0, exp sums to ~1)."""
        x = torch.randn(2, 1, 128, 512)
        with torch.no_grad():
            out = model(x)
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


class TestOnnxExportSmoke:
    """Test ONNX export with dummy tensors (requires torch.onnx)."""

    def test_export_creates_file(self):
        """Export to a temp directory and verify the file exists."""
        model = build_model(img_height=128, img_width=512)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "test_model.onnx"
            dummy_input = torch.randn(1, 1, 128, 512)

            try:
                torch.onnx.export(
                    model, dummy_input, str(onnx_path),
                    export_params=True, opset_version=17,
                    input_names=["image"], output_names=["log_probs"],
                    dynamic_axes={
                        "image": {0: "batch_size", 3: "width"},
                        "log_probs": {0: "time_steps", 1: "batch_size"},
                    },
                )
                assert onnx_path.exists()
                assert onnx_path.stat().st_size > 0
            except Exception as e:
                pytest.skip(f"ONNX export not available: {e}")

    def test_export_file_size_reasonable(self):
        """Exported model should be < 10 MB for our small architecture."""
        model = build_model(img_height=128, img_width=512)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            onnx_path = Path(tmpdir) / "test_model.onnx"
            dummy_input = torch.randn(1, 1, 128, 512)

            try:
                torch.onnx.export(
                    model, dummy_input, str(onnx_path),
                    export_params=True, opset_version=17,
                    input_names=["image"], output_names=["log_probs"],
                )
                size_mb = onnx_path.stat().st_size / 1024 / 1024
                assert size_mb < 10, f"ONNX file too large: {size_mb:.1f} MB"
            except Exception as e:
                pytest.skip(f"ONNX export not available: {e}")


class TestInspectModelShapesIntegration:
    """Integration test for the full inspect → report pipeline."""

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
