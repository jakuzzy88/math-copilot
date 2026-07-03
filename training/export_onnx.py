"""
export_onnx.py -- ONNX export for the CNN-CTC model.

Sprint C: Export Readiness (prepared but not executed on real checkpoints).

This module provides:
  1. Model shape inspection (input/output shapes, parameter count)
  2. ONNX export function with dynamic axes for batch and width
  3. Comprehensive export validation:
     - File existence and minimum size check (>= 1 MB)
     - onnx.checker.check_model
     - onnxruntime inference test
     - Output shape matching between PyTorch and ONNX
     - Numerical equivalence (max abs diff < 1e-4)
  4. Smoke test with dummy tensors

Usage (inspect only):
    python training/export_onnx.py --inspect

Usage (export from checkpoint):
    python training/export_onnx.py \
        --checkpoint training/runs/synthetic_v2_full/best_model.pt \
        --out training/runs/synthetic_v2_full/model.onnx

Usage (smoke test with dummy tensor):
    python training/export_onnx.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.models.vocabulary import VOCAB_SIZE
from training.models.cnn_ctc import build_model, CNNCTC


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum acceptable ONNX file size for our ~1.4M param model.
# float32 weights alone are ~5.5 MB; anything under 1 MB is suspicious.
MIN_ONNX_FILE_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB

# Maximum acceptable numerical difference between PyTorch and ONNX outputs.
MAX_NUMERICAL_DIFF = 1e-4

# Default ONNX opset version.  Using 17 for broad compatibility.
DEFAULT_OPSET_VERSION = 17


# ---------------------------------------------------------------------------
# Shape inspection
# ---------------------------------------------------------------------------

def inspect_model_shapes(
    img_height: int = 128,
    img_width: int = 512,
    num_classes: int = VOCAB_SIZE,
    use_bilstm: bool = True,
    lstm_hidden: int = 128,
) -> dict:
    """Inspect model input/output shapes and parameter counts.

    Returns a dict with:
        input_shape, output_shape, total_params, trainable_params,
        feature_h, feature_w, time_steps, num_classes
    """
    model = build_model(
        img_height=img_height,
        img_width=img_width,
        num_classes=num_classes,
        use_bilstm=use_bilstm,
        lstm_hidden=lstm_hidden,
    )
    model.eval()

    # Compute shapes with a dummy forward pass.
    dummy_input = torch.randn(1, 1, img_height, img_width)
    with torch.no_grad():
        output = model(dummy_input)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "input_shape": list(dummy_input.shape),        # [B, 1, H, W]
        "output_shape": list(output.shape),             # [T, B, C]
        "total_params": total_params,
        "trainable_params": trainable_params,
        "feature_h": model.feature_h,
        "feature_w": model.feature_w,
        "time_steps": output.shape[0],
        "num_classes": num_classes,
        "use_bilstm": use_bilstm,
        "lstm_hidden": lstm_hidden,
        "img_height": img_height,
        "img_width": img_width,
    }


def print_model_inspection(info: dict) -> None:
    """Print a formatted model inspection report."""
    print("\n" + "=" * 60)
    print("MODEL SHAPE INSPECTION")
    print("=" * 60)
    print(f"  Input shape:       {info['input_shape']}  (B, C, H, W)")
    print(f"  Output shape:      {info['output_shape']}  (T, B, num_classes)")
    print(f"  Image size:        {info['img_height']} x {info['img_width']}")
    print(f"  Time steps (T):    {info['time_steps']}")
    print(f"  Num classes:       {info['num_classes']}")
    print(f"  BiLSTM:            {info['use_bilstm']} (hidden={info['lstm_hidden']})")
    print(f"  Feature map:       {info['feature_h']} x {info['feature_w']}")
    print(f"  Total params:      {info['total_params']:,}")
    print(f"  Trainable params:  {info['trainable_params']:,}")
    print(f"  Model size (est):  ~{info['total_params'] * 4 / 1024 / 1024:.1f} MB (float32)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    img_height: int = 128,
    img_width: int = 512,
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> Path:
    """Export a CNN-CTC model to ONNX format.

    Uses dynamic axes for batch size and input width to allow
    variable-length equation images at inference time.

    Key: Forces ``dynamo=False`` and ``external_data=False`` for
    PyTorch 2.12+ compatibility, ensuring all weights are embedded
    in a single .onnx file rather than split into .onnx + .onnx.data.

    Args:
        model: Trained CNNCTC model instance.
        output_path: Path for the output .onnx file.
        img_height: Expected input height.
        img_width: Default input width (dynamic axis allows variation).
        opset_version: ONNX opset version (default 17).

    Returns:
        The resolved output path.

    Raises:
        RuntimeError: If the exported file is missing or suspiciously small.
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 1, img_height, img_width)

    # Use the legacy (TorchScript-based) export path for stability.
    # PyTorch 2.12+ defaults to dynamo=True which may produce
    # external data files.  We force both flags explicitly.
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["log_probs"],
        dynamic_axes={
            "image": {0: "batch_size", 3: "width"},
            "log_probs": {0: "time_steps", 1: "batch_size"},
        },
        dynamo=False,
        external_data=False,
    )

    # --- Immediate sanity checks ---
    if not output_path.exists():
        raise RuntimeError(f"ONNX export failed: {output_path} was not created")

    file_size = output_path.stat().st_size
    print(f"ONNX model exported to {output_path}")
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB ({file_size:,} bytes)")

    if file_size < MIN_ONNX_FILE_SIZE_BYTES:
        raise RuntimeError(
            f"ONNX export produced a suspiciously small file: "
            f"{file_size:,} bytes (< {MIN_ONNX_FILE_SIZE_BYTES:,} byte minimum).  "
            f"This likely means weights were not embedded.  "
            f"Check for a stray .onnx.data file."
        )

    return output_path


# ---------------------------------------------------------------------------
# ONNX validation
# ---------------------------------------------------------------------------

def validate_onnx(
    onnx_path: str | Path,
    model: nn.Module | None = None,
    img_height: int = 128,
    img_width: int = 512,
) -> bool:
    """Validate an exported ONNX model comprehensively.

    Checks:
      1. File exists and meets minimum size threshold.
      2. onnx.checker.check_model passes.
      3. onnxruntime can load the model and run inference.
      4. ONNX output shape matches PyTorch output shape.
      5. Numerical difference between PyTorch and ONNX outputs is small.

    If ``onnx`` or ``onnxruntime`` are not installed, prints a clear
    error message and returns False (validation failure, not silent skip).

    Args:
        onnx_path: Path to the exported .onnx file.
        model: Optional PyTorch model for numerical comparison.
               If None, only structural validation is performed.
        img_height: Image height used during export.
        img_width: Image width used during export.

    Returns:
        True if all validation checks pass.

    Raises:
        ImportError: Printed as clear message; returns False.
        AssertionError: On shape or numerical mismatch.
    """
    onnx_path = Path(onnx_path)

    # -- Step 0: Check file existence and size --
    if not onnx_path.exists():
        print(f"  [FAIL] ONNX file does not exist: {onnx_path}")
        return False

    file_size = onnx_path.stat().st_size
    if file_size < MIN_ONNX_FILE_SIZE_BYTES:
        print(
            f"  [FAIL] ONNX file is suspiciously small: "
            f"{file_size:,} bytes (minimum: {MIN_ONNX_FILE_SIZE_BYTES:,})"
        )
        return False
    print(f"  [PASS] File size: {file_size / 1024 / 1024:.2f} MB")

    # -- Step 1: Check onnx package availability --
    try:
        import onnx
    except ImportError:
        print("  [FAIL] 'onnx' package is not installed.")
        print("         Install with: pip install onnx")
        print("         Validation CANNOT proceed without this package.")
        return False

    # -- Step 2: onnx.checker.check_model --
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("  [PASS] onnx.checker.check_model")
    except Exception as e:
        print(f"  [FAIL] onnx.checker.check_model: {e}")
        return False

    # -- Step 3: Check onnxruntime availability --
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [FAIL] 'onnxruntime' package is not installed.")
        print("         Install with: pip install onnxruntime")
        print("         Validation CANNOT proceed without this package.")
        return False

    # -- Step 4: Load model and run inference via onnxruntime --
    import numpy as np

    try:
        session = ort.InferenceSession(str(onnx_path))
    except Exception as e:
        print(f"  [FAIL] onnxruntime could not load model: {e}")
        return False
    print("  [PASS] onnxruntime loaded model")

    # Use a fixed seed for deterministic dummy input.
    rng = np.random.RandomState(42)
    dummy_input_np = rng.randn(1, 1, img_height, img_width).astype(np.float32)

    try:
        ort_outputs = session.run(None, {"image": dummy_input_np})
    except Exception as e:
        print(f"  [FAIL] onnxruntime inference failed: {e}")
        return False

    onnx_output = ort_outputs[0]
    expected_t = img_width // 16
    expected_shape = (expected_t, 1, VOCAB_SIZE)

    if onnx_output.shape != expected_shape:
        print(
            f"  [FAIL] ONNX output shape mismatch: "
            f"{onnx_output.shape} != expected {expected_shape}"
        )
        return False
    print(f"  [PASS] ONNX output shape: {onnx_output.shape}")

    # -- Step 5: Numerical comparison with PyTorch (if model provided) --
    if model is not None:
        model.eval()
        dummy_input_pt = torch.from_numpy(dummy_input_np)
        with torch.no_grad():
            pt_output = model(dummy_input_pt).numpy()

        if pt_output.shape != onnx_output.shape:
            print(
                f"  [FAIL] Shape mismatch: PyTorch {pt_output.shape} "
                f"vs ONNX {onnx_output.shape}"
            )
            return False
        print(f"  [PASS] PyTorch output shape matches ONNX: {pt_output.shape}")

        max_diff = float(np.max(np.abs(pt_output - onnx_output)))
        mean_diff = float(np.mean(np.abs(pt_output - onnx_output)))
        print(f"  [INFO] Numerical diff: max={max_diff:.6e}, mean={mean_diff:.6e}")

        if max_diff > MAX_NUMERICAL_DIFF:
            print(
                f"  [FAIL] Numerical difference too large: "
                f"max_diff={max_diff:.6e} > threshold={MAX_NUMERICAL_DIFF:.1e}"
            )
            return False
        print(f"  [PASS] Numerical equivalence (max diff < {MAX_NUMERICAL_DIFF:.1e})")
    else:
        print("  [SKIP] Numerical comparison (no PyTorch model provided)")

    print("  [PASS] All validation checks passed")
    return True


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test() -> bool:
    """Run a smoke test: build model, export to temp ONNX, verify shapes.

    This does NOT require a trained checkpoint — uses random weights.
    Returns True if all checks pass.
    """
    print("\n" + "=" * 60)
    print("ONNX EXPORT SMOKE TEST")
    print("=" * 60)

    img_h, img_w = 128, 512

    # 1. Build model with random weights.
    model = build_model(img_height=img_h, img_width=img_w)
    model.eval()
    print("  [1/5] Model built (random weights)")

    # 2. Forward pass with dummy tensor.
    dummy = torch.randn(2, 1, img_h, img_w)
    with torch.no_grad():
        output = model(dummy)
    expected_t = img_w // 16  # 32
    assert output.shape == (expected_t, 2, VOCAB_SIZE), \
        f"Output shape mismatch: {output.shape}"
    print(f"  [2/5] Forward pass OK: {tuple(output.shape)}")

    # 3. Export to temp file.
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "test_model.onnx"
        try:
            export_to_onnx(model, onnx_path, img_height=img_h, img_width=img_w)
            file_size = onnx_path.stat().st_size
            print(f"  [3/5] ONNX export OK: {file_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            print(f"  [3/5] ONNX export FAILED: {e}")
            print("  [NOTE] torch.onnx.export may require 'onnx' package.")
            print("         pip install onnx")
            print("  Continuing with shape validation only...")
            print("=" * 60)
            return False

        # Verify no stray external data file was created.
        data_file = Path(str(onnx_path) + ".data")
        if data_file.exists():
            print(f"  [WARN] Stray external data file found: {data_file}")
            print("         Weights may not be embedded in the .onnx file!")

        # 4. Full validation including numerical comparison.
        validated = validate_onnx(
            onnx_path, model=model, img_height=img_h, img_width=img_w,
        )
        status = "OK" if validated else "FAILED"
        print(f"  [4/5] ONNX validation: {status}")

        if not validated:
            print("\n  SMOKE TEST: FAILED")
            print("=" * 60)
            return False

    # 5. Test with different batch sizes (shape flexibility).
    for batch_size in [1, 4, 8]:
        dummy_b = torch.randn(batch_size, 1, img_h, img_w)
        with torch.no_grad():
            out_b = model(dummy_b)
        assert out_b.shape == (expected_t, batch_size, VOCAB_SIZE)
    print("  [5/5] Batch size flexibility: OK (tested B=1,4,8)")

    print("\n  SMOKE TEST: ALL PASSED")
    print("=" * 60)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNX export toolkit for CNN-CTC model.",
    )
    parser.add_argument("--inspect", action="store_true",
                        help="Print model shape inspection report.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run export smoke test with dummy tensors.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained checkpoint (.pt) for export.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for the ONNX file.")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET_VERSION,
                        help=f"ONNX opset version (default: {DEFAULT_OPSET_VERSION}).")
    parser.add_argument("--height", type=int, default=128,
                        help="Image height (default: 128).")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width (default: 512).")
    args = parser.parse_args()

    if args.inspect:
        info = inspect_model_shapes(img_height=args.height, img_width=args.width)
        print_model_inspection(info)
        return

    if args.smoke_test:
        success = run_smoke_test()
        sys.exit(0 if success else 1)

    if args.checkpoint:
        if not args.out:
            args.out = str(Path(args.checkpoint).parent / "model.onnx")

        device = torch.device("cpu")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        train_args = ckpt.get("args", {})
        img_h = train_args.get("height", args.height)
        img_w = train_args.get("width", args.width)

        model = build_model(img_height=img_h, img_width=img_w, num_classes=VOCAB_SIZE)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

        info = inspect_model_shapes(img_height=img_h, img_width=img_w)
        print_model_inspection(info)

        # Remove any stale external data file from prior bad exports.
        stale_data = Path(args.out + ".data")
        if stale_data.exists():
            stale_data.unlink()
            print(f"  Removed stale external data file: {stale_data}")

        export_to_onnx(
            model, args.out,
            img_height=img_h, img_width=img_w,
            opset_version=args.opset,
        )

        print("\n--- Validating exported ONNX model ---")
        valid = validate_onnx(
            args.out, model=model,
            img_height=img_h, img_width=img_w,
        )
        if not valid:
            print("\n[ERROR] ONNX validation FAILED. The exported model is unreliable.")
            sys.exit(1)

        print("\n[OK] ONNX export and validation completed successfully.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
