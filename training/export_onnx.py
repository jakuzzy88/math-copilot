"""
export_onnx.py -- ONNX export for the CNN-CTC model.

Sprint C: Export Readiness (prepared but not executed on real checkpoints).

This module provides:
  1. Model shape inspection (input/output shapes, parameter count)
  2. ONNX export function with dynamic axes for batch and width
  3. Export validation via onnxruntime (if available)
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
    opset_version: int = 17,
) -> Path:
    """Export a CNN-CTC model to ONNX format.

    Uses dynamic axes for batch size and input width to allow
    variable-length equation images at inference time.

    Args:
        model: Trained CNNCTC model instance.
        output_path: Path for the output .onnx file.
        img_height: Expected input height.
        img_width: Default input width (dynamic axis allows variation).
        opset_version: ONNX opset version (default 17).

    Returns:
        The resolved output path.
    """
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 1, img_height, img_width)

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
    )

    print(f"ONNX model exported to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")
    return output_path


# ---------------------------------------------------------------------------
# ONNX validation (optional, requires onnxruntime)
# ---------------------------------------------------------------------------

def validate_onnx(
    onnx_path: str | Path,
    img_height: int = 128,
    img_width: int = 512,
) -> bool:
    """Validate an ONNX model by running inference with onnxruntime.

    Returns True if validation passes.
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  [SKIP] onnx / onnxruntime not installed. Install with:")
        print("         pip install onnx onnxruntime")
        return False

    onnx_path = Path(onnx_path)

    # Check model structure.
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print("  ONNX model structure: valid")

    # Run inference.
    session = ort.InferenceSession(str(onnx_path))
    import numpy as np
    dummy_input = np.random.randn(1, 1, img_height, img_width).astype(np.float32)
    outputs = session.run(None, {"image": dummy_input})

    output_shape = outputs[0].shape
    expected_t = img_width // 16
    assert output_shape[0] == expected_t, f"Time steps mismatch: {output_shape[0]} != {expected_t}"
    assert output_shape[1] == 1, f"Batch size mismatch: {output_shape[1]} != 1"
    assert output_shape[2] == VOCAB_SIZE, f"Vocab size mismatch: {output_shape[2]} != {VOCAB_SIZE}"

    print(f"  ONNX inference: OK (output shape {output_shape})")
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
            print(f"  [3/5] ONNX export OK: {onnx_path.stat().st_size / 1024:.1f} KB")
        except Exception as e:
            print(f"  [3/5] ONNX export FAILED: {e}")
            print("  [NOTE] torch.onnx.export may require 'onnx' package.")
            print("         pip install onnx")
            print("  Continuing with shape validation only...")
            print("=" * 60)
            return False

        # 4. Validate with onnxruntime (if available).
        validated = validate_onnx(onnx_path, img_height=img_h, img_width=img_w)
        status = "OK" if validated else "SKIPPED (onnxruntime not installed)"
        print(f"  [4/5] ONNX validation: {status}")

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
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17).")
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
        print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

        info = inspect_model_shapes(img_height=img_h, img_width=img_w)
        print_model_inspection(info)

        export_to_onnx(model, args.out, img_height=img_h, img_width=img_w, opset_version=args.opset)
        validate_onnx(args.out, img_height=img_h, img_width=img_w)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
