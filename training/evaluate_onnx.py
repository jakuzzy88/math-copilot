"""
evaluate_onnx.py – ONNX Runtime evaluation on real dataset images.

Loads the exported ONNX model, runs inference on synthetic_v2/test images
using the same preprocessing as PyTorch training, decodes CTC outputs via
greedy decode, and compares results against ground-truth labels.

The final report cross-references against the stored PyTorch evaluation
baseline to confirm ONNX accuracy matches PyTorch accuracy.

Usage:
    python training/evaluate_onnx.py \
        --onnx training/runs/synthetic_v2_full_50ep/model.onnx \
        --dataset training/datasets/synthetic_v2 \
        --split test \
        --pytorch-eval training/runs/synthetic_v2_full_50ep/evaluation_test.json

    # Also runs per-sample PyTorch vs ONNX comparison when checkpoint is given:
    python training/evaluate_onnx.py \
        --onnx training/runs/synthetic_v2_full_50ep/model.onnx \
        --dataset training/datasets/synthetic_v2 \
        --split test \
        --checkpoint training/runs/synthetic_v2_full_50ep/best_model.pt \
        --pytorch-eval training/runs/synthetic_v2_full_50ep/evaluation_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# CTC Greedy Decode (NumPy-only, no PyTorch dependency for pure ONNX path)
# ---------------------------------------------------------------------------

def _ctc_greedy_decode_numpy(
    log_probs: np.ndarray,
    idx_to_char: dict[int, str],
    blank_idx: int = 0,
) -> str:
    """CTC greedy decode from a (T, C) numpy array of log-probabilities.

    Args:
        log_probs: (T, num_classes) array.
        idx_to_char: mapping from class index to character.
        blank_idx: index of the CTC blank token.

    Returns:
        Decoded string after CTC collapse.
    """
    # Argmax over classes for each time step.
    indices = log_probs.argmax(axis=-1)  # (T,)

    # CTC collapse: merge consecutive duplicates, then drop blanks.
    collapsed: list[int] = []
    prev = -1
    for idx in indices:
        idx = int(idx)
        if idx != prev:
            collapsed.append(idx)
        prev = idx

    # Remove blanks and convert to characters.
    chars: list[str] = []
    for idx in collapsed:
        if idx == blank_idx:
            continue
        chars.append(idx_to_char[idx])
    return "".join(chars)


# ---------------------------------------------------------------------------
# Metrics (same as evaluate.py)
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _exact_match(predictions: list[str], targets: list[str]) -> float:
    if not predictions:
        return 0.0
    return sum(p == t for p, t in zip(predictions, targets)) / len(predictions)


def _char_accuracy(predictions: list[str], targets: list[str]) -> float:
    if not predictions:
        return 0.0
    total = 0.0
    for pred, target in zip(predictions, targets):
        max_len = max(len(pred), len(target))
        if max_len == 0:
            total += 1.0
            continue
        matches = sum(p == t for p, t in zip(pred, target))
        total += matches / max_len
    return total / len(predictions)


def _avg_edit_distance(predictions: list[str], targets: list[str]) -> float:
    if not predictions:
        return 0.0
    return sum(
        _edit_distance(p, t) for p, t in zip(predictions, targets)
    ) / len(predictions)


# ---------------------------------------------------------------------------
# Image loading (same preprocessing as EquationDataset)
# ---------------------------------------------------------------------------

def _load_image(img_path: Path, height: int, width: int) -> np.ndarray:
    """Load a single greyscale image and preprocess for inference.

    Returns a (1, 1, H, W) float32 numpy array normalised to [0, 1],
    matching the EquationDataset preprocessing pipeline.
    """
    from PIL import Image

    img = Image.open(img_path).convert("L")
    img = img.resize((width, height), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # (H, W) → (1, 1, H, W)
    return arr[np.newaxis, np.newaxis, :, :]


# ---------------------------------------------------------------------------
# ONNX Evaluation
# ---------------------------------------------------------------------------

def evaluate_onnx(
    onnx_path: str | Path,
    dataset_root: str | Path,
    split: str = "test",
    img_height: int = 128,
    img_width: int = 512,
    batch_size: int = 16,
) -> dict:
    """Run ONNX Runtime inference on a dataset split and compute metrics.

    Returns a dict with:
        exact_accuracy, char_accuracy, avg_edit_distance,
        num_samples, predictions, targets, inference_time_ms
    """
    import onnxruntime as ort

    # Build vocabulary mapping (matches vocabulary.py).
    CHARACTERS = "0123456789x+-=/() "
    BLANK_IDX = 0
    idx_to_char: dict[int, str] = {}
    for i, ch in enumerate(CHARACTERS, start=1):
        idx_to_char[i] = ch

    # Load ONNX model.
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    # Collect test samples.
    dataset_root = Path(dataset_root)
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not lbl_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {lbl_dir}")

    img_files = sorted(img_dir.glob("*.png"))
    samples: list[tuple[Path, Path]] = []
    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            samples.append((img_path, lbl_path))

    if not samples:
        raise RuntimeError(f"No image/label pairs found in {img_dir}")

    print(f"  Loaded {len(samples)} samples from {split} split")

    # Run inference in batches.
    all_preds: list[str] = []
    all_targets: list[str] = []
    total_inference_ms = 0.0

    for batch_start in range(0, len(samples), batch_size):
        batch_samples = samples[batch_start : batch_start + batch_size]
        batch_images: list[np.ndarray] = []
        batch_labels: list[str] = []

        for img_path, lbl_path in batch_samples:
            img = _load_image(img_path, img_height, img_width)
            label = lbl_path.read_text(encoding="utf-8").strip()
            batch_images.append(img)
            batch_labels.append(label)

        # Stack into (B, 1, H, W).
        batch_input = np.concatenate(batch_images, axis=0)

        # Run ONNX inference.
        t0 = time.perf_counter()
        outputs = session.run(None, {input_name: batch_input})
        t1 = time.perf_counter()
        total_inference_ms += (t1 - t0) * 1000

        # outputs[0] shape: (T, B, C)
        log_probs = outputs[0]
        B = log_probs.shape[1]

        for b in range(B):
            pred_str = _ctc_greedy_decode_numpy(
                log_probs[:, b, :], idx_to_char, BLANK_IDX,
            )
            all_preds.append(pred_str)
            all_targets.append(batch_labels[b])

    # Compute metrics.
    exact_acc = _exact_match(all_preds, all_targets)
    char_acc = _char_accuracy(all_preds, all_targets)
    avg_ed = _avg_edit_distance(all_preds, all_targets)

    return {
        "exact_accuracy": exact_acc,
        "char_accuracy": char_acc,
        "avg_edit_distance": avg_ed,
        "num_samples": len(all_preds),
        "predictions": all_preds,
        "targets": all_targets,
        "inference_time_ms": total_inference_ms,
        "ms_per_sample": total_inference_ms / len(all_preds) if all_preds else 0,
    }


# ---------------------------------------------------------------------------
# Per-sample PyTorch vs ONNX comparison
# ---------------------------------------------------------------------------

def compare_pytorch_onnx_per_sample(
    onnx_path: str | Path,
    checkpoint_path: str | Path,
    dataset_root: str | Path,
    split: str = "test",
    img_height: int = 128,
    img_width: int = 512,
    max_samples: int | None = None,
) -> dict:
    """Run both PyTorch and ONNX inference on the same images and compare.

    Returns a dict with:
        num_samples, num_matching, num_mismatching,
        match_rate, mismatches (list of dicts),
        max_output_diff, mean_output_diff
    """
    import torch
    import onnxruntime as ort
    from training.models.vocabulary import VOCAB_SIZE, ctc_greedy_decode
    from training.models.cnn_ctc import build_model

    # Load PyTorch model.
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    img_h = train_args.get("height", img_height)
    img_w = train_args.get("width", img_width)

    model = build_model(img_height=img_h, img_width=img_w, num_classes=VOCAB_SIZE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load ONNX session.
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name

    # Build vocabulary mapping for numpy decode.
    CHARACTERS = "0123456789x+-=/() "
    BLANK_IDX = 0
    idx_to_char: dict[int, str] = {}
    for i, ch in enumerate(CHARACTERS, start=1):
        idx_to_char[i] = ch

    # Load samples.
    dataset_root = Path(dataset_root)
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split
    img_files = sorted(img_dir.glob("*.png"))
    samples: list[tuple[Path, Path]] = []
    for img_path in img_files:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            samples.append((img_path, lbl_path))

    if max_samples is not None:
        samples = samples[:max_samples]

    # Compare per-sample.
    mismatches: list[dict] = []
    all_max_diffs: list[float] = []
    all_mean_diffs: list[float] = []

    for i, (img_path, lbl_path) in enumerate(samples):
        label = lbl_path.read_text(encoding="utf-8").strip()
        img_np = _load_image(img_path, img_h, img_w)  # (1, 1, H, W)

        # PyTorch inference.
        img_pt = torch.from_numpy(img_np)
        with torch.no_grad():
            pt_output = model(img_pt)  # (T, 1, C)
        pt_pred = ctc_greedy_decode(pt_output[:, 0, :])
        pt_output_np = pt_output.numpy()

        # ONNX inference.
        onnx_outputs = session.run(None, {input_name: img_np})
        onnx_output = onnx_outputs[0]  # (T, 1, C)
        onnx_pred = _ctc_greedy_decode_numpy(
            onnx_output[:, 0, :], idx_to_char, BLANK_IDX,
        )

        # Numerical comparison.
        max_diff = float(np.max(np.abs(pt_output_np - onnx_output)))
        mean_diff = float(np.mean(np.abs(pt_output_np - onnx_output)))
        all_max_diffs.append(max_diff)
        all_mean_diffs.append(mean_diff)

        if pt_pred != onnx_pred:
            mismatches.append({
                "index": i,
                "file": img_path.name,
                "label": label,
                "pytorch_pred": pt_pred,
                "onnx_pred": onnx_pred,
                "max_output_diff": max_diff,
            })

    num_matching = len(samples) - len(mismatches)
    return {
        "num_samples": len(samples),
        "num_matching": num_matching,
        "num_mismatching": len(mismatches),
        "match_rate": num_matching / len(samples) if samples else 0,
        "mismatches": mismatches,
        "max_output_diff": max(all_max_diffs) if all_max_diffs else 0,
        "mean_output_diff": np.mean(all_mean_diffs) if all_mean_diffs else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an ONNX model on real dataset images.",
    )
    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to the exported ONNX model.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset root (e.g. training/datasets/synthetic_v2).")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="Dataset split to evaluate (default: test).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference (default: 16).")
    parser.add_argument("--pytorch-eval", type=str, default=None,
                        help="Path to PyTorch evaluation_*.json for cross-reference.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to PyTorch checkpoint for per-sample comparison.")
    parser.add_argument("--height", type=int, default=128,
                        help="Image height (default: 128).")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width (default: 512).")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for evaluation results JSON.")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("ONNX RUNTIME EVALUATION ON REAL DATASET IMAGES")
    print("=" * 70)
    print(f"  ONNX model:    {args.onnx}")
    print(f"  Dataset:       {args.dataset}")
    print(f"  Split:         {args.split}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Image size:    {args.height} x {args.width}")
    print()

    # ---- Step 1: ONNX evaluation ----
    print("--- Step 1: ONNX Runtime Inference ---")
    onnx_metrics = evaluate_onnx(
        onnx_path=args.onnx,
        dataset_root=args.dataset,
        split=args.split,
        img_height=args.height,
        img_width=args.width,
        batch_size=args.batch_size,
    )

    print(f"\n{'=' * 70}")
    print("ONNX EVALUATION RESULTS")
    print(f"{'=' * 70}")
    print(f"  Split:              {args.split}")
    print(f"  Samples:            {onnx_metrics['num_samples']}")
    print(f"  Exact accuracy:     {onnx_metrics['exact_accuracy']:.2%}")
    print(f"  Character accuracy: {onnx_metrics['char_accuracy']:.2%}")
    print(f"  Avg edit distance:  {onnx_metrics['avg_edit_distance']:.2f}")
    print(f"  Total inference:    {onnx_metrics['inference_time_ms']:.1f} ms")
    print(f"  Per sample:         {onnx_metrics['ms_per_sample']:.2f} ms")
    print(f"{'=' * 70}")

    # Sample predictions table.
    print("\nSample Predictions (first 20):")
    print(f"{'#':>4s}  {'ONNX Prediction':<25s}  {'Target':<25s}  {'Match':>5s}  {'ED':>3s}")
    print("-" * 68)
    preds = onnx_metrics["predictions"][:20]
    targets = onnx_metrics["targets"][:20]
    for i, (p, t) in enumerate(zip(preds, targets), start=1):
        match = "✓" if p == t else "✗"
        ed = _edit_distance(p, t)
        print(f"{i:4d}  {p!r:<25s}  {t!r:<25s}  {match:>5s}  {ed:3d}")
    print()

    # ---- Step 2: Cross-reference with PyTorch baseline ----
    if args.pytorch_eval:
        print("--- Step 2: Cross-Reference with PyTorch Baseline ---")
        pt_eval_path = Path(args.pytorch_eval)
        if not pt_eval_path.exists():
            print(f"  [WARN] PyTorch evaluation file not found: {pt_eval_path}")
        else:
            with open(pt_eval_path, "r", encoding="utf-8") as f:
                pt_eval = json.load(f)

            pt_exact = pt_eval["exact_accuracy"]
            pt_char = pt_eval["char_accuracy"]
            pt_ed = pt_eval["avg_edit_distance"]
            pt_n = pt_eval["num_samples"]

            onnx_exact = round(onnx_metrics["exact_accuracy"], 4)
            onnx_char = round(onnx_metrics["char_accuracy"], 4)
            onnx_ed = round(onnx_metrics["avg_edit_distance"], 4)

            print(f"\n{'Metric':<25s}  {'PyTorch':>12s}  {'ONNX':>12s}  {'Match':>7s}")
            print("-" * 60)
            print(
                f"{'Exact accuracy':<25s}  {pt_exact:>11.2%}  "
                f"{onnx_exact:>11.2%}  "
                f"{'  ✓' if abs(pt_exact - onnx_exact) < 1e-6 else '  ✗'}"
            )
            print(
                f"{'Character accuracy':<25s}  {pt_char:>11.2%}  "
                f"{onnx_char:>11.2%}  "
                f"{'  ✓' if abs(pt_char - onnx_char) < 1e-6 else '  ✗'}"
            )
            print(
                f"{'Avg edit distance':<25s}  {pt_ed:>12.4f}  "
                f"{onnx_ed:>12.4f}  "
                f"{'  ✓' if abs(pt_ed - onnx_ed) < 1e-6 else '  ✗'}"
            )
            print(
                f"{'Num samples':<25s}  {pt_n:>12d}  "
                f"{onnx_metrics['num_samples']:>12d}  "
                f"{'  ✓' if pt_n == onnx_metrics['num_samples'] else '  ✗'}"
            )

            # Verdict.
            all_match = (
                abs(pt_exact - onnx_exact) < 1e-6
                and abs(pt_char - onnx_char) < 1e-6
                and abs(pt_ed - onnx_ed) < 1e-6
                and pt_n == onnx_metrics["num_samples"]
            )

            print()
            if all_match:
                print("  ✅ ONNX ACCURACY MATCHES PYTORCH EVALUATION — EXACT MATCH")
            else:
                # Check if within floating-point tolerance (rounding).
                close = (
                    abs(pt_exact - onnx_exact) < 0.005
                    and abs(pt_char - onnx_char) < 0.005
                    and abs(pt_ed - onnx_ed) < 0.01
                )
                if close:
                    print("  ⚠️  ONNX accuracy is CLOSE to PyTorch (within rounding tolerance)")
                    print(f"     Exact diff: {abs(pt_exact - onnx_exact):.6f}")
                    print(f"     Char diff:  {abs(pt_char - onnx_char):.6f}")
                    print(f"     ED diff:    {abs(pt_ed - onnx_ed):.6f}")
                else:
                    print("  ❌ ONNX ACCURACY DOES NOT MATCH PYTORCH EVALUATION")
                    print(f"     Exact diff: {abs(pt_exact - onnx_exact):.6f}")
                    print(f"     Char diff:  {abs(pt_char - onnx_char):.6f}")
                    print(f"     ED diff:    {abs(pt_ed - onnx_ed):.6f}")
            print()

    # ---- Step 3: Per-sample PyTorch vs ONNX output comparison ----
    if args.checkpoint:
        print("--- Step 3: Per-Sample PyTorch vs ONNX Comparison ---")
        comparison = compare_pytorch_onnx_per_sample(
            onnx_path=args.onnx,
            checkpoint_path=args.checkpoint,
            dataset_root=args.dataset,
            split=args.split,
            img_height=args.height,
            img_width=args.width,
        )

        print(f"\n  Samples compared:   {comparison['num_samples']}")
        print(f"  Predictions match:  {comparison['num_matching']}")
        print(f"  Predictions differ: {comparison['num_mismatching']}")
        print(f"  Match rate:         {comparison['match_rate']:.2%}")
        print(f"  Max output diff:    {comparison['max_output_diff']:.6e}")
        print(f"  Mean output diff:   {comparison['mean_output_diff']:.6e}")

        if comparison["mismatches"]:
            print(f"\n  Mismatched predictions ({len(comparison['mismatches'])}):")
            print(f"  {'#':>4s}  {'File':<12s}  {'Label':<18s}  {'PyTorch':<18s}  {'ONNX':<18s}  {'MaxDiff':>10s}")
            print("  " + "-" * 84)
            for m in comparison["mismatches"][:20]:
                print(
                    f"  {m['index']:4d}  {m['file']:<12s}  "
                    f"{m['label']:<18s}  {m['pytorch_pred']:<18s}  "
                    f"{m['onnx_pred']:<18s}  {m['max_output_diff']:>10.2e}"
                )
        else:
            print("\n  ✅ ALL per-sample predictions are IDENTICAL between PyTorch and ONNX")

        if comparison["match_rate"] == 1.0:
            print("\n  ✅ PERFECT MATCH: ONNX Runtime produces identical CTC decodes to PyTorch")
        elif comparison["match_rate"] >= 0.99:
            print(f"\n  ⚠️  NEAR MATCH: {comparison['match_rate']:.2%} of predictions are identical")
        else:
            print(f"\n  ❌ SIGNIFICANT DIVERGENCE: only {comparison['match_rate']:.2%} match")
        print()

    # ---- Save results ----
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "engine": "onnxruntime",
            "onnx_model": str(args.onnx),
            "split": args.split,
            "num_samples": onnx_metrics["num_samples"],
            "exact_accuracy": round(onnx_metrics["exact_accuracy"], 4),
            "char_accuracy": round(onnx_metrics["char_accuracy"], 4),
            "avg_edit_distance": round(onnx_metrics["avg_edit_distance"], 4),
            "inference_time_ms": round(onnx_metrics["inference_time_ms"], 1),
            "ms_per_sample": round(onnx_metrics["ms_per_sample"], 2),
        }
        report_path = out_dir / f"onnx_evaluation_{args.split}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"ONNX evaluation report saved to {report_path}")

        # Save prediction samples.
        sample_count = min(len(onnx_metrics["predictions"]), 100)
        samples_data = []
        for p, t in zip(
            onnx_metrics["predictions"][:sample_count],
            onnx_metrics["targets"][:sample_count],
        ):
            ed = _edit_distance(p, t)
            samples_data.append({
                "label": t,
                "onnx_prediction": p,
                "exact": p == t,
                "edit_distance": ed,
            })
        samples_path = out_dir / f"onnx_prediction_samples_{args.split}.json"
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples_data, f, indent=2)
        print(f"ONNX prediction samples ({len(samples_data)} rows) saved to {samples_path}")

    print("\n[DONE] ONNX Runtime evaluation complete.")


if __name__ == "__main__":
    main()
