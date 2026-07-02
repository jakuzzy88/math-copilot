"""
evaluate.py – Evaluation script for CNN-CTC equation recognition model.

Loads a trained checkpoint and evaluates it on a dataset split,
reporting exact sequence accuracy, character accuracy, average edit
distance, and a sample predictions table.

Usage:
    python training/evaluate.py \\
        --dataset training/datasets/synthetic_v1 \\
        --checkpoint training/runs/synthetic_v1_smoke/best_model.pt \\
        --split valid --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.models.vocabulary import VOCAB_SIZE, ctc_greedy_decode
from training.models.equation_dataset import EquationDataset, collate_fn
from training.models.cnn_ctc import build_model


# ---------------------------------------------------------------------------
# Metrics
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
    return sum(_edit_distance(p, t) for p, t in zip(predictions, targets)) / len(predictions)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run evaluation, return metrics and all predictions."""
    model.eval()
    all_preds: list[str] = []
    all_targets: list[str] = []

    for images, labels, label_lengths, raw_labels in loader:
        images = images.to(device)
        log_probs = model(images)  # (T, B, C)
        T, B, C = log_probs.size()

        for b in range(B):
            pred_str = ctc_greedy_decode(log_probs[:, b, :])
            all_preds.append(pred_str)
            all_targets.append(raw_labels[b])

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
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a CNN-CTC checkpoint on a dataset split.",
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset root.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--split", type=str, default="valid",
                        choices=["train", "valid", "test"],
                        help="Dataset split to evaluate (default: valid).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16).")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for evaluation JSON and prediction samples.")
    args = parser.parse_args()

    # Device.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load checkpoint to extract training args for model construction.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    img_height = train_args.get("height", 128)
    img_width = train_args.get("width", 512)

    # Dataset.
    ds = EquationDataset(
        root=args.dataset, split=args.split,
        height=img_height, width=img_width,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
        num_workers=0,
    )
    print(f"Evaluating on {args.split} split: {len(ds)} samples")

    # Model.
    model = build_model(
        img_height=img_height,
        img_width=img_width,
        num_classes=VOCAB_SIZE,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Evaluate.
    metrics = evaluate(model, loader, device)

    # Report.
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Split:              {args.split}")
    print(f"Samples:            {metrics['num_samples']}")
    print(f"Exact accuracy:     {metrics['exact_accuracy']:.2%}")
    print(f"Character accuracy: {metrics['char_accuracy']:.2%}")
    print(f"Avg edit distance:  {metrics['avg_edit_distance']:.2f}")
    print("=" * 60)

    # Sample predictions table.
    print("\nSample Predictions (first 20):")
    print(f"{'#':>4s}  {'Prediction':<25s}  {'Target':<25s}  {'Match':>5s}  {'ED':>3s}")
    print("-" * 68)
    preds = metrics["predictions"][:20]
    targets = metrics["targets"][:20]
    for i, (p, t) in enumerate(zip(preds, targets), start=1):
        match = "✓" if p == t else "✗"
        ed = _edit_distance(p, t)
        print(f"{i:4d}  {p!r:<25s}  {t!r:<25s}  {match:>5s}  {ed:3d}")

    print()

    # Save results if --out is provided.
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save evaluation metrics.
        eval_report = {
            "split": args.split,
            "num_samples": metrics["num_samples"],
            "exact_accuracy": round(metrics["exact_accuracy"], 4),
            "char_accuracy": round(metrics["char_accuracy"], 4),
            "avg_edit_distance": round(metrics["avg_edit_distance"], 4),
            "checkpoint": str(args.checkpoint),
            "epoch": ckpt.get("epoch", None),
        }
        eval_path = out_dir / f"evaluation_{args.split}.json"
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)
        print(f"Evaluation report saved to {eval_path}")

        # Save prediction samples (all samples, up to 100).
        all_preds = metrics["predictions"]
        all_targets = metrics["targets"]
        sample_count = min(len(all_preds), 100)
        samples = []
        for p, t in zip(all_preds[:sample_count], all_targets[:sample_count]):
            ed = _edit_distance(p, t)
            samples.append({
                "label": t,
                "prediction": p,
                "exact": p == t,
                "edit_distance": ed,
            })
        samples_path = out_dir / f"prediction_samples_{args.split}.json"
        with open(samples_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
        print(f"Prediction samples ({len(samples)} rows) saved to {samples_path}")


if __name__ == "__main__":
    main()
