"""
analyze_predictions.py – Per-equation-form accuracy analysis.

Loads a trained checkpoint, runs predictions on a dataset split,
classifies each label into its canonical equation form, and reports
per-form metrics: sample count, exact accuracy, character accuracy,
and average edit distance.

Also saves a prediction_samples_<split>.json with ≥50 sample rows.

Usage:
    python training/analyze_predictions.py \
        --dataset training/datasets/synthetic_v2 \
        --checkpoint training/runs/synthetic_v2_full/best_model.pt \
        --split test --batch-size 32 \
        --out training/runs/synthetic_v2_full
"""

from __future__ import annotations

import argparse
import json
import re
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
# Equation form classifier
# ---------------------------------------------------------------------------

# Patterns ordered from most specific to least specific to avoid
# premature matching.  All patterns are anchored (^ ... $).
# We use named groups for clarity.

_FORM_PATTERNS: list[tuple[str, re.Pattern]] = [
    # a(x+b)=c
    ("a(x+b)=c", re.compile(
        r"^-?\d+\(x\+-?\d+\)=-?\d+$"
    )),
    # a(x-b)=c
    ("a(x-b)=c", re.compile(
        r"^-?\d+\(x--?\d+\)=-?\d+$"
    )),
    # ax/b=c
    ("ax/b=c", re.compile(
        r"^-?\d+x/-?\d+=-?\d+$"
    )),
    # ax+b=c
    ("ax+b=c", re.compile(
        r"^-?\d+x\+-?\d+=-?\d+$"
    )),
    # ax-b=c
    ("ax-b=c", re.compile(
        r"^-?\d+x--?\d+=-?\d+$"
    )),
    # ax=b
    ("ax=b", re.compile(
        r"^-?\d+x=-?\d+$"
    )),
    # x/a=b
    ("x/a=b", re.compile(
        r"^x/-?\d+=-?\d+$"
    )),
    # x+a=b
    ("x+a=b", re.compile(
        r"^x\+-?\d+=-?\d+$"
    )),
    # x-a=b
    ("x-a=b", re.compile(
        r"^x--?\d+=-?\d+$"
    )),
]


def classify_equation_form(label: str) -> str:
    """Classify a label string into a canonical equation form.

    Returns one of the known form names or ``'unknown'``.
    """
    # Strip any whitespace for matching.
    label = label.strip().replace(" ", "")
    for form_name, pattern in _FORM_PATTERNS:
        if pattern.match(label):
            return form_name
    return "unknown"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
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
def run_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[str], list[str]]:
    """Run prediction on the entire loader.

    Returns (all_preds, all_targets).
    """
    model.eval()
    all_preds: list[str] = []
    all_targets: list[str] = []

    for images, labels, label_lengths, raw_labels in loader:
        images = images.to(device)
        log_probs = model(images)

        for b in range(log_probs.size(1)):
            pred_str = ctc_greedy_decode(log_probs[:, b, :])
            all_preds.append(pred_str)
            all_targets.append(raw_labels[b])

    return all_preds, all_targets


def compute_per_form_metrics(
    predictions: list[str],
    targets: list[str],
) -> dict[str, dict]:
    """Group predictions by equation form and compute per-form metrics.

    Returns a dict keyed by form name, each containing:
        sample_count, exact_accuracy, char_accuracy, avg_edit_distance
    """
    # Group by form.
    form_groups: dict[str, tuple[list[str], list[str]]] = {}
    for pred, target in zip(predictions, targets):
        form = classify_equation_form(target)
        if form not in form_groups:
            form_groups[form] = ([], [])
        form_groups[form][0].append(pred)
        form_groups[form][1].append(target)

    results: dict[str, dict] = {}
    for form, (preds, tgts) in sorted(form_groups.items()):
        results[form] = {
            "sample_count": len(preds),
            "exact_accuracy": round(_exact_match(preds, tgts), 4),
            "char_accuracy": round(_char_accuracy(preds, tgts), 4),
            "avg_edit_distance": round(_avg_edit_distance(preds, tgts), 4),
        }

    return results


def build_prediction_samples(
    predictions: list[str],
    targets: list[str],
    max_samples: int = 100,
) -> list[dict]:
    """Build a list of prediction sample dicts for export."""
    samples = []
    for pred, target in zip(predictions[:max_samples], targets[:max_samples]):
        ed = _edit_distance(pred, target)
        samples.append({
            "label": target,
            "prediction": pred,
            "exact": pred == target,
            "edit_distance": ed,
        })
    return samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-equation-form accuracy analysis.",
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset root.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt).")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "valid", "test"],
                        help="Dataset split to analyse (default: test).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16).")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--out", type=str, required=True,
                        help="Output directory for analysis JSON files.")
    args = parser.parse_args()

    # Device.
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load checkpoint.
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
    print(f"Analysing {args.split} split: {len(ds)} samples")

    # Model.
    model = build_model(
        img_height=img_height,
        img_width=img_width,
        num_classes=VOCAB_SIZE,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Run predictions.
    predictions, targets = run_predictions(model, loader, device)

    # Overall metrics.
    overall_exact = _exact_match(predictions, targets)
    overall_char = _char_accuracy(predictions, targets)
    overall_ed = _avg_edit_distance(predictions, targets)

    # Per-form metrics.
    per_form = compute_per_form_metrics(predictions, targets)

    # Build report.
    report = {
        "split": args.split,
        "num_samples": len(predictions),
        "overall": {
            "exact_accuracy": round(overall_exact, 4),
            "char_accuracy": round(overall_char, 4),
            "avg_edit_distance": round(overall_ed, 4),
        },
        "per_form": per_form,
    }

    # Print summary.
    print("\n" + "=" * 70)
    print(f"PREDICTION ANALYSIS — {args.split} split ({len(predictions)} samples)")
    print("=" * 70)
    print(f"  Overall exact accuracy:     {overall_exact:.2%}")
    print(f"  Overall char accuracy:      {overall_char:.2%}")
    print(f"  Overall avg edit distance:  {overall_ed:.2f}")
    print()
    print(f"  {'Form':<16s}  {'Count':>6s}  {'Exact%':>7s}  {'Char%':>7s}  {'AvgED':>6s}")
    print("  " + "-" * 48)
    for form, m in per_form.items():
        print(
            f"  {form:<16s}  {m['sample_count']:6d}  "
            f"{m['exact_accuracy']:7.2%}  {m['char_accuracy']:7.2%}  "
            f"{m['avg_edit_distance']:6.2f}"
        )
    print("=" * 70)

    # Save analysis JSON.
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = out_dir / f"prediction_analysis_{args.split}.json"
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nAnalysis saved to {analysis_path}")

    # Save prediction samples.
    samples = build_prediction_samples(predictions, targets, max_samples=100)
    samples_path = out_dir / f"prediction_samples_{args.split}.json"
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)
    print(f"Prediction samples ({len(samples)} rows) saved to {samples_path}")


if __name__ == "__main__":
    main()
