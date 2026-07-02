"""
debug_predictions.py -- Comprehensive model debugging toolkit.

Sprint A: Evaluation/Debug Readiness.

Tools:
  1. Enhanced prediction sample viewer (colour-coded diffs)
  2. Per-character confusion report
  3. Failed examples grouped by equation form
  4. Top N worst predictions (by edit distance)

Usage:
    python training/debug_predictions.py \
        --dataset training/datasets/synthetic_v2 \
        --checkpoint training/runs/synthetic_v2_full/best_model.pt \
        --split test --batch-size 32 \
        --out training/runs/synthetic_v2_full/debug
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.models.vocabulary import VOCAB_SIZE, CHARACTERS, ctc_greedy_decode
from training.models.equation_dataset import EquationDataset, collate_fn
from training.models.cnn_ctc import build_model


def _edit_distance(a: str, b: str) -> int:
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


_FORM_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("a(x+b)=c", re.compile(r"^-?\d+\(x\+-?\d+\)=-?\d+$")),
    ("a(x-b)=c", re.compile(r"^-?\d+\(x--?\d+\)=-?\d+$")),
    ("ax/b=c",   re.compile(r"^-?\d+x/-?\d+=-?\d+$")),
    ("ax+b=c",   re.compile(r"^-?\d+x\+-?\d+=-?\d+$")),
    ("ax-b=c",   re.compile(r"^-?\d+x--?\d+=-?\d+$")),
    ("ax=b",     re.compile(r"^-?\d+x=-?\d+$")),
    ("x/a=b",    re.compile(r"^x/-?\d+=-?\d+$")),
    ("x+a=b",    re.compile(r"^x\+-?\d+=-?\d+$")),
    ("x-a=b",    re.compile(r"^x--?\d+=-?\d+$")),
]


def classify_equation_form(label: str) -> str:
    label = label.strip().replace(" ", "")
    for form_name, pattern in _FORM_PATTERNS:
        if pattern.match(label):
            return form_name
    return "unknown"


@torch.no_grad()
def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> list[dict]:
    """Run predictions and return list of sample dicts."""
    model.eval()
    results: list[dict] = []
    for images, labels, label_lengths, raw_labels in loader:
        images = images.to(device)
        log_probs = model(images)
        for b in range(log_probs.size(1)):
            pred_str = ctc_greedy_decode(log_probs[:, b, :])
            target = raw_labels[b]
            ed = _edit_distance(pred_str, target)
            results.append({
                "target": target, "prediction": pred_str,
                "exact": pred_str == target, "edit_distance": ed,
                "form": classify_equation_form(target),
            })
    return results


# --- Tool 1: Enhanced Prediction Viewer ---

def _char_diff_markers(target: str, prediction: str) -> str:
    max_len = max(len(target), len(prediction))
    markers: list[str] = []
    for i in range(max_len):
        t_ch = target[i] if i < len(target) else ""
        p_ch = prediction[i] if i < len(prediction) else ""
        if t_ch == p_ch:
            markers.append(" ")
        elif t_ch == "":
            markers.append("+")
        elif p_ch == "":
            markers.append("-")
        else:
            markers.append("^")
    return "".join(markers)


def print_prediction_viewer(samples: list[dict], count: int = 30) -> str:
    lines: list[str] = ["", "=" * 90, "PREDICTION SAMPLE VIEWER", "=" * 90]
    lines.append(f"{'#':>4s}  {'Target':<25s}  {'Prediction':<25s}  {'Match':>5s}  {'ED':>3s}  {'Form':<12s}")
    lines.append("-" * 90)
    shown = 0
    for i, s in enumerate(samples):
        if shown >= count:
            break
        icon = "Y" if s["exact"] else "N"
        lines.append(f"{i+1:4d}  {s['target']!r:<25s}  {s['prediction']!r:<25s}  {icon:>5s}  {s['edit_distance']:3d}  {s['form']:<12s}")
        if not s["exact"]:
            diff = _char_diff_markers(s["target"], s["prediction"])
            lines.append(f"      {'diff:':<25s}  {diff:<25s}")
        shown += 1
    lines.append("-" * 90)
    total = len(samples)
    correct = sum(1 for s in samples if s["exact"])
    lines.append(f"Total: {total} | Correct: {correct} | Failed: {total-correct} | Acc: {correct/max(total,1):.2%}")
    lines.append("=" * 90)
    output = "\n".join(lines)
    print(output)
    return output


# --- Tool 2: Character Confusion Report ---

def build_char_confusion(samples: list[dict]) -> dict:
    report: dict[str, dict] = {}
    for ch in CHARACTERS:
        report[ch] = {"total": 0, "correct": 0, "confused_with": collections.Counter(), "missed": 0}
    report["<extra>"] = {"total": 0, "correct": 0, "confused_with": collections.Counter(), "missed": 0}
    for s in samples:
        target, pred = s["target"], s["prediction"]
        for i in range(max(len(target), len(pred))):
            t_ch = target[i] if i < len(target) else None
            p_ch = pred[i] if i < len(pred) else None
            if t_ch is not None and t_ch in report:
                report[t_ch]["total"] += 1
                if p_ch == t_ch:
                    report[t_ch]["correct"] += 1
                elif p_ch is None:
                    report[t_ch]["missed"] += 1
                else:
                    report[t_ch]["confused_with"][p_ch] += 1
            if t_ch is None and p_ch is not None:
                report["<extra>"]["total"] += 1
                report["<extra>"]["confused_with"][p_ch] += 1
    return report


def print_char_confusion(report: dict) -> str:
    lines: list[str] = ["", "=" * 80, "CHARACTER CONFUSION REPORT", "=" * 80]
    lines.append(f"  {'Char':<6s}  {'Total':>6s}  {'Correct':>7s}  {'Acc%':>6s}  {'Missed':>6s}  {'Top confusions':<30s}")
    lines.append("  " + "-" * 74)
    for ch, data in sorted(report.items()):
        if data["total"] == 0:
            continue
        acc = data["correct"] / data["total"]
        top = data["confused_with"].most_common(3)
        conf_str = ", ".join(f"'{c}'->{n}" for c, n in top) if top else "-"
        ch_d = repr(ch) if len(ch) == 1 else ch
        lines.append(f"  {ch_d:<6s}  {data['total']:6d}  {data['correct']:7d}  {acc:6.1%}  {data['missed']:6d}  {conf_str:<30s}")
    lines.append("=" * 80)
    output = "\n".join(lines)
    print(output)
    return output


def serialize_char_confusion(report: dict) -> dict:
    return {ch: {"total": d["total"], "correct": d["correct"],
                 "confused_with": dict(d["confused_with"]), "missed": d["missed"]}
            for ch, d in report.items()}


# --- Tool 3: Failed Examples by Equation Form ---

def build_failures_by_form(samples: list[dict]) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for s in samples:
        form = s["form"]
        if form not in groups:
            groups[form] = {"total": 0, "failed": 0, "examples": []}
        groups[form]["total"] += 1
        if not s["exact"]:
            groups[form]["failed"] += 1
            groups[form]["examples"].append({"target": s["target"], "prediction": s["prediction"], "edit_distance": s["edit_distance"]})
    for d in groups.values():
        d["failure_rate"] = d["failed"] / d["total"] if d["total"] > 0 else 0.0
    return dict(sorted(groups.items(), key=lambda x: -x[1]["failure_rate"]))


def print_failures_by_form(failures: dict[str, dict], max_ex: int = 5) -> str:
    lines: list[str] = ["", "=" * 80, "FAILED EXAMPLES BY EQUATION FORM", "=" * 80]
    for form, data in failures.items():
        if data["failed"] == 0:
            lines.append(f"\n  {form:<16s}  total={data['total']}  failed=0  All correct")
            continue
        lines.append(f"\n  {form:<16s}  total={data['total']}  failed={data['failed']}  rate={data['failure_rate']:.2%}")
        lines.append(f"  {'-' * 60}")
        for ex in data["examples"][:max_ex]:
            lines.append(f"    target:     {ex['target']!r}")
            lines.append(f"    prediction: {ex['prediction']!r}  (ED={ex['edit_distance']})")
            lines.append("")
        rem = data["failed"] - max_ex
        if rem > 0:
            lines.append(f"    ... and {rem} more failures")
    lines.append("\n" + "=" * 80)
    output = "\n".join(lines)
    print(output)
    return output


# --- Tool 4: Top N Worst Predictions ---

def get_worst_predictions(samples: list[dict], n: int = 20) -> list[dict]:
    failed = [s for s in samples if not s["exact"]]
    return sorted(failed, key=lambda x: -x["edit_distance"])[:n]


def print_worst_predictions(worst: list[dict], n: int = 20) -> str:
    lines: list[str] = ["", "=" * 90, f"TOP {n} WORST PREDICTIONS", "=" * 90]
    lines.append(f"{'Rank':>4s}  {'Target':<25s}  {'Prediction':<25s}  {'ED':>3s}  {'Form':<12s}")
    lines.append("-" * 90)
    for rank, s in enumerate(worst, start=1):
        lines.append(f"{rank:4d}  {s['target']!r:<25s}  {s['prediction']!r:<25s}  {s['edit_distance']:3d}  {s['form']:<12s}")
        diff = _char_diff_markers(s["target"], s["prediction"])
        lines.append(f"      {'':25s}  {diff:<25s}")
    lines.append("-" * 90)
    if worst:
        avg_ed = sum(s["edit_distance"] for s in worst) / len(worst)
        lines.append(f"  Avg ED in worst-{n}: {avg_ed:.1f}  |  Max ED: {worst[0]['edit_distance']}")
    lines.append("=" * 90)
    output = "\n".join(lines)
    print(output)
    return output


# --- Main ---

def main() -> None:
    parser = argparse.ArgumentParser(description="Debug toolkit: inspect CNN-CTC predictions.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--viewer-count", type=int, default=30)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = ckpt.get("args", {})
    img_h = train_args.get("height", 128)
    img_w = train_args.get("width", 512)

    ds = EquationDataset(root=args.dataset, split=args.split, height=img_h, width=img_w)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    print(f"Loaded {args.split} split: {len(ds)} samples")

    model = build_model(img_height=img_h, img_width=img_w, num_classes=VOCAB_SIZE).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    print("Running predictions...", flush=True)
    samples = collect_predictions(model, loader, device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    viewer_out = print_prediction_viewer(samples, count=args.viewer_count)
    confusion = build_char_confusion(samples)
    confusion_out = print_char_confusion(confusion)
    failures = build_failures_by_form(samples)
    failures_out = print_failures_by_form(failures)
    worst = get_worst_predictions(samples, n=args.top_n)
    worst_out = print_worst_predictions(worst, n=args.top_n)

    # Save JSON reports
    debug_report = {
        "split": args.split, "num_samples": len(samples),
        "num_correct": sum(1 for s in samples if s["exact"]),
        "num_failed": sum(1 for s in samples if not s["exact"]),
        "exact_accuracy": round(sum(1 for s in samples if s["exact"]) / max(len(samples), 1), 4),
        "checkpoint": str(args.checkpoint), "epoch": ckpt.get("epoch", None),
    }
    with open(out_dir / f"debug_report_{args.split}.json", "w") as f:
        json.dump(debug_report, f, indent=2)

    with open(out_dir / f"char_confusion_{args.split}.json", "w") as f:
        json.dump(serialize_char_confusion(confusion), f, indent=2)

    fail_ser = {form: {"total": d["total"], "failed": d["failed"],
                       "failure_rate": round(d["failure_rate"], 4),
                       "examples": d["examples"][:10]}
                for form, d in failures.items()}
    with open(out_dir / f"failures_by_form_{args.split}.json", "w") as f:
        json.dump(fail_ser, f, indent=2)

    with open(out_dir / f"worst_predictions_{args.split}.json", "w") as f:
        json.dump(worst, f, indent=2)

    text_report = "\n\n".join([viewer_out, confusion_out, failures_out, worst_out])
    with open(out_dir / f"debug_full_report_{args.split}.txt", "w") as f:
        f.write(text_report)

    print(f"\nAll debug artifacts saved to {out_dir}/")


if __name__ == "__main__":
    main()
