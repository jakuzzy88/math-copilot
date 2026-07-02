"""
summarize_training_run.py – Summarise a training run from its history.

Reads ``training_history.json`` from a run directory and produces a
concise summary, both printed and saved as ``run_summary.json``.

Usage:
    python training/summarize_training_run.py \
        --run training/runs/synthetic_v2_full
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_training_history(run_dir: str | Path) -> dict:
    """Load training_history.json from a run directory."""
    path = Path(run_dir) / "training_history.json"
    if not path.exists():
        raise FileNotFoundError(f"Training history not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize(history: dict) -> dict:
    """Produce a summary dict from the training history.

    Returns dict with keys:
        total_epochs, best_epoch, best_valid_loss,
        final_train_loss, final_valid_loss,
        final_exact_accuracy, final_char_accuracy,
        early_stopped, scheduler, augment
    """
    epochs = history.get("epochs", [])
    if not epochs:
        raise ValueError("No epoch records found in training history.")

    # Find best epoch by minimum valid_loss.
    best_epoch_record = min(epochs, key=lambda e: e["valid_loss"])

    final = epochs[-1]

    return {
        "total_epochs": len(epochs),
        "best_epoch": best_epoch_record["epoch"],
        "best_valid_loss": best_epoch_record["valid_loss"],
        "final_train_loss": final["train_loss"],
        "final_valid_loss": final["valid_loss"],
        "final_exact_accuracy": final["exact_accuracy"],
        "final_char_accuracy": final["char_accuracy"],
        "final_learning_rate": final.get("learning_rate", None),
        "early_stopped": history.get("early_stopped", False),
        "scheduler": history.get("scheduler", "unknown"),
        "augment": history.get("augment", False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise a training run.",
    )
    parser.add_argument("--run", type=str, required=True,
                        help="Path to the training run directory.")
    args = parser.parse_args()

    run_dir = Path(args.run)
    history = load_training_history(run_dir)
    summary = summarize(history)

    # Print summary.
    print("\n" + "=" * 55)
    print("TRAINING RUN SUMMARY")
    print("=" * 55)
    print(f"  Run directory:        {run_dir}")
    print(f"  Total epochs:         {summary['total_epochs']}")
    print(f"  Best epoch:           {summary['best_epoch']}")
    print(f"  Best valid loss:      {summary['best_valid_loss']:.6f}")
    print(f"  Final train loss:     {summary['final_train_loss']:.6f}")
    print(f"  Final valid loss:     {summary['final_valid_loss']:.6f}")
    print(f"  Final exact accuracy: {summary['final_exact_accuracy']:.2%}")
    print(f"  Final char accuracy:  {summary['final_char_accuracy']:.2%}")
    print(f"  Final learning rate:  {summary['final_learning_rate']}")
    print(f"  Early stopped:        {summary['early_stopped']}")
    print(f"  Scheduler:            {summary['scheduler']}")
    print(f"  Augmentation:         {summary['augment']}")
    print("=" * 55)

    # Save summary JSON.
    summary_path = run_dir / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
