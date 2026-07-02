"""
train_ctc.py – Training loop for the CNN-CTC equation recogniser.

Usage (smoke test):
    python training/train_ctc.py \\
        --dataset training/datasets/synthetic_v1 \\
        --epochs 1 --batch-size 8 --seed 42 \\
        --out training/runs/synthetic_v1_smoke \\
        --smoke-test

Usage (full training):
    python training/train_ctc.py \\
        --dataset training/datasets/synthetic_v1 \\
        --epochs 20 --batch-size 16 --lr 1e-3 --seed 42 \\
        --out training/runs/synthetic_v1_full

Sprint 4A additions:
    --scheduler cosine|reduce_on_plateau|none
    --early-stopping-patience N
    --min-delta D
    --augment
    --overfit-small-batch

Sprint 4B additions:
    --log-every N   (batch-level progress logging interval)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.models.vocabulary import (
    VOCAB_SIZE,
    ctc_greedy_decode,
    decode_indices,
)
from training.models.equation_dataset import EquationDataset, collate_fn
from training.models.cnn_ctc import build_model


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_duration(seconds: float) -> str:
    """Format seconds as HH:MM:SS or MM:SS."""
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _exact_match(predictions: list[str], targets: list[str]) -> float:
    """Fraction of predictions that exactly match their target."""
    if len(predictions) == 0:
        return 0.0
    correct = sum(p == t for p, t in zip(predictions, targets))
    return correct / len(predictions)


def _char_accuracy(predictions: list[str], targets: list[str]) -> float:
    """Character-level accuracy averaged over all samples.

    For each sample, accuracy = (matching chars) / max(len_pred, len_target).
    """
    if len(predictions) == 0:
        return 0.0
    total_acc = 0.0
    for pred, target in zip(predictions, targets):
        max_len = max(len(pred), len(target))
        if max_len == 0:
            total_acc += 1.0
            continue
        matches = sum(
            p == t for p, t in zip(pred, target)
        )
        total_acc += matches / max_len
    return total_acc / len(predictions)


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
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


def _avg_edit_distance(predictions: list[str], targets: list[str]) -> float:
    """Average edit distance over all samples."""
    if not predictions:
        return 0.0
    return sum(_edit_distance(p, t) for p, t in zip(predictions, targets)) / len(predictions)


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int = 1,
    total_epochs: int = 1,
    log_every: int = 10,
) -> float:
    """Train for one epoch, return average loss.

    Prints batch-level progress every *log_every* batches.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_batches = len(loader)
    epoch_t0 = time.time()

    for images, labels, label_lengths, _raw in loader:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        # Forward pass: model output is (T, B, C).
        log_probs = model(images)
        T, B, C = log_probs.size()

        # CTC loss expects input_lengths for every batch element.
        input_lengths = torch.full((B,), T, dtype=torch.int32, device=device)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to stabilise CTC training.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Batch-level progress logging.
        if log_every > 0 and (num_batches % log_every == 0 or num_batches == total_batches):
            avg_loss = total_loss / num_batches
            elapsed = time.time() - epoch_t0
            print(
                f"  train batch {num_batches:03d}/{total_batches:03d} | "
                f"loss={loss.item():.4f} | avg_loss={avg_loss:.4f} | "
                f"elapsed={_format_duration(elapsed)}",
                flush=True,
            )

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> dict:
    """Run validation, return metrics dict."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds: list[str] = []
    all_targets: list[str] = []

    for images, labels, label_lengths, raw_labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        log_probs = model(images)
        T, B, C = log_probs.size()
        input_lengths = torch.full((B,), T, dtype=torch.int32, device=device)

        loss = criterion(log_probs, labels, input_lengths, label_lengths)
        total_loss += loss.item()
        num_batches += 1

        # Greedy decode each sample in the batch.
        for b in range(B):
            pred_str = ctc_greedy_decode(log_probs[:, b, :])  # (T, C)
            all_preds.append(pred_str)
            all_targets.append(raw_labels[b])

    avg_loss = total_loss / max(num_batches, 1)
    exact_acc = _exact_match(all_preds, all_targets)
    char_acc = _char_accuracy(all_preds, all_targets)
    avg_ed = _avg_edit_distance(all_preds, all_targets)

    return {
        "valid_loss": avg_loss,
        "exact_accuracy": exact_acc,
        "char_accuracy": char_acc,
        "avg_edit_distance": avg_ed,
        "num_samples": len(all_preds),
        "predictions": all_preds,
        "targets": all_targets,
    }


# ---------------------------------------------------------------------------
# Scheduler helpers
# ---------------------------------------------------------------------------

SCHEDULER_CHOICES = ("none", "cosine", "reduce_on_plateau")


def build_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> object | None:
    """Create a learning-rate scheduler by name.

    Returns:
        A PyTorch scheduler instance, or ``None`` for ``'none'``.
    """
    if name == "none":
        return None
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )
    raise ValueError(f"Unknown scheduler: {name!r}")


def scheduler_step(scheduler, name: str, val_loss: float | None = None) -> None:
    """Advance the scheduler by one epoch."""
    if scheduler is None:
        return
    if name == "reduce_on_plateau":
        scheduler.step(val_loss)
    else:
        scheduler.step()


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the current learning rate from the first param group."""
    return optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Track validation loss and signal when to stop.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum improvement to qualify as progress.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self.counter: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """Update state with the latest validation loss.

        Returns ``True`` if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
            return True
        return False


# ---------------------------------------------------------------------------
# Overfit sanity mode helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _overfit_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Compute exact-match and char accuracy on the overfit subset."""
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
    return _exact_match(all_preds, all_targets), _char_accuracy(all_preds, all_targets)


# ---------------------------------------------------------------------------
# Console logging helpers
# ---------------------------------------------------------------------------

_SEP = "=" * 60
_LINE = "-" * 60


def _print_startup_banner(
    args: argparse.Namespace,
    device: torch.device,
    total_params: int,
    train_count: int,
    valid_count: int,
    epochs: int,
    overfit_mode: bool,
) -> None:
    """Print a structured startup summary banner."""
    aug_status = "enabled" if args.augment else "disabled"
    if args.early_stopping_patience > 0:
        es_status = f"patience={args.early_stopping_patience}, min_delta={args.min_delta}"
    else:
        es_status = "disabled"

    print(f"\n{_SEP}")
    print("OFFLINE MATH COPILOT — CNN-CTC TRAINING")
    print(_SEP)
    print(f"Run:              {args.out}")
    print(f"Dataset:          {args.dataset}")
    print(f"Device:           {device}")
    print(f"Model parameters: {total_params:,}")
    print(f"Train samples:    {train_count}")
    print(f"Valid samples:    {valid_count}")
    print(f"Batch size:       {args.batch_size}")
    print(f"Epochs:           {epochs}")
    print(f"LR:               {args.lr}")
    print(f"Scheduler:        {args.scheduler}")
    print(f"Augmentation:     {aug_status}")
    print(f"Early stopping:   {es_status}")
    print(f"Image size:       {args.height}x{args.width}")
    print(f"Seed:             {args.seed}")
    if overfit_mode:
        print(f"Mode:             overfit-small-batch")
    if args.smoke_test:
        print(f"Mode:             smoke-test")
    print(f"Log every:        {args.log_every} batches")
    print(f"{_SEP}\n")


def _print_epoch_summary(
    epoch: int,
    epochs: int,
    train_loss: float,
    val_metrics: dict,
    current_lr: float,
    elapsed: float,
    best_loss: float,
    saved_best: bool,
    early_stopper: EarlyStopping | None,
    overfit_mode: bool,
    overfit_exact: float | None = None,
    overfit_char: float | None = None,
) -> None:
    """Print a structured epoch summary."""
    es_patience = early_stopper.patience if early_stopper else 0
    es_counter = early_stopper.counter if early_stopper else 0
    es_str = f"{es_counter}/{es_patience}" if early_stopper else "disabled"
    ckpt_str = "✓ saved best_model.pt" if saved_best else "—"

    print(f"\n{_LINE}")
    print(f"Epoch {epoch}/{epochs} complete")
    print(f"  Train loss:      {train_loss:.4f}")
    print(f"  Valid loss:       {val_metrics['valid_loss']:.4f}")
    print(f"  Valid exact acc:  {val_metrics['exact_accuracy']:.2%}")
    print(f"  Valid char acc:   {val_metrics['char_accuracy']:.2%}")
    print(f"  Avg edit dist:    {val_metrics['avg_edit_distance']:.2f}")
    print(f"  Learning rate:    {current_lr:.2e}")
    print(f"  Duration:         {_format_duration(elapsed)}")
    print(f"  Best valid loss:  {best_loss:.4f}")
    print(f"  Checkpoint:       {ckpt_str}")
    print(f"  Early stopping:   {es_str}")
    if overfit_mode and overfit_exact is not None:
        print(f"  Overfit exact:    {overfit_exact:.2%}")
        print(f"  Overfit char:     {overfit_char:.2%}")
    print(_LINE)


def _print_prediction_preview(
    predictions: list[str],
    targets: list[str],
    count: int = 5,
) -> None:
    """Print a compact prediction preview table."""
    print("\n  Prediction preview:")
    for pred, target in zip(predictions[:count], targets[:count]):
        status = "✓" if pred == target else "✗"
        print(f"    label: {target:<16s}  pred: {pred:<16s}  {status}")


def _print_end_of_run(
    history: list[dict],
    early_stopped: bool,
    best_epoch: int,
    best_loss: float,
    out_dir: Path,
    dataset: str,
    batch_size: int,
) -> None:
    """Print a structured end-of-run summary."""
    print(f"\n{_SEP}")
    print("TRAINING COMPLETE")
    print(_SEP)
    print(f"  Completed epochs: {len(history)}")
    print(f"  Early stopped:    {str(early_stopped).lower()}")
    print(f"  Best epoch:       {best_epoch}")
    print(f"  Best valid loss:  {best_loss:.4f}")
    print(f"  Best checkpoint:")
    print(f"    {out_dir / 'best_model.pt'}")
    print(f"  Last checkpoint:")
    print(f"    {out_dir / 'last_model.pt'}")
    print(f"  History:")
    print(f"    {out_dir / 'training_history.json'}")
    print()
    print("  Next command:")
    print(f"  python3 training/evaluate.py \\")
    print(f"    --dataset {dataset} \\")
    print(f"    --checkpoint {out_dir / 'best_model.pt'} \\")
    print(f"    --split test \\")
    print(f"    --batch-size {batch_size} \\")
    print(f"    --out {out_dir}")
    print(_SEP)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN-CTC model for handwritten equation recognition.",
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset root (e.g. training/datasets/synthetic_v1).")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for Adam (default: 1e-3).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    parser.add_argument("--width", type=int, default=512,
                        help="Image width (default: 512).")
    parser.add_argument("--height", type=int, default=128,
                        help="Image height (default: 128).")
    parser.add_argument("--out", type=str, default="training/runs/default",
                        help="Output directory for checkpoints and logs.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto' (default: auto).")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run a quick smoke test with minimal data and 1 epoch.")

    # Sprint 4A: Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=SCHEDULER_CHOICES,
                        help="LR scheduler: none, cosine, reduce_on_plateau (default: cosine).")

    # Sprint 4A: Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=0,
                        help="Stop if valid_loss doesn't improve for this many epochs. 0 = disabled.")
    parser.add_argument("--min-delta", type=float, default=0.0,
                        help="Minimum improvement for early stopping (default: 0.0).")

    # Sprint 4A: Augmentation
    parser.add_argument("--augment", action="store_true",
                        help="Apply online image augmentation to training data.")

    # Sprint 4A: Overfit sanity mode
    parser.add_argument("--overfit-small-batch", action="store_true",
                        help="Train on 16 samples repeatedly to prove model can overfit tiny data.")

    # Sprint 4B: Logging
    parser.add_argument("--log-every", type=int, default=10,
                        help="Print batch-level progress every N batches (default: 10).")

    args = parser.parse_args()

    # ---- Seed ----
    torch.manual_seed(args.seed)

    # ---- Device ----
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # ---- Output dir ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Smoke test overrides ----
    max_samples = None
    epochs = args.epochs
    if args.smoke_test:
        max_samples = 32
        epochs = 1

    # ---- Datasets ----
    train_ds = EquationDataset(
        root=args.dataset, split="train",
        height=args.height, width=args.width,
        max_samples=max_samples,
        augment=args.augment,
    )
    valid_ds = EquationDataset(
        root=args.dataset, split="valid",
        height=args.height, width=args.width,
        max_samples=max_samples,
        augment=False,  # Never augment validation.
    )

    # ---- Overfit-small-batch mode ----
    overfit_mode = args.overfit_small_batch
    if overfit_mode:
        overfit_size = min(16, len(train_ds))
        overfit_subset = Subset(train_ds, list(range(overfit_size)))
        train_loader = DataLoader(
            overfit_subset, batch_size=min(args.batch_size, overfit_size),
            shuffle=True, collate_fn=collate_fn,
            num_workers=0,
        )
        # Validation loader is the same tiny subset.
        valid_loader = DataLoader(
            overfit_subset, batch_size=min(args.batch_size, overfit_size),
            shuffle=False, collate_fn=collate_fn,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=True, collate_fn=collate_fn,
            num_workers=0,
        )
        valid_loader = DataLoader(
            valid_ds, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_fn,
            num_workers=0,
        )

    # ---- Model ----
    model = build_model(
        img_height=args.height,
        img_width=args.width,
        num_classes=VOCAB_SIZE,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    # ---- Loss & optimiser ----
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ---- Scheduler ----
    scheduler = build_scheduler(args.scheduler, optimizer, epochs)

    # ---- Early stopping ----
    early_stopper = None
    if args.early_stopping_patience > 0:
        early_stopper = EarlyStopping(
            patience=args.early_stopping_patience,
            min_delta=args.min_delta,
        )

    # ---- Startup banner ----
    train_count = len(train_ds) if not overfit_mode else min(16, len(train_ds))
    valid_count = len(valid_ds) if not overfit_mode else train_count
    _print_startup_banner(
        args, device, total_params,
        train_count, valid_count, epochs, overfit_mode,
    )

    # ---- Training loop ----
    history: list[dict] = []
    best_loss = float("inf")
    best_epoch = 0
    early_stopped = False

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        print(f"Epoch {epoch}/{epochs}")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, total_epochs=epochs, log_every=args.log_every,
        )

        print(f"  validating...", flush=True)
        val_metrics = validate(model, valid_loader, criterion, device)

        # Step scheduler.
        scheduler_step(scheduler, args.scheduler, val_loss=val_metrics["valid_loss"])
        current_lr = get_current_lr(optimizer)

        elapsed = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "valid_loss": round(val_metrics["valid_loss"], 6),
            "exact_accuracy": round(val_metrics["exact_accuracy"], 4),
            "char_accuracy": round(val_metrics["char_accuracy"], 4),
            "learning_rate": current_lr,
            "elapsed_seconds": round(elapsed, 2),
        }
        history.append(record)

        # Overfit mode: compute accuracy on the tiny subset.
        overfit_exact = None
        overfit_char = None
        if overfit_mode:
            overfit_exact, overfit_char = _overfit_accuracy(model, train_loader, device)

        # Save last checkpoint.
        last_ckpt = out_dir / "last_model.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "valid_loss": val_metrics["valid_loss"],
            "args": vars(args),
        }, last_ckpt)

        # Save best checkpoint.
        saved_best = False
        if val_metrics["valid_loss"] < best_loss:
            best_loss = val_metrics["valid_loss"]
            best_epoch = epoch
            best_ckpt = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "valid_loss": val_metrics["valid_loss"],
                "args": vars(args),
            }, best_ckpt)
            saved_best = True

        # Epoch summary.
        _print_epoch_summary(
            epoch, epochs, train_loss, val_metrics,
            current_lr, elapsed, best_loss, saved_best,
            early_stopper, overfit_mode,
            overfit_exact, overfit_char,
        )

        # Prediction preview.
        _print_prediction_preview(
            val_metrics["predictions"], val_metrics["targets"],
        )

        # Early stopping check.
        if early_stopper is not None:
            if early_stopper.step(val_metrics["valid_loss"]):
                early_stopped = True
                print(
                    f"\n⏹ Early stopping triggered at epoch {epoch} "
                    f"(patience={args.early_stopping_patience})."
                )
                break

    # ---- Save training history ----
    history_meta = {
        "epochs": history,
        "early_stopped": early_stopped,
        "scheduler": args.scheduler,
        "augment": args.augment,
        "overfit_small_batch": overfit_mode,
    }
    history_path = out_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_meta, f, indent=2)

    # ---- End-of-run summary ----
    _print_end_of_run(
        history, early_stopped, best_epoch, best_loss,
        out_dir, args.dataset, args.batch_size,
    )


if __name__ == "__main__":
    main()
