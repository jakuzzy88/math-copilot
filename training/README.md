# Training Pipeline — Offline Handwritten Math Copilot

End-to-end guide: environment setup → dataset generation → training → evaluation → analysis.

---

## 1. Environment Setup

### Prerequisites

- Python 3.10+
- pip

### Install Dependencies

```bash
# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install required packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow numpy pytest
```

**Tested versions:**

| Package      | Version        |
|-------------|----------------|
| Python      | 3.12.3         |
| PyTorch     | 2.12.1+cpu     |
| torchvision | 0.27.1+cpu     |
| Pillow      | 12.2.0         |
| NumPy       | 2.4.4          |
| pytest      | 9.0.3          |

> **GPU (optional):** For CUDA, replace the pip install line with the appropriate
> PyTorch CUDA wheel from https://pytorch.org/get-started/locally/

---

## 2. Generate Synthetic Dataset

The training pipeline uses rendered synthetic equation images. Generate the `synthetic_v2` dataset (5000 samples):

```bash
# Run from the math-copilot/ directory
python3 training/build_synthetic_dataset.py \
  --out training/datasets/synthetic_v2 \
  --samples 5000 \
  --seed 42
```

This creates:
```
training/datasets/synthetic_v2/
├── images/
│   ├── train/    (4000 images)
│   ├── valid/    (500 images)
│   └── test/     (500 images)
└── labels/
    ├── train/    (4000 .txt files)
    ├── valid/    (500 .txt files)
    └── test/     (500 .txt files)
```

### Verify Dataset

```bash
python3 training/verify_synthetic_dataset.py \
  --dataset training/datasets/synthetic_v2
```

---

## 3. Run Tests

Run the full test suite before training to make sure everything works:

```bash
python3 -m pytest training/tests/ -v
```

Expected: **123 passed, 1 warning** (the warning is a benign PyTorch scheduler note in unit tests).

---

## 4. Training

### Smoke Test (quick sanity check, ~6 min)

```bash
python3 training/train_ctc.py \
  --dataset training/datasets/synthetic_v2 \
  --epochs 1 --batch-size 8 --seed 42 \
  --out training/runs/synthetic_v2_smoke \
  --smoke-test
```

### Overfit Sanity Check (~60 min)

Confirms the model can memorise a tiny subset (loss should drop steadily):

```bash
python3 training/train_ctc.py \
  --dataset training/datasets/synthetic_v2 \
  --epochs 10 --batch-size 16 --seed 42 \
  --out training/runs/synthetic_v2_overfit \
  --overfit-small-batch
```

### Full Training (Sprint 4B)

```bash
python3 training/train_ctc.py \
  --dataset training/datasets/synthetic_v2 \
  --epochs 75 \
  --batch-size 32 \
  --lr 0.001 \
  --seed 42 \
  --out training/runs/synthetic_v2_full \
  --scheduler cosine \
  --augment \
  --early-stopping-patience 12 \
  --min-delta 0.0005
```

### ⏱ Estimated Training Time

| Setup              | Per Epoch | 30 Epochs | 75 Epochs |
|--------------------|-----------|-----------|-----------|
| **CPU** (measured) | ~6 min    | ~3 hours  | ~7.5 hours|
| **GPU** (estimate) | ~30 sec   | ~15 min   | ~40 min   |

> The per-epoch time of **~356 seconds** was measured on CPU with batch_size=32,
> 4000 train samples, 500 validation samples, image size 128×512.
> If CPU training is too slow, reduce `--epochs` to 30 and increase later.

### Training Arguments Reference

| Argument                      | Default   | Description                                    |
|-------------------------------|-----------|------------------------------------------------|
| `--dataset`                   | required  | Path to dataset root                           |
| `--epochs`                    | 5         | Number of training epochs                      |
| `--batch-size`                | 16        | Batch size                                     |
| `--lr`                        | 0.001     | Learning rate (Adam)                           |
| `--seed`                      | 42        | Random seed                                    |
| `--width`                     | 512       | Image width in pixels                          |
| `--height`                    | 128       | Image height in pixels                         |
| `--out`                       | `training/runs/default` | Output directory              |
| `--device`                    | auto      | `cpu`, `cuda`, or `auto`                       |
| `--smoke-test`                | off       | Quick 1-epoch, 32-sample run                   |
| `--scheduler`                 | cosine    | `cosine`, `reduce_on_plateau`, or `none`       |
| `--early-stopping-patience`   | 0 (off)   | Stop after N epochs without improvement        |
| `--min-delta`                 | 0.0       | Minimum loss improvement for early stopping    |
| `--augment`                   | off       | Enable online image augmentation (train only)  |
| `--overfit-small-batch`       | off       | Train on 16 samples to verify learning ability |

### Training Outputs

After training, the run directory contains:

```
training/runs/synthetic_v2_full/
├── best_model.pt              # Best checkpoint (lowest valid_loss)
├── last_model.pt              # Last epoch checkpoint
└── training_history.json      # Per-epoch metrics log
```

---

## 5. Evaluation

### Evaluate on Test Split

```bash
python3 training/evaluate.py \
  --dataset training/datasets/synthetic_v2 \
  --checkpoint training/runs/synthetic_v2_full/best_model.pt \
  --split test \
  --batch-size 32 \
  --out training/runs/synthetic_v2_full
```

### Evaluate on Validation Split

```bash
python3 training/evaluate.py \
  --dataset training/datasets/synthetic_v2 \
  --checkpoint training/runs/synthetic_v2_full/best_model.pt \
  --split valid \
  --batch-size 32 \
  --out training/runs/synthetic_v2_full
```

This prints metrics and (when `--out` is provided) saves:
- `evaluation_<split>.json` — metrics summary
- `prediction_samples_<split>.json` — up to 100 sample predictions

---

## 6. Per-Form Accuracy Analysis

Analyse how the model performs on each equation form (e.g. `x+a=b`, `ax=b`, `a(x+b)=c`):

```bash
python3 training/analyze_predictions.py \
  --dataset training/datasets/synthetic_v2 \
  --checkpoint training/runs/synthetic_v2_full/best_model.pt \
  --split test \
  --batch-size 32 \
  --out training/runs/synthetic_v2_full
```

Saves:
- `prediction_analysis_test.json` — per-form metrics breakdown
- `prediction_samples_test.json` — sample predictions with edit distances

---

## 7. Training Run Summary

Generate a concise summary from the training history:

```bash
python3 training/summarize_training_run.py \
  --run training/runs/synthetic_v2_full
```

Saves `run_summary.json` with best epoch, best valid loss, final metrics, and early stopping status.

---

## 8. Full Sprint 4B Pipeline (copy-paste)

Run everything in sequence after training completes:

```bash
# 1. Evaluate test split
python3 training/evaluate.py \
  --dataset training/datasets/synthetic_v2 \
  --checkpoint training/runs/synthetic_v2_full/best_model.pt \
  --split test --batch-size 32 \
  --out training/runs/synthetic_v2_full

# 2. Evaluate valid split
python3 training/evaluate.py \
  --dataset training/datasets/synthetic_v2 \
  --checkpoint training/runs/synthetic_v2_full/best_model.pt \
  --split valid --batch-size 32 \
  --out training/runs/synthetic_v2_full

# 3. Per-form analysis
python3 training/analyze_predictions.py \
  --dataset training/datasets/synthetic_v2 \
  --checkpoint training/runs/synthetic_v2_full/best_model.pt \
  --split test --batch-size 32 \
  --out training/runs/synthetic_v2_full

# 4. Training summary
python3 training/summarize_training_run.py \
  --run training/runs/synthetic_v2_full
```

---

## 9. How to Interpret Training Results

### Key Metrics

| Metric                | What It Tells You                                    | Good Range       |
|-----------------------|------------------------------------------------------|------------------|
| **Train loss**        | How well the model fits training data                | Decreasing       |
| **Valid loss**        | Generalisation — are we overfitting?                 | Decreasing or flat |
| **Exact accuracy**    | % of equations predicted exactly correct             | ≥ 70% for export |
| **Character accuracy**| Average per-character match rate                     | ≥ 85%            |
| **Avg edit distance** | Mean Levenshtein distance (lower = better)           | ≤ 1.5            |

### Reading the Training History

The `training_history.json` records per-epoch metrics. Look for:

1. **Loss convergence**: Both train and valid loss should decrease. If train loss drops but valid loss stalls or rises, the model is overfitting.
2. **Learning rate schedule**: With cosine annealing, the LR follows a smooth cosine curve. Check `learning_rate` values are decreasing.
3. **Early stopping**: If `early_stopped` is `true`, training was halted because valid loss didn't improve for N epochs.

### Per-Form Analysis

Run `analyze_predictions.py` to see accuracy per equation form. Common patterns:

| Symptom                        | Likely Cause                          | Fix                        |
|--------------------------------|---------------------------------------|----------------------------|
| `a(x+b)=c` has low accuracy   | Parentheses are harder to render/read | More training data         |
| `ax/b=c` fails often           | Division sign confusion               | Check slash rendering      |
| Short equations (e.g. `ax=b`) are fine, long ones fail | Model runs out of time steps | Increase image width |

### Debug Toolkit

For deeper investigation, use the debug toolkit:

```bash
python3 training/debug_predictions.py \
  --dataset training/datasets/synthetic_v2 \
  --checkpoint training/runs/synthetic_v2_full/best_model.pt \
  --split test --batch-size 32 \
  --out training/runs/synthetic_v2_full/debug
```

This produces:

| Output File                       | Contents                                  |
|-----------------------------------|-------------------------------------------|
| `debug_report_test.json`          | Overall accuracy summary                  |
| `char_confusion_test.json`        | Per-character confusion matrix             |
| `failures_by_form_test.json`      | Failed predictions grouped by form        |
| `worst_predictions_test.json`     | Top 20 highest edit-distance predictions  |
| `debug_full_report_test.txt`      | Combined human-readable text report       |

### What to Look For

- **Character confusion**: If `(` is frequently confused with `1`, the renderer's font choice may be ambiguous.
- **Worst predictions**: If the worst predictions are all empty strings, the model may not be learning alignment (CTC blank dominance).
- **Form-specific failures**: If one equation form has significantly lower accuracy, consider generating more training samples of that form.

---

## 10. Decision Gate

After evaluation, check the **test exact accuracy**:

| Test Exact Accuracy | Next Step                              |
|---------------------|----------------------------------------|
| **≥ 70%**           | → Sprint 5A: ONNX Export               |
| **< 70%**           | → Sprint 4C: Model/Data Improvements   |

---

## Project Structure

```
training/
├── train_ctc.py                 # Training loop (CNN-CTC)
├── evaluate.py                  # Checkpoint evaluation
├── analyze_predictions.py       # Per-equation-form analysis
├── debug_predictions.py         # Debug toolkit (Sprint A)
├── export_onnx.py               # ONNX export (Sprint C)
├── summarize_training_run.py    # Training run summary
├── build_synthetic_dataset.py   # Dataset generator
├── verify_synthetic_dataset.py  # Dataset verification
├── models/
│   ├── cnn_ctc.py               # CNN-CTC model architecture
│   ├── equation_dataset.py      # PyTorch Dataset + augmentation
│   └── vocabulary.py            # CTC vocabulary & decoding
├── equation_generator/
│   └── generate_equations.py    # Equation string generator
├── synthetic_renderer/          # Image rendering pipeline
├── datasets/
│   ├── synthetic_v1/            # 500-sample dataset
│   └── synthetic_v2/            # 5000-sample dataset
├── runs/                        # Training run outputs
└── tests/                       # Test suite (pytest)
```
