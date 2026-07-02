# Training Plan – CNN-CTC Handwriting Recognition

> **Status:** Sprint 3 complete – CNN-CTC baseline model training operational.

## Approach

1. **Synthetic data generation** – render equations with varied fonts, strokes,
   and augmentations to bootstrap training without real samples.
2. **CNN-CTC architecture** – convolutional feature extractor followed by a CTC
   decoder that outputs a symbol sequence directly.
3. **Export** – convert trained model to ONNX and TFLite for on-device inference.

---

## Sprint 2: Synthetic Data Generation

### Overview

Sprint 2 implements a complete pipeline for generating labeled synthetic
equation images suitable for training a CNN+CTC handwriting recogniser.

The pipeline has three stages:

1. **Equation Generator** – produces random valid linear equation labels
2. **Synthetic Renderer** – renders each label into an augmented greyscale PNG
3. **Dataset Builder** – orchestrates generation and creates train/valid/test splits

### Equation Generator

**File:** `training/equation_generator/generate_equations.py`

Generates single-variable linear equations in 9 canonical forms:

| Form | Example |
|---|---|
| `x+a=b` | `x+3=7` |
| `x-a=b` | `x-2=5` |
| `ax=b` | `4x=12` |
| `ax+b=c` | `3x+4=10` |
| `ax-b=c` | `5x-2=18` |
| `x/a=b` | `x/2=5` |
| `ax/b=c` | `3x/4=6` |
| `a(x+b)=c` | `2(x+1)=10` |
| `a(x-b)=c` | `3(x-2)=9` |

**Constraints:**
- Integers only
- No division by zero
- Prefers integer solutions
- Canonical labels have no spaces
- Supported characters: `0123456789x+-=/() `
- Deterministic seed support
- Deduplication support

### Synthetic Renderer

**File:** `training/synthetic_renderer/render_equation.py`

Renders equation strings into 512×128 greyscale PNG images with randomised
augmentations to simulate handwriting variation:

| Augmentation | Range |
|---|---|
| Font size | 20–42px (varies with height) |
| X/Y offset | 5–50px / 5–20px |
| Character spacing | 0–6px |
| Rotation | ±3° |
| Brightness | 0.7–1.0 |
| Blur radius | 0, 0.5, or 1.0 |
| Noise (σ) | 0–5% |

**Font handling:**
- Discovers system TrueType fonts automatically
- Prefers DejaVu, Liberation, Noto, Ubuntu families
- Falls back to Pillow default font if none available

### Dataset Structure

```
training/datasets/synthetic_v1/
├── images/
│   ├── train/     # 80% of samples
│   │   ├── 000001.png
│   │   └── ...
│   ├── valid/     # 10% of samples
│   └── test/      # 10% of samples
├── labels/
│   ├── train/
│   │   ├── 000001.txt   (content: "3x+4=10")
│   │   └── ...
│   ├── valid/
│   └── test/
└── metadata.json
```

Each label file has the same basename as its image and contains the
canonical equation string (no spaces, no newline).

### Commands

**Build dataset (500 samples):**

```bash
python training/build_synthetic_dataset.py \
  --out training/datasets/synthetic_v1 \
  --samples 500 \
  --seed 42
```

**Build dataset (5000 samples, production):**

```bash
python training/build_synthetic_dataset.py \
  --out training/datasets/synthetic_v1 \
  --samples 5000 \
  --seed 42
```

**CLI arguments:**

| Arg | Default | Description |
|---|---|---|
| `--out` | `training/datasets/synthetic_v1` | Output directory |
| `--samples` | `5000` | Total number of samples |
| `--seed` | `42` | Random seed |
| `--width` | `512` | Image width (px) |
| `--height` | `128` | Image height (px) |

**Verify dataset:**

```bash
python training/verify_synthetic_dataset.py \
  --dataset training/datasets/synthetic_v1
```

**Run tests:**

```bash
python -m pytest training/tests/ -v
```

### Test Coverage

| Test | File |
|---|---|
| Supported characters only | `test_equation_generator.py` |
| Single equals sign | `test_equation_generator.py` |
| No division by zero | `test_equation_generator.py` |
| Deterministic seed | `test_equation_generator.py` |
| Deduplication | `test_equation_generator.py` |
| All 9 forms produce valid labels | `test_equation_generator.py` |
| Generates 500+ unique equations | `test_equation_generator.py` |
| Non-empty PNG created | `test_synthetic_renderer.py` |
| Correct dimensions | `test_synthetic_renderer.py` |
| Greyscale mode | `test_synthetic_renderer.py` |
| Text actually rendered (not blank) | `test_synthetic_renderer.py` |
| Image/label file pairs created | `test_synthetic_renderer.py` |
| Label content matches input | `test_synthetic_renderer.py` |
| Small dataset integration test | `test_synthetic_renderer.py` |

### Known Limitations

1. **No real handwriting fonts** – augmentations simulate variation but use
   standard system fonts, not actual handwriting typefaces. Real-world
   performance will benefit from fine-tuning on real samples.
2. **Limited augmentation variety** – no elastic deformation, ink bleed,
   or perspective distortion. These can be added in future sprints.
3. **Integer coefficients only** – all generated equations use small integers
   (typically −20 to +20). Larger ranges may be needed later.
4. **No TypeScript Math Core cross-validation** – the verifier uses a
   lightweight Python regex-based grammar check rather than invoking the
   Sprint 1 TypeScript solver directly. Full cross-validation can be added
   when a Python↔TS bridge is practical.
5. **Single variable (x) only** – by design for Sprint 1/2 scope.
6. **No negative coefficients in front of x** – forms like `-3x+4=10` are
   not generated yet; can be added in a future iteration.

---

## Sprint 3: CNN-CTC Model Training

### Overview

Sprint 3 implements the first offline handwriting recognition baseline model.
A small CNN-CTC architecture is trained on the synthetic equation dataset
from Sprint 2, with full training, evaluation, checkpointing, and smoke
testing support.

### Model Architecture

**File:** `training/models/cnn_ctc.py`

```
Input: (B, 1, 128, 512) greyscale image

CNN Encoder (4 blocks):
  Block 1: Conv2d(1→32)  → BN → ReLU → MaxPool(2,2)  → (B, 32, 64, 256)
  Block 2: Conv2d(32→64) → BN → ReLU → MaxPool(2,2)  → (B, 64, 32, 128)
  Block 3: Conv2d(64→128)→ BN → ReLU → MaxPool(2,2)  → (B, 128, 16, 64)
  Block 4: Conv2d(128→128)→ BN → ReLU → MaxPool(2,2) → (B, 128, 8, 32)

Reshape: (B, 128, 8, 32) → (B, 1024, 32) → permute → (32, B, 1024)
   Width becomes time axis (T=32 time steps for W=512 input)

BiLSTM: (32, B, 1024) → (32, B, 256)
   Single-layer bidirectional LSTM (hidden=128 per direction)

Linear: (32, B, 256) → (32, B, 19)
   Projects to vocabulary size (19 = blank + 18 printable chars)

Output: (T, B, C) log-probabilities via log_softmax
```

**Parameters:** ~1.4M (suitable for later mobile export)

### CTC Vocabulary

**File:** `training/models/vocabulary.py`

| Index | Token |
|---|---|
| 0 | `<blank>` (CTC blank) |
| 1–10 | `0 1 2 3 4 5 6 7 8 9` |
| 11 | `x` |
| 12–15 | `+ - = /` |
| 16–17 | `( )` |
| 18 | ` ` (space) |

**Total vocabulary size:** 19

### Training Command

**Full training:**

```bash
python3 training/train_ctc.py \
  --dataset training/datasets/synthetic_v1 \
  --epochs 20 \
  --batch-size 16 \
  --lr 1e-3 \
  --seed 42 \
  --out training/runs/synthetic_v1_full
```

**CLI arguments:**

| Arg | Default | Description |
|---|---|---|
| `--dataset` | (required) | Path to dataset root |
| `--epochs` | `5` | Number of training epochs |
| `--batch-size` | `16` | Batch size |
| `--lr` | `1e-3` | Learning rate (Adam) |
| `--seed` | `42` | Random seed |
| `--width` | `512` | Image width |
| `--height` | `128` | Image height |
| `--out` | `training/runs/default` | Output directory |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |
| `--smoke-test` | off | Quick validation mode |

### Smoke Test Command

```bash
python3 training/train_ctc.py \
  --dataset training/datasets/synthetic_v1 \
  --epochs 1 \
  --batch-size 8 \
  --seed 42 \
  --out training/runs/synthetic_v1_smoke \
  --smoke-test
```

Smoke test behaviour:
- Uses only 32 samples for train and valid
- Runs 1 epoch
- Confirms forward pass, backward pass, checkpoint save, and evaluation work
- Not for accuracy – purely for pipeline validation

### Evaluation Command

```bash
python3 training/evaluate.py \
  --dataset training/datasets/synthetic_v1 \
  --checkpoint training/runs/synthetic_v1_smoke/best_model.pt \
  --split valid \
  --batch-size 8
```

**CLI arguments:**

| Arg | Default | Description |
|---|---|---|
| `--dataset` | (required) | Path to dataset root |
| `--checkpoint` | (required) | Path to `.pt` checkpoint |
| `--split` | `valid` | `train`, `valid`, or `test` |
| `--batch-size` | `16` | Batch size |
| `--device` | `auto` | `cpu`, `cuda`, or `auto` |

### Metrics Explained

| Metric | Description |
|---|---|
| **Train loss** | Average CTC loss over training batches |
| **Valid loss** | Average CTC loss over validation set |
| **Exact sequence accuracy** | Fraction of predictions that match the target string exactly |
| **Character accuracy** | Per-sample character-level accuracy, averaged over all samples |
| **Average edit distance** | Mean Levenshtein edit distance between prediction and target |

### Output Structure

```
training/runs/<run_name>/
├── best_model.pt           # Checkpoint with lowest validation loss
├── last_model.pt           # Checkpoint from the final epoch
└── training_history.json   # Per-epoch metrics log
```

Each checkpoint contains:
- `epoch`: training epoch number
- `model_state_dict`: model weights
- `optimizer_state_dict`: optimizer state
- `train_loss`: training loss at that epoch
- `valid_loss`: validation loss at that epoch
- `args`: original CLI arguments

### Test Coverage (Sprint 3)

| Test | File |
|---|---|
| All chars map to unique indices | `test_vocabulary.py` |
| Unsupported chars raise ValueError | `test_vocabulary.py` |
| char↔idx roundtrip | `test_vocabulary.py` |
| Vocab size includes blank | `test_vocabulary.py` |
| encode_label encodes correctly | `test_vocabulary.py` |
| decode_indices skips blanks | `test_vocabulary.py` |
| CTC greedy decode from index list | `test_vocabulary.py` |
| CTC greedy decode from 1-D tensor | `test_vocabulary.py` |
| CTC greedy decode from 2-D logits | `test_vocabulary.py` |
| Dataset loads image/label pairs | `test_equation_dataset.py` |
| Image shape (1, H, W) | `test_equation_dataset.py` |
| Image normalised to [0, 1] | `test_equation_dataset.py` |
| Label dtype int32 | `test_equation_dataset.py` |
| collate_fn returns valid CTC tensors | `test_equation_dataset.py` |
| Model forward pass correct shape | `test_cnn_ctc.py` |
| Output is log-probabilities | `test_cnn_ctc.py` |
| Works with different image sizes | `test_cnn_ctc.py` |
| Works without BiLSTM | `test_cnn_ctc.py` |
| Param count < 5M | `test_cnn_ctc.py` |
| CTC loss computes finite scalar | `test_cnn_ctc.py` |
| Backward pass produces gradients | `test_cnn_ctc.py` |

### Known Limitations

1. **Smoke test accuracy is 0%** – expected; 1 epoch on 32 samples is
   purely for pipeline validation, not model quality.
2. **No learning rate scheduler** – a fixed learning rate is used; adding
   cosine annealing or reduce-on-plateau would improve convergence.
3. **No data augmentation at training time** – the dataset images have
   baked-in augmentations from rendering, but no additional online
   augmentation (e.g. random crops, elastic distortion) is applied.
4. **CPU-only training validated** – GPU training should work but has not
   been explicitly tested in this sprint.
5. **Small dataset** – the 500-sample synthetic_v1 dataset is minimal;
   training on 5000+ samples with more epochs is needed for real accuracy.
6. **No early stopping** – training runs for the full number of epochs;
   early stopping based on validation loss would be beneficial.
7. **Single BiLSTM layer** – deeper recurrent layers or attention could
   improve sequence modelling but would increase model size.

---

## Data pipeline

```
generate_equations.py  →  equation strings + ground truth
render_equation.py     →  synthetic images (varied styles)
datasets/              →  versioned dataset directories
```

## Model training (Sprint 3 – complete ✓)

```
vocabulary.py        →  CTC vocabulary and encoding
equation_dataset.py  →  PyTorch dataset loader
cnn_ctc.py           →  CNN-CTC model definition
train_ctc.py         →  training loop with checkpointing
evaluate.py          →  evaluation with metrics reporting
```

## Model export (Sprint 4 – not yet implemented)

```
export_onnx.py   →  ONNX export for cross-platform
export_tflite.py →  TFLite export for mobile
```

## Milestones

- [x] Generate 500+ synthetic equations (Sprint 2) ✓
- [x] CNN-CTC model training pipeline (Sprint 3) ✓
- [ ] Generate 10k synthetic equations (scale up)
- [ ] Train baseline CNN-CTC to > 90% symbol accuracy
- [ ] Export and benchmark on-device inference (Sprint 4)
