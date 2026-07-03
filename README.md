# Offline Handwritten Math Copilot

**Point your camera at a handwritten equation → get the solution with step-by-step explanations — entirely offline.**

---

## Vision

The Offline Handwritten Math Copilot is a privacy-first mobile application that recognises handwritten linear equations from a live camera feed, solves them symbolically, and explains each algebraic step in plain language.

**Core principles:**

- **Offline / Privacy-first** — everything runs on-device. No cloud APIs, no data leaves the phone.
- **AI reads, code solves** — a CNN-CTC neural network recognises handwriting into a symbol string. A deterministic, fully tested symbolic engine then tokenises, parses, solves, and explains the math. The AI never "does math"; it only reads ink.
- **Pedagogical** — the explanation engine teaches *why* each step works (inverse operations, balance, isolating the variable), not just *what* happens.

---

## MVP Scope

The current MVP targets **single-variable linear equations** with one variable **x**.

### Supported

| Equation | Solution |
|---|---|
| `x+5=9` | `x = 4` |
| `2x=8` | `x = 4` |
| `3x+4=10` | `x = 2` |
| `5x-2=18` | `x = 4` |
| `x/2=5` | `x = 10` |
| `3x/4=6` | `x = 8` |
| `2(x+1)=10` | `x = 4` |

**Supports:** integer constants, simple fractions, parentheses, implicit multiplication (`2x`, `2(x+1)`).

### Not Yet Supported

- Quadratic equations
- Square roots / exponents
- Matrices
- Systems of equations
- Calculus
- Multiple variables

---

## Architecture Overview

```
Camera Frame
  → Image Preprocessing (greyscale, crop, normalise)
  → CNN-CTC OCR Model (ONNX / TFLite)
  → Candidate Beam (top-k predictions)
  → OCR Normalisation (fix common misreads: O→0, I→1)
  → Grammar Validation (EBNF structure check)
  → Parser (recursive-descent → AST)
  → Symbolic Solver (ax + b = c → x = …)
  → Action Log (each algebraic step recorded)
  → Explanation Engine (pedagogical natural-language steps)
  → Stability Aggregator (temporal smoothing across frames)
  → UI Overlay (solution + step-by-step display)
```

### Major Components

| Component | Description |
|---|---|
| **Math Core** (`app/src/`) | Tokeniser, recursive-descent parser, AST-based linear solver, action logger, and explanation engine. Fully deterministic and tested — no AI here. |
| **OCR Candidate Pipeline** (`app/src/pipeline/`) | Accepts OCR beam candidates, normalises common misrecognitions, validates against the grammar, parses and solves, and ranks results by confidence. |
| **Stability Aggregator** (`app/src/pipeline/`) | Sliding-window temporal filter that prevents UI flickering by requiring consistent recognition across multiple frames before displaying a result. |
| **Synthetic Data Pipeline** (`training/`) | Generates thousands of rendered equation images with randomised fonts, sizes, rotations, and noise for training. |
| **CNN-CTC Training Pipeline** (`training/`) | PyTorch-based training loop with CTC loss, cosine annealing, online augmentation, early stopping, and comprehensive evaluation/debug tooling. |
| **Evaluation & Debug Tools** (`training/`) | Per-form accuracy analysis, character confusion matrices, worst-prediction reports, and training run summaries. |
| **Local UI Prototype** (`ui/`) | Static HTML/CSS/JS browser demo of the solver and explanation engine. No camera — type an equation, see the solution and steps. |
| **Mobile App** (`app/`) | React Native app shell with Android native project. Currently runs in demo mode with UI overlays, guide box, and explanation display. Real camera and ONNX inference are next. |

---

## Repository Structure

```
math-copilot/
├── app/                          # React Native app + TypeScript Math Core
│   ├── src/
│   │   ├── parser/               #   Tokeniser, parser, AST definitions
│   │   ├── solver/               #   Linear solver, simplifier, action log
│   │   ├── explanation/          #   Pedagogical explanation engine
│   │   ├── grammar/              #   Grammar validator
│   │   ├── pipeline/             #   OCR candidate pipeline, stability aggregator
│   │   ├── inference/            #   Camera frame processing, ONNX stubs
│   │   ├── ui/                   #   UI state adapter, overlay components
│   │   ├── screens/              #   MathCameraScreen, real-mode stubs
│   │   ├── diagnostics/          #   Runtime diagnostic utilities
│   │   └── __tests__/            #   Jest test suite (387 tests)
│   ├── android/                  #   Android native project (Gradle, Kotlin)
│   ├── index.js                  #   React Native entry point
│   ├── package.json
│   └── tsconfig.json
│
├── training/                     # Python ML training pipeline
│   ├── train_ctc.py              #   CNN-CTC training loop
│   ├── evaluate.py               #   Checkpoint evaluation
│   ├── analyze_predictions.py    #   Per-form accuracy analysis
│   ├── debug_predictions.py      #   Debug toolkit (confusion, worst predictions)
│   ├── summarize_training_run.py #   Training run summary generator
│   ├── build_synthetic_dataset.py#   Synthetic dataset generator
│   ├── verify_synthetic_dataset.py#  Dataset verification
│   ├── export_onnx.py            #   ONNX model export
│   ├── models/                   #   CNN-CTC model, dataset loader, vocabulary
│   ├── equation_generator/       #   Equation string generator
│   ├── synthetic_renderer/       #   Image rendering pipeline
│   └── tests/                    #   pytest test suite
│
├── shared/                       # Shared specifications
│   ├── supported_grammar.md      #   EBNF grammar definition
│   ├── symbols.json              #   Symbol set metadata
│   └── test_equations.json       #   Canonical test equations
│
├── docs/                         # Project documentation
│   ├── architecture.md           #   Architecture overview
│   ├── mvp_scope.md              #   MVP scope definition
│   ├── training_plan.md          #   Training strategy and milestones
│   ├── onnx_export_plan.md       #   ONNX export plan
│   └── git_workflow.md           #   Branching and merge workflow
│
├── ui/                           # Static browser prototype
│   ├── index.html                #   Main page
│   ├── solver.js                 #   JS solver port
│   ├── app.js                    #   UI logic
│   └── style.css                 #   Styles
│
├── .gitignore
└── README.md                     # ← You are here
```

### Generated Folders (Git-ignored)

These are created locally and must **never** be committed:

| Folder | Contents |
|---|---|
| `training/datasets/` | Generated synthetic image datasets |
| `training/runs/` | Training outputs (checkpoints, metrics, logs) |
| `.mathsolvervenv/` | Python virtual environment |
| `node_modules/` | Node.js dependencies |
| `*.pt`, `*.onnx`, `*.tflite` | Model checkpoint and export files |

---

## Git Workflow

- **`main`** contains stable, tested, working code.
- All development happens on **feature branches** and is merged after tests pass.

### Branch Naming

| Prefix | Purpose | Example |
|---|---|---|
| `feature/` | New functionality | `feature/onnx-export`, `feature/camera-poc` |
| `fix/` | Bug fixes | `fix/explanation-quality` |
| `experiment/` | Exploratory work | `experiment/model-v2` |
| `docs/` | Documentation only | `docs/api-reference` |

### Rules

- Run all tests before merging.
- Never commit generated datasets, training runs, checkpoints, virtual environments, or `node_modules/`.

> 📄 Full details: [docs/git_workflow.md](docs/git_workflow.md)

---

## Prerequisites

### Linux / macOS

| Requirement | Notes |
|---|---|
| Git | |
| Python 3.10+ | |
| Node.js 18+ | |
| npm | |
| CUDA GPU | Optional — CPU training works, just slower |

### Windows

| Requirement | Notes |
|---|---|
| Git for Windows | |
| Python 3.10+ | |
| Node.js 18+ | |
| PowerShell | Ships with Windows |
| CUDA GPU | Optional |

**Recommended on Windows:**
- [Windows Terminal](https://aka.ms/terminal) for a better shell experience.
- [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) for a smoother ML workflow (all Linux commands work natively).

---

## Clone the Repository

**Linux / macOS / Windows PowerShell:**

```bash
git clone https://github.com/jakuzzy88/math-copilot.git
cd math-copilot
```

**SSH alternative:**

```bash
git clone git@github.com:jakuzzy88/math-copilot.git
cd math-copilot
```

---

## Python Environment Setup

All Python commands should be run from the **repository root** (`math-copilot/`).

### Linux / macOS

```bash
python3 -m venv .mathsolvervenv
source .mathsolvervenv/bin/activate
python3 -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow numpy pytest
```

### Windows PowerShell

```powershell
py -3 -m venv .mathsolvervenv
.\.mathsolvervenv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow numpy pytest
```

> **PowerShell activation blocked?** Run this once first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

> **GPU training:** Replace the PyTorch install line with the appropriate CUDA wheel from https://pytorch.org/get-started/locally/

---

## App / TypeScript Setup

Install the TypeScript dependencies inside `app/`:

```bash
cd app
npm install
npm test
cd ..
```

This works identically on Linux, macOS, and Windows.

---

## Run All Tests

### Python / Training Tests

**Linux / macOS:**

```bash
python3 -m pytest training/tests/ -v
```

**Windows:**

```powershell
python -m pytest training/tests/ -v
```

### App / TypeScript Tests

```bash
cd app
npm test
cd ..
```

Both test suites should pass fully. The Python suite may emit a benign PyTorch scheduler warning in unit tests — this is expected.

---

## Local UI Prototype

The `ui/` folder contains a static browser demo of the Math Core. No camera, no model — you type an equation and see the solver + explanation engine in action.

### How to Run

**Option A — Open directly:**

Open `ui/index.html` in any browser.

**Option B — Local server:**

*Linux / macOS:*

```bash
python3 -m http.server 8080 -d ui
```

*Windows:*

```powershell
python -m http.server 8080 -d ui
```

Then open [http://localhost:8080](http://localhost:8080).

### Examples to Try

- `3x+2=17`
- `5x-2=18`
- `x/2=5`
- `2(x+1)=10`

---

## Android App — Build & Deploy to Phone

The app can be built as a standalone APK and installed on any Android phone. It currently runs in **demo mode** (fake OCR, placeholder camera) — no real camera or model required.

### Prerequisites

| Requirement | Notes |
|---|---|
| Node.js 18+ | |
| Java 17 (OpenJDK) | `java -version` to check |
| Android SDK | Via [Android Studio](https://developer.android.com/studio) |
| `ANDROID_HOME` env var | Set to your SDK path (e.g. `~/Android/Sdk`) |
| USB debugging on phone | Enable in phone's Developer Options |
| ADB | Comes with Android SDK, verify with `adb devices` |

### Build Offline Release APK (recommended)

This creates a standalone APK that runs without a computer connected:

```bash
cd app
npm install               # first time only
npm run build:release-apk # bundles JS + builds release APK
```

The APK is created at:

```
app/android/app/build/outputs/apk/release/app-release.apk
```

### Install on Phone

Connect your phone via USB, then:

```bash
# Verify device is connected:
adb devices

# Install the APK:
adb install -r android/app/build/outputs/apk/release/app-release.apk

# Launch:
adb shell am start -n com.mathcopilot/.MainActivity
```

Or simply tap the **MathCopilot** icon in your phone's app launcher.

### Debug Build (with Metro hot-reload)

For development with live code reloading:

```bash
# Terminal 1 — start Metro bundler:
cd app
npm start

# Terminal 2 — build, install, and launch:
cd app
npm run android

# If the app shows a white screen, forward the port:
adb reverse tcp:8081 tcp:8081
```

### Available Scripts

| Script | What It Does |
|---|---|
| `npm start` | Start Metro dev server |
| `npm run android` | Debug build + install (requires Metro) |
| `npm run bundle:android` | Bundle JS for release |
| `npm run assemble:release` | Gradle release build |
| `npm run build:release-apk` | **One-command offline APK** (bundle + build) |
| `npm test` | Run Jest test suite |

### Troubleshooting

| Problem | Fix |
|---|---|
| `adb devices` empty | Enable USB debugging in Developer Options |
| Device shows "unauthorized" | Accept the USB debugging prompt on phone |
| "Unable to load script" | You're running the debug APK without Metro — use `build:release-apk` instead |
| White screen with Metro running | Run `adb reverse tcp:8081 tcp:8081` |
| Gradle build fails | Check `java -version` shows 17, and `ANDROID_HOME` is set |

> ⚠️ **The release APK uses a debug keystore for local testing only.** Replace with proper release signing before publishing. See [React Native Signed APK docs](https://reactnative.dev/docs/signed-apk-android).

---

## Generate Synthetic Dataset

The OCR model trains on synthetic images of rendered equations. The generator creates randomised variations with different fonts, sizes, rotations, and noise levels.

### Generate

**Linux / macOS:**

```bash
python3 training/build_synthetic_dataset.py \
  --out training/datasets/synthetic_v2 \
  --samples 5000 \
  --seed 42
```

**Windows PowerShell:**

```powershell
python training/build_synthetic_dataset.py `
  --out training/datasets/synthetic_v2 `
  --samples 5000 `
  --seed 42
```

### Verify

**Linux / macOS:**

```bash
python3 training/verify_synthetic_dataset.py --dataset training/datasets/synthetic_v2
```

**Windows:**

```powershell
python training/verify_synthetic_dataset.py --dataset training/datasets/synthetic_v2
```

### Resulting Structure

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

---

## Training Guide

### Smoke Test (quick sanity check)

Runs 1 epoch on a tiny subset to verify the pipeline works end-to-end:

**Linux / macOS:**

```bash
python3 training/train_ctc.py \
  --dataset training/datasets/synthetic_v2 \
  --epochs 1 \
  --batch-size 8 \
  --seed 42 \
  --out training/runs/synthetic_v2_smoke \
  --smoke-test \
  --log-every 2
```

**Windows PowerShell:**

```powershell
python training/train_ctc.py `
  --dataset training/datasets/synthetic_v2 `
  --epochs 1 `
  --batch-size 8 `
  --seed 42 `
  --out training/runs/synthetic_v2_smoke `
  --smoke-test `
  --log-every 2
```

### First CPU Training Run (recommended starting point)

**Linux / macOS:**

```bash
python3 training/train_ctc.py \
  --dataset training/datasets/synthetic_v2 \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.001 \
  --seed 42 \
  --out training/runs/synthetic_v2_full_20ep \
  --scheduler cosine \
  --augment \
  --early-stopping-patience 8 \
  --min-delta 0.0005 \
  --log-every 10
```

**Windows PowerShell:**

```powershell
python training/train_ctc.py `
  --dataset training/datasets/synthetic_v2 `
  --epochs 20 `
  --batch-size 32 `
  --lr 0.001 `
  --seed 42 `
  --out training/runs/synthetic_v2_full_20ep `
  --scheduler cosine `
  --augment `
  --early-stopping-patience 8 `
  --min-delta 0.0005 `
  --log-every 10
```

### Training Progression Recommendations

| Milestone | Epochs | Purpose |
|---|---|---|
| First run | 20 | Verify loss decreases, get baseline metrics |
| Meaningful checkpoint | 30–50 | Character accuracy and edit distance should improve significantly |
| MVP OCR baseline | 50–75 | Target ≥ 70% exact accuracy for export readiness |

**What to watch first:** valid loss, character accuracy, and average edit distance will improve well before exact sequence accuracy. Don't be discouraged by low exact accuracy early in CTC training — this is normal.

### Training Outputs

Each training run produces:

```
training/runs/<run_name>/
├── best_model.pt              # Best checkpoint (lowest validation loss)
├── last_model.pt              # Final epoch checkpoint
└── training_history.json      # Per-epoch metrics log
```

> 📄 Full training arguments, timing estimates, and metrics guide: [training/README.md](training/README.md)

---

## Evaluation Guide

After training, evaluate and analyse the checkpoint:

### Evaluate on Test Split

**Linux / macOS:**

```bash
python3 training/evaluate.py \
  --dataset training/datasets/synthetic_v2 \
  --checkpoint training/runs/synthetic_v2_full_20ep/best_model.pt \
  --split test \
  --batch-size 32 \
  --out training/runs/synthetic_v2_full_20ep
```

**Windows PowerShell:**

```powershell
python training/evaluate.py `
  --dataset training/datasets/synthetic_v2 `
  --checkpoint training/runs/synthetic_v2_full_20ep/best_model.pt `
  --split test `
  --batch-size 32 `
  --out training/runs/synthetic_v2_full_20ep
```

### Per-Form Accuracy Analysis

```bash
python3 training/analyze_predictions.py \
  --dataset training/datasets/synthetic_v2 \
  --checkpoint training/runs/synthetic_v2_full_20ep/best_model.pt \
  --split test \
  --batch-size 32 \
  --out training/runs/synthetic_v2_full_20ep
```

### Debug Predictions

```bash
python3 training/debug_predictions.py \
  --samples training/runs/synthetic_v2_full_20ep/prediction_samples_test.json \
  --out training/runs/synthetic_v2_full_20ep/debug_report
```

### Training Run Summary

```bash
python3 training/summarize_training_run.py \
  --run training/runs/synthetic_v2_full_20ep
```

### Key Metrics

| Metric | What It Tells You | Target |
|---|---|---|
| **Valid loss** | Generalisation quality — is the model overfitting? | Decreasing or flat |
| **Exact sequence accuracy** | % of equations predicted perfectly | ≥ 70% for export |
| **Character accuracy** | Average per-character match rate | ≥ 85% |
| **Avg edit distance** | Mean Levenshtein distance (lower = better) | ≤ 1.5 |
| **Per-form accuracy** | Accuracy broken down by equation form | Uniform across forms |
| **Prediction samples** | Side-by-side ground truth vs. predicted | Spot-check for patterns |

---

## ONNX Export

ONNX export is prepared but should only be used **after training a good checkpoint** (≥ 70% test exact accuracy).

### Inspect Model Architecture

```bash
python3 training/export_onnx.py --inspect
```

### Smoke Test Export

```bash
python3 training/export_onnx.py --smoke-test
```

### Export a Checkpoint

**Linux / macOS:**

```bash
python3 training/export_onnx.py \
  --checkpoint training/runs/synthetic_v2_full_20ep/best_model.pt \
  --out training/runs/synthetic_v2_full_20ep/model.onnx
```

**Windows PowerShell:**

```powershell
python training/export_onnx.py `
  --checkpoint training/runs/synthetic_v2_full_20ep/best_model.pt `
  --out training/runs/synthetic_v2_full_20ep/model.onnx
```

> ⚠️ **Do not rush to export.** First train, evaluate, and iterate on your model until it reaches acceptable accuracy. ONNX/TFLite export is a deployment step, not a training step.

---

## Troubleshooting

### "python: can't open file"

You're likely running the command from inside `.mathsolvervenv/` or another subdirectory. All Python commands should be run from the **repository root** (`math-copilot/`).

### Virtual Environment Won't Activate (Windows)

PowerShell may block script execution by default. Run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then retry `.\.mathsolvervenv\Scripts\Activate.ps1`.

### `ModuleNotFoundError: No module named 'torch'`

Your virtual environment is not activated, or PyTorch was not installed. Activate the venv and re-run the pip install commands from [Python Environment Setup](#python-environment-setup).

### Dataset Path Missing

If training or evaluation says the dataset path doesn't exist, make sure you've run the `build_synthetic_dataset.py` generator first. See [Generate Synthetic Dataset](#generate-synthetic-dataset).

### Git SSH Permission Denied

If `git clone git@github.com:...` fails, either:
- Use the HTTPS URL instead: `https://github.com/jakuzzy88/math-copilot.git`
- Or add your SSH key to GitHub: https://docs.github.com/en/authentication/connecting-to-github-with-ssh

### Accidentally Committed Large Files

If you accidentally stage generated datasets, checkpoints, or `node_modules/`, unstage them:

```bash
git reset HEAD training/datasets/ training/runs/ .mathsolvervenv/ node_modules/
```

The `.gitignore` should prevent this, but always run `git status` before committing.

### Training Is Slow on CPU

CPU training takes ~6 minutes per epoch with 4000 samples. This is expected. For faster training:
- Use a CUDA GPU (install the CUDA PyTorch wheel).
- Reduce `--epochs` for initial experiments.
- Use `--smoke-test` to verify the pipeline works before committing to a long run.

### Exact Accuracy Stays Low Early in Training

This is **normal for CTC training**. The model learns character-level alignment before it produces perfect sequences. Watch character accuracy and edit distance first — exact accuracy will follow as training progresses. Expect meaningful exact accuracy improvements after 20–30 epochs.

---

## Developer Checklist

Quick-start checklist for a new contributor:

- [ ] Clone the repository
- [ ] Create Python virtual environment (`.mathsolvervenv`)
- [ ] Install Python dependencies (`torch`, `torchvision`, `pillow`, `numpy`, `pytest`)
- [ ] Run `npm install` in `app/`
- [ ] Run Python tests: `python3 -m pytest training/tests/ -v`
- [ ] Run app tests: `cd app && npm test`
- [ ] Open the UI demo: `ui/index.html`
- [ ] Build Android APK: `cd app && npm run build:release-apk`
- [ ] Install on phone: `adb install -r android/app/build/outputs/apk/release/app-release.apk`
- [ ] Generate synthetic dataset: `build_synthetic_dataset.py`
- [ ] Verify dataset: `verify_synthetic_dataset.py`
- [ ] Run smoke training: `train_ctc.py --smoke-test`
- [ ] Run real training: `train_ctc.py --epochs 20`
- [ ] Evaluate checkpoint: `evaluate.py`

---

## License

MIT
