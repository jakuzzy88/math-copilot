# Git Branching Workflow

## Branch Strategy

### `main` — Stable Only

The `main` branch is reserved exclusively for **stable, tested, working code**.

- Never commit directly to `main` with untested changes.
- All work must be done on feature branches and merged into `main` only after all tests pass.

### Feature Branches

Create a new branch for each sprint, feature, fix, or experiment.

#### Branch Naming Convention

| Prefix | Purpose | Example |
|---|---|---|
| `feature/` | New functionality | `feature/onnx-export`, `feature/camera-poc` |
| `fix/` | Bug fixes | `fix/explanation-quality`, `fix/tokenizer-edge-case` |
| `experiment/` | Exploratory work | `experiment/model-v2`, `experiment/augmentation-tuning` |
| `docs/` | Documentation only | `docs/api-reference` |

#### Creating a Branch

```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

## Before Merging into `main`

Every merge into `main` must satisfy these checks:

### 1. Run App Tests

```bash
cd app
npm test
cd ..
```

### 2. Run Training Tests

```bash
python3 -m pytest training/tests/ -v
```

### 3. Inspect Git Status

```bash
git status --short
```

Verify no generated files, datasets, checkpoints, or virtual environments are staged.

### 4. Review Changes

```bash
git diff main --stat
```

## Merge Options

### Squash Merge (Preferred for Feature Branches)

Combines all feature branch commits into a single clean commit on `main`:

```bash
git checkout main
git merge --squash feature/your-feature-name
git commit -m "Add feature: description"
git push origin main
```

### Normal Merge (For Larger Sprint History)

Preserves full commit history when the individual commits are meaningful:

```bash
git checkout main
git merge feature/your-feature-name
git push origin main
```

### After Merging

Delete the merged branch:

```bash
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

## What Must NEVER Be Committed

The `.gitignore` enforces these exclusions, but always verify before committing:

| Excluded | Reason |
|---|---|
| `.mathsolvervenv/`, `.venv/` | Python virtual environments |
| `node_modules/` | Node.js dependencies |
| `training/datasets/` | Generated synthetic datasets |
| `training/runs/` | Training run outputs (logs, metrics) |
| `*.pt`, `*.pth`, `*.onnx`, `*.tflite`, `*.ckpt` | Model checkpoints and exports |
| `__pycache__/`, `.pytest_cache/` | Python cache |
| `dist/`, `build/` | Build outputs |

## What SHOULD Be Committed

- Source code (`app/src/`, `training/*.py`, `ui/`)
- Tests (`app/src/__tests__/`, `training/tests/`)
- Documentation (`docs/`)
- Configuration files (`package.json`, `tsconfig.json`, `jest.config.ts`)
- Small shared metadata/config files
