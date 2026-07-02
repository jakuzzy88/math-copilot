# Architecture – Offline Handwritten Math Copilot

## Overview

The app recognises handwritten linear equations via the device camera,
solves them symbolically (no `eval`), and explains each step to the learner.

Everything runs **offline** – no cloud APIs.

## High-level pipeline

```
Camera Frame
  → Pre-processing (greyscale, crop)
  → CNN-CTC recognition model (ONNX / TFLite)
  → Symbol sequence (e.g. "3x+4=10")
  → Tokeniser → Parser → AST
  → Symbolic Solver  →  Action Log
  → Explanation Engine → Step-by-step text
  → UI Overlay
```

## Module map

| Module | Purpose |
|---|---|
| `parser/tokenizer.ts` | Converts raw symbol string into token stream |
| `parser/parser.ts` | Recursive-descent parser producing an AST |
| `parser/ast.ts` | AST node type definitions |
| `solver/linearSolver.ts` | Solves `ax + b = c` symbolically |
| `solver/simplify.ts` | AST simplification utilities |
| `solver/actionLog.ts` | Records each algebraic step |
| `explanation/` | Converts action logs into natural-language steps |
| `grammar/` | Validates and corrects recognised symbol sequences |
| `diagnostics/` | Runtime diagnostics for recognition accuracy |

## Sprint plan

1. **Sprint 1** – Deterministic Math Core (tokeniser → solver → explanation)
2. Sprint 2 – Synthetic training data generation
3. Sprint 3 – CNN-CTC model training
4. Sprint 4 – On-device inference integration
5. Sprint 5 – Camera pipeline & UI
