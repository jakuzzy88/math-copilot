# MVP Scope – Sprint 1: Deterministic Math Core

## Goal

Implement and fully test the **deterministic** path from a raw equation string
to a solved result with step-by-step explanations.

No camera, no AI model, no cloud.

## In scope

- Tokeniser for digits, `x`, `+`, `-`, `*`, `/`, `=`, `(`, `)`
- Implicit-multiplication normalisation (`2x` → `2*x`)
- Recursive-descent parser producing an AST
- AST types: `NumberLiteral`, `Variable`, `BinaryOp`, `UnaryMinus`, `Equation`
- Symbolic linear solver (`ax + b = c → x = (c - b) / a`)
- Step-by-step action log
- Template-based explanation engine
- Grammar validator (supported symbols & structure)
- Comprehensive Jest test suite

## Supported equation forms

| Equation | Expected solution |
|---|---|
| `x+5=9` | `x = 4` |
| `2x=8` | `x = 4` |
| `3x+4=10` | `x = 2` |
| `5x-2=18` | `x = 4` |
| `x/2=5` | `x = 10` |
| `3x/4=6` | `x = 8` |
| `2(x+1)=10` | `x = 4` |

## Out of scope (Sprint 1)

- Camera integration
- AI/ML model inference
- Quadratic or higher-order equations
- Multiple variables
- Cloud connectivity
