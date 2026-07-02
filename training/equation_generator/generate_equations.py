"""
generate_equations.py – Synthetic equation generator

Generates random single-variable linear equations matching the supported
MVP grammar. Every generated equation uses only the supported symbol set:
    0 1 2 3 4 5 6 7 8 9 x + - = / ( )

Supported canonical forms (no spaces):
    x+a=b       x-a=b       ax=b
    ax+b=c      ax-b=c      x/a=b
    ax/b=c      a(x+b)=c    a(x-b)=c

All generated equations use integers only, avoid division by zero,
and prefer equations with integer solutions.

Output: list of canonical label strings.

Sprint 2 implementation.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional


# Supported symbols for label validation
SUPPORTED_CHARS = set("0123456789x+-=/() ")


def _randint_nonzero(lo: int, hi: int, rng: random.Random) -> int:
    """Return a random non-zero integer in [lo, hi]."""
    while True:
        v = rng.randint(lo, hi)
        if v != 0:
            return v


def _format_int(n: int) -> str:
    """Format an integer, using parenthesised negative if needed."""
    return str(n)


def generate_form_x_plus_a_eq_b(rng: random.Random) -> str:
    """x+a=b  or  x-a=b  (a > 0 always, sign baked into form)."""
    a = _randint_nonzero(1, 20, rng)
    # Choose x so solution is integer
    x_val = rng.randint(-20, 20)
    op = rng.choice(["+", "-"])
    if op == "+":
        b = x_val + a
        return f"x+{a}={b}"
    else:
        b = x_val - a
        return f"x-{a}={b}"


def generate_form_ax_eq_b(rng: random.Random) -> str:
    """ax=b  where b = a*x_val so x_val is integer."""
    a = _randint_nonzero(2, 12, rng)
    x_val = rng.randint(-10, 10)
    b = a * x_val
    return f"{a}x={b}"


def generate_form_ax_plus_b_eq_c(rng: random.Random) -> str:
    """ax+b=c  or  ax-b=c  with integer solution."""
    a = _randint_nonzero(2, 9, rng)
    b = _randint_nonzero(1, 15, rng)
    x_val = rng.randint(-10, 10)
    op = rng.choice(["+", "-"])
    if op == "+":
        c = a * x_val + b
        return f"{a}x+{b}={c}"
    else:
        c = a * x_val - b
        return f"{a}x-{b}={c}"


def generate_form_x_div_a_eq_b(rng: random.Random) -> str:
    """x/a=b  with integer solution x = a*b."""
    a = _randint_nonzero(2, 10, rng)
    b = rng.randint(-10, 10)
    # x = a*b, equation: x/a = b
    return f"x/{a}={b}"


def generate_form_ax_div_b_eq_c(rng: random.Random) -> str:
    """ax/b=c  with integer solution.  a*x must be divisible by b."""
    b = _randint_nonzero(2, 8, rng)
    a = _randint_nonzero(2, 9, rng)
    x_val = rng.randint(-8, 8)
    # ensure a*x_val divisible by b for clean c
    # c = a*x_val / b  →  pick c, then x_val = b*c/a  ... simpler: pick c directly
    c = rng.randint(-10, 10)
    # x_val = b*c / a  — need integer
    # Instead: pick x_val such that a*x_val % b == 0
    # Easiest: x_val = b * k for some k, then c = a*k
    k = rng.randint(-5, 5)
    x_val = b * k
    c = a * k
    return f"{a}x/{b}={c}"


def generate_form_a_paren_x_pm_b_eq_c(rng: random.Random) -> str:
    """a(x+b)=c  or  a(x-b)=c  with integer solution."""
    a = _randint_nonzero(2, 8, rng)
    b = _randint_nonzero(1, 10, rng)
    x_val = rng.randint(-8, 8)
    op = rng.choice(["+", "-"])
    if op == "+":
        c = a * (x_val + b)
        return f"{a}(x+{b})={c}"
    else:
        c = a * (x_val - b)
        return f"{a}(x-{b})={c}"


# Registry of all form generators with relative weights
_FORM_GENERATORS = [
    (generate_form_x_plus_a_eq_b, 2),
    (generate_form_ax_eq_b, 2),
    (generate_form_ax_plus_b_eq_c, 3),
    (generate_form_x_div_a_eq_b, 1),
    (generate_form_ax_div_b_eq_c, 1),
    (generate_form_a_paren_x_pm_b_eq_c, 2),
]


def validate_label(label: str) -> bool:
    """Check that a label only contains supported characters and has exactly one '='."""
    allowed = set("0123456789x+-=/() ")
    if not all(c in allowed for c in label):
        return False
    if label.count("=") != 1:
        return False
    # Must not be empty on either side of =
    lhs, rhs = label.split("=")
    if not lhs.strip() or not rhs.strip():
        return False
    return True


def generate_equations(
    count: int = 500,
    seed: Optional[int] = None,
    deduplicate: bool = True,
) -> List[str]:
    """
    Generate `count` valid single-variable linear equation labels.

    Args:
        count: Number of equations to generate.
        seed: Random seed for reproducibility.
        deduplicate: If True, remove duplicate equations (may return fewer).

    Returns:
        List of canonical equation label strings.
    """
    rng = random.Random(seed)

    generators = [g for g, _ in _FORM_GENERATORS]
    weights = [w for _, w in _FORM_GENERATORS]

    equations: List[str] = []
    attempts = 0
    max_attempts = count * 5  # prevent infinite loop

    seen: set[str] = set()

    while len(equations) < count and attempts < max_attempts:
        attempts += 1
        gen = rng.choices(generators, weights=weights, k=1)[0]
        eq = gen(rng)

        if not validate_label(eq):
            continue

        if deduplicate:
            if eq in seen:
                continue
            seen.add(eq)

        equations.append(eq)

    return equations


def save_equations_json(equations: List[str], output_path: str | Path) -> None:
    """Save equations to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "count": len(equations),
        "equations": equations,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    # Generate default 500 equations and save to shared/
    eqs = generate_equations(count=500, seed=42)
    out_path = Path(__file__).resolve().parent.parent.parent / "shared" / "test_equations_generated.json"
    save_equations_json(eqs, out_path)
    print(f"Generated {len(eqs)} equations → {out_path}")
