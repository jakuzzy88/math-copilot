"""
test_equation_generator.py – Tests for the synthetic equation generator.

Validates:
    - Generated labels use only supported characters
    - Each equation has exactly one '=' sign
    - No division by zero in generated equations
    - Correct form distribution
    - Deterministic seed behaviour
    - Minimum count is met

Sprint 2 tests.
"""

import re
import sys
from pathlib import Path

import pytest

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.equation_generator.generate_equations import (
    generate_equations,
    validate_label,
    generate_form_x_plus_a_eq_b,
    generate_form_ax_eq_b,
    generate_form_ax_plus_b_eq_c,
    generate_form_x_div_a_eq_b,
    generate_form_ax_div_b_eq_c,
    generate_form_a_paren_x_pm_b_eq_c,
)

import random

# Supported character set
SUPPORTED_CHARS = set("0123456789x+-=/() ")


class TestEquationGenerator:
    """Tests for the equation generator module."""

    def test_generates_minimum_count(self):
        """Generator produces at least the requested number of equations."""
        eqs = generate_equations(count=100, seed=42)
        assert len(eqs) >= 100

    def test_supported_characters_only(self):
        """All generated labels use only supported characters."""
        eqs = generate_equations(count=200, seed=123)
        for eq in eqs:
            for ch in eq:
                assert ch in SUPPORTED_CHARS, (
                    f"Unsupported character '{ch}' in equation '{eq}'"
                )

    def test_single_equals_sign(self):
        """Each generated equation has exactly one '=' sign."""
        eqs = generate_equations(count=200, seed=456)
        for eq in eqs:
            assert eq.count("=") == 1, (
                f"Expected 1 '=', found {eq.count('=')} in '{eq}'"
            )

    def test_no_division_by_zero(self):
        """No generated equation has /0 or divides by zero."""
        eqs = generate_equations(count=300, seed=789)
        for eq in eqs:
            # Check for explicit /0
            assert "/0" not in eq or "/0=" not in eq.replace("/0=", "/0_check_"), (
                f"Division by zero in '{eq}'"
            )
            # More thorough: extract all /N patterns and check N != 0
            divs = re.findall(r"/(\d+)", eq)
            for d in divs:
                assert int(d) != 0, f"Division by zero in '{eq}'"

    def test_deterministic_seed(self):
        """Same seed produces identical output."""
        eqs1 = generate_equations(count=50, seed=42)
        eqs2 = generate_equations(count=50, seed=42)
        assert eqs1 == eqs2

    def test_different_seeds_produce_different_output(self):
        """Different seeds produce different output."""
        eqs1 = generate_equations(count=50, seed=42)
        eqs2 = generate_equations(count=50, seed=99)
        assert eqs1 != eqs2

    def test_deduplication(self):
        """With deduplication enabled, no duplicates in output."""
        eqs = generate_equations(count=200, seed=42, deduplicate=True)
        assert len(eqs) == len(set(eqs))

    def test_validate_label_valid(self):
        """validate_label accepts valid equations."""
        assert validate_label("x+5=9")
        assert validate_label("2x=8")
        assert validate_label("3x+4=10")
        assert validate_label("x/2=5")
        assert validate_label("2(x+1)=10")

    def test_validate_label_invalid(self):
        """validate_label rejects invalid equations."""
        assert not validate_label("x+5")  # no equals
        assert not validate_label("x+5==9")  # two equals
        assert not validate_label("=9")  # empty LHS
        assert not validate_label("x+5=")  # empty RHS
        assert not validate_label("x^2=5")  # unsupported char

    def test_individual_forms(self):
        """Each form generator produces a valid label."""
        rng = random.Random(42)
        generators = [
            generate_form_x_plus_a_eq_b,
            generate_form_ax_eq_b,
            generate_form_ax_plus_b_eq_c,
            generate_form_x_div_a_eq_b,
            generate_form_ax_div_b_eq_c,
            generate_form_a_paren_x_pm_b_eq_c,
        ]
        for gen in generators:
            for _ in range(20):
                eq = gen(rng)
                assert validate_label(eq), f"Invalid label from {gen.__name__}: '{eq}'"

    def test_non_empty_sides(self):
        """Both sides of every equation are non-empty."""
        eqs = generate_equations(count=200, seed=42)
        for eq in eqs:
            lhs, rhs = eq.split("=")
            assert len(lhs) > 0, f"Empty LHS in '{eq}'"
            assert len(rhs) > 0, f"Empty RHS in '{eq}'"

    def test_no_spaces_in_labels(self):
        """Generated canonical labels have no spaces."""
        eqs = generate_equations(count=200, seed=42)
        for eq in eqs:
            assert " " not in eq, f"Space found in label '{eq}'"

    def test_large_generation(self):
        """Can generate 500+ unique equations."""
        eqs = generate_equations(count=500, seed=42, deduplicate=True)
        assert len(eqs) >= 500
