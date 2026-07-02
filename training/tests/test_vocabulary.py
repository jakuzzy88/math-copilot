"""
Tests for training.models.vocabulary
"""

from __future__ import annotations

import pytest
import torch

import sys
from pathlib import Path

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.models.vocabulary import (
    BLANK_IDX,
    BLANK_TOKEN,
    CHARACTERS,
    VOCAB_SIZE,
    char_to_idx,
    ctc_greedy_decode,
    decode_indices,
    encode_label,
    idx_to_char,
)


class TestCharToIdx:
    """Tests for char_to_idx mapping."""

    def test_all_supported_chars_have_indices(self):
        """Every character in CHARACTERS must map to a valid index."""
        for ch in CHARACTERS:
            idx = char_to_idx(ch)
            assert isinstance(idx, int)
            assert idx >= 1  # 0 is reserved for blank

    def test_blank_is_not_in_char_to_idx(self):
        """The blank token is not a printable character."""
        with pytest.raises(ValueError):
            char_to_idx(BLANK_TOKEN)

    def test_unsupported_char_raises(self):
        """Characters outside the vocabulary must raise ValueError."""
        for ch in ["@", "#", "!", "a", "b", "π", "²"]:
            with pytest.raises(ValueError, match="Unsupported character"):
                char_to_idx(ch)

    def test_indices_are_unique(self):
        """No two characters should share the same index."""
        indices = [char_to_idx(ch) for ch in CHARACTERS]
        assert len(indices) == len(set(indices))


class TestIdxToChar:
    """Tests for idx_to_char mapping."""

    def test_blank_idx_returns_blank_token(self):
        assert idx_to_char(BLANK_IDX) == BLANK_TOKEN

    def test_roundtrip(self):
        """char_to_idx → idx_to_char should roundtrip for all chars."""
        for ch in CHARACTERS:
            assert idx_to_char(char_to_idx(ch)) == ch

    def test_invalid_idx_raises(self):
        with pytest.raises(ValueError):
            idx_to_char(9999)


class TestVocabSize:
    """Test vocabulary size constant."""

    def test_vocab_size_includes_blank(self):
        # blank + all printable characters
        assert VOCAB_SIZE == len(CHARACTERS) + 1


class TestEncodeLabel:
    """Tests for encode_label."""

    def test_simple_equation(self):
        encoded = encode_label("3x+4=10")
        assert len(encoded) == 7
        # Verify each character maps correctly.
        for idx, ch in zip(encoded, "3x+4=10"):
            assert idx == char_to_idx(ch)

    def test_equation_with_parens(self):
        encoded = encode_label("2(x+1)=10")
        assert len(encoded) == 9

    def test_empty_string(self):
        assert encode_label("") == []

    def test_unsupported_char_raises(self):
        with pytest.raises(ValueError):
            encode_label("3x+4=10#")

    def test_all_supported_chars(self):
        """Encoding all supported chars should produce VOCAB_SIZE-1 indices."""
        encoded = encode_label(CHARACTERS)
        assert len(encoded) == len(CHARACTERS)


class TestDecodeIndices:
    """Tests for decode_indices."""

    def test_roundtrip(self):
        label = "5x-2=18"
        encoded = encode_label(label)
        decoded = decode_indices(encoded)
        assert decoded == label

    def test_blanks_are_skipped(self):
        encoded = encode_label("x+1")
        # Insert blanks.
        with_blanks = [BLANK_IDX, encoded[0], BLANK_IDX, encoded[1], encoded[2], BLANK_IDX]
        decoded = decode_indices(with_blanks)
        assert decoded == "x+1"

    def test_empty(self):
        assert decode_indices([]) == ""


class TestCTCGreedyDecode:
    """Tests for ctc_greedy_decode."""

    def test_from_index_list(self):
        """Repeated indices are collapsed, blanks removed."""
        # Simulate CTC output: 0=blank, then indices for "x+1"
        x_idx = char_to_idx("x")
        plus_idx = char_to_idx("+")
        one_idx = char_to_idx("1")
        indices = [0, x_idx, x_idx, 0, plus_idx, 0, 0, one_idx, one_idx, 0]
        result = ctc_greedy_decode(indices)
        assert result == "x+1"

    def test_from_1d_tensor(self):
        x_idx = char_to_idx("x")
        indices = torch.tensor([0, x_idx, x_idx, 0, x_idx])
        result = ctc_greedy_decode(indices)
        # x_idx, x_idx → "x", then blank, then x_idx → "x" → "xx"
        assert result == "xx"

    def test_from_2d_logits(self):
        """2-D tensor (T, C) should argmax then collapse."""
        T = 5
        C = VOCAB_SIZE
        logits = torch.zeros(T, C)
        # Set high value at specific positions.
        x_idx = char_to_idx("x")
        logits[0, BLANK_IDX] = 10.0
        logits[1, x_idx] = 10.0
        logits[2, x_idx] = 10.0
        logits[3, BLANK_IDX] = 10.0
        logits[4, x_idx] = 10.0
        result = ctc_greedy_decode(logits)
        assert result == "xx"

    def test_all_blanks(self):
        result = ctc_greedy_decode([0, 0, 0, 0])
        assert result == ""

    def test_empty(self):
        result = ctc_greedy_decode([])
        assert result == ""
