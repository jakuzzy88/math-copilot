"""
vocabulary.py – CTC vocabulary for handwritten equation recognition.

Defines the character set, encoding/decoding helpers, and CTC greedy
decode logic used throughout training and evaluation.

Supported characters:
    blank  0 1 2 3 4 5 6 7 8 9 x + - = / ( )

The CTC blank token is always index 0.
"""

from __future__ import annotations

from typing import List, Sequence, Union

import torch

# ---------------------------------------------------------------------------
# Vocabulary definition
# ---------------------------------------------------------------------------

# CTC blank is index 0, followed by the printable symbols.
CHARACTERS: str = "0123456789x+-=/() "
BLANK_TOKEN: str = "<blank>"
BLANK_IDX: int = 0

# Build forward and reverse mappings.
# Index 0 → blank, indices 1..N → printable characters.
_char_list: list[str] = list(CHARACTERS)
_idx_to_char: dict[int, str] = {0: BLANK_TOKEN}
_char_to_idx: dict[str, int] = {}

for _i, _ch in enumerate(_char_list, start=1):
    _idx_to_char[_i] = _ch
    _char_to_idx[_ch] = _i

VOCAB_SIZE: int = len(_idx_to_char)  # blank + printable characters


def char_to_idx(ch: str) -> int:
    """Return the integer index for a single character.

    Raises ``ValueError`` for unsupported characters.
    """
    if ch not in _char_to_idx:
        raise ValueError(
            f"Unsupported character {ch!r}. "
            f"Supported: {CHARACTERS!r}"
        )
    return _char_to_idx[ch]


def idx_to_char(idx: int) -> str:
    """Return the character for a given index (0 = blank)."""
    if idx not in _idx_to_char:
        raise ValueError(
            f"Index {idx} out of vocabulary range [0, {VOCAB_SIZE - 1}]."
        )
    return _idx_to_char[idx]


# ---------------------------------------------------------------------------
# Label encoding / decoding
# ---------------------------------------------------------------------------

def encode_label(label: str) -> list[int]:
    """Encode a label string into a list of integer indices.

    Each character in *label* is mapped through :func:`char_to_idx`.
    The blank token is **not** inserted – CTC training handles alignment.

    Raises ``ValueError`` if any character is unsupported.
    """
    indices: list[int] = []
    for ch in label:
        indices.append(char_to_idx(ch))
    return indices


def decode_indices(indices: Sequence[int]) -> str:
    """Decode a list of integer indices back to a string.

    Blank indices (0) are skipped – use :func:`ctc_greedy_decode`
    for full CTC collapse logic.
    """
    chars: list[str] = []
    for idx in indices:
        idx = int(idx)
        if idx == BLANK_IDX:
            continue
        chars.append(idx_to_char(idx))
    return "".join(chars)


# ---------------------------------------------------------------------------
# CTC greedy decoding
# ---------------------------------------------------------------------------

def ctc_greedy_decode(
    logits_or_indices: Union[torch.Tensor, Sequence[int], List[int]],
) -> str:
    """Perform CTC greedy decoding.

    Accepts either:
    * A 1-D tensor / list of **class indices** (already argmax-ed), or
    * A 2-D tensor of shape ``(time_steps, num_classes)`` containing raw
      logits – argmax is applied along the class dimension.

    The standard CTC collapse rule is applied: repeated identical indices
    are merged, then blanks are removed.
    """
    if isinstance(logits_or_indices, torch.Tensor):
        t = logits_or_indices
        if t.dim() == 2:
            # (T, C) → take argmax over class axis
            indices = t.argmax(dim=-1).tolist()
        elif t.dim() == 1:
            indices = t.tolist()
        else:
            raise ValueError(
                f"Expected 1-D or 2-D tensor, got {t.dim()}-D."
            )
    else:
        indices = list(logits_or_indices)

    # CTC collapse: merge consecutive duplicates, then drop blanks.
    collapsed: list[int] = []
    prev = -1
    for idx in indices:
        idx = int(idx)
        if idx != prev:
            collapsed.append(idx)
        prev = idx

    # Remove blanks.
    result_chars: list[str] = []
    for idx in collapsed:
        if idx == BLANK_IDX:
            continue
        result_chars.append(idx_to_char(idx))
    return "".join(result_chars)
