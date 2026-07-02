"""
Tests for training.models.cnn_ctc
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Ensure project root is on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.models.cnn_ctc import CNNCTC, build_model
from training.models.vocabulary import VOCAB_SIZE, BLANK_IDX, encode_label


class TestCNNCTCModel:
    """Tests for the CNN-CTC model architecture."""

    def test_build_model_returns_cnnctc(self):
        model = build_model()
        assert isinstance(model, CNNCTC)

    def test_forward_output_shape(self):
        """Output should be (T, B, num_classes) where T = W // 16."""
        model = build_model(img_height=128, img_width=512)
        B = 4
        x = torch.randn(B, 1, 128, 512)
        out = model(x)

        expected_T = 512 // 16  # = 32
        assert out.shape == (expected_T, B, VOCAB_SIZE)

    def test_forward_output_is_log_prob(self):
        """Output should be log-probabilities (sum to ~1 after exp)."""
        model = build_model(img_height=128, img_width=512)
        x = torch.randn(2, 1, 128, 512)
        out = model(x)
        # exp(log_softmax) sums to 1 along class axis.
        probs = out.exp().sum(dim=2)
        assert torch.allclose(probs, torch.ones_like(probs), atol=1e-5)

    def test_forward_different_sizes(self):
        """Model should handle different H/W configurations."""
        for H, W in [(64, 256), (128, 512), (128, 1024)]:
            model = build_model(img_height=H, img_width=W)
            x = torch.randn(2, 1, H, W)
            out = model(x)
            expected_T = W // 16
            assert out.shape == (expected_T, 2, VOCAB_SIZE)

    def test_without_bilstm(self):
        """Model should work without BiLSTM."""
        model = build_model(use_bilstm=False)
        x = torch.randn(2, 1, 128, 512)
        out = model(x)
        assert out.shape == (32, 2, VOCAB_SIZE)

    def test_model_param_count_reasonable(self):
        """Model should be reasonably small for mobile export."""
        model = build_model()
        total = sum(p.numel() for p in model.parameters())
        # Should be under 5M parameters for mobile viability.
        assert total < 5_000_000
        # And should have some meaningful number of parameters.
        assert total > 10_000


class TestCTCLossIntegration:
    """Test that the model output works with CTCLoss."""

    def test_ctc_loss_computes(self):
        """CTCLoss should compute a finite scalar on model output."""
        model = build_model(img_height=128, img_width=512)
        model.train()

        B = 4
        x = torch.randn(B, 1, 128, 512)
        log_probs = model(x)  # (T, B, C)
        T = log_probs.size(0)

        # Create dummy labels.
        label_strs = ["x+1=2", "3x=6", "x-1=0", "2x+3=7"]
        all_targets = []
        target_lengths = []
        for s in label_strs:
            enc = encode_label(s)
            all_targets.extend(enc)
            target_lengths.append(len(enc))

        targets = torch.tensor(all_targets, dtype=torch.int32)
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32)
        input_lengths = torch.full((B,), T, dtype=torch.int32)

        criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        assert torch.isfinite(loss)
        assert loss.item() > 0

    def test_backward_pass_works(self):
        """Backward pass should complete without error."""
        model = build_model(img_height=128, img_width=512)
        model.train()

        x = torch.randn(2, 1, 128, 512)
        log_probs = model(x)
        T, B, C = log_probs.size()

        targets = torch.tensor(encode_label("x+1=2"), dtype=torch.int32)
        target_lengths = torch.tensor([len("x+1=2")], dtype=torch.int32)
        # Duplicate for batch.
        targets = targets.repeat(B)
        target_lengths = target_lengths.repeat(B)
        input_lengths = torch.full((B,), T, dtype=torch.int32)

        criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        loss.backward()

        # Check gradients exist.
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
