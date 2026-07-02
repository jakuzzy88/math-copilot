"""
cnn_ctc.py – CNN-CTC model for handwritten equation recognition.

Architecture overview:
    Input:  (B, 1, H, W) greyscale image  (default H=128, W=512)
    
    Convolutional encoder:
        4 conv blocks, each: Conv2d → BatchNorm2d → ReLU → MaxPool2d
        Height is progressively reduced by pooling; width is partially
        preserved to form the time-step axis for CTC.

    Sequence projection:
        Reshape (B, C, H', W') → (B, C*H', W') then permute to (W', B, C*H')
        This treats each column of the feature map as one time step.

    Optional BiLSTM:
        Single-layer bidirectional LSTM over the time axis to add
        contextual modelling without much parameter overhead.

    Linear classifier:
        Projects each time step to vocabulary size (num_classes).

    Output: (T, B, num_classes) – log-probabilities via log_softmax,
            compatible with torch.nn.CTCLoss.

The model is kept intentionally small (~300K params) for eventual
mobile export via ONNX / TFLite.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.models.vocabulary import VOCAB_SIZE


class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_kernel: tuple[int, int] = (2, 2),
        pool_stride: tuple[int, int] = (2, 2),
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class CNNCTC(nn.Module):
    """Small CNN-CTC model for equation recognition.

    Args:
        img_height: Expected input image height (default 128).
        img_width: Expected input image width (default 512).
        num_classes: Vocabulary size including CTC blank (default from vocab).
        use_bilstm: If ``True``, add a single-layer BiLSTM after the CNN.
        lstm_hidden: Hidden size per direction for the BiLSTM.
    """

    def __init__(
        self,
        img_height: int = 128,
        img_width: int = 512,
        num_classes: int = VOCAB_SIZE,
        use_bilstm: bool = True,
        lstm_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.num_classes = num_classes
        self.use_bilstm = use_bilstm

        # ----- Convolutional encoder -----
        # Each block halves H and W via (2,2) pooling.
        #
        # Input : (B, 1, 128, 512)
        # Block1: (B, 32, 64, 256)
        # Block2: (B, 64, 32, 128)
        # Block3: (B, 128, 16, 64)
        # Block4: (B, 128, 8, 32)
        self.cnn = nn.Sequential(
            ConvBlock(1, 32),     # → (B, 32, H/2, W/2)
            ConvBlock(32, 64),    # → (B, 64, H/4, W/4)
            ConvBlock(64, 128),   # → (B, 128, H/8, W/8)
            ConvBlock(128, 128),  # → (B, 128, H/16, W/16)
        )

        # After 4 pools of (2,2): feature_h = H // 16, feature_w = W // 16
        self.feature_h = img_height // 16   # 128 // 16 = 8
        self.feature_w = img_width // 16    # 512 // 16 = 32

        # Sequence dimension = channels * collapsed_height
        seq_features = 128 * self.feature_h  # 128 * 8 = 1024

        # ----- Optional BiLSTM -----
        if use_bilstm:
            self.lstm = nn.LSTM(
                input_size=seq_features,
                hidden_size=lstm_hidden,
                num_layers=1,
                batch_first=False,
                bidirectional=True,
            )
            classifier_in = lstm_hidden * 2  # bidirectional → 2× hidden
        else:
            self.lstm = None
            classifier_in = seq_features

        # ----- Linear classifier -----
        self.fc = nn.Linear(classifier_in, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``(B, 1, H, W)`` input images.

        Returns:
            ``(T, B, num_classes)`` log-probabilities where
            ``T = W // 16`` (number of time steps).
        """
        # CNN feature extraction.
        # (B, 1, H, W) → (B, 128, H/16, W/16)
        conv_out = self.cnn(x)

        B, C, H, W = conv_out.size()
        # Collapse height into channels: (B, C, H, W) → (B, C*H, W)
        conv_out = conv_out.view(B, C * H, W)
        # Permute to (W, B, C*H) – W becomes time axis for CTC.
        seq = conv_out.permute(2, 0, 1)  # (T, B, features)

        # Optional BiLSTM contextual modelling.
        if self.lstm is not None:
            seq, _ = self.lstm(seq)  # (T, B, 2*lstm_hidden)

        # Linear projection to vocabulary.
        logits = self.fc(seq)  # (T, B, num_classes)

        # Log-softmax for CTCLoss (expects log-probabilities).
        return F.log_softmax(logits, dim=2)


def build_model(
    img_height: int = 128,
    img_width: int = 512,
    num_classes: int = VOCAB_SIZE,
    use_bilstm: bool = True,
    lstm_hidden: int = 128,
) -> CNNCTC:
    """Factory function to create a new CNN-CTC model instance."""
    return CNNCTC(
        img_height=img_height,
        img_width=img_width,
        num_classes=num_classes,
        use_bilstm=use_bilstm,
        lstm_hidden=lstm_hidden,
    )
