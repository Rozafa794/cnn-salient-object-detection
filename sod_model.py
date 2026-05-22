"""
sod_model.py
------------
Defines the CNN Encoder-Decoder architecture for Salient Object Detection.

Architecture overview:
- Encoder: 4 Conv2D blocks (Conv -> BatchNorm -> ReLU -> MaxPool)
           Each block doubles the number of channels and halves the spatial size.
- Bottleneck: Conv2D + BatchNorm + ReLU + Dropout
- Decoder: 4 ConvTranspose2D blocks (upsampling) with BatchNorm + ReLU
- Output : 1x1 Conv2D + Sigmoid -> single-channel saliency mask in [0, 1]

Improvements built into this model (Requirement #8):
  1) BatchNorm after every Conv layer  -> faster, more stable training
  2) Dropout in the bottleneck         -> reduces overfitting

Input  shape : (B, 3, 128, 128)
Output shape : (B, 1, 128, 128)
"""

import torch
import torch.nn as nn


# --------------------------- Reusable conv blocks --------------------------- #
class ConvBlock(nn.Module):
    """A single encoder block: Conv -> BN -> ReLU (no pooling here)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """A single decoder block: ConvTranspose -> BN -> ReLU. Doubles spatial size."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=2, stride=2,   # stride=2 -> upsamples by 2x
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# ------------------------------- Full model ------------------------------- #
class SODModel(nn.Module):
    """
    CNN Encoder-Decoder for Salient Object Detection.

    Spatial sizes assuming a 128x128 input:
        Input          : 128 x 128
        After enc1+pool:  64 x 64    (channels:  32)
        After enc2+pool:  32 x 32    (channels:  64)
        After enc3+pool:  16 x 16    (channels: 128)
        After enc4+pool:   8 x 8     (channels: 256)
        Bottleneck      :   8 x 8     (channels: 256)
        After up1       :  16 x 16    (channels: 128)
        After up2       :  32 x 32    (channels:  64)
        After up3       :  64 x 64    (channels:  32)
        After up4       : 128 x 128   (channels:  16)
        Final 1x1 conv  : 128 x 128 x 1 -> Sigmoid
    """

    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.3):
        super().__init__()

        # ----------- Encoder ----------- #
        self.enc1 = ConvBlock(in_channels, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.enc4 = ConvBlock(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # halves spatial size

        # ----------- Bottleneck (with Dropout) ----------- #
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),     # <-- regularization (improvement)
        )

        # ----------- Decoder ----------- #
        self.up1 = UpBlock(256, 128)
        self.up2 = UpBlock(128, 64)
        self.up3 = UpBlock(64, 32)
        self.up4 = UpBlock(32, 16)

        # ----------- Output layer ----------- #
        # 1x1 conv to produce a 1-channel saliency map, then sigmoid -> [0, 1]
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.pool(self.enc1(x))   # 128 -> 64
        x = self.pool(self.enc2(x))   # 64  -> 32
        x = self.pool(self.enc3(x))   # 32  -> 16
        x = self.pool(self.enc4(x))   # 16  -> 8

        # Bottleneck
        x = self.bottleneck(x)        # 8 -> 8

        # Decoder
        x = self.up1(x)               # 8  -> 16
        x = self.up2(x)               # 16 -> 32
        x = self.up3(x)               # 32 -> 64
        x = self.up4(x)               # 64 -> 128

        # Output
        x = self.out_conv(x)
        x = self.sigmoid(x)
        return x


# ------------------------------- Quick test ------------------------------- #
if __name__ == "__main__":
    # Sanity check: forward a random tensor and inspect output shape.
    model = SODModel()
    dummy = torch.randn(2, 3, 128, 128)        # batch of 2 images
    with torch.no_grad():
        out = model(dummy)
    print("Input :", dummy.shape)
    print("Output:", out.shape)                # expect (2, 1, 128, 128)
    print("Output range:", out.min().item(), out.max().item())  # in [0, 1]

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")