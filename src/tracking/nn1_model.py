"""
SADHA — NN1 CNN Architecture: BeaconHeatmapNet

Lightweight encoder-decoder CNN that produces a spatial probability heatmap
from a monochrome sensor frame.  Peaks in the heatmap correspond to beacon
centroid locations.

Architecture choice (document in technical report):
  Heatmap regression was chosen over direct coordinate regression because it
  naturally supports 0-to-N detections per frame without a fixed output
  dimension, handles multi-beacon scenarios through simple peak extraction,
  and provides spatially interpretable confidence via peak intensity.

  The encoder is deliberately shallow (4 layers, ~85 K parameters) to clear
  the ≥20 FPS budget with headroom on CPU.  Deeper variants can be swapped
  in behind the same output contract if GPU is available.

Input:  (B, 1, H, W)  monochrome uint8 normalised to [0, 1]
Output: (B, 1, H/4, W/4)  heatmap in [0, 1] at quarter resolution
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeaconHeatmapNet(nn.Module):
    """Compact encoder-decoder for beacon centroid heatmap regression."""

    def __init__(self):
        super().__init__()

        # ---- Encoder (downsampling path) ----
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # ---- Decoder (upsampling path) ----
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # ---- Heatmap head ----
        self.head = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) normalised monochrome frame
        Returns:
            heatmap: (B, 1, H/4, W/4) beacon probability map in [0, 1]
        """
        # Encoder
        e1 = self.enc1(x)    # (B, 16, H/2, W/2)
        e2 = self.enc2(e1)   # (B, 32, H/4, W/4)
        e3 = self.enc3(e2)   # (B, 64, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(e3)  # (B, 64, H/8, W/8)

        # Decoder with skip connection
        d1 = self.dec1(b)    # (B, 32, H/4, W/4)
        d1 = d1 + e2         # Skip connection from enc2
        d2 = self.dec2(d1)   # (B, 16, H/4, W/4)

        # Heatmap output
        heatmap = self.head(d2)  # (B, 1, H/4, W/4)
        return heatmap

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(device: torch.device = None) -> BeaconHeatmapNet:
    """Factory: create and initialise a BeaconHeatmapNet on the given device."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BeaconHeatmapNet().to(device)
    _init_weights(model)
    return model


def _init_weights(model: nn.Module):
    """Kaiming initialisation for Conv layers, constant for BatchNorm."""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
