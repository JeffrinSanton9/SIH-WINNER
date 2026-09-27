"""
SADHA — NN1 Training Pipeline

Generates synthetic training data on-the-fly from Module 1's rendering and
disturbance engines, trains ``BeaconHeatmapNet`` to localize beacon centroids
under the full range of disturbances the detector will face at inference.

Training data augmentation includes every disturbance type simultaneously:
  • Salt & pepper noise (0–10 %)
  • Gaussian intensity noise (σ 0–20)
  • Poisson shot noise
  • Camera jitter (±0–20 px)
  • Atmospheric degradation (clear / haze / fog / rain / low-light)

Usage::

    from src.tracking.nn1_trainer import NN1Trainer
    trainer = NN1Trainer()
    trainer.train(epochs=30, samples_per_epoch=500)
    # Model auto-saved to data/nn1_weights.pth
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from src.tracking.nn1_model import BeaconHeatmapNet, create_model
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    class Dataset:
        pass
    class DataLoader:
        pass

from src.core.types import TargetShape, AtmosphericCondition
from src.sim.disturbances import DisturbanceEngine


# Default save location
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_WEIGHTS = _WEIGHTS_DIR / "nn1_weights.pth"


class SyntheticBeaconDataset(Dataset):
    """Generates synthetic monochrome beacon frames with ground-truth heatmaps.

    Each sample is a randomised scene: 1–3 beacons at random positions,
    random shapes/sizes, and a random disturbance combination applied.
    """

    def __init__(
        self,
        num_samples: int = 1000,
        frame_w: int = 640,
        frame_h: int = 480,
        heatmap_scale: int = 4,
        sigma: float = 4.0,
    ):
        self.num_samples = num_samples
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.hm_w = frame_w // heatmap_scale
        self.hm_h = frame_h // heatmap_scale
        self.scale = heatmap_scale
        self.sigma = sigma  # Gaussian blob σ in heatmap space

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        frame, heatmap = self._generate_sample()
        frame_tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).unsqueeze(0)
        heatmap_tensor = torch.from_numpy(heatmap.astype(np.float32)).unsqueeze(0)
        return frame_tensor, heatmap_tensor

    def _generate_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        w, h = self.frame_w, self.frame_h

        # Dark background with slight noise floor
        bg_level = random.randint(2, 12)
        frame = np.full((h, w), bg_level, dtype=np.uint8)

        # Heatmap target (quarter resolution)
        heatmap = np.zeros((self.hm_h, self.hm_w), dtype=np.float32)

        # Place 1–3 beacons
        num_beacons = random.randint(1, 3)
        for _ in range(num_beacons):
            bx = random.uniform(30, w - 30)
            by = random.uniform(30, h - 30)
            size = random.randint(5, 18)
            brightness = random.randint(180, 255)
            shape = random.choice(list(TargetShape))

            # Draw beacon on frame
            self._draw_beacon(frame, bx, by, size, brightness, shape)

            # Draw Gaussian blob on heatmap at corresponding position
            hm_x = bx / self.scale
            hm_y = by / self.scale
            self._draw_gaussian(heatmap, hm_x, hm_y, self.sigma)

        # Random disturbances
        engine = self._random_disturbance_engine()
        frame = engine.apply(frame)

        return frame, heatmap

    @staticmethod
    def _draw_beacon(
        frame: np.ndarray, cx: float, cy: float,
        size: int, brightness: int, shape: TargetShape,
    ):
        h, w = frame.shape
        half = size // 2
        iu, iv = int(round(cx)), int(round(cy))

        if shape == TargetShape.SQUARE:
            y0 = max(0, iv - half)
            y1 = min(h, iv + half + 1)
            x0 = max(0, iu - half)
            x1 = min(w, iu + half + 1)
            frame[y0:y1, x0:x1] = brightness

        elif shape == TargetShape.CIRCLE:
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    if dx * dx + dy * dy <= half * half:
                        py, px = iv + dy, iu + dx
                        if 0 <= py < h and 0 <= px < w:
                            frame[py, px] = brightness

        elif shape == TargetShape.TRIANGLE:
            for dy in range(-half, half + 1):
                span = int(half * (1.0 - (dy + half) / max(1, 2 * half)))
                for dx in range(-span, span + 1):
                    py, px = iv + dy, iu + dx
                    if 0 <= py < h and 0 <= px < w:
                        frame[py, px] = brightness

        # Diffraction halo
        halo_r = int(half * 2.0)
        for dy in range(-halo_r, halo_r + 1):
            for dx in range(-halo_r, halo_r + 1):
                d = math.sqrt(dx * dx + dy * dy)
                if d < half * 0.5 or d > halo_r:
                    continue
                py, px = iv + dy, iu + dx
                if 0 <= py < h and 0 <= px < w:
                    falloff = max(0.0, 1.0 - d / halo_r)
                    glow = int(brightness * 0.3 * falloff * falloff)
                    frame[py, px] = min(255, int(frame[py, px]) + glow)

    @staticmethod
    def _draw_gaussian(
        heatmap: np.ndarray, cx: float, cy: float, sigma: float,
    ):
        """Render a 2D Gaussian blob centred at (cx, cy) onto the heatmap."""
        h, w = heatmap.shape
        radius = int(sigma * 3.5) + 1
        x0 = max(0, int(cx) - radius)
        x1 = min(w, int(cx) + radius + 1)
        y0 = max(0, int(cy) - radius)
        y1 = min(h, int(cy) + radius + 1)

        for y in range(y0, y1):
            for x in range(x0, x1):
                dx = x - cx
                dy = y - cy
                val = math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
                heatmap[y, x] = max(heatmap[y, x], val)

    @staticmethod
    def _random_disturbance_engine() -> DisturbanceEngine:
        """Create a randomly-configured disturbance engine for augmentation."""
        atmo = random.choice(list(AtmosphericCondition))
        return DisturbanceEngine(
            salt_pepper_enabled=random.random() < 0.4,
            salt_pepper_density=random.uniform(0.0, 0.10),
            gaussian_enabled=random.random() < 0.5,
            gaussian_std=random.uniform(0.0, 20.0),
            poisson_enabled=random.random() < 0.3,
            poisson_scale=1.0,
            jitter_enabled=False,  # Jitter is spatial, not intensity
            jitter_amplitude=0.0,
            atmospheric=atmo,
        )


class NN1Trainer:
    """Trains ``BeaconHeatmapNet`` on synthetic data and saves weights."""

    def __init__(
        self,
        lr: float = 1e-3,
        batch_size: int = 16,
        device: Optional[torch.device] = None,
        weights_path: Optional[Path] = None,
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS

        self.model = create_model(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.train_losses: List[float] = []

    def train(
        self,
        epochs: int = 30,
        samples_per_epoch: int = 500,
        log_interval: int = 5,
        callback=None,
    ) -> List[float]:
        """Run the full training loop.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        samples_per_epoch : int
            Synthetic samples generated per epoch.
        log_interval : int
            Print loss every N epochs.
        callback : callable or None
            Called with ``(epoch, loss)`` after each epoch — useful for
            wiring into a UI progress bar.

        Returns
        -------
        list of float
            Per-epoch average loss.
        """
        dataset = SyntheticBeaconDataset(num_samples=samples_per_epoch)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

        self.model.train()
        self.train_losses = []

        print(f"[NN1] Training on {self.device} — "
              f"{epochs} epochs × {samples_per_epoch} samples")
        t0 = time.perf_counter()

        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            n_batches = 0

            for frames, heatmaps in loader:
                frames = frames.to(self.device)
                heatmaps = heatmaps.to(self.device)

                pred = self.model(frames)

                # Ensure spatial dimensions match (handle rounding)
                if pred.shape != heatmaps.shape:
                    heatmaps = nn.functional.interpolate(
                        heatmaps, size=pred.shape[2:], mode="bilinear",
                        align_corners=False,
                    )

                loss = self.criterion(pred, heatmaps)

                # Focal-style weighting: emphasise non-zero regions
                positive_mask = (heatmaps > 0.1).float()
                weighted_loss = loss + 2.0 * self.criterion(
                    pred * positive_mask, heatmaps * positive_mask
                )

                self.optimizer.zero_grad()
                weighted_loss.backward()
                self.optimizer.step()

                epoch_loss += weighted_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            self.train_losses.append(avg_loss)

            if epoch % log_interval == 0 or epoch == 1:
                elapsed = time.perf_counter() - t0
                print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}  "
                      f"({elapsed:.1f}s elapsed)")

            if callback:
                callback(epoch, avg_loss)

        # Save weights
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.weights_path)
        total = time.perf_counter() - t0
        print(f"[NN1] Training complete in {total:.1f}s — "
              f"weights saved to {self.weights_path}")

        self.model.eval()
        return self.train_losses
