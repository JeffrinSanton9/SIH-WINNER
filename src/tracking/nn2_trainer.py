"""
SADHA — NN2 Trainer: Synthetic Trajectory Generation + Training Pipeline

Generates trajectories across all motion types NN2 must handle:
  - Circular (constant angular velocity)
  - Figure-8 (Lissajous curve)
  - Spiral (expanding/contracting radius)
  - Ballistic (quadratic arc under gravity)
  - Linear (baseline — Kalman already handles this, but included for completeness)

Each trajectory sample is a window of 9 positions + disturbance descriptors,
with the target being the position 3 frames ahead of the window end.

Disturbance descriptors are randomly generated per-sample to teach the model
to weight noisy history differently from clean.
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
    from torch.utils.data import Dataset, DataLoader
    from src.tracking.nn2_model import (
        BeaconMotionGRU, create_nn2_model,
        INPUT_DIM, DISTURBANCE_FEATURES, POSITION_FEATURES,
    )
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    pass


# Default paths
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_WEIGHTS = _WEIGHTS_DIR / "nn2_weights.pth"

# Training hyperparameters
WINDOW_SIZE = 5           # Input history length (matches nn2_predictor)
PREDICTION_HORIZON = 1    # Predict 1 frame ahead (matches nn2_predictor)
SENSOR_W = 640.0
SENSOR_H = 480.0


def _generate_trajectory(
    motion_type: str,
    length: int,
    dt: float = 1.0 / 30.0,
) -> List[Tuple[float, float]]:
    """Generate a trajectory of (x, y) positions in sensor pixel coordinates."""

    cx = random.uniform(150, SENSOR_W - 150)
    cy = random.uniform(120, SENSOR_H - 120)

    if motion_type == "circular":
        radius = random.uniform(20, 180)
        omega = random.uniform(0.5, 6.5) * random.choice([-1, 1])
        phase = random.uniform(0, 2 * math.pi)
        return [
            (cx + radius * math.cos(omega * i * dt + phase),
             cy + radius * math.sin(omega * i * dt + phase))
            for i in range(length)
        ]

    elif motion_type == "figure8":
        rx = random.uniform(40, 220)
        ry = random.uniform(30, 140)
        omega = random.uniform(0.5, 7.0)
        phase = random.uniform(0, 2 * math.pi)
        return [
            (cx + rx * math.sin(omega * i * dt + phase),
             cy + ry * math.sin(2 * omega * i * dt + phase))
            for i in range(length)
        ]

    elif motion_type == "spiral":
        r0 = random.uniform(10, 40)
        growth = random.uniform(0.3, 2.5)
        omega = random.uniform(1.0, 5.0) * random.choice([-1, 1])
        return [
            (cx + (r0 + growth * i * dt) * math.cos(omega * i * dt),
             cy + (r0 + growth * i * dt) * math.sin(omega * i * dt))
            for i in range(length)
        ]

    elif motion_type == "ballistic":
        vx = random.uniform(-120, 120)
        vy = random.uniform(-80, -10)  # upward initial velocity
        g = random.uniform(50, 250)    # gravity (px/s^2)
        x0 = random.uniform(100, SENSOR_W - 100)
        y0 = random.uniform(200, SENSOR_H - 50)
        return [
            (x0 + vx * i * dt,
             y0 + vy * i * dt + 0.5 * g * (i * dt) ** 2)
            for i in range(length)
        ]

    else:  # linear
        vx = random.uniform(-100, 100)
        vy = random.uniform(-80, 80)
        x0 = random.uniform(100, SENSOR_W - 100)
        y0 = random.uniform(80, SENSOR_H - 80)
        return [
            (x0 + vx * i * dt, y0 + vy * i * dt)
            for i in range(length)
        ]


def _random_disturbance_descriptor() -> List[float]:
    """Generate a random disturbance descriptor vector."""
    desc = [0.0] * DISTURBANCE_FEATURES
    # Randomly activate 0–3 disturbance types
    n_active = random.randint(0, 3)
    indices = random.sample(range(DISTURBANCE_FEATURES), min(n_active, DISTURBANCE_FEATURES))
    for idx in indices:
        desc[idx] = random.uniform(0.1, 1.0)
    return desc


def _add_position_noise(
    positions: List[Tuple[float, float]],
    noise_std: float = 2.0,
) -> List[Tuple[float, float]]:
    """Add Gaussian noise to positions (simulating NN1 detection jitter)."""
    return [
        (x + random.gauss(0, noise_std), y + random.gauss(0, noise_std))
        for x, y in positions
    ]


if _TORCH_AVAILABLE:
    class TrajectoryDataset(Dataset):
        """Synthetic trajectory dataset for NN2 training.

        Each sample: (input_sequence, target_delta)
          - input_sequence: (WINDOW_SIZE, INPUT_DIM) tensor
          - target_delta: (2,) tensor — displacement from last input position
            to the position PREDICTION_HORIZON frames ahead
        """

        MOTION_TYPES = ["circular", "figure8", "spiral", "ballistic", "linear"]

        def __init__(self, num_samples: int = 2000, noise_std: float = 2.0):
            self.num_samples = num_samples
            self.noise_std = noise_std
            self._data = []
            self._generate()

        def _generate(self):
            total_len = WINDOW_SIZE + PREDICTION_HORIZON
            for _ in range(self.num_samples):
                motion = random.choice(self.MOTION_TYPES)
                traj = _generate_trajectory(motion, total_len)

                # Add NN1-like detection noise
                noisy = _add_position_noise(traj, self.noise_std)

                # Input window: first WINDOW_SIZE positions
                window = noisy[:WINDOW_SIZE]

                # Target: true position PREDICTION_HORIZON frames after window end
                # (use clean trajectory for target, not noisy)
                target_pos = traj[WINDOW_SIZE + PREDICTION_HORIZON - 1]
                last_pos = noisy[WINDOW_SIZE - 1]

                # Delta from last observed position to target
                delta_x = (target_pos[0] - last_pos[0]) / SENSOR_W
                delta_y = (target_pos[1] - last_pos[1]) / SENSOR_H

                # Build feature vectors per timestep
                disturbance = _random_disturbance_descriptor()
                features = []
                for t in range(WINDOW_SIZE):
                    x_norm = window[t][0] / SENSOR_W
                    y_norm = window[t][1] / SENSOR_H
                    if t > 0:
                        dx = (window[t][0] - window[t-1][0]) / SENSOR_W
                        dy = (window[t][1] - window[t-1][1]) / SENSOR_H
                    else:
                        dx = 0.0
                        dy = 0.0
                    feat = [x_norm, y_norm, dx, dy] + disturbance
                    features.append(feat)

                self._data.append((
                    torch.tensor(features, dtype=torch.float32),
                    torch.tensor([delta_x, delta_y], dtype=torch.float32),
                ))

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]


def train_nn2(
    num_samples: int = 3000,
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    weights_path: Optional[Path] = None,
    noise_std: float = 2.0,
) -> Path:
    """Train NN2 and save weights.

    Returns the path to the saved weights file.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for NN2 training")

    save_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[NN2] Training on {device} — {epochs} epochs × {num_samples} samples")

    # Dataset + loader
    dataset = TrajectoryDataset(num_samples=num_samples, noise_std=noise_std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = create_nn2_model(device)
    print(f"[NN2] Model params: {model.param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Huber loss — less sensitive to outlier trajectories than MSE
    criterion = torch.nn.SmoothL1Loss()

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_batches)
        elapsed = time.time() - t0
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}  "
                  f"({elapsed:.1f}s elapsed)")

    torch.save(model.state_dict(), save_path)
    print(f"[NN2] Training complete in {time.time()-t0:.1f}s — "
          f"weights saved to {save_path}")
    return save_path


if __name__ == "__main__":
    train_nn2()
