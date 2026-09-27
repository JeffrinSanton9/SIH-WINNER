"""
SADHA — Output Interface Contract (Module 1 → Module 2)

Per-frame data packet emitted by the Virtual Environment.
Module 2 consumes this identically whether the source is
Module 1's renderer or an external .mp4 file (Benchmark-2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class BeaconGroundTruth:
    """Ground-truth beacon position in camera-frame sensor coordinates.
    Exposed for performance logging and tracking-error scoring ONLY —
    must NEVER be fed into detection or tracking algorithms.
    """
    beacon_id: int
    u: float          # Sensor-frame x position (px)
    v: float          # Sensor-frame y position (px)
    world_x: float    # World-frame x position (px)
    world_y: float    # World-frame y position (px)
    in_fov: bool      # Whether this beacon is within the camera's FOV


@dataclass
class FrameData:
    """Per-frame output packet — the contract between Module 1 and Module 2.

    Fields marked 'telemetry' are only available when the source is
    Module 1 (simulated); external video sources leave them at defaults.
    Module 2's processing code-path must not branch on their presence.
    """
    # ---- Always present ----
    frame_index: int
    timestamp: float                     # Seconds since simulation start
    image: np.ndarray = None             # Monochrome sensor frame (H, W), uint8

    # ---- Camera telemetry (Module 1 only) ----
    camera_pan_deg: float = 0.0
    camera_tilt_deg: float = 0.0
    capture_rate_hz: float = 30.0

    # ---- Disturbance descriptor for NN2 input history ----
    # [salt_density, gauss_stddev_norm, poisson_flag, jitter_amp_norm, atmo_level]
    disturbance_descriptor: List[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0]
    )

    # ---- Ground truth (scoring only — never feed to tracking) ----
    ground_truth: List[BeaconGroundTruth] = field(default_factory=list)
