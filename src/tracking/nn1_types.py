"""
SADHA — NN1 Output Contract Types (v1)

These are the stable, versioned record types that NN2, the Kalman filter,
and the Hungarian association layer all consume.  NN1's internal model
architecture may change freely without touching any downstream code, as
long as it continues to emit ``NN1Result`` records through this interface.

Design invariants:
  • ``detections`` is ALWAYS a list — even for single-beacon, even for zero
    detections.  Downstream code uses one path for 0-to-N.
  • No beacon identity or track ID is attached — association is done by the
    Hungarian layer sitting above NN1.
  • ``disturbance_descriptor`` is passed through unchanged from the input
    so NN2 can window over it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True, slots=True)
class Detection:
    """A single raw, unlabeled detection in camera-frame pixel coordinates."""
    x: float               # Centroid x (px), sub-pixel precision
    y: float               # Centroid y (px), sub-pixel precision
    confidence: float       # Detection quality score ∈ [0, 1]
    bbox_x: float = 0.0    # Optional bounding box left
    bbox_y: float = 0.0    # Optional bounding box top
    bbox_w: float = 0.0    # Optional bounding box width
    bbox_h: float = 0.0    # Optional bounding box height
    peak_intensity: int = 0 # Raw peak pixel value at detection site


@dataclass(slots=True)
class NN1Result:
    """Per-frame output record emitted by NN1.

    This is the unit that NN2 windows over, the Kalman filter measures from,
    and the Hungarian layer matches against existing tracks.
    """
    frame_index: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)

    # Pass-through fields (NN1 doesn't interpret these — just forwards them)
    disturbance_descriptor: Optional[List[float]] = None
    camera_pan_deg: Optional[float] = None
    camera_tilt_deg: Optional[float] = None
    capture_rate_hz: Optional[float] = None

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0
