"""
SADHA — NN2 Output Contract Types (v1)

Stable, versioned record types for NN2 beacon motion prediction.
Consumed by:
  • Confidence scoring (self-evaluation on target frame arrival)
  • Loss-recovery ladder (pan/tilt slew alternative to Kalman velocity)
  • Analytics / performance display
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(slots=True)
class NN2Prediction:
    """Single prediction record: where a beacon will be N frames from now."""
    track_id: int
    predicted_position: Tuple[float, float]   # (x, y) in sensor pixel coords
    frame_index: int                          # Frame this prediction was made on
    target_frame_index: int                   # Future frame this describes
    timestamp: float = 0.0                    # When this prediction was made


@dataclass(slots=True)
class NN2ConfidenceScore:
    """Confidence evaluation: how accurate was a past NN2 prediction?"""
    track_id: int
    frame_index: int              # The target frame that arrived
    predicted_x: float            # What NN2 predicted
    predicted_y: float
    actual_x: float               # What NN1 actually observed
    actual_y: float
    error_px: float               # Euclidean distance between them
    confidence: float             # 0–1 score (1 = perfect, 0 = ≥threshold)
    threshold_px: float = 10.0    # Tracking error target from PDF


@dataclass(slots=True)
class NN2Result:
    """Per-frame NN2 output (may be empty on non-prediction frames)."""
    frame_index: int
    timestamp: float = 0.0
    predictions: List[NN2Prediction] = field(default_factory=list)
    confidence_scores: List[NN2ConfidenceScore] = field(default_factory=list)
    ran_this_frame: bool = False   # True only on every 3rd frame
