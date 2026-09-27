"""
SADHA — Tracker Output Contract Types (v1)

Stable, versioned record types consumed by:
  • Module 1 camera control (per-terminal pan/tilt commands)
  • NN2 (per-track history buffer routing)
  • Loss-recovery ladder logic
  • Analytics / UI display

Design invariants:
  • One ``TrackState`` per tracked beacon, per frame.
  • ``track_id`` is stable across frames for the life of a track.
  • ``disturbance_descriptor`` is passed through from NN1 for NN2.
  • ``nn1_raw_history`` carries the per-track NN1 position window for NN2,
    NOT this stage's smoothed estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class LockState(Enum):
    """Per-track lock status."""
    LOCKED = "locked"
    LOST = "lost"
    SEARCHING = "searching"


@dataclass(slots=True)
class TrackState:
    """Per-track, per-frame output record."""
    track_id: int
    terminal_id: int                      # Bound terminal/camera index

    # Smoothed Kalman estimates
    position: Tuple[float, float]         # (x, y) filtered position
    velocity: Tuple[float, float]         # (vx, vy) estimated velocity
    predicted_position: Tuple[float, float]  # One-frame-ahead prediction

    # Frame identity (pass-through)
    frame_index: int = 0
    timestamp: float = 0.0

    # Match status
    matched: bool = True
    consecutive_missed_frames: int = 0
    time_since_last_match: float = 0.0
    lock_state: LockState = LockState.LOCKED

    # Detection that was matched (if any)
    matched_detection_index: int = -1
    matched_confidence: float = 0.0

    # Pass-through for NN2
    disturbance_descriptor: Optional[List[float]] = None

    # Per-track NN1 raw position history for NN2 input
    # List of (x, y, timestamp) from NN1 detections, NOT smoothed estimates
    nn1_raw_history: List[Tuple[float, float, float]] = field(
        default_factory=list
    )


@dataclass(slots=True)
class AssignmentResult:
    """Per-frame Hungarian assignment outcome — exposed for NN2 and scoring."""
    frame_index: int
    timestamp: float
    # (track_id, detection_index) pairs for matched tracks
    matches: List[Tuple[int, int]] = field(default_factory=list)
    # detection indices that spawned new tracks
    new_track_indices: List[int] = field(default_factory=list)
    # track_ids that had no detection this frame
    unmatched_track_ids: List[int] = field(default_factory=list)


@dataclass(slots=True)
class TrackerResult:
    """Complete per-frame tracker output."""
    frame_index: int
    timestamp: float
    tracks: List[TrackState] = field(default_factory=list)
    assignment: Optional[AssignmentResult] = None
    disturbance_descriptor: Optional[List[float]] = None

    @property
    def num_active_tracks(self) -> int:
        return len(self.tracks)

    @property
    def locked_tracks(self) -> List[TrackState]:
        return [t for t in self.tracks if t.lock_state == LockState.LOCKED]

    @property
    def lost_tracks(self) -> List[TrackState]:
        return [t for t in self.tracks if t.lock_state == LockState.LOST]
