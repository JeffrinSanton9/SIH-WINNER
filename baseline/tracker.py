"""
Conventional / Baseline Tracker Orchestrator

Implements standard classical tracking:
  1. Linear Constant-Velocity Kalman Filter for basic state smoothing
  2. Greedy Nearest Neighbor (GNN) Data Association (No Hungarian Bipartite Matching)
  3. No Recurrent Trajectory Predictor (No GRU; purely linear extrapolation)
  4. Reactive Proportional Pan-Tilt Gimbal Control (No Velocity Feedforward)
  5. Static 30 Hz Frame Rate (No Dynamic Loss Ramping)

Known failure modes under real-world conditions:
  - Identity swapping during multi-beacon crossing paths
  - Track coalescence when targets are in close proximity
  - Severe lag / divergence during non-linear accelerations (spirals, figure-8)
  - Slow re-acquisition when line-of-sight is temporarily broken
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.tracking.nn1_types import NN1Result, Detection
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.tracker_types import (
    TrackState, AssignmentResult, TrackerResult, LockState,
)


class _BaselineTrackEntry:
    def __init__(self, track_id: int, terminal_id: int, x: float, y: float, timestamp: float):
        self.track_id = track_id
        self.terminal_id = terminal_id
        self.kf = KalmanFilter(x, y)
        self.consecutive_missed = 0
        self.time_since_match = 0.0
        self.last_match_time = timestamp
        self.lock_state = LockState.SEARCHING
        self.matched_this_frame = True
        self.matched_det_idx = -1
        self.matched_confidence = 0.0


class BaselineTracker:
    """Conventional tracker using Greedy Nearest Neighbor and linear Kalman filter."""

    def __init__(
        self,
        gate_threshold_px: float = 80.0,
        loss_trigger_frames: int = 3,
        retirement_time_sec: float = 1.0,
        max_terminals: int = 5,
        sensor_width: int = 640,
        sensor_height: int = 480,
        fov_h_deg: float = 4.0,
        fov_v_deg: float = 3.0,
    ):
        self.gate_threshold = gate_threshold_px
        self.loss_trigger_frames = loss_trigger_frames
        self.retirement_time = retirement_time_sec
        self.max_terminals = max_terminals

        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self._deg_per_px_h = fov_h_deg / sensor_width
        self._deg_per_px_v = fov_v_deg / sensor_height

        self._tracks: Dict[int, _BaselineTrackEntry] = {}
        self._next_track_id = 1
        self._available_terminals: List[int] = list(range(max_terminals))
        self._prev_timestamp: Optional[float] = None

    def update(self, nn1: NN1Result) -> TrackerResult:
        """Process one frame of detections using Greedy Nearest Neighbor."""
        timestamp = nn1.timestamp
        frame_index = nn1.frame_index

        dt = 1.0 / 30.0
        if self._prev_timestamp is not None and timestamp > self._prev_timestamp:
            dt = timestamp - self._prev_timestamp
        self._prev_timestamp = timestamp

        # Step 1: Predict existing tracks using standard Constant-Velocity KF
        predicted_positions: Dict[int, Tuple[float, float]] = {}
        for tid, entry in self._tracks.items():
            pred = entry.kf.predict(dt)
            predicted_positions[tid] = (float(pred[0]), float(pred[1]))

        # Step 2: Greedy Nearest Neighbor (GNN) Association
        # (Classical heuristic: for each track, find closest detection without global optimization)
        detections = nn1.detections
        matched_tracks = set()
        matched_detections = set()
        matches: List[Tuple[int, int]] = []

        for tid, (tx, ty) in predicted_positions.items():
            best_det_idx = -1
            best_dist = float("inf")

            for dj, det in enumerate(detections):
                if dj in matched_detections:
                    # Note: in naive GNN, detections are claimed one-by-one
                    continue
                dist = math.sqrt((tx - det.x) ** 2 + (ty - det.y) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = dj

            # Gating check
            if best_det_idx >= 0 and best_dist <= self.gate_threshold:
                matches.append((tid, best_det_idx))
                matched_tracks.add(tid)
                matched_detections.add(best_det_idx)

        # Step 3: Update matched tracks
        for tid, det_idx in matches:
            entry = self._tracks[tid]
            det = detections[det_idx]
            entry.kf.update(det.x, det.y)
            entry.consecutive_missed = 0
            entry.time_since_match = 0.0
            entry.last_match_time = timestamp
            entry.lock_state = LockState.LOCKED
            entry.matched_this_frame = True
            entry.matched_det_idx = det_idx
            entry.matched_confidence = det.confidence

        # Step 4: Handle unmatched tracks
        unmatched_track_ids = [tid for tid in self._tracks.keys() if tid not in matched_tracks]
        for tid in unmatched_track_ids:
            entry = self._tracks[tid]
            entry.consecutive_missed += 1
            entry.time_since_match = timestamp - entry.last_match_time
            entry.matched_this_frame = False
            entry.matched_det_idx = -1
            entry.matched_confidence = 0.0

            if entry.consecutive_missed >= self.loss_trigger_frames:
                entry.lock_state = LockState.LOST

        # Step 5: Retire stale tracks
        retired_ids = [
            tid for tid in unmatched_track_ids
            if (timestamp - self._tracks[tid].last_match_time) >= self.retirement_time
        ]
        for tid in retired_ids:
            entry = self._tracks.pop(tid)
            if entry.terminal_id not in self._available_terminals:
                self._available_terminals.append(entry.terminal_id)
                self._available_terminals.sort()

        # Step 6: Spawn new tracks for unmatched detections
        unmatched_det_indices = [j for j in range(len(detections)) if j not in matched_detections]
        for dj in unmatched_det_indices:
            if not self._available_terminals or len(self._tracks) >= self.max_terminals:
                break
            det = detections[dj]
            tid = self._next_track_id
            self._next_track_id += 1
            term_id = self._available_terminals.pop(0)

            entry = _BaselineTrackEntry(tid, term_id, det.x, det.y, timestamp)
            entry.matched_confidence = det.confidence
            self._tracks[tid] = entry

        # Step 7: Build output
        track_states = []
        for tid, entry in self._tracks.items():
            ts = TrackState(
                track_id=tid,
                terminal_id=entry.terminal_id,
                position=entry.kf.position,
                velocity=entry.kf.velocity,
                predicted_position=entry.kf.predicted_position,
                frame_index=frame_index,
                timestamp=timestamp,
                matched=entry.matched_this_frame,
                consecutive_missed_frames=entry.consecutive_missed,
                time_since_last_match=entry.time_since_match,
                lock_state=entry.lock_state,
                matched_detection_index=entry.matched_det_idx,
                matched_confidence=entry.matched_confidence,
            )
            track_states.append(ts)

        assignment = AssignmentResult(
            frame_index=frame_index,
            timestamp=timestamp,
            matches=matches,
            new_track_indices=unmatched_det_indices,
            unmatched_track_ids=unmatched_track_ids,
        )

        return TrackerResult(
            frame_index=frame_index,
            timestamp=timestamp,
            tracks=track_states,
            assignment=assignment,
        )

    def compute_pan_tilt_commands(self) -> Dict[int, Tuple[float, float]]:
        """Compute conventional reactive Proportional gimbal rates (No Velocity Feedforward)."""
        cx = self.sensor_width * 0.5
        cy = self.sensor_height * 0.5
        commands = {}

        for tid, entry in self._tracks.items():
            if entry.lock_state == LockState.LOST:
                # Conventional approach: Stop or freeze when lost (no predictive slew)
                pan_rate = 0.0
                tilt_rate = 0.0
            else:
                # Conventional pure P-controller (No velocity feedforward)
                px, py = entry.kf.position
                dx = px - cx
                dy = py - cy
                # Standard proportional gain
                pan_rate = (dx * 4.5) * self._deg_per_px_h
                tilt_rate = (dy * 4.5) * self._deg_per_px_v

            commands[entry.terminal_id] = (pan_rate, tilt_rate)

        return commands
