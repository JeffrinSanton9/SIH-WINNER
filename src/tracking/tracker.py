"""
SADHA — Multi-Beacon Tracker Orchestrator

Manages the full per-frame pipeline:
  1. Predict all existing tracks (Kalman predict step).
  2. Build cost matrix and solve Hungarian assignment.
  3. Gate assignments, update matched tracks, handle unmatched.
  4. Spawn new tracks for unmatched detections.
  5. Apply loss-trigger (3 consecutive misses) and retirement (1 s).
  6. Maintain per-track terminal binding and NN1 raw-history buffers.
  7. Generate per-terminal pan/tilt commands from predicted positions.

Runs on every frame, unconditionally.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from src.tracking.nn1_types import NN1Result, Detection
from src.tracking.kalman_filter import KalmanFilter
from src.tracking.hungarian import build_cost_matrix, solve_assignment
from src.tracking.tracker_types import (
    TrackState, AssignmentResult, TrackerResult, LockState,
)


# ---------------------------------------------------------------------------
# Configuration constants & Architectural Invariants
# ---------------------------------------------------------------------------

# [INTENTIONAL DESIGN DECISION]: Two-Tier Loss vs. Retirement Architecture
# 1. Tier 1 - Loss Trigger (Frame Count: 3 consecutive misses):
#    Immediate state transition to LockState.LOST occurs after 3 missed frames (~100 ms at 30 Hz).
#    This provides high-agility reaction to momentary optical dropouts (e.g. beam scintillation,
#    dense smoke puffs, or rapid maneuvering across sensor boundaries), immediately alerting the
#    gimbal controller and triggering adaptive capture-rate ramp-up.
LOSS_TRIGGER_FRAMES = 3        # Consecutive misses before marking "lost"

# 2. Tier 2 - Track Retirement (Elapsed Real Time: 1.0 second):
#    Track destruction and terminal deallocation are deliberately measured in elapsed physical
#    time (seconds) rather than frame count. Because SADHA dynamically ramps capture rate between
#    30 Hz (nominal) and 60 Hz (loss recovery), a frame-based retirement threshold would expire
#    twice as fast in real-time at 60 Hz (30 frames = 0.5 s) compared to 30 Hz (30 frames = 1.0 s).
#    Measuring retirement in continuous elapsed seconds guarantees an invariant physical window
#    of 1.0 s for target re-acquisition regardless of sensor FPS fluctuations.
RETIREMENT_TIME_SEC = 1.0      # Elapsed time without match → retire track

# [INTENTIONAL DESIGN DECISION]: Hungarian Gating Threshold Rationale
# Standard association gates in stationary-sensor tracking are typically 30-50 px.
# However, in mobile FSOC terminals, high-acceleration beacon maneuvers (e.g., Lemniscate figure-8
# and projectile drops) compounded with active platform vibrations can produce apparent
# frame-to-frame pixel displacements of up to 100-120 px when sampling at 30 Hz.
# Setting GATE_THRESHOLD_PX = 160.0 px prevents premature track divergence during aggressive
# platform transients while still rejecting false associations from distant optical noise spikes.
GATE_THRESHOLD_PX = 160.0      # Max distance for valid assignment (px) — accommodates fast maneuvers

NN1_HISTORY_LENGTH = 15        # Sliding window of raw NN1 positions for NN2

# [INTENTIONAL DESIGN DECISION]: One-Terminal-Per-Beacon Dedicated Binding
# In physical FSOC optical architectures, a single gimballed telescope/collimator has a single
# physical line-of-sight and cannot simultaneously steer towards multiple spatially separated targets.
# To cleanly model multi-beacon environments, SADHA allocates a discrete terminal ID (0 to 4)
# from a managed pool for each active beacon track. This guarantees isolated pan/tilt command
# generation per terminal without shared-camera pointing contention.
MAX_TERMINALS = 5              # Max simultaneous beacon terminals


class _TrackEntry:
    """Internal bookkeeping for a single active track."""

    __slots__ = (
        "track_id", "terminal_id", "kf",
        "consecutive_missed", "time_since_match", "last_match_time",
        "lock_state", "nn1_history", "matched_this_frame",
        "matched_det_idx", "matched_confidence",
    )

    def __init__(
        self, track_id: int, terminal_id: int,
        x: float, y: float, timestamp: float,
    ):
        self.track_id = track_id
        self.terminal_id = terminal_id
        self.kf = KalmanFilter(x, y)
        self.consecutive_missed = 0
        self.time_since_match = 0.0
        self.last_match_time = timestamp
        self.lock_state = LockState.SEARCHING
        self.nn1_history: deque = deque(maxlen=NN1_HISTORY_LENGTH)
        self.nn1_history.append((x, y, timestamp))
        self.matched_this_frame = True
        self.matched_det_idx = -1
        self.matched_confidence = 0.0


class MultiBeaconTracker:
    """Top-level multi-beacon tracking and data association engine.
    
    Subsystem: Module 2 (State Estimation & Target Association).
    
    Consumes raw detections emitted by NN1, manages track initiation, solves
    optimal bipartite data association via Hungarian matching with distance gating,
    updates per-beacon Kalman filters, monitors two-tier loss/retirement lifecycles,
    and synthesizes closed-loop pan/tilt rate commands for camera steering.
    """

    def __init__(
        self,
        gate_threshold: float = GATE_THRESHOLD_PX,
        loss_trigger_frames: int = LOSS_TRIGGER_FRAMES,
        retirement_time: float = RETIREMENT_TIME_SEC,
        max_terminals: int = MAX_TERMINALS,
        sensor_width: int = 640,
        sensor_height: int = 480,
        fov_h_deg: float = 4.0,
        fov_v_deg: float = 3.0,
    ):
        self.gate_threshold = gate_threshold
        self.loss_trigger_frames = loss_trigger_frames
        self.retirement_time = retirement_time
        self.max_terminals = max_terminals

        # Camera geometry for pan/tilt command generation
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.fov_h_deg = fov_h_deg
        self.fov_v_deg = fov_v_deg
        self._deg_per_px_h = fov_h_deg / sensor_width
        self._deg_per_px_v = fov_v_deg / sensor_height

        # Track registry
        self._tracks: Dict[int, _TrackEntry] = {}
        self._next_track_id = 1

        # Terminal pool: set of available terminal IDs
        self._available_terminals: List[int] = list(range(max_terminals))
        self._used_terminals: Dict[int, int] = {}  # track_id → terminal_id

        # Previous timestamp for dt computation
        self._prev_timestamp: Optional[float] = None

    # ------------------------------------------------------------------
    # Main per-frame entry point
    # ------------------------------------------------------------------
    def update(self, nn1: NN1Result) -> TrackerResult:
        """Process one frame's NN1 detections and return tracker output.

        Called once per frame, synchronously, unconditionally.
        """
        timestamp = nn1.timestamp
        frame_index = nn1.frame_index

        # Compute dt
        dt = 1.0 / 30.0  # default
        if self._prev_timestamp is not None and timestamp > self._prev_timestamp:
            dt = timestamp - self._prev_timestamp
        self._prev_timestamp = timestamp

        # Step 1: Predict all existing tracks
        track_ids = list(self._tracks.keys())
        predicted_positions: Dict[int, Tuple[float, float]] = {}
        for tid in track_ids:
            entry = self._tracks[tid]
            pred = entry.kf.predict(dt)
            predicted_positions[tid] = (float(pred[0]), float(pred[1]))

        # Step 2: Build cost matrix
        det_positions = [(d.x, d.y) for d in nn1.detections]
        track_id_list = list(predicted_positions.keys())
        pred_list = [predicted_positions[tid] for tid in track_id_list]

        cost = build_cost_matrix(pred_list, det_positions)

        # Step 3: Hungarian assignment with gating
        matches, unmatched_trk_idxs, unmatched_det_idxs = solve_assignment(
            cost, gate_threshold=self.gate_threshold
        )

        # Build assignment result
        assignment = AssignmentResult(
            frame_index=frame_index,
            timestamp=timestamp,
        )

        # Step 4: Update matched tracks
        for trk_idx, det_idx in matches:
            tid = track_id_list[trk_idx]
            det = nn1.detections[det_idx]
            entry = self._tracks[tid]

            entry.kf.update(det.x, det.y)
            entry.consecutive_missed = 0
            entry.time_since_match = 0.0
            entry.last_match_time = timestamp
            entry.matched_this_frame = True
            entry.matched_det_idx = det_idx
            entry.matched_confidence = det.confidence
            entry.nn1_history.append((det.x, det.y, timestamp))

            # Transition to LOCKED after first successful match
            if entry.lock_state == LockState.SEARCHING:
                entry.lock_state = LockState.LOCKED
            elif entry.lock_state == LockState.LOST:
                entry.lock_state = LockState.LOCKED  # re-acquired

            assignment.matches.append((tid, det_idx))

        # Step 5: Handle unmatched tracks
        for trk_idx in unmatched_trk_idxs:
            tid = track_id_list[trk_idx]
            entry = self._tracks[tid]
            entry.consecutive_missed += 1
            entry.time_since_match = timestamp - entry.last_match_time
            entry.matched_this_frame = False
            entry.matched_det_idx = -1
            entry.matched_confidence = 0.0

            # Loss trigger: 3 consecutive misses
            if (entry.consecutive_missed >= self.loss_trigger_frames and
                    entry.lock_state != LockState.LOST):
                entry.lock_state = LockState.LOST

            assignment.unmatched_track_ids.append(tid)

        # Step 6: Retire tracks exceeding retirement time
        retired_ids = []
        for tid, entry in self._tracks.items():
            if not entry.matched_this_frame:
                elapsed = timestamp - entry.last_match_time
                if elapsed >= self.retirement_time:
                    retired_ids.append(tid)

        for tid in retired_ids:
            self._retire_track(tid)

        # Step 7: Spawn new tracks for unmatched detections
        for det_idx in unmatched_det_idxs:
            det = nn1.detections[det_idx]
            new_tid = self._spawn_track(det.x, det.y, timestamp)
            if new_tid is not None:
                entry = self._tracks[new_tid]
                entry.matched_confidence = det.confidence
                assignment.new_track_indices.append(det_idx)
                assignment.matches.append((new_tid, det_idx))

        # Step 8: Build output
        track_states = []
        for tid, entry in self._tracks.items():
            pos = entry.kf.position
            vel = entry.kf.velocity
            pred_pos = entry.kf.predicted_position

            ts = TrackState(
                track_id=tid,
                terminal_id=entry.terminal_id,
                position=pos,
                velocity=vel,
                predicted_position=pred_pos,
                frame_index=frame_index,
                timestamp=timestamp,
                matched=entry.matched_this_frame,
                consecutive_missed_frames=entry.consecutive_missed,
                time_since_last_match=entry.time_since_match,
                lock_state=entry.lock_state,
                matched_detection_index=entry.matched_det_idx,
                matched_confidence=entry.matched_confidence,
                disturbance_descriptor=(
                    list(nn1.disturbance_descriptor)
                    if nn1.disturbance_descriptor else None
                ),
                nn1_raw_history=list(entry.nn1_history),
            )
            track_states.append(ts)

        return TrackerResult(
            frame_index=frame_index,
            timestamp=timestamp,
            tracks=track_states,
            assignment=assignment,
            disturbance_descriptor=(
                list(nn1.disturbance_descriptor)
                if nn1.disturbance_descriptor else None
            ),
        )

    # ------------------------------------------------------------------
    # Track lifecycle
    # ------------------------------------------------------------------
    def _spawn_track(
        self, x: float, y: float, timestamp: float
    ) -> Optional[int]:
        """Create a new track and allocate a terminal. Returns track_id or None."""
        if not self._available_terminals:
            return None  # All terminals occupied

        terminal_id = self._available_terminals.pop(0)
        track_id = self._next_track_id
        self._next_track_id += 1

        entry = _TrackEntry(track_id, terminal_id, x, y, timestamp)
        self._tracks[track_id] = entry
        self._used_terminals[track_id] = terminal_id

        return track_id

    def _retire_track(self, track_id: int) -> None:
        """Destroy a track and release its terminal binding."""
        entry = self._tracks.pop(track_id, None)
        if entry is not None:
            terminal_id = self._used_terminals.pop(track_id, None)
            if terminal_id is not None:
                self._available_terminals.append(terminal_id)
                self._available_terminals.sort()

    # ------------------------------------------------------------------
    # Pan/tilt command generation
    # ------------------------------------------------------------------
    def compute_pan_tilt_commands(self) -> Dict[int, Tuple[float, float]]:
        """Compute per-terminal pan/tilt rate commands from predicted positions.

        Returns a dict mapping terminal_id → (pan_rate_deg_s, tilt_rate_deg_s).
        Pixel offset from frame centre → angular delta, combined with velocity
        feedforward to track maneuvering beacons (figure-8, curves) without lag.
        Module 1 enforces the per-terminal max slew-rate clamp.
        """
        cx = self.sensor_width / 2.0
        cy = self.sensor_height / 2.0
        commands = {}

        for tid, entry in self._tracks.items():
            if entry.lock_state == LockState.LOST:
                # Lost: slew toward last known velocity projection
                px, py = entry.kf.predicted_position
                vx, vy = entry.kf.velocity
                pan_rate = vx * self._deg_per_px_h * 1.2
                tilt_rate = vy * self._deg_per_px_v * 1.2
            else:
                # Locked / Searching: point toward predicted position with velocity feedforward
                px, py = entry.kf.predicted_position
                vx, vy = entry.kf.velocity
                dx = px - cx
                dy = py - cy

                # Proportional error term (deg/s) + velocity feedforward term (deg/s)
                pan_rate = (dx * 7.5 + vx * 0.9) * self._deg_per_px_h
                tilt_rate = (dy * 7.5 + vy * 0.9) * self._deg_per_px_v

            commands[entry.terminal_id] = (pan_rate, tilt_rate)

        return commands

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------
    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def get_track(self, track_id: int) -> Optional[_TrackEntry]:
        return self._tracks.get(track_id)

    def get_all_track_ids(self) -> List[int]:
        return list(self._tracks.keys())
