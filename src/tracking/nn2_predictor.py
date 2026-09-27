"""
SADHA — NN2 Predictor: Per-Track Motion Prediction Orchestrator

Manages:
  • One NN2 state per tracked beacon (matching Kalman's 1-per-track model)
  • Fixed 1-in-3-frame cadence (never adaptive)
  • Confidence scoring when target frames arrive
  • Classical fallback (quadratic extrapolation) when torch unavailable

Architecture: GRU model when weights available, else classical curve-fitting.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.tracking.nn2_types import (
    NN2Prediction, NN2ConfidenceScore, NN2Result,
)
from src.tracking.tracker_types import TrackerResult, TrackState

# Soft torch dependency
_TORCH_AVAILABLE = False
try:
    import torch
    from src.tracking.nn2_model import (
        BeaconMotionGRU, create_nn2_model,
        INPUT_DIM, DISTURBANCE_FEATURES,
    )
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    INPUT_DIM = 12
    DISTURBANCE_FEATURES = 8

# Defaults
_WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "data"
_DEFAULT_WEIGHTS = _WEIGHTS_DIR / "nn2_weights.pth"

PREDICTION_CADENCE = 1    # Run every frame
WINDOW_SIZE = 5         # Input history length for GRU
CLASSICAL_WINDOW = 4    # Short window for quadratic extrapolation
PREDICTION_HORIZON = 1   # Predict this many frames ahead
SENSOR_W = 640.0
SENSOR_H = 480.0
CONFIDENCE_THRESHOLD = 15.0  # px — tracking error target
KALMAN_BLEND_WEIGHT = 0.35  # Blend weight for Kalman 1-ahead in classical mode


class _PerTrackState:
    """Internal per-beacon NN2 bookkeeping."""
    __slots__ = (
        "track_id", "frame_counter", "pending_predictions",
    )

    def __init__(self, track_id: int):
        self.track_id = track_id
        self.frame_counter = 0
        # Predictions awaiting target-frame arrival for confidence scoring
        # keyed by target_frame_index
        self.pending_predictions: Dict[int, NN2Prediction] = {}


class NN2Predictor:
    """Per-track beacon motion predictor — runs every 3rd frame per track.

    Parameters
    ----------
    weights_path : Path or None
        Override default NN2 weights file location.
    mode : str
        ``"auto"`` — GRU if weights exist, else classical.
        ``"gru"`` — force GRU (raises if no weights).
        ``"classical"`` — force quadratic extrapolation.
    """

    def __init__(
        self,
        weights_path: Optional[Path] = None,
        mode: str = "auto",
    ):
        self._weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS
        self._mode = mode
        self._use_gru = False

        # GRU model state
        self._model = None
        self._device = None

        # Per-track state
        self._track_states: Dict[int, _PerTrackState] = {}

        # Confidence log
        self._confidence_log: List[NN2ConfidenceScore] = []

        # Attempt to load GRU model
        if mode in ("auto", "gru") and _TORCH_AVAILABLE:
            self._try_load_model()

        if mode == "gru" and not self._use_gru:
            raise RuntimeError("NN2 GRU mode requested but weights not available")

    def _try_load_model(self):
        """Load pre-trained GRU weights if available."""
        if self._weights_path.exists():
            try:
                self._device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
                self._model = create_nn2_model(self._device)
                state = torch.load(
                    self._weights_path, map_location=self._device,
                    weights_only=True,
                )
                self._model.load_state_dict(state)
                self._model.eval()
                self._use_gru = True
            except Exception:
                self._use_gru = False

    # ------------------------------------------------------------------
    # Main per-frame entry point (Interface Boundary: Module 2.2 → Module 2.3)
    # ------------------------------------------------------------------
    def update(
        self,
        tracker_result: TrackerResult,
        frame_index: int,
        timestamp: float,
    ) -> NN2Result:
        """Process one frame of tracking state and evaluate/generate predictions.

        Parameters
        ----------
        tracker_result : TrackerResult
            Current frame's Kalman tracker output containing per-track kinematic
            states, sliding-window raw NN1 history, and disturbance descriptors.
        frame_index : int
            Current sequential frame number.
        timestamp : float
            Current simulation elapsed time in seconds.

        Returns
        -------
        NN2Result
            Output record containing:
            - `predictions`: List of `NN2Prediction` records for future target frames.
            - `confidence_scores`: List of `NN2ConfidenceScore` records for past forecasts
              whose target frames have matured and been verified against actual NN1 observations.
            - `ran_this_frame`: Boolean indicating whether model inference executed this tick.

        Interface Boundary
        ------------------
        Consumes the output of Module 2.2 and emits predictions and empirical confidence
        evaluations to the application dashboard and trajectory visualization displays.
        """
        result = NN2Result(frame_index=frame_index, timestamp=timestamp)

        # Prune track states for retired tracks
        active_ids = {ts.track_id for ts in tracker_result.tracks}
        retired = [tid for tid in self._track_states if tid not in active_ids]
        for tid in retired:
            del self._track_states[tid]

        # Process each active track
        for ts in tracker_result.tracks:
            tid = ts.track_id

            # Ensure per-track state exists
            if tid not in self._track_states:
                self._track_states[tid] = _PerTrackState(tid)

            pstate = self._track_states[tid]
            pstate.frame_counter += 1

            # --- Confidence scoring: check if any pending predictions matured ---
            # [INTENTIONAL DESIGN DECISION]: Empirical Ground-Truth Confidence Scoring
            # Rather than relying on self-reported neural network softmax/variance (which is
            # notoriously uncalibrated and prone to overconfidence during abrupt maneuvers),
            # SADHA implements true empirical validation:
            # 1. When NN2 predicts a target position at frame T for future frame T+H, the prediction
            #    is buffered in `pending_predictions`.
            # 2. When frame T+H subsequently arrives and is matched by NN1, the actual sensor
            #    observation (actual_x, actual_y) is compared against the earlier forecast:
            #       error = || pred_pos - actual_pos ||_2
            # 3. Confidence is calculated using a bounded linear-decay metric against the 15 px
            #    tracking error target:
            #       confidence = max(0.0, 1.0 - error / CONFIDENCE_THRESHOLD)
            # This directly quantifies empirical forecasting performance against physical reality.
            matured = []
            for target_fi, pred in pstate.pending_predictions.items():
                if frame_index >= target_fi and ts.matched:
                    # NN1's actual observed position for this track this frame
                    actual_x, actual_y = ts.position[0], ts.position[1]
                    # Use the matched NN1 raw position if available
                    if ts.nn1_raw_history:
                        last_raw = ts.nn1_raw_history[-1]
                        actual_x, actual_y = last_raw[0], last_raw[1]

                    error = math.sqrt(
                        (pred.predicted_position[0] - actual_x) ** 2 +
                        (pred.predicted_position[1] - actual_y) ** 2
                    )
                    confidence = max(0.0, 1.0 - error / CONFIDENCE_THRESHOLD)

                    score = NN2ConfidenceScore(
                        track_id=tid,
                        frame_index=target_fi,
                        predicted_x=pred.predicted_position[0],
                        predicted_y=pred.predicted_position[1],
                        actual_x=actual_x,
                        actual_y=actual_y,
                        error_px=error,
                        confidence=confidence,
                    )
                    result.confidence_scores.append(score)
                    self._confidence_log.append(score)
                    matured.append(target_fi)

            for fi in matured:
                del pstate.pending_predictions[fi]

            # --- Prediction: every Nth frame for this track ---
            if pstate.frame_counter % PREDICTION_CADENCE != 0:
                continue

            # Need enough history
            history = ts.nn1_raw_history
            if len(history) < 2:
                continue

            # Run prediction (pass full TrackState for Kalman access)
            dist_desc = ts.disturbance_descriptor
            pred_pos = self._predict_single(history, dist_desc, ts)

            if pred_pos is not None:
                target_fi = frame_index + PREDICTION_HORIZON
                prediction = NN2Prediction(
                    track_id=tid,
                    predicted_position=pred_pos,
                    frame_index=frame_index,
                    target_frame_index=target_fi,
                    timestamp=timestamp,
                )
                result.predictions.append(prediction)
                pstate.pending_predictions[target_fi] = prediction
                result.ran_this_frame = True

        return result

    # ------------------------------------------------------------------
    # Prediction backends
    # ------------------------------------------------------------------
    def _predict_single(
        self,
        history: List[Tuple[float, float, float]],
        disturbance_descriptor: Optional[List[float]],
        track_state: Optional[TrackState] = None,
    ) -> Optional[Tuple[float, float]]:
        """Predict future position for a single track.

        Ensemble blending:
        Combines classical quadratic extrapolation, Kalman filter 1-ahead
        kinematic prediction, and GRU neural trajectory prediction.
        """
        c_pred = self._predict_classical(history, track_state)
        k_pred = track_state.predicted_position if track_state else None

        if not self._use_gru or self._model is None:
            return c_pred

        g_pred = self._predict_gru(history, disturbance_descriptor, track_state)
        if c_pred and k_pred and g_pred:
            # 60% classical (responsive to local curvature) + 25% Kalman (smooth) + 15% GRU (disturbance-aware)
            return (
                0.60 * c_pred[0] + 0.25 * k_pred[0] + 0.15 * g_pred[0],
                0.60 * c_pred[1] + 0.25 * k_pred[1] + 0.15 * g_pred[1],
            )
        return c_pred or g_pred

    def _predict_gru(
        self,
        history: List[Tuple[float, float, float]],
        disturbance_descriptor: Optional[List[float]],
        track_state: Optional[TrackState] = None,
    ) -> Optional[Tuple[float, float]]:
        """GRU-based prediction, blended with Kalman 1-ahead."""
        if self._model is None:
            return self._predict_classical(history, track_state)

        # Take last WINDOW_SIZE entries (or pad if fewer)
        window = list(history[-WINDOW_SIZE:])

        # Pad to WINDOW_SIZE if needed
        while len(window) < WINDOW_SIZE:
            window.insert(0, window[0])

        # [INTENTIONAL DESIGN DECISION]: Disturbance-Conditioned Recurrent Sequence Modeling
        # In free-space optical environments, observation quality fluctuates wildly due to
        # atmospheric turbulence, fog scattering, and camera vibration. When noise is severe,
        # apparent centroid movement is dominated by optical distortion rather than true physical
        # acceleration. By concatenating the 8-dimensional disturbance descriptor vector
        # (characterizing noise std, SNR, jitter amplitude, and transmission factor) directly into
        # every temporal time-step of the GRU input tensor alongside [x, y, dx, dy], the GRU
        # learns to adaptively condition its hidden state: it relies more heavily on kinematic momentum
        # under turbulent conditions and tightens its lookahead response when conditions are clear.
        dist = [0.0] * DISTURBANCE_FEATURES
        if disturbance_descriptor and len(disturbance_descriptor) >= DISTURBANCE_FEATURES:
            dist = disturbance_descriptor[:DISTURBANCE_FEATURES]
        elif disturbance_descriptor:
            for i, v in enumerate(disturbance_descriptor):
                if i < DISTURBANCE_FEATURES:
                    dist[i] = v

        # Build feature tensor: [x_norm, y_norm, dx_norm, dy_norm] + [8 disturbance channels]
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
            feat = [x_norm, y_norm, dx, dy] + dist
            features.append(feat)

        with torch.no_grad():
            x = torch.tensor([features], dtype=torch.float32, device=self._device)
            delta = self._model(x)  # (1, 2)
            dx = delta[0, 0].item() * SENSOR_W
            dy = delta[0, 1].item() * SENSOR_H

        last_x, last_y = window[-1][0], window[-1][1]
        gru_pred = (last_x + dx, last_y + dy)

        # Blend with Kalman 1-ahead if available
        if track_state is not None and track_state.predicted_position:
            kp = track_state.predicted_position
            w = KALMAN_BLEND_WEIGHT
            return (
                gru_pred[0] * (1 - w) + kp[0] * w,
                gru_pred[1] * (1 - w) + kp[1] * w,
            )
        return gru_pred

    def _predict_classical(
        self,
        history: List[Tuple[float, float, float]],
        track_state: Optional[TrackState] = None,
    ) -> Optional[Tuple[float, float]]:
        """Classical prediction: quadratic extrapolation blended with Kalman.

        Uses a SHORT window (CLASSICAL_WINDOW=4) so the quadratic doesn't
        over-fit the wrong curvature on fast-changing trajectories.
        When the Kalman filter's 1-ahead prediction is available, it is
        blended in to provide complementary smoothing.
        """
        n = min(len(history), CLASSICAL_WINDOW)

        # --- Kalman 1-ahead baseline (always available after frame 1) ---
        kalman_pred = None
        if track_state is not None and track_state.predicted_position:
            kalman_pred = track_state.predicted_position

        # Not enough history for quadratic → fall back to Kalman or linear
        if n < 3:
            if n >= 2:
                # Simple linear extrapolation from last 2 points
                dx = history[-1][0] - history[-2][0]
                dy = history[-1][1] - history[-2][1]
                linear = (
                    history[-1][0] + dx * PREDICTION_HORIZON,
                    history[-1][1] + dy * PREDICTION_HORIZON,
                )
                if kalman_pred:
                    w = KALMAN_BLEND_WEIGHT
                    return (
                        linear[0] * (1 - w) + kalman_pred[0] * w,
                        linear[1] * (1 - w) + kalman_pred[1] * w,
                    )
                return linear
            return kalman_pred  # May be None

        recent = list(history[-n:])
        xs = np.array([p[0] for p in recent])
        ys = np.array([p[1] for p in recent])
        ts = np.arange(n, dtype=np.float64)

        # Fit quadratic: pos = a*t^2 + b*t + c
        try:
            cx = np.polyfit(ts, xs, min(2, n - 1))
            cy = np.polyfit(ts, ys, min(2, n - 1))
        except (np.linalg.LinAlgError, ValueError):
            # Fallback to linear extrapolation
            vx = (xs[-1] - xs[0]) / max(1, n - 1)
            vy = (ys[-1] - ys[0]) / max(1, n - 1)
            quad_pred = (
                float(xs[-1] + vx * PREDICTION_HORIZON),
                float(ys[-1] + vy * PREDICTION_HORIZON),
            )
            if kalman_pred:
                w = KALMAN_BLEND_WEIGHT
                return (
                    quad_pred[0] * (1 - w) + kalman_pred[0] * w,
                    quad_pred[1] * (1 - w) + kalman_pred[1] * w,
                )
            return quad_pred

        # Evaluate at t = n - 1 + PREDICTION_HORIZON
        t_pred = float(n - 1 + PREDICTION_HORIZON)
        pred_x = float(np.polyval(cx, t_pred))
        pred_y = float(np.polyval(cy, t_pred))
        quad_pred = (pred_x, pred_y)

        # Blend with Kalman 1-ahead
        if kalman_pred:
            w = KALMAN_BLEND_WEIGHT
            return (
                quad_pred[0] * (1 - w) + kalman_pred[0] * w,
                quad_pred[1] * (1 - w) + kalman_pred[1] * w,
            )
        return quad_pred

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------
    @property
    def confidence_log(self) -> List[NN2ConfidenceScore]:
        return self._confidence_log

    @property
    def avg_confidence(self) -> float:
        if not self._confidence_log:
            return 0.0
        return sum(s.confidence for s in self._confidence_log) / len(self._confidence_log)

    @property
    def avg_error_px(self) -> float:
        if not self._confidence_log:
            return 0.0
        return sum(s.error_px for s in self._confidence_log) / len(self._confidence_log)

    @property
    def using_gru(self) -> bool:
        return self._use_gru
