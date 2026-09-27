"""
SADHA — Module 2: Recursive Kinematic State Estimation (Kalman Filter)
Per-beacon constant-velocity model tracking [x, y, vx, vy]^T in sensor pixel space.

[INTENTIONAL DESIGN DECISION]: Discrete Frame-Unit Coordinates vs. Continuous-Time Scaling
In typical Kalman filter formulations with continuous time, the process noise covariance Q
scales with dt^4 / 4 for position and dt^3 / 2 for cross terms. At high frame rates (30–60 Hz),
dt = 0.016–0.033 s, which causes position process noise to evaluate to near-zero (~10^-6).
This causes the Kalman gain K to collapse towards zero, treating the system as nearly deterministic
and resulting in severe filter lag (30–50 px) behind agile maneuvering beacons.
By reformulating the state transition and process covariance in discrete **frame-unit coordinates**
(where state transition F advances dt_frames = 1 per call and accel_noise_std directly represents
allowable px/frame^2 change):
  1. The filter adapts rapidly to sudden accelerations without numerical collapse.
  2. Covariance updates use the stabilized Joseph form (I - KH) P (I - KH)^T + K R K^T to guarantee
     positive semi-definiteness across millions of frames.
"""

from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Single-beacon constant-velocity Kalman filter (frame-unit coords).

    State: [x, y, vx, vy] — positions in px, velocities in px/frame.

    Parameters
    ----------
    x0, y0 : float
        Initial position from the first matched detection.
    accel_noise_std : float
        Standard deviation of unmodelled acceleration in px/frame².
        Controls how quickly the filter adapts to velocity changes.
        A value of 8–12 suits beacons with moderate-to-fast maneuvers.
    measurement_noise_std : float
        Standard deviation of measurement noise in px.
        Tuned to NN1 detection accuracy (~2–5 px for CNN, <1 px classical).
    """

    DIM_STATE = 4       # [x, y, vx, vy]
    DIM_MEAS = 2        # [x, y]

    def __init__(
        self,
        x0: float,
        y0: float,
        accel_noise_std: float = 10.0,
        measurement_noise_std: float = 3.0,
    ):
        # ---- State vector (positions in px, velocities in px/frame) ----
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)

        # ---- Covariance ----
        # Large initial velocity uncertainty so the filter quickly adapts
        # its velocity estimate from the very first measurements.
        self.P = np.diag([
            measurement_noise_std ** 2,   # x uncertainty  (px²)
            measurement_noise_std ** 2,   # y uncertainty  (px²)
            100.0 ** 2,                   # vx uncertainty (px/frame)² — deliberately large
            100.0 ** 2,                   # vy uncertainty (px/frame)²
        ])

        # ---- Measurement matrix ----
        # z = H @ x  →  [x, y] = [[1,0,0,0],[0,1,0,0]] @ [x,y,vx,vy]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # ---- Measurement noise ----
        self.R = np.diag([
            measurement_noise_std ** 2,
            measurement_noise_std ** 2,
        ])

        # ---- Process noise ----
        self._accel_std = accel_noise_std

        # ---- Constant matrices (dt_frames = 1 always) ----
        self.F = self._build_F()
        self.Q = self._build_Q()

    # ------------------------------------------------------------------
    # Matrix builders
    # ------------------------------------------------------------------
    @staticmethod
    def _build_F() -> np.ndarray:
        """Constant-velocity state transition (dt = 1 frame)."""
        return np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

    def _build_Q(self) -> np.ndarray:
        """Process noise covariance (discrete white-noise acceleration model).

        With dt_frames = 1 the standard formula simplifies to:
            Q = σ_a² * G G^T
        where G = [0.5, 0.5, 1, 1]^T is the acceleration coupling vector.

        Result:
            Q_pos  = σ_a² / 4   (px²)
            Q_cross = σ_a² / 2
            Q_vel  = σ_a²        (px/frame)²
        """
        q = self._accel_std ** 2
        return q * np.array([
            [0.25, 0,    0.5, 0  ],
            [0,    0.25, 0,   0.5],
            [0.5,  0,    1.0, 0  ],
            [0,    0.5,  0,   1.0],
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Predict step
    # ------------------------------------------------------------------
    def predict(self, dt: float = None) -> np.ndarray:
        """Propagate state one frame forward.

        The ``dt`` parameter is accepted for API compatibility with the
        tracker but is **not used** — in frame-unit coordinates every
        call advances exactly one frame.

        Returns the predicted position [x, y].
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[:2].copy()

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------
    def update(self, z_x: float, z_y: float) -> np.ndarray:
        """Incorporate a measurement [z_x, z_y] and return the updated position.

        Parameters
        ----------
        z_x, z_y : float
            Measured beacon centroid from NN1 detection.

        Returns
        -------
        np.ndarray
            Updated position [x, y].
        """
        z = np.array([z_x, z_y], dtype=np.float64)

        # Innovation (measurement residual)
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.DIM_STATE) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        return self.x[:2].copy()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def position(self) -> tuple:
        """Current smoothed position (x, y)."""
        return (float(self.x[0]), float(self.x[1]))

    @property
    def velocity(self) -> tuple:
        """Current velocity estimate (vx, vy) in px/frame."""
        return (float(self.x[2]), float(self.x[3]))

    @property
    def predicted_position(self) -> tuple:
        """One-frame-ahead prediction WITHOUT modifying internal state."""
        x_pred = self.F @ self.x
        return (float(x_pred[0]), float(x_pred[1]))

    @property
    def position_uncertainty(self) -> float:
        """Square root of the trace of the position covariance block."""
        return float(np.sqrt(self.P[0, 0] + self.P[1, 1]))

    def innovation_distance(self, z_x: float, z_y: float) -> float:
        """Mahalanobis-like distance between a measurement and the predicted state.

        Used by the gated assignment to evaluate match quality.
        """
        z = np.array([z_x, z_y], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        # Squared Mahalanobis distance
        d2 = float(y.T @ np.linalg.inv(S) @ y)
        return float(np.sqrt(max(0, d2)))
