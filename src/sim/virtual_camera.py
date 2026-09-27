"""
SADHA — Virtual Pan-Tilt Camera Simulation
2-axis gimbal kinematics with hard slew-rate clamping, FOV projection,
and adaptive 30–60 Hz capture rate command interface.
"""

from __future__ import annotations

import math


class VirtualCamera:
    """Simulated monochrome pan-tilt sensor for FSOC coarse alignment.

    Command Interface
    -----------------
    Module 2 (or any driver) sends:
      • ``set_commanded_rates(pan_rate, tilt_rate)`` — angular velocity commands
      • ``set_capture_rate(hz)`` — adapt capture rate (30–60 Hz range)
    Module 1 hard-clamps these to the configured limits before applying.

    Telemetry
    ---------
    Emits current ``pan``, ``tilt``, ``current_fps`` alongside each frame.
    """

    def __init__(
        self,
        sensor_width: int = 640,
        sensor_height: int = 480,
        fov_h: float = 4.0,
        fov_v: float = 3.0,
        baseline_fps: float = 30.0,
        loss_ramp_fps: float = 60.0,
        max_pan_speed: float = 5.0,
        max_tilt_speed: float = 5.0,
        initial_pan: float = 0.0,
        initial_tilt: float = 0.0,
    ):
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.fov_h = fov_h    # degrees
        self.fov_v = fov_v

        # Capture rate limits
        self.baseline_fps = baseline_fps
        self.loss_ramp_fps = loss_ramp_fps
        self.current_fps = baseline_fps

        # Slew-rate hard limits (enforced regardless of caller)
        self.max_pan_speed = max_pan_speed    # deg/s
        self.max_tilt_speed = max_tilt_speed  # deg/s

        # Gimbal state (degrees)
        self.pan = initial_pan
        self.tilt = initial_tilt
        self._pan_rate = 0.0
        self._tilt_rate = 0.0

        # Commanded rates from Module 2
        self._cmd_pan_rate = 0.0
        self._cmd_tilt_rate = 0.0

        # World-space scale factor (px per degree)
        self.deg_to_world = 150.0

        # World aim-point (centre of gaze)
        self.aim_x = 0.0
        self.aim_y = 0.0

        # Frame timing accumulator
        self._frame_timer = 0.0
        self.should_capture = False

    # ------------------------------------------------------------------
    # Command Interface (Module 2 → Module 1 Interface Boundary)
    # ------------------------------------------------------------------
    def set_commanded_rates(self, pan_rate: float, tilt_rate: float) -> None:
        """Accept angular-velocity steering commands from Module 2.

        Parameters
        ----------
        pan_rate : float
            Commanded azimuth rate in degrees per second (positive = right).
        tilt_rate : float
            Commanded elevation rate in degrees per second (positive = down).

        Notes
        -----
        [INTENTIONAL DESIGN DECISION]: Module 1 unconditionally enforces hard slew-rate
        clamping to `[-max_pan_speed, max_pan_speed]` and `[-max_tilt_speed, max_tilt_speed]`
        regardless of the value emitted by Module 2. This enforces physical motor/actuator
        kinematic feasibility, ensuring simulated tracking reflects realistic physical gimbal limits.
        """
        self._cmd_pan_rate = _clamp(pan_rate, -self.max_pan_speed, self.max_pan_speed)
        self._cmd_tilt_rate = _clamp(tilt_rate, -self.max_tilt_speed, self.max_tilt_speed)

    def set_capture_rate(self, hz: float) -> None:
        """Accept an adaptive frame capture rate command from Module 2.

        Parameters
        ----------
        hz : float
            Desired sensor acquisition rate in Hertz (frames per second).

        Notes
        -----
        [INTENTIONAL DESIGN DECISION]: Capture-Rate Ramp Behavior on Beacon Loss:
        Under nominal locked tracking, the camera runs at `baseline_fps` (30 Hz) to
        minimize power and computational throughput. When Module 2 detects target loss
        (LockState.LOST), it commands `loss_ramp_fps` (up to 60 Hz). Doubling the temporal
        sampling rate halves the inter-frame spatial displacement of fast-moving beacons,
        providing twice as many measurement updates to the Hungarian association and
        Kalman filter to enable rapid re-acquisition before the target leaves the camera FOV.
        """
        self.current_fps = _clamp(hz, self.baseline_fps, self.loss_ramp_fps)

    # ------------------------------------------------------------------
    # FOV geometry helpers
    # ------------------------------------------------------------------
    @property
    def world_fov_w(self) -> float:
        return self.fov_h * self.deg_to_world

    @property
    def world_fov_h(self) -> float:
        return self.fov_v * self.deg_to_world

    def frustum_corners(self):
        """World-space rectangle of the camera's FOV footprint."""
        hw = self.world_fov_w * 0.5
        hh = self.world_fov_h * 0.5
        ax, ay = self.aim_x, self.aim_y
        return [
            (ax - hw, ay - hh), (ax + hw, ay - hh),
            (ax + hw, ay + hh), (ax - hw, ay + hh),
        ]

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def update(
        self,
        dt: float,
        platform_offset_x: float = 0.0,
        platform_offset_y: float = 0.0,
        world_w: float = 2400.0,
        world_h: float = 2400.0,
    ) -> None:
        # Frame capture tick
        self._frame_timer += dt
        interval = 1.0 / self.current_fps
        if self._frame_timer >= interval:
            self.should_capture = True
            self._frame_timer %= interval
        else:
            self.should_capture = False

        # Smooth gimbal rate tracking with hard clamp
        target_pr = _clamp(self._cmd_pan_rate, -self.max_pan_speed, self.max_pan_speed)
        target_tr = _clamp(self._cmd_tilt_rate, -self.max_tilt_speed, self.max_tilt_speed)
        self._pan_rate = _lerp(self._pan_rate, target_pr, 0.4)
        self._tilt_rate = _lerp(self._tilt_rate, target_tr, 0.4)

        # Integrate gimbal angles
        self.pan += self._pan_rate * dt
        self.tilt += self._tilt_rate * dt

        # Map to world aim-point
        base_cx = world_w * 0.5 + platform_offset_x
        base_cy = world_h * 0.5 + platform_offset_y
        self.aim_x = base_cx + self.pan * self.deg_to_world
        self.aim_y = base_cy + self.tilt * self.deg_to_world

    # ------------------------------------------------------------------
    # Coordinate Transforms
    # ------------------------------------------------------------------
    def world_to_sensor(
        self, wx: float, wy: float, jitter_x: float = 0.0, jitter_y: float = 0.0
    ) -> tuple:
        """Project world coordinate (wx, wy) → sensor pixel (u, v, in_fov)."""
        dx = wx - self.aim_x
        dy = wy - self.aim_y
        norm_u = dx / self.world_fov_w
        norm_v = dy / self.world_fov_h
        u = (norm_u + 0.5) * self.sensor_width + jitter_x
        v = (norm_v + 0.5) * self.sensor_height + jitter_y
        margin = 2
        in_fov = (-margin <= u <= self.sensor_width + margin and
                  -margin <= v <= self.sensor_height + margin)
        return u, v, in_fov

    def sensor_to_world(self, u: float, v: float) -> tuple:
        """Inverse: sensor pixel → world coordinate."""
        norm_u = u / self.sensor_width - 0.5
        norm_v = v / self.sensor_height - 0.5
        wx = self.aim_x + norm_u * self.world_fov_w
        wy = self.aim_y + norm_v * self.world_fov_h
        return wx, wy


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _lerp(a, b, t):
    return a + (b - a) * t
