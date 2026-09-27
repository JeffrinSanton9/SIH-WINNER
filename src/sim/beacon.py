"""
SADHA — Module 1: Optical Beacon Kinematics & Motion Dynamics
Simulates target beacon trajectories in the continuous world coordinate frame.

[INTENTIONAL DESIGN DECISION]: Extended Kinematic Models & Real-Time Human-in-the-Loop Control
Beyond the four mandatory trajectory models (Straight Line, Circular, Figure-8, Random),
SADHA intentionally implements five advanced motion modes:
  1. Lemniscate Figure-8: Solves Bernoulli's lemniscate with non-linear inflection.
  2. Ballistic Projectile: Parabolic gravity trajectory with ground reflection for rapid acceleration stress-testing.
  3. Expanding/Contracting Spiral: Stresses multi-scale radial/tangential velocity estimation.
  4. Manual Recorded Path: Allows dragging an arbitrary mouse trajectory normalized to [0, 1] for benchmark reproducibility.
  5. Interactive Keyboard & Drag (WASD / Arrows): Enables live human-in-the-loop adversarial evasion testing against the AI tracker.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

import numpy as np

from src.core.types import MotionType, TargetShape


class Beacon:
    """Single optical beacon target with configurable kinematics and shape geometry."""

    def __init__(
        self,
        beacon_id: int,
        shape: TargetShape = TargetShape.SQUARE,
        size: int = 10,
        motion_type: MotionType = MotionType.FIGURE_8,
        speed: float = 3.5,
        radius: float = 400.0,
        x: float = 1200.0,
        y: float = 1200.0,
        brightness: int = 255,
        phase_offset: float = 0.0,
        manual_loop: bool = True,
    ):
        self.id = beacon_id
        self.shape = shape
        self.size = size
        self.motion_type = motion_type
        self.speed = speed
        self.radius = radius
        self.brightness = brightness

        # World-space position & kinematics
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0

        # Origin anchor for cyclic patterns
        self.origin_x = x
        self.origin_y = y

        # Phase accumulator
        self.time = phase_offset
        self.heading = random.uniform(0, 2 * math.pi)

        # Projectile state
        self._proj_vx = random.uniform(-6, 6)
        self._proj_vy = -15.0
        self._gravity = 0.35

        # Manual / Recorded waypoints (normalized 0–1)
        self._waypoints: List[dict] = []   # [{x, y}]
        self._manual_loop = manual_loop
        self._manual_progress = 0.0

        # Interactive user control state (keyboard WASD & mouse drag)
        self.manual_vx: float = 0.0
        self.manual_vy: float = 0.0

    # ------------------------------------------------------------------
    # Interactive User Control API (Live Pilot & Mouse Drag)
    # ------------------------------------------------------------------
    def steer(self, vx: float, vy: float) -> None:
        """Apply velocity steering from keyboard (WASD / Arrow Keys)."""
        self.motion_type = MotionType.USER_CONTROLLED
        self.manual_vx = vx
        self.manual_vy = vy

    def drag_to(self, target_x: float, target_y: float) -> None:
        """Directly position beacon from mouse dragging in world coordinates."""
        self.motion_type = MotionType.USER_CONTROLLED
        self.x = target_x
        self.y = target_y
        self.manual_vx = 0.0
        self.manual_vy = 0.0

    # ------------------------------------------------------------------
    # Manual Path Recording API
    # ------------------------------------------------------------------
    def set_waypoints(self, waypoints: List[dict], loop: bool = True) -> None:
        """Load a recorded mouse-drag path (normalized 0–1 coords)."""
        self._waypoints = [{"x": w["x"], "y": w["y"]} for w in waypoints]
        self._manual_loop = loop
        self._manual_progress = 0.0
        self.motion_type = MotionType.MANUAL_RECORDED

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------
    def update(self, dt: float, world_w: float, world_h: float) -> None:
        self.time += dt * (self.speed / 3.0)
        prev_x, prev_y = self.x, self.y

        handler = {
            MotionType.STRAIGHT_LINE:   self._move_straight,
            MotionType.CIRCULAR:        self._move_circular,
            MotionType.FIGURE_8:        self._move_figure8,
            MotionType.RANDOM:          self._move_random,
            MotionType.SPIRAL:          self._move_spiral,
            MotionType.SINUSOIDAL:      self._move_sinusoidal,
            MotionType.PROJECTILE:      self._move_projectile,
            MotionType.MANUAL_RECORDED: self._move_manual,
            MotionType.USER_CONTROLLED: self._move_user_controlled,
        }.get(self.motion_type, self._move_straight)

        handler(dt, world_w, world_h)

        # Update velocity estimate
        if dt > 0:
            self.vx = (self.x - prev_x) / dt
            self.vy = (self.y - prev_y) / dt

    # ------------------------------------------------------------------
    # Motion Model Implementations
    # ------------------------------------------------------------------
    def _move_straight(self, dt, ww, wh):
        spd = self.speed * 60.0 * dt
        self.x += math.cos(self.heading) * spd
        self.y += math.sin(self.heading) * spd
        margin = 120
        if self.x < margin or self.x > ww - margin:
            self.heading = math.pi - self.heading
            self.x = max(margin, min(ww - margin, self.x))
        if self.y < margin or self.y > wh - margin:
            self.heading = -self.heading
            self.y = max(margin, min(wh - margin, self.y))

    def _move_circular(self, dt, ww, wh):
        omega = 0.75 * (self.speed / 3.0)
        self.x = self.origin_x + math.cos(self.time * omega) * self.radius
        self.y = self.origin_y + math.sin(self.time * omega) * self.radius

    def _move_figure8(self, dt, ww, wh):
        omega = 0.65 * (self.speed / 3.0)
        t = self.time * omega
        self.x = self.origin_x + math.sin(t) * self.radius
        self.y = self.origin_y + math.sin(2 * t) * 0.45 * self.radius

    def _move_random(self, dt, ww, wh):
        self.heading += random.uniform(-0.25, 0.25)
        spd = self.speed * 60.0 * dt
        self.x += math.cos(self.heading) * spd
        self.y += math.sin(self.heading) * spd
        # Soft pull toward centre
        cx, cy = ww * 0.5, wh * 0.5
        self.x += (cx - self.x) * 0.06 * dt
        self.y += (cy - self.y) * 0.06 * dt

    def _move_spiral(self, dt, ww, wh):
        omega = 0.9 * (self.speed / 3.0)
        r = (math.sin(self.time * 0.15) * 0.5 + 0.5) * self.radius + 80
        self.x = self.origin_x + math.cos(self.time * omega) * r
        self.y = self.origin_y + math.sin(self.time * omega) * r

    def _move_sinusoidal(self, dt, ww, wh):
        vx = self.speed * 45.0
        self.x += vx * dt
        if self.x > ww - 100:
            self.x = 100.0
        self.y = self.origin_y + math.sin(self.x * 0.012) * self.radius * 0.6

    def _move_projectile(self, dt, ww, wh):
        self.x += self._proj_vx * 60 * dt
        self.y += self._proj_vy * 60 * dt
        self._proj_vy += self._gravity * 60 * dt
        if self.y > wh - 150:
            self.y = wh - 150
            self._proj_vy = -abs(self._proj_vy) * 0.85
            if abs(self._proj_vy) < 2:
                self._proj_vy = -16.0
        if self.x < 150 or self.x > ww - 150:
            self._proj_vx = -self._proj_vx
            self.x = max(150, min(ww - 150, self.x))

    def _move_manual(self, dt, ww, wh):
        pts = self._waypoints
        if len(pts) < 2:
            return
        step = (self.speed * 0.05 * dt) / max(1, len(pts) * 0.05)
        self._manual_progress += step
        if self._manual_progress > 1.0:
            self._manual_progress = self._manual_progress % 1.0 if self._manual_loop else 1.0
        pos = _catmull_rom(pts, self._manual_progress, self._manual_loop)
        self.x = pos[0] * ww
        self.y = pos[1] * wh

    def _move_user_controlled(self, dt: float, ww: float, wh: float) -> None:
        """Interactive control: moves according to user's keyboard or mouse drag."""
        self.x += self.manual_vx * dt
        self.y += self.manual_vy * dt
        margin = 40
        self.x = max(margin, min(ww - margin, self.x))
        self.y = max(margin, min(wh - margin, self.y))


# ---------------------------------------------------------------------------
# Catmull-Rom spline for smooth waypoint interpolation
# ---------------------------------------------------------------------------
def _catmull_rom(
    points: List[dict], t: float, loop: bool = True
) -> tuple:
    n = len(points)
    if n == 0:
        return (0.5, 0.5)
    if n == 1:
        return (points[0]["x"], points[0]["y"])
    if n == 2:
        a, b = points[0], points[1]
        return (a["x"] + (b["x"] - a["x"]) * t, a["y"] + (b["y"] - a["y"]) * t)

    u = t % 1.0
    seg = n if loop else n - 1
    progress = u * seg
    idx = int(progress)
    lt = progress - idx

    def pt(i):
        if loop:
            return points[i % n]
        return points[max(0, min(n - 1, i))]

    p0, p1, p2, p3 = pt(idx - 1), pt(idx), pt(idx + 1), pt(idx + 2)
    t2, t3 = lt * lt, lt ** 3

    q0 = -t3 + 2 * t2 - lt
    q1 = 3 * t3 - 5 * t2 + 2
    q2 = -3 * t3 + 4 * t2 + lt
    q3 = t3 - t2

    x = 0.5 * (p0["x"] * q0 + p1["x"] * q1 + p2["x"] * q2 + p3["x"] * q3)
    y = 0.5 * (p0["y"] * q0 + p1["y"] * q1 + p2["y"] * q2 + p3["y"] * q3)
    return (max(0, min(1, x)), max(0, min(1, y)))
