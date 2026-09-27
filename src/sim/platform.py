"""
SADHA — Mobile Platform Motion Simulation
Generates dynamic base displacement for the mobile FSOC terminal.
Separate from target beacon motion — both apply simultaneously.
"""

from __future__ import annotations

import math
import random

from src.core.types import PlatformMotionType


class PlatformMotion:
    """Simulates platform/vehicle base motion that shifts the camera mounting point."""

    def __init__(
        self,
        enabled: bool = True,
        motion_type: PlatformMotionType = PlatformMotionType.LINEAR,
        max_displacement: float = 8.0,
        frequency: float = 0.04,
    ):
        self.enabled = enabled
        self.motion_type = motion_type
        self.max_displacement = max_displacement
        self.frequency = frequency

        # Current platform offset from nominal position (px)
        self.offset_x = 0.0
        self.offset_y = 0.0

        # Internals
        self._time = 0.0
        self._random_angle = random.uniform(0, 2 * math.pi)

    def update(self, dt: float) -> None:
        if not self.enabled or self.motion_type == PlatformMotionType.NONE:
            self.offset_x = 0.0
            self.offset_y = 0.0
            return

        self._time += dt
        amp = self.max_displacement * 15.0  # spatial envelope scale
        t_phase = self._time * self.frequency * 2 * math.pi

        if self.motion_type == PlatformMotionType.LINEAR:
            self.offset_x = math.sin(t_phase) * amp
            self.offset_y = math.cos(t_phase) * amp * 0.4

        elif self.motion_type == PlatformMotionType.CIRCULAR:
            self.offset_x = math.cos(t_phase) * amp
            self.offset_y = math.sin(t_phase) * amp

        elif self.motion_type == PlatformMotionType.FIGURE_8:
            self.offset_x = math.sin(t_phase) * amp
            self.offset_y = math.sin(2 * t_phase) * 0.5 * amp

        elif self.motion_type == PlatformMotionType.SPIRAL:
            r = (math.sin(self._time * 0.1) * 0.5 + 0.5) * amp
            self.offset_x = math.cos(t_phase) * r
            self.offset_y = math.sin(t_phase) * r

        elif self.motion_type == PlatformMotionType.RANDOM:
            self._random_angle += random.uniform(-0.3, 0.3)
            target_x = math.cos(self._random_angle) * amp
            target_y = math.sin(self._random_angle) * amp
            self.offset_x += (target_x - self.offset_x) * 0.05
            self.offset_y += (target_y - self.offset_y) * 0.05
