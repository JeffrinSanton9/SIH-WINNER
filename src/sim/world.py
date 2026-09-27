"""
SADHA — Module 1: World Simulation Orchestrator
Ties together beacons, platform, camera, and disturbances.
Generates FrameData packets per the output interface contract.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from src.core.types import TargetShape, MotionType, AtmosphericCondition
from src.core.config import SimulationConfig
from src.core.frame_data import FrameData, BeaconGroundTruth
from src.sim.beacon import Beacon
from src.sim.platform import PlatformMotion
from src.sim.virtual_camera import VirtualCamera
from src.sim.disturbances import DisturbanceEngine


class WorldSimulation:
    """Module 1 top-level orchestrator.

    Responsibilities:
      • Owns every scene entity (beacons, platform, camera, disturbances).
      • Advances the simulation by one time-step.
      • Renders a monochrome sensor frame and emits a FrameData packet.
      • Exposes command channels for Module 2 to drive the camera.
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.cfg = config or SimulationConfig()
        self.world_w = self.cfg.world.width
        self.world_h = self.cfg.world.height
        self.frame_index = 0
        self.sim_time = 0.0

        # ---- Sub-systems ----
        self.beacons: List[Beacon] = []
        self._init_beacons()

        self.platform = PlatformMotion(
            enabled=self.cfg.platform.enabled,
            motion_type=self.cfg.platform.motion_type,
            max_displacement=self.cfg.platform.max_displacement,
            frequency=self.cfg.platform.frequency,
        )

        self.camera = VirtualCamera(
            sensor_width=self.cfg.camera.sensor_width,
            sensor_height=self.cfg.camera.sensor_height,
            fov_h=self.cfg.camera.fov_h,
            fov_v=self.cfg.camera.fov_v,
            baseline_fps=self.cfg.camera.baseline_fps,
            loss_ramp_fps=self.cfg.camera.loss_ramp_fps,
            max_pan_speed=self.cfg.camera.max_pan_speed,
            max_tilt_speed=self.cfg.camera.max_tilt_speed,
            initial_pan=self.cfg.camera.initial_pan,
            initial_tilt=self.cfg.camera.initial_tilt,
        )

        dc = self.cfg.disturbances
        self.disturbances = DisturbanceEngine(
            salt_pepper_enabled=dc.salt_pepper.enabled,
            salt_pepper_density=dc.salt_pepper.density,
            gaussian_enabled=dc.gaussian.enabled,
            gaussian_std=dc.gaussian.std_dev,
            poisson_enabled=dc.poisson.enabled,
            poisson_scale=dc.poisson.scale,
            jitter_enabled=dc.camera_jitter.enabled,
            jitter_amplitude=dc.camera_jitter.amplitude,
            atmospheric=dc.atmospheric,
        )

        # ---- Mouse-drag path recording state ----
        self._recording_path = False
        self._recorded_waypoints: List[dict] = []

    # ------------------------------------------------------------------
    # Beacon initialisation
    # ------------------------------------------------------------------
    def _init_beacons(self) -> None:
        tc = self.cfg.target
        self.beacons = []
        for i in range(tc.count):
            angle = (i / max(1, tc.count)) * 2 * math.pi
            ox = self.world_w * 0.5 + (math.cos(angle) * 200 if tc.count > 1 else 0)
            oy = self.world_h * 0.5 + (math.sin(angle) * 200 if tc.count > 1 else 0)
            self.beacons.append(Beacon(
                beacon_id=i + 1,
                shape=tc.shape,
                size=tc.size,
                motion_type=tc.motion_type,
                speed=tc.speed * (1.0 + i * 0.15),
                radius=tc.radius,
                x=ox, y=oy,
                brightness=tc.brightness,
                phase_offset=i * 1.5,
                manual_loop=tc.manual_loop,
            ))

    # ------------------------------------------------------------------
    # Reconfiguration (apply new config without restarting app)
    # ------------------------------------------------------------------
    def reconfigure(self, config: SimulationConfig) -> None:
        self.cfg = config
        self.world_w = config.world.width
        self.world_h = config.world.height
        self.frame_index = 0
        self.sim_time = 0.0
        self._init_beacons()
        self.platform = PlatformMotion(
            enabled=config.platform.enabled,
            motion_type=config.platform.motion_type,
            max_displacement=config.platform.max_displacement,
            frequency=config.platform.frequency,
        )
        cam = config.camera
        self.camera = VirtualCamera(
            sensor_width=cam.sensor_width, sensor_height=cam.sensor_height,
            fov_h=cam.fov_h, fov_v=cam.fov_v,
            baseline_fps=cam.baseline_fps, loss_ramp_fps=cam.loss_ramp_fps,
            max_pan_speed=cam.max_pan_speed, max_tilt_speed=cam.max_tilt_speed,
            initial_pan=cam.initial_pan, initial_tilt=cam.initial_tilt,
        )
        dc = config.disturbances
        self.disturbances = DisturbanceEngine(
            salt_pepper_enabled=dc.salt_pepper.enabled,
            salt_pepper_density=dc.salt_pepper.density,
            gaussian_enabled=dc.gaussian.enabled,
            gaussian_std=dc.gaussian.std_dev,
            poisson_enabled=dc.poisson.enabled,
            poisson_scale=dc.poisson.scale,
            jitter_enabled=dc.camera_jitter.enabled,
            jitter_amplitude=dc.camera_jitter.amplitude,
            atmospheric=dc.atmospheric,
        )

    # ------------------------------------------------------------------
    # Mouse-drag path recording
    # ------------------------------------------------------------------
    def start_path_recording(self) -> None:
        self._recording_path = True
        self._recorded_waypoints = []

    def add_path_waypoint(self, norm_x: float, norm_y: float) -> None:
        if self._recording_path:
            self._recorded_waypoints.append({"x": norm_x, "y": norm_y})

    def finish_path_recording(self, loop: bool = True) -> bool:
        self._recording_path = False
        if len(self._recorded_waypoints) >= 2:
            for b in self.beacons:
                b.set_waypoints(self._recorded_waypoints, loop)
            return True
        return False

    # ------------------------------------------------------------------
    # Interactive real-time user steering & dragging
    # ------------------------------------------------------------------
    def steer_primary_beacon(self, vx: float, vy: float) -> None:
        """Drive primary beacon via real-time velocity steering (WASD / Arrows)."""
        if self.beacons:
            self.beacons[0].steer(vx, vy)

    def drag_primary_beacon(self, wx: float, wy: float) -> None:
        """Directly position primary beacon via mouse dragging in world coordinates."""
        if self.beacons:
            self.beacons[0].drag_to(wx, wy)

    # ------------------------------------------------------------------
    # Simulation step (Interface Boundary: Module 1 → Module 2)
    # ------------------------------------------------------------------
    def step(self, dt: float) -> Optional[FrameData]:
        """Advance the physical world simulation by time delta dt.

        Parameters
        ----------
        dt : float
            Elapsed simulation time delta in seconds (typically 1/60 s for 60 Hz physics).

        Returns
        -------
        Optional[FrameData]
            A fully populated `FrameData` packet if the camera's internal capture
            timer fired this tick, or `None` during intermediate sub-frame ticks
            when the camera shutter is closed.

        Side Effects
        ------------
        - Updates continuous kinematics for all active beacons in world coordinates.
        - Advances host platform vibration offsets.
        - Advances high-frequency spatial camera jitter translations.
        - Integrates 2-axis camera pan/tilt gimbal rates and clamps to slew limits.
        - Advances simulation time accumulator and increments frame_index.

        Interface Boundary
        ------------------
        This function is the primary emitter for Module 1. When a FrameData packet is returned,
        it is dispatched directly into Module 2's tracking pipeline.
        """
        self.sim_time += dt

        # 1. Platform dynamics
        self.platform.update(dt)

        # 2. Beacon kinematics
        for b in self.beacons:
            b.update(dt, self.world_w, self.world_h)

        # 3. Spatial jitter
        self.disturbances.update(dt)

        # 4. Camera gimbal + capture timer
        self.camera.update(
            dt,
            self.platform.offset_x,
            self.platform.offset_y,
            self.world_w,
            self.world_h,
        )

        if not self.camera.should_capture:
            return None  # sub-frame tick — no frame emitted

        self.frame_index += 1
        return self._render_frame()

    # ------------------------------------------------------------------
    # Monochrome sensor frame rendering
    # ------------------------------------------------------------------
    def _render_frame(self) -> FrameData:
        sw = self.camera.sensor_width
        sh = self.camera.sensor_height

        # Black background — optical dark current
        frame = np.full((sh, sw), 5, dtype=np.uint8)

        jx = self.disturbances.jitter_x
        jy = self.disturbances.jitter_y
        ground_truth: List[BeaconGroundTruth] = []

        for b in self.beacons:
            u, v, in_fov = self.camera.world_to_sensor(b.x, b.y, jx, jy)
            ground_truth.append(BeaconGroundTruth(
                beacon_id=b.id, u=u, v=v,
                world_x=b.x, world_y=b.y, in_fov=in_fov,
            ))

            if not in_fov:
                continue

            iu, iv = int(round(u)), int(round(v))
            half = b.size // 2

            if b.shape == TargetShape.SQUARE:
                y0 = max(0, iv - half)
                y1 = min(sh, iv + half + 1)
                x0 = max(0, iu - half)
                x1 = min(sw, iu + half + 1)
                frame[y0:y1, x0:x1] = b.brightness

            elif b.shape == TargetShape.CIRCLE:
                rr = half
                for dy in range(-rr, rr + 1):
                    for dx in range(-rr, rr + 1):
                        if dx * dx + dy * dy <= rr * rr:
                            py, px = iv + dy, iu + dx
                            if 0 <= py < sh and 0 <= px < sw:
                                frame[py, px] = b.brightness

            elif b.shape == TargetShape.TRIANGLE:
                for dy in range(-half, half + 1):
                    span = int(half * (1.0 - (dy + half) / max(1, 2 * half)))
                    for dx in range(-span, span + 1):
                        py, px = iv + dy, iu + dx
                        if 0 <= py < sh and 0 <= px < sw:
                            frame[py, px] = b.brightness

            # Airy-disk diffraction halo (optical point-spread function)
            halo_r = int(half * 2.5)
            for dy in range(-halo_r, halo_r + 1):
                for dx in range(-halo_r, halo_r + 1):
                    d = math.sqrt(dx * dx + dy * dy)
                    if d < half * 0.5 or d > halo_r:
                        continue
                    py, px = iv + dy, iu + dx
                    if 0 <= py < sh and 0 <= px < sw:
                        falloff = max(0, 1.0 - d / halo_r)
                        glow = int(b.brightness * 0.35 * falloff * falloff)
                        frame[py, px] = min(255, int(frame[py, px]) + glow)

        # Apply intensity-domain disturbances
        frame = self.disturbances.apply(frame)

        return FrameData(
            frame_index=self.frame_index,
            timestamp=self.sim_time,
            image=frame,
            camera_pan_deg=self.camera.pan,
            camera_tilt_deg=self.camera.tilt,
            capture_rate_hz=self.camera.current_fps,
            disturbance_descriptor=self.disturbances.get_descriptor(),
            ground_truth=ground_truth,
        )
