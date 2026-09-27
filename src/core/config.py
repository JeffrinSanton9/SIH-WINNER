"""
SADHA — Configuration & Predefined Scenario Presets
All user-configurable parameters with defaults matching SIH PS4 spec values.
Preset scenarios are first-class saved configs for Benchmark-1 evaluation.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, Any

from src.core.types import (
    MotionType, PlatformMotionType, TargetShape, AtmosphericCondition,
)


@dataclass
class WorldConfig:
    """World coordinate simulation canvas bounds.
    
    Subsystem: Module 1 (Physical World Simulation).
    Governs the spatial boundary in which beacon kinematics and platform
    trajectories are integrated. Beacons wrap or bounce at canvas edges.
    """
    width: int = 2400
    """World width in pixels.
    - Default: 2400 px
    - Valid Range: [2000, 10000] px (SIH PS4 specifies >= 2000 px)
    - Units: Pixels (world-space)
    - Affects: Horizontal spatial boundary and camera FOV footprint coverage.
    """

    height: int = 2400
    """World height in pixels.
    - Default: 2400 px
    - Valid Range: [2000, 10000] px (SIH PS4 specifies >= 2000 px)
    - Units: Pixels (world-space)
    - Affects: Vertical spatial boundary and camera FOV footprint coverage.
    """


@dataclass
class CameraConfig:
    """Virtual pan-tilt gimbal camera optical and kinematic specifications.
    
    Subsystem: Module 1 (Virtual Camera) & Module 2 (Closed-loop Gimbal Steering).
    Defines optical sensor resolution, angular field of view, 2-axis gimbal slew rates,
    and adaptive capture rate timer.
    """
    sensor_width: int = 640
    """Monochrome sensor horizontal resolution.
    - Default: 640 px
    - Valid Range: [320, 1920] px (Nominal 640x480 standard)
    - Units: Pixels
    - Affects: Rendered FrameData width, NN1 input tensor shape, optical pixel scale.
    """

    sensor_height: int = 480
    """Monochrome sensor vertical resolution.
    - Default: 480 px
    - Valid Range: [240, 1080] px
    - Units: Pixels
    - Affects: Rendered FrameData height, NN1 input tensor shape, optical pixel scale.
    """

    fov_h: float = 4.0
    """Horizontal field-of-view angular span.
    - Default: 4.0 deg
    - Valid Range: [1.0, 30.0] deg (Nominal FSOC coarse acquisition FOV)
    - Units: Degrees
    - Affects: Angular coverage per horizontal pixel (fov_h / sensor_width deg/px).
    """

    fov_v: float = 3.0
    """Vertical field-of-view angular span.
    - Default: 3.0 deg
    - Valid Range: [1.0, 30.0] deg
    - Units: Degrees
    - Affects: Angular coverage per vertical pixel (fov_v / sensor_height deg/px).
    """

    baseline_fps: float = 30.0
    """Nominal sensor frame acquisition frequency.
    - Default: 30.0 Hz
    - Valid Range: [15.0, 60.0] Hz (SIH PS4 standard baseline)
    - Units: Hertz (frames per second)
    - Affects: Nominal frame capture interval (1.0 / baseline_fps sec).
    """

    loss_ramp_fps: float = 60.0
    """Maximum ramped frame capture frequency activated upon target track loss.
    - Default: 60.0 Hz
    - Valid Range: [30.0, 120.0] Hz
    - Units: Hertz
    - [INTENTIONAL DESIGN DECISION]: Dynamic capture-rate ramping doubles temporal
      sampling density upon beacon loss, giving Hungarian assignment and Kalman filter
      twice as many opportunities per second to reacquire high-speed targets before
      they escape the narrow 4°x3° FOV, optimizing telemetry bandwidth dynamically.
    - Affects: Emergency capture interval during LockState.LOST conditions.
    """

    max_pan_speed: float = 5.0
    """Hard slew-rate limit for azimuth (horizontal) gimbal angular velocity.
    - Default: 5.0 deg/s
    - Valid Range: [1.0, 20.0] deg/s (SIH PS4 specified range 5-10 deg/s)
    - Units: Degrees per second
    - Affects: Clamps commanded rate commands in VirtualCamera.set_commanded_rates.
    """

    max_tilt_speed: float = 5.0
    """Hard slew-rate limit for elevation (vertical) gimbal angular velocity.
    - Default: 5.0 deg/s
    - Valid Range: [1.0, 20.0] deg/s (SIH PS4 specified range 5-10 deg/s)
    - Units: Degrees per second
    - Affects: Clamps commanded rate commands in VirtualCamera.set_commanded_rates.
    """

    initial_pan: float = 0.0
    """Initial horizontal gimbal boresight angle.
    - Default: 0.0 deg
    - Valid Range: [-180.0, 180.0] deg
    - Units: Degrees
    - Affects: Starting aim-point on world canvas.
    """

    initial_tilt: float = 0.0
    """Initial vertical gimbal boresight angle.
    - Default: 0.0 deg
    - Valid Range: [-90.0, 90.0] deg
    - Units: Degrees
    - Affects: Starting aim-point on world canvas.
    """

    monochrome: bool = True
    """Forces monochrome (single-channel 8-bit grayscale) optical output.
    - Default: True
    - Valid Range: {True, False}
    - Units: Boolean flag
    - Affects: Frame rendering depth matching operational 850/1550 nm FSOC cameras.
    """


@dataclass
class TargetConfig:
    """Optical beacon target kinematic and geometric configuration.
    
    Subsystem: Module 1 (Beacon Simulation) & Module 2 (Multi-Beacon Tracker).
    Controls beacon population, shape morphology, brightness, scale, trajectory pattern,
    and interactive steering behaviors.
    """
    count: int = 1
    """Number of simultaneous active optical beacons on the world canvas.
    - Default: 1
    - Valid Range: [1, 5] (Supported up to MAX_TERMINALS = 5)
    - [INTENTIONAL DESIGN DECISION]: Multi-beacon support allows testing multi-terminal
      gated association, Hungarian assignment, and crossing scenarios beyond the
      single-beacon minimum requirement.
    - Affects: Spawns N beacon instances with discrete terminal bindings.
    """

    shape: TargetShape = TargetShape.SQUARE
    """Geometric morphology of the beacon's optical spot on the sensor plane.
    - Default: TargetShape.SQUARE
    - Valid Range: {SQUARE, CIRCLE, TRIANGLE} (All 3 mandatory shapes)
    - Units: Categorical enum
    - Affects: Rasterization footprint and sub-pixel centroid profile in FrameData.
    """

    size: int = 10
    """Characteristic optical spot diameter / side-length.
    - Default: 10 px
    - Valid Range: [5, 20] px (SIH PS4 specified range 5-20 px)
    - Units: Pixels (sensor plane)
    - Affects: Bounding box size and diffraction halo convolution envelope.
    """

    motion_type: MotionType = MotionType.FIGURE_8
    """Active kinematic trajectory pattern for the beacon.
    - Default: MotionType.FIGURE_8
    - Valid Range: {STRAIGHT_LINE, CIRCULAR, FIGURE_8, RANDOM, SPIRAL, SINUSOIDAL,
      PROJECTILE, MANUAL_RECORDED, USER_CONTROLLED}
    - Units: Categorical enum
    - Affects: Kinematic equations of motion integrated per simulation step.
    """

    speed: float = 3.5
    """Velocity multiplier for autonomous beacon trajectories.
    - Default: 3.5
    - Valid Range: [0.5, 10.0]
    - Units: Dimensionless velocity scaling factor
    - Affects: Linear/angular step size per second along trajectory.
    """

    radius: float = 400.0
    """Spatial radius or major amplitude for cyclic motion patterns (Figure-8, Circular, Spiral).
    - Default: 400.0 px
    - Valid Range: [50.0, 1000.0] px
    - Units: Pixels (world-space)
    - Affects: Extent of orbit on world canvas.
    """

    manual_loop: bool = True
    """Whether custom mouse-drag recorded waypoints repeat continuously upon reaching end.
    - Default: True
    - Valid Range: {True, False}
    - Units: Boolean flag
    - Affects: Waypoint progression wrap-around in Beacon.update.
    """

    brightness: int = 255
    """Peak optical emission intensity for the beacon core.
    - Default: 255 (Full saturation)
    - Valid Range: [50, 255]
    - Units: 8-bit Grayscale Digital Number (DN)
    - Affects: Signal-to-Noise Ratio (SNR) against dark current and background noise.
    """


@dataclass
class PlatformConfig:
    """Mobile FSOC host platform vibration and base displacement dynamics.
    
    Subsystem: Module 1 (Platform Motion Simulation).
    Simulates base movement (e.g. ship mast, UAV gimbal mount, ground vehicle)
    that translates the camera physical origin independently of target motion.
    """
    enabled: bool = True
    """Enables or disables host platform movement.
    - Default: True
    - Valid Range: {True, False}
    - Affects: Adds platform offset to camera base anchor point.
    """

    motion_type: PlatformMotionType = PlatformMotionType.LINEAR
    """Pattern of host platform displacement.
    - Default: PlatformMotionType.LINEAR
    - Valid Range: {NONE, LINEAR, CIRCULAR, RANDOM, SPIRAL, FIGURE_8}
    - Affects: Base mount trajectory equations.
    """

    max_displacement: float = 8.0
    """Maximum platform displacement magnitude per step.
    - Default: 8.0 px/frame
    - Valid Range: [0.0, 20.0] px/frame (SIH PS4 specifies max +-20 px)
    - Units: Pixels per frame envelope scale
    - Affects: Physical translation amplitude of camera coordinate frame.
    """

    frequency: float = 0.04
    """Oscillation frequency for cyclic platform motion modes.
    - Default: 0.04 Hz
    - Valid Range: [0.01, 0.5] Hz
    - Units: Hertz
    - Affects: Temporal rate of platform attitude fluctuation.
    """


@dataclass
class SaltPepperConfig:
    """Impulsive Salt-and-Pepper noise configuration.
    
    Subsystem: Module 1 (Disturbance Engine).
    Simulates bit-flip sensor readout transmission errors and dead/hot pixels.
    """
    enabled: bool = False
    """Enables impulsive salt-and-pepper noise channel.
    - Default: False
    - Valid Range: {True, False}
    """

    density: float = 0.08
    """Proportion of corrupted image pixels set to 0 or 255.
    - Default: 0.08 (~8% of total pixels, testing up to 10% spec)
    - Valid Range: [0.0, 0.20]
    - Units: Fractional area (0.0 to 1.0)
    - Affects: Number of random saturated / zeroed pixels per frame.
    """


@dataclass
class GaussianNoiseConfig:
    """Additive Gaussian noise configuration.
    
    Subsystem: Module 1 (Disturbance Engine).
    Simulates thermal read noise and electronic amplifier noise in the intensity domain.
    """
    enabled: bool = False
    """Enables additive Gaussian noise channel.
    - Default: False
    - Valid Range: {True, False}
    """

    std_dev: float = 18.0
    """Standard deviation of Gaussian intensity perturbation.
    - Default: 18.0 DN
    - Valid Range: [0.0, 50.0] DN (Intensity-domain mapping of SIH 20 px figure)
    - Units: Grayscale Digital Numbers (DN)
    - [INTENTIONAL DESIGN DECISION]: In physical sensor modeling, detector thermal noise
      is an intensity-domain phenomenon (modulating brightness), whereas physical mount
      instability is a spatial phenomenon (shifting coordinates). Gaussian noise is mapped
      to pixel intensity variation, with spatial jitter handled by CameraJitterConfig.
    - Affects: Pixel intensity variance across the entire image.
    """


@dataclass
class PoissonNoiseConfig:
    """Signal-dependent Poisson shot noise configuration.
    
    Subsystem: Module 1 (Disturbance Engine).
    Simulates quantum photon arrival statistics where noise variance equals signal mean.
    """
    enabled: bool = False
    """Enables quantum photon shot noise.
    - Default: False
    - Valid Range: {True, False}
    """

    scale: float = 1.0
    """Scaling factor for Poisson noise strength.
    - Default: 1.0
    - Valid Range: [0.1, 5.0]
    - Units: Dimensionless scaling multiplier
    - Affects: Intensity-dependent variance on bright beacon regions.
    """


@dataclass
class CameraJitterConfig:
    """High-frequency spatial camera focal plane jitter configuration.
    
    Subsystem: Module 1 (Disturbance Engine).
    Simulates mechanical vibrations, wind buffeting, and gimbal servo ripple.
    """
    enabled: bool = False
    """Enables spatial camera coordinate jitter.
    - Default: False
    - Valid Range: {True, False}
    """

    amplitude: float = 6.0
    """Maximum spatial displacement of focal plane per frame.
    - Default: 6.0 px
    - Valid Range: [0.0, 20.0] px (SIH PS4 specifies max +-20 px)
    - Units: Pixels (sensor coordinate frame)
    - Affects: Instantaneous translation (jitter_x, jitter_y) applied to world_to_sensor projection.
    """


@dataclass
class DisturbanceConfig:
    """Composite environmental and sensor disturbances container.
    
    Subsystem: Module 1 (Disturbance Engine).
    Aggregates image noise, camera jitter, and atmospheric transmission degradation.
    """
    salt_pepper: SaltPepperConfig = field(default_factory=SaltPepperConfig)
    gaussian: GaussianNoiseConfig = field(default_factory=GaussianNoiseConfig)
    poisson: PoissonNoiseConfig = field(default_factory=PoissonNoiseConfig)
    camera_jitter: CameraJitterConfig = field(default_factory=CameraJitterConfig)
    atmospheric: AtmosphericCondition = AtmosphericCondition.CLEAR
    """Atmospheric transmission condition affecting optical visibility.
    - Default: AtmosphericCondition.CLEAR
    - Valid Range: {CLEAR, HAZE, FOG, RAIN, LOW_LIGHT}
    - Affects: Contrast attenuation factor, path radiance backscatter, and global transmission.
    """


@dataclass
class SimulationConfig:
    """Complete simulation configuration containing all subsystem parameters.
    
    Provides deep-copy capabilities and structured access for the WorldSimulation orchestrator.
    """
    world: WorldConfig = field(default_factory=WorldConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    platform: PlatformConfig = field(default_factory=PlatformConfig)
    disturbances: DisturbanceConfig = field(default_factory=DisturbanceConfig)

    def deep_copy(self) -> "SimulationConfig":
        """Produce an isolated deep clone of this configuration."""
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Predefined Benchmark-1 Scenarios
# ---------------------------------------------------------------------------

def _make_preset(name: str, description: str, overrides: Dict[str, Any]) -> Dict:
    """Build a preset entry (config is constructed lazily from overrides)."""
    return {"name": name, "description": description, "overrides": overrides}


PRESET_SCENARIOS = {
    "NOMINAL_LOW_DYNAMICS": _make_preset(
        name="Benchmark-1 · Nominal Low-Dynamics",
        description=(
            "Baseline straight-line beacon motion, static platform, clear atmosphere. "
            "Tests basic acquisition and steady-state tracking."
        ),
        overrides={
            "target.motion_type": MotionType.STRAIGHT_LINE,
            "target.speed": 2.5,
            "platform.enabled": False,
            "platform.motion_type": PlatformMotionType.NONE,
            "disturbances.atmospheric": AtmosphericCondition.CLEAR,
        },
    ),
    "HIGH_DYNAMICS_FIG8": _make_preset(
        name="Benchmark-1 · High-Dynamics Figure-8",
        description=(
            "Lemniscate figure-8 beacon trajectory with active linear platform motion "
            "and camera jitter. Stresses non-linear prediction."
        ),
        overrides={
            "target.motion_type": MotionType.FIGURE_8,
            "target.speed": 4.5,
            "target.radius": 450.0,
            "platform.enabled": True,
            "platform.motion_type": PlatformMotionType.LINEAR,
            "platform.max_displacement": 10.0,
            "disturbances.gaussian.enabled": True,
            "disturbances.gaussian.std_dev": 12.0,
            "disturbances.camera_jitter.enabled": True,
            "disturbances.camera_jitter.amplitude": 8.0,
        },
    ),
    "SEVERE_FOG_NOISE": _make_preset(
        name="Benchmark-1 · Severe Fog & Noise",
        description=(
            "Circular beacon under heavy fog, low-light conditions, and 8 % "
            "salt-and-pepper noise. Tests robustness under degraded visibility."
        ),
        overrides={
            "target.shape": TargetShape.CIRCLE,
            "target.motion_type": MotionType.CIRCULAR,
            "target.speed": 3.2,
            "target.size": 12,
            "platform.enabled": True,
            "platform.motion_type": PlatformMotionType.RANDOM,
            "platform.max_displacement": 6.0,
            "disturbances.salt_pepper.enabled": True,
            "disturbances.salt_pepper.density": 0.08,
            "disturbances.gaussian.enabled": True,
            "disturbances.gaussian.std_dev": 15.0,
            "disturbances.poisson.enabled": True,
            "disturbances.camera_jitter.enabled": True,
            "disturbances.camera_jitter.amplitude": 6.0,
            "disturbances.atmospheric": AtmosphericCondition.FOG,
        },
    ),
    "MULTI_BEACON_CROSSING": _make_preset(
        name="Benchmark-1 · Multi-Beacon Crossing",
        description=(
            "Two optical beacons on intersecting figure-8 paths with haze. "
            "Tests Hungarian data association under proximity."
        ),
        overrides={
            "target.count": 2,
            "target.motion_type": MotionType.FIGURE_8,
            "target.speed": 3.8,
            "platform.enabled": True,
            "platform.motion_type": PlatformMotionType.CIRCULAR,
            "platform.max_displacement": 6.0,
            "disturbances.gaussian.enabled": True,
            "disturbances.gaussian.std_dev": 10.0,
            "disturbances.atmospheric": AtmosphericCondition.HAZE,
        },
    ),
    "SPIRAL_AGGRESSIVE": _make_preset(
        name="Benchmark-1 · Spiral Acceleration & Rain",
        description=(
            "Expanding-contracting spiral with figure-8 platform motion, rain, "
            "and all noise channels active. Maximum stress test."
        ),
        overrides={
            "target.shape": TargetShape.TRIANGLE,
            "target.motion_type": MotionType.SPIRAL,
            "target.speed": 4.8,
            "target.size": 12,
            "platform.enabled": True,
            "platform.motion_type": PlatformMotionType.FIGURE_8,
            "platform.max_displacement": 14.0,
            "disturbances.salt_pepper.enabled": True,
            "disturbances.salt_pepper.density": 0.05,
            "disturbances.gaussian.enabled": True,
            "disturbances.gaussian.std_dev": 18.0,
            "disturbances.poisson.enabled": True,
            "disturbances.camera_jitter.enabled": True,
            "disturbances.camera_jitter.amplitude": 12.0,
            "disturbances.atmospheric": AtmosphericCondition.RAIN,
        },
    ),
}


def apply_preset(preset_key: str) -> SimulationConfig:
    """Create a SimulationConfig with a preset's overrides applied.

    The returned config is fully editable — preset originals are never mutated.
    """
    cfg = SimulationConfig()
    preset = PRESET_SCENARIOS[preset_key]

    for dotpath, value in preset["overrides"].items():
        parts = dotpath.split(".")
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    return cfg
