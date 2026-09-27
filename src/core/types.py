"""
SADHA — Core Type Definitions, Enumerations, and System Constants
Aligned with SIH Problem Statement 4 mandatory parameters.
"""

from enum import Enum


class MotionType(Enum):
    """Beacon target motion patterns (≥4 mandatory, 4 enhancement)."""
    STRAIGHT_LINE = "straight_line"
    CIRCULAR = "circular"
    FIGURE_8 = "figure_8"
    RANDOM = "random"
    # Enhancements (flag in technical report)
    SPIRAL = "spiral"
    SINUSOIDAL = "sinusoidal"
    PROJECTILE = "projectile"
    MANUAL_RECORDED = "manual_recorded"
    USER_CONTROLLED = "user_controlled"


class PlatformMotionType(Enum):
    """Mobile platform base motion patterns."""
    NONE = "none"
    LINEAR = "linear"
    CIRCULAR = "circular"
    RANDOM = "random"
    SPIRAL = "spiral"
    FIGURE_8 = "figure_8"


class TargetShape(Enum):
    """Optical beacon shape geometry."""
    SQUARE = "square"
    CIRCLE = "circle"
    TRIANGLE = "triangle"


class AtmosphericCondition(Enum):
    """Atmospheric degradation conditions affecting contrast & brightness."""
    CLEAR = "clear"
    HAZE = "haze"
    FOG = "fog"
    RAIN = "rain"
    LOW_LIGHT = "low_light"


class LockState(Enum):
    """Tracking lock state for the coarse alignment system."""
    LOCKED = "locked"
    LOST = "lost"
    SEARCHING = "searching"


class SystemMode(Enum):
    """Application operating mode — simulated scene or external video input."""
    SIMULATED = "simulated"
    EXTERNAL_VIDEO = "external_video"


# ---------------------------------------------------------------------------
# Performance benchmark targets — SIH PS4 scoring rubric
# ---------------------------------------------------------------------------
BENCHMARK_TARGETS = {
    "max_acquisition_time_sec": 2.0,
    "max_tracking_error_px": 10.0,
    "max_target_loss_pct": 5.0,
    "max_reacquisition_time_sec": 1.0,
    "min_processing_fps": 20.0,
}
