# Core Data Types, Configuration & Infrastructure

## 1. Overview & Architectural Role

The `src/core/` package provides the **immutable foundational schemas, shared data transfer records, and system-wide configuration primitives** for the entire SADHA codebase. By centralizing all interface schemas here, circular imports between Module 1 and Module 2 are eliminated, establishing clear architectural boundaries.

---

## 2. Responsibilities & Boundaries

### What `src/core/` IS Responsible For:
- **Interface Contract Records**: Defining versioned, strongly-typed dataclasses (`FrameData`, `BeaconGroundTruth`) that mediate between the simulation environment and the tracking pipeline.
- **Domain Enumerations**: Standardizing valid operational modes (`MotionType`, `PlatformMotionType`, `TargetShape`, `AtmosphericCondition`, `LockState`, `SystemMode`).
- **Configuration Hierarchy**: Defining typed dataclasses for every configurable simulation parameter (`WorldConfig`, `CameraConfig`, `TargetConfig`, `PlatformConfig`, `DisturbanceConfig`, `SimulationConfig`) with defaults matching the SIH PS4 benchmark specification.
- **Scenario Presets**: Defining first-class configurations for the five standardized Benchmark-1 evaluation scenarios.
- **External Video Ingestion**: Frame-by-frame decoding and format normalization for recorded mission telemetry (`VideoPlayer`).

### What `src/core/` is NOT Responsible For:
- Does not integrate differential equations or execute graphics loops.
- Does not contain neural network layers or matrix algebra routines.

---

## 3. Package File Map

```
src/core/
├── __init__.py         # Package namespace exports
├── types.py            # Enums (MotionType, LockState, etc.) & benchmark targets
├── frame_data.py       # FrameData & BeaconGroundTruth interface contract
├── config.py           # SimulationConfig hierarchy & PRESET_SCENARIOS
├── video_player.py     # OpenCV-based external video stream player
└── README.md           # This documentation specification
```

---

## 4. Key Contracts & Types

### 4.1 `FrameData` (Module 1 $\rightarrow$ Module 2 Interface)
```python
@dataclass
class FrameData:
    frame_index: int
    timestamp: float
    image: np.ndarray                   # Grayscale sensor frame (H, W) uint8
    camera_pan_deg: float = 0.0         # Telemetry (simulated only)
    camera_tilt_deg: float = 0.0
    capture_rate_hz: float = 30.0
    disturbance_descriptor: List[float] # 8-D noise summary vector
    ground_truth: List[BeaconGroundTruth] # Oracle scoring only (never fed to tracking)
```

### 4.2 Benchmark Target Rubric (`BENCHMARK_TARGETS`)
```python
BENCHMARK_TARGETS = {
    "max_acquisition_time_sec": 2.0,     # Time to initial lock <= 2.0 s
    "max_tracking_error_px": 10.0,       # Mean tracking error <= 10.0 px
    "max_target_loss_pct": 5.0,          # Loss rate <= 5.0%
    "max_reacquisition_time_sec": 1.0,   # Re-acquisition time <= 1.0 s
    "min_processing_fps": 20.0,          # Processing throughput >= 20.0 FPS
}
```
