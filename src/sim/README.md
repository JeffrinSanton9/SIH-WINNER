# Module 1: Physical World & Virtual Camera Simulation

## 1. Overview & Architectural Role

Module 1 serves as the **high-fidelity physical simulation and optical sensor environment** for SADHA (Smart Adaptive Disturbance-aware Hybrid Acquisition). It models continuous-space optical beacon dynamics, host platform disturbances, atmospheric degradation, optical blur (point-spread function), and a 2-axis gimballed virtual camera with hard kinematic constraints.

Module 1 acts as an empirical testbed and ground-truth oracle. It operates independently of any neural networks or tracking algorithms, generating synthetic optical frames that emulate real-world free-space optical communication (FSOC) sensor telemetry.

---

## 2. Responsibilities & Boundaries

### What Module 1 IS Responsible For:
- **Continuous World Simulation**: Simulating a $2400 \times 2400$ pixel world canvas where target beacons navigate along complex kinematic trajectories.
- **Beacon Kinematics**: Propagating multi-beacon equations of motion (Lemniscate figure-8, expanding/contracting spiral, circular orbits, ballistic projectile dynamics with gravity/reflection, linear trajectories, and recorded mouse-drag paths).
- **Mobile Host Platform Vibrations**: Generating multi-axis base displacement (linear, circular, harmonic figure-8, random walk) simulating vehicle/ship mast vibrations up to $\pm 20$ px/frame.
- **Environmental & Optical Noise**: Applying intensity-domain Gaussian thermal noise, Poisson photon shot noise, impulsive salt-and-pepper noise, atmospheric extinction/backscatter (fog, haze, rain), and sub-pixel optical diffraction halos.
- **Focal Plane Spatial Jitter**: Simulating high-frequency mechanical mount vibrations up to $\pm 20$ px.
- **Virtual Gimbal Camera**: Integrating 2-axis angular pan/tilt gimbal rates, clamping commands to physical slew limits ($5^\circ/\text{s}$), and projecting world coordinates into sensor pixel space ($640 \times 480$, $4^\circ \times 3^\circ$ FOV).
- **Adaptive Frame Capture Timer**: Emitting discrete `FrameData` packets at 30–60 Hz based on the commanded capture rate.

### What Module 1 is NOT Responsible For:
- **Target Detection & Centroiding**: Handled strictly by Module 2.1 (NN1).
- **State Filtering & Temporal Association**: Handled strictly by Module 2.2 (Kalman Filter & Hungarian algorithm).
- **Lookahead Trajectory Prediction**: Handled strictly by Module 2.3 (NN2).
- **Gimbal Steering Decisions**: Module 1 accepts rate commands via `set_commanded_rates()`, but never calculates where the camera should point.

---

## 3. Interface Contracts

### 3.1 Input Contract (Module 2 $\rightarrow$ Module 1)
Module 1 exposes two thread-safe command methods called by Module 2's closed-loop controller:

```python
def set_commanded_rates(self, pan_rate: float, tilt_rate: float) -> None
```
- **`pan_rate`** (`float`): Azimuth angular rate command in degrees per second ($\pm 5.0^\circ/\text{s}$).
- **`tilt_rate`** (`float`): Elevation angular rate command in degrees per second ($\pm 5.0^\circ/\text{s}$).
- **Behavior**: Hard-clamped to `[-max_pan_speed, max_pan_speed]` and `[-max_tilt_speed, max_tilt_speed]`, then blended with a 0.4 lerp factor to model motor inertia.

```python
def set_capture_rate(self, hz: float) -> None
```
- **`hz`** (`float`): Target capture rate in Hertz ($\in [30.0, 60.0]$ Hz).
- **Behavior**: Adjusts the internal accumulator timer interval ($1.0 / \text{hz}$) to dynamically double sampling frequency during target loss recovery.

### 3.2 Output Contract (Module 1 $\rightarrow$ Module 2)
On every camera acquisition tick, `WorldSimulation.step(dt)` emits a structured `FrameData` packet:

| Field | Type | Description |
|---|---|---|
| `frame_index` | `int` | Strictly monotonically increasing frame sequence number |
| `timestamp` | `float` | Simulation elapsed time in seconds |
| `image` | `np.ndarray` | $480 \times 640$ 8-bit unsigned integer monochrome sensor frame (`uint8`) |
| `camera_pan_deg` | `float` | Current camera horizontal gimbal angle (degrees) |
| `camera_tilt_deg` | `float` | Current camera vertical gimbal angle (degrees) |
| `capture_rate_hz` | `float` | Current acquisition frequency in Hertz |
| `disturbance_descriptor` | `List[float]` | 8-D environmental noise summary vector |
| `ground_truth` | `List[BeaconGroundTruth]` | **Oracle evaluation data only** (beacon ID, sensor $u, v$, world $x, y$, in-FOV flag). *Never accessible to tracking models.* |

---

## 4. Module Architecture & File Map

```
src/sim/
├── __init__.py           # Package namespace definition
├── world.py              # WorldSimulation orchestrator (entity manager & step engine)
├── virtual_camera.py     # 2-axis gimbal kinematics, projection matrix, FOV frustum
├── beacon.py             # Beacon entity with 8 distinct kinematic equations of motion
├── platform.py           # Mobile vehicle / mast displacement generator
├── disturbances.py       # Intensity-domain noise, atmospheric models, and spatial jitter
└── README.md             # This documentation specification
```

---

## 5. Intentional Design Decisions & Deviations

1. **Separation of Intensity Noise vs. Spatial Jitter**:
   - Gaussian and Poisson noise are implemented strictly in the **intensity domain** (Digital Numbers 0–255), representing detector dark current and photon shot noise.
   - Sensor vibrations are modeled separately as **spatial focal plane jitter** ($\Delta x, \Delta y$), preserving geometric consistency.
2. **Extended Kinematic Motion Library**:
   - Beyond the four baseline motion types in the problem specification, Module 1 adds:
     - *Lemniscate Figure-8*: Models Bernoulli inflection curves.
     - *Ballistic Projectile*: Gravitational acceleration with elastic ground reflection.
     - *Expanding/Contracting Spiral*: Continuous multi-scale radius acceleration.
     - *Manual Recorded Drag*: Interactive waypoint recording normalized to $[0, 1]$.
     - *Interactive Keyboard Steering (WASD)*: Live human-in-the-loop pilot control.
3. **Diffraction Halo Modeling (Airy Disk)**:
   - Emulates optical point-spread function (PSF) diffraction with quadratic falloff, providing realistic sub-pixel gradient profiles for the NN1 sub-pixel centroid estimator.
