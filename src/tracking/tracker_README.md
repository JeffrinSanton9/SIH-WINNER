# Module 2.2: Multi-Beacon Gated Assignment & Kalman Filter Tracking Engine

## 1. Overview & Architectural Role

Module 2.2 forms the **temporal state estimation and multi-target tracking backbone** of SADHA. Positioned directly between the stateless detection stage (NN1) and the trajectory prediction stage (NN2), it solves the multi-target correspondence problem, maintains recursive kinematic state estimates $[x, y, v_x, v_y]^T$, manages track lifecycles, and generates closed-loop pan/tilt rate steering commands for the physical gimbal.

---

## 2. Responsibilities & Boundaries

### What Module 2.2 IS Responsible For:
- **Recursive State Estimation**: Filtering measurement noise and estimating continuous velocities using per-track discrete-time linear Kalman filters.
- **Optimal Bipartite Data Association**: Constructing Euclidean distance cost matrices and solving the global minimum-cost pairing via the Hungarian (Munkres) algorithm.
- **Spatial Distance Gating**: Enforcing a $160\text{ px}$ distance threshold to reject physically impossible associations during extreme target maneuvers.
- **Two-Tier Loss & Retirement Lifecycle**:
  - *Tier 1 Loss Trigger*: Marking a track as `LockState.LOST` after 3 consecutive missed frames.
  - *Tier 2 Track Retirement*: Deleting the track and freeing its hardware terminal after 1.0 second of continuous absence.
- **Multi-Terminal Pool Management**: Allocating dedicated, isolated virtual terminal IDs ($0 \dots 4$) to active tracks.
- **Closed-Loop Camera Steering Commands**: Computing proportional-plus-velocity feedforward rate commands $(\dot{\theta}_{\text{pan}}, \dot{\theta}_{\text{tilt}})$ to guide the virtual camera boresight toward the tracked beacon.

### What Module 2.2 is NOT Responsible For:
- **Raw Pixel Processing**: Image decoding, filtering, and centroid extraction (handled strictly by NN1).
- **Non-Linear Sequence Forecasting**: Deep GRU lookahead prediction across long horizons (handled by NN2).
- **Frame Rendering & Camera Kinematics**: Simulating physical motor slew and image projection (handled by Module 1).

---

## 3. Interface Contracts

### 3.1 Input Contract (NN1 $\rightarrow$ Tracker)
Consumes an `NN1Result` record emitted by Module 2.1 on every frame:

```python
def update(self, nn1: NN1Result) -> TrackerResult
```

### 3.2 Output Contract (Tracker $\rightarrow$ NN2 & GUI)
Returns a structured `TrackerResult` record containing full temporal history and state:

```python
@dataclass
class TrackState:
    track_id: int
    terminal_id: int
    position: Tuple[float, float]               # Filtered (x, y) in sensor px
    velocity: Tuple[float, float]               # Estimated (vx, vy) in px/frame
    predicted_position: Tuple[float, float]     # 1-frame-ahead kinematic projection
    frame_index: int
    timestamp: float
    matched: bool                               # True if matched to a detection this frame
    consecutive_missed_frames: int              # Miss counter for Tier-1 loss trigger
    time_since_last_match: float                # Elapsed seconds for Tier-2 retirement
    lock_state: LockState                       # SEARCHING, LOCKED, or LOST
    matched_detection_index: int
    matched_confidence: float
    disturbance_descriptor: Optional[List[float]]
    nn1_raw_history: List[Tuple[float, float, float]]  # Sliding window of (x, y, t) for NN2
```

### 3.3 Steering Command Contract (Tracker $\rightarrow$ Virtual Camera)
Generates per-terminal rate commands:

```python
def compute_pan_tilt_commands(self) -> Dict[int, Tuple[float, float]]
```
Returns a mapping `{terminal_id: (pan_rate_deg_s, tilt_rate_deg_s)}`.
- Pixel offset from boresight: $e_x = x_{\text{pred}} - x_{\text{center}}$, $e_y = y_{\text{pred}} - y_{\text{center}}$
- Control law:
  $$\dot{\theta}_{\text{pan}} = (7.5 \cdot e_x + 0.9 \cdot v_x) \cdot K_{\text{deg/px\_h}}$$
  $$\dot{\theta}_{\text{tilt}} = (7.5 \cdot e_y + 0.9 \cdot v_y) \cdot K_{\text{deg/px\_v}}$$

---

## 4. Key Architectural Rationale

### 1. Two-Tier Loss vs. Retirement Architecture
- **Loss Trigger ($N = 3\text{ frames}$)**: Provides high-agility reaction (~100 ms at 30 Hz) to momentary optical dropouts (beam scintillation, dust puffs, high-G turns), instantly alerting the gimbal controller and triggering the capture-rate ramp.
- **Track Retirement ($\Delta t = 1.0\text{ s}$)**: Track deletion and terminal reallocation are deliberately measured in **continuous elapsed physical time (seconds)** rather than frame count. Because capture rate dynamically ramps between 30 Hz and 60 Hz during target loss, a frame-based retirement threshold would expire twice as fast at 60 Hz as at 30 Hz. Measuring retirement in elapsed seconds guarantees a stable, invariant physical window for target re-acquisition.

### 2. Distance-Gating Threshold ($160\text{ px}$)
While typical tracking gates in stationary cameras are 30–50 px, mobile FSOC terminals experience rapid angular accelerations compounded by platform vibration, causing apparent frame-to-frame displacements of up to 100–120 px at 30 Hz. Setting the gate threshold to $160\text{ px}$ prevents premature track divergence during aggressive maneuvers while still rejecting false associations from distant optical noise spikes.

### 3. Dedicated One-Terminal-Per-Beacon Binding
In operational FSOC systems, a physical pan/tilt coarse alignment gimbal has a single optical line-of-sight and cannot simultaneously point toward multiple dispersed targets. Binding each active track to a dedicated virtual terminal ID ($0 \dots 4$) from an available pool isolates control channels, preventing multi-target pointing conflicts and enabling seamless terminal switching.

---

## 5. Module Architecture & File Map

```
src/tracking/
├── tracker_types.py   # TrackState, AssignmentResult, TrackerResult, LockState
├── kalman_filter.py   # Discrete-time linear Kalman filter (frame-unit coords)
├── hungarian.py       # Cost matrix construction, gating, SciPy/Munkres solver
├── tracker.py         # MultiBeaconTracker orchestrator & pan/tilt command engine
└── tracker_README.md  # This documentation specification
```
