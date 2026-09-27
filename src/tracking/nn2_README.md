# Module 2.3: NN2 Motion Prediction & Ground-Truth Confidence Engine

## 1. Overview & Architectural Role

Module 2.3 (NN2) provides **predictive lookahead trajectory forecasting** and **empirical confidence scoring** for SADHA. While the Kalman filter in Module 2.2 provides smooth 1-frame kinematic filtering under linear constant-velocity assumptions, NN2 captures complex non-linear motion patterns (such as Lemniscate figure-8 inflection curves and expanding spirals) by conditioning a recurrent Gated Recurrent Unit (GRU) network on both kinematic trajectory history and environmental disturbance descriptors.

Furthermore, NN2 implements an **objective ground-truth confidence scoring engine** that quantifies true predictive efficacy by comparing past forecasts against future sensor observations as they arrive.

---

## 2. Responsibilities & Boundaries

### What NN2 IS Responsible For:
- **Non-Linear Trajectory Forecasting**: Predicting beacon sensor coordinates $N$ frames ahead ($H = 1$ to $5$ frames) using deep recurrent sequence modeling.
- **Disturbance-Conditioned Recurrent Inference**: Ingesting an 8-dimensional disturbance summary vector concatenated with normalized kinematic velocities $[x, y, \Delta x, \Delta y]$ at every recurrent time-step.
- **Ensemble Trajectory Blending**: Blending predictions across three complementary mechanisms:
  - *GRU Neural Prediction* (disturbance-aware, non-linear patterns)
  - *Classical Quadratic Extrapolation* (short-window local curvature fitting)
  - *Kalman 1-Ahead Projection* (kinematic smoothing baseline)
- **Empirical Confidence Scoring**: Buffering past forecasts and computing true Euclidean error metrics when target frames arrive from the sensor pipeline.
- **Classical Extrapolation Fallback**: Automatic, seamless fallback to numerical quadratic curve-fitting when PyTorch or pre-trained weights are unavailable.

### What NN2 is NOT Responsible For:
- **Raw Sensor Detection**: Handled strictly by NN1.
- **Track Spawning & Data Association**: Handled strictly by the Hungarian layer in Module 2.2.
- **Primary Camera Rate Commands**: The primary gimbal closed-loop rate commands are driven by the Kalman filter in Module 2.2; NN2 predictions are used for trajectory visualization, feedforward lookahead, and predictive loss recovery.

---

## 3. Interface Contracts

### 3.1 Input Contract (Tracker $\rightarrow$ NN2)
NN2 consumes the output of Module 2.2:

```python
def update(
    self,
    tracker_result: TrackerResult,
    frame_index: int,
    timestamp: float,
) -> NN2Result
```
- **`tracker_result`**: Contains per-track `nn1_raw_history` (sliding window of raw $(x, y, t)$ coordinates) and the current frame's `disturbance_descriptor`.
- **`frame_index`**: Current frame sequence index.
- **`timestamp`**: Current mission elapsed time.

### 3.2 Output Contract (NN2 $\rightarrow$ Application / Analytics)
Returns an `NN2Result` record containing forecasts and matured confidence scores:

```python
@dataclass(slots=True)
class NN2Prediction:
    track_id: int
    predicted_position: Tuple[float, float]   # (x, y) in sensor pixel coords
    frame_index: int                          # Frame prediction was generated on
    target_frame_index: int                   # Target frame index in the future
    timestamp: float

@dataclass(slots=True)
class NN2ConfidenceScore:
    track_id: int
    frame_index: int                          # Matured target frame index
    predicted_x: float                        # What NN2 previously predicted
    predicted_y: float
    actual_x: float                           # What NN1 actually observed
    actual_y: float
    error_px: float                           # Euclidean distance (px)
    confidence: float                         # Empirical score in [0.0, 1.0]
    threshold_px: float = 15.0                # Tracking error tolerance threshold

@dataclass(slots=True)
class NN2Result:
    frame_index: int
    timestamp: float
    predictions: List[NN2Prediction]
    confidence_scores: List[NN2ConfidenceScore]
    ran_this_frame: bool
```

---

## 4. Key Architectural Rationale

### 1. Empirical Ground-Truth Confidence Metric
Self-reported neural network confidence metrics (such as softmax probabilities or latent Gaussian variance) are notoriously overconfident and poorly calibrated under out-of-distribution turbulence or erratic target maneuvers. 

SADHA implements **empirical delayed ground-truth verification**:
1. When NN2 generates a prediction at frame $T$ for future target frame $T + H$, the forecast is stored in a pending queue.
2. When frame $T + H$ subsequently arrives from NN1, the actual observed centroid $(x_{\text{actual}}, y_{\text{actual}})$ is matched against the earlier forecast:
   $$\text{Error}_{\text{px}} = \sqrt{(x_{\text{pred}} - x_{\text{actual}})^2 + (y_{\text{pred}} - y_{\text{actual}})^2}$$
3. The confidence score is computed against the 15 px tracking benchmark tolerance:
   $$\text{Confidence} = \max\left(0.0, \, 1.0 - \frac{\text{Error}_{\text{px}}}{15.0}\right)$$
This delivers an objective, verifiable measurement of prediction accuracy grounded in physical sensor reality.

### 2. Disturbance-Conditioned Recurrent Sequence Modeling
In free-space optical links, apparent beacon jitter may arise from real target acceleration or from optical scintillation, thermal beam wander, and mount vibrations. 
By concatenating the 8-dimensional disturbance descriptor vector directly into every recurrent time-step of the GRU input alongside normalized kinematics $[x_{\text{norm}}, y_{\text{norm}}, \Delta x_{\text{norm}}, \Delta y_{\text{norm}}]$, the recurrent cell adaptively weights its predictions: when turbulence is severe, it dampens high-frequency corrections and relies on kinematic momentum; under clear conditions, it tightens its tracking response.

---

## 5. Module Architecture & File Map

```
src/tracking/
├── nn2_types.py      # Contract dataclasses: NN2Prediction, NN2ConfidenceScore, NN2Result
├── nn2_model.py      # PyTorch BeaconMotionGRU recurrent architecture
├── nn2_predictor.py  # NN2Predictor orchestrator, ensemble blending, confidence engine
├── nn2_trainer.py    # Training loop and synthetic sequence generator
└── nn2_README.md     # This documentation specification
```

---

## 6. Standalone Execution & Training

### Re-training the NN2 GRU Predictor:
```bash
python -m src.tracking.nn2_trainer --epochs 25 --sequences 3000 --horizon 1
```
Trained weights are automatically saved to `data/nn2_weights.pth`.
