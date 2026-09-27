# Module 2.1: NN1 Beacon Detection & Disturbance Characterization

## 1. Overview & Architectural Role

Module 2.1 (NN1) is the **front-line visual perception engine** of SADHA. It processes raw monochrome sensor frames ($640 \times 480 \times 1$) from Module 1 or external video streams, identifies candidate optical beacon centroids with sub-pixel precision, and computes an 8-dimensional disturbance characterization descriptor.

NN1 operates on a **per-frame, stateless basis**. It does not maintain temporal tracks, assign beacon identities, or predict future motion — its sole responsibility is extracting high-fidelity spatial detections and environmental metrics from single frames.

---

## 2. Responsibilities & Boundaries

### What NN1 IS Responsible For:
- **Optical Beacon Detection**: Locating candidate beacon spots in extreme noise (Gaussian noise $\sigma \le 20$, salt-and-pepper density $\le 10\%$, Poisson shot noise, atmospheric fog/haze/rain).
- **Sub-Pixel Centroid Estimation**: Computing $(x, y)$ coordinates with sub-pixel accuracy using intensity-weighted spatial moments over the candidate region.
- **Confidence Scoring**: Assigning an empirical detection quality score $\in [0.0, 1.0]$ based on peak signal-to-noise ratio (PSNR) and spatial compactness.
- **Disturbance Vector Extraction**: Computing or extracting an 8-dimensional environmental summary vector (measuring noise standard deviation, background luminance, contrast degradation, and spatial jitter variance) to condition downstream neural predictors.
- **Architectural Dual-Backend Support**:
  - *Learned CNN Pathway (`BeaconHeatmapNet`)*: Fully convolutional U-Net-style heatmap regression architecture trained to isolate optical beacons under heavy disturbances.
  - *Classical Algorithmic Fallback*: Adaptive thresholding, morphologic filtering, and connected-component centroiding when PyTorch weights are unavailable.

### What NN1 is NOT Responsible For:
- **Temporal Identity Association**: Associating detections across consecutive frames (handled by Hungarian matching in Module 2.2).
- **Target Tracking & State Estimation**: Kalman position/velocity smoothing (handled by Module 2.2).
- **Lookahead Forecasting**: Trajectory prediction (handled by NN2 in Module 2.3).
- **Camera Gimbal Control**: Pan/tilt steering rate computation.

---

## 3. Interface Contracts

### 3.1 Input Contract (Module 1 $\rightarrow$ NN1)
NN1 consumes a single `FrameData` object:

```python
def detect(self, frame_data: FrameData) -> NN1Result
```
- **`frame_data.image`** (`np.ndarray`): $480 \times 640$ 8-bit unsigned integer grayscale frame (`uint8`).
- **`frame_data.frame_index`** (`int`): Current sequential frame index.
- **`frame_data.timestamp`** (`float`): Current simulation/mission timestamp in seconds.
- **`frame_data.disturbance_descriptor`** (`List[float]`): Optional environmental telemetry.

### 3.2 Output Contract (NN1 $\rightarrow$ Module 2.2 Kalman Tracker)
NN1 returns an immutable, strongly-typed `NN1Result` record:

```python
@dataclass(slots=True)
class NN1Result:
    frame_index: int
    timestamp: float
    detections: List[Detection]               # 0 to N raw, unlabeled detections
    disturbance_descriptor: Optional[List[float]]
    camera_pan_deg: Optional[float]
    camera_tilt_deg: Optional[float]
    capture_rate_hz: Optional[float]
```

Each item in `detections` is a `Detection` instance:
| Field | Type | Description |
|---|---|---|
| `x` | `float` | Centroid horizontal coordinate in sensor frame (sub-pixel px) |
| `y` | `float` | Centroid vertical coordinate in sensor frame (sub-pixel px) |
| `confidence` | `float` | Detection score $\in [0.0, 1.0]$ |
| `bbox_x`, `bbox_y` | `float` | Top-left bounding box coordinates |
| `bbox_w`, `bbox_h` | `float` | Bounding box spatial dimensions |
| `peak_intensity` | `int` | Maximum pixel intensity ($0 \dots 255$) at detection centroid |

**Design Invariant**: `detections` is always an initialized Python list (empty list if zero targets detected). Downstream modules never have to guard against `None`.

---

## 4. Module Architecture & File Map

```
src/tracking/
├── nn1_types.py      # Contract dataclasses: Detection, NN1Result
├── nn1_model.py      # PyTorch BeaconHeatmapNet CNN architecture
├── nn1_detector.py   # NN1Detector orchestrator (CNN + Classical fallback)
├── nn1_trainer.py    # Synthetic dataset generator & training loop for NN1
├── nn1_test.py       # Standalone test runner & precision-recall evaluator
└── nn1_README.md     # This documentation specification
```

---

## 5. Standalone Execution & Training

### Running Standalone Evaluation:
```bash
python -m src.tracking.nn1_test
```

### Re-training the NN1 Heatmap Model:
```bash
python -m src.tracking.nn1_trainer --epochs 20 --samples 2000 --batch-size 16
```
Trained weights are automatically saved to `data/nn1_weights.pth`.
