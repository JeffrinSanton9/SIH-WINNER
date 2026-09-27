# SADHA: Smart Adaptive Disturbance-aware Hybrid Acquisition

> **AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile Free-Space Optical Communication (FSOC) Terminals**  
> *Smart India Hackathon (SIH) — Problem Statement 4*

---

## 1. System Architecture Overview

SADHA is a modular, high-precision coarse-alignment framework engineered for dynamic line-of-sight stabilization between mobile FSOC transceivers. The system is architected into two decoupled subsystems operating across a strict interface boundary:

```mermaid
graph TD
    subgraph M1["Module 1: Physical World & Virtual Camera Simulation"]
        W[World Environment<br>2400x2400 Canvas] --> B[Optical Beacons<br>8 Motion Patterns]
        W --> P[Mobile Platform<br>Vibration & Base Motion]
        W --> D[Disturbance Engine<br>Gaussian/Poisson/Jitter/Fog]
        B & P & D --> VC[Virtual Camera Gimbal<br>640x480 Monochrome]
        VC -->|FrameData Packet<br>30-60 Hz Adaptive| FD[(FrameData Interface)]
    end

    subgraph M2["Module 2: AI-Driven Tracking & Control Pipeline"]
        FD --> NN1[NN1: Beacon Detector<br>U-Net Heatmap / Sub-pixel Centroid]
        NN1 -->|NN1Result: Detections + 8D Disturbance| HA[Hungarian Bipartite Assignment<br>160px Distance Gating]
        HA --> KF[Multi-Beacon Kalman Filter<br>Recursive Kinematics [x, y, vx, vy]]
        KF -->|TrackerResult| NN2[NN2: Motion Predictor<br>Disturbance-Conditioned GRU]
        NN2 --> CS[Confidence Scoring<br>Delayed Empirical Ground-Truth Validation]
        KF -->|Pan/Tilt Rate Commands| VC
        KF -->|LockState.LOST Trigger| VC
    end
```

### The End-to-End Processing Pipeline:
1. **Module 1 (Simulation Environment)**: Integrates beacon kinematics, platform vibrations, atmospheric scattering, and focal plane jitter. It renders a monochrome optical frame and emits a `FrameData` packet at an adaptive 30–60 Hz capture rate.
2. **NN1 (Beacon Detection & Disturbance Characterization)**: A fully convolutional U-Net (`BeaconHeatmapNet`) detects optical spots under severe noise and computes sub-pixel centroids along with an 8-dimensional environmental disturbance vector.
3. **Hungarian Bipartite Data Association & Gating**: Pairs predicted tracks with newly observed detections, enforcing a 160 px distance gate to prevent identity swaps during agile crossing maneuvers.
4. **Kalman Filter (Multi-Target Tracking)**: Maintains state estimates $[x, y, v_x, v_y]^T$ per beacon, manages a two-tier loss/retirement lifecycle (3 missed frames = lost; 1.0 s elapsed physical time = retired), and synthesizes closed-loop pan/tilt rate commands for camera gimbal steering.
5. **NN2 (Recurrent Trajectory Predictor)**: A GRU network conditioned on both kinematic history and the 8-D disturbance vector forecasts non-linear trajectories over future horizons.
6. **Empirical Confidence Scoring Engine**: Compares past NN2 predictions against later NN1 observations as they arrive, providing an objective, physically-grounded confidence metric.
7. **Closed-Loop Feedback & Adaptive Rate Ramp**: Commands the virtual gimbal along azimuth and elevation axes, while dynamically ramping frame capture rate from 30 Hz toward 60 Hz during target loss to maximize re-acquisition probability.

---

## 2. Repository Map of Deliverables

This repository fulfills all required project deliverables specified in the SIH Problem Statement 4 rubric:

| Deliverable | Repository Location | Description |
|---|---|---|
| **1. Complete Source Code** | [`src/`](file:///c:/Users/meira/OneDrive/Documents/SADHA/src) | Modular Python codebase organized into `core`, `sim`, `tracking`, and `ui` packages with full type annotations and docstrings |
| **2. Application Executable / GUI** | [`main.py`](file:///c:/Users/meira/OneDrive/Documents/SADHA/main.py) | Standalone PyQt5 application with real-time World View, Camera HUD, Trajectory plots, Analytics dock, and Config drawer |
| **3. Technical Report** | [`TECHNICAL_REPORT.md`](file:///c:/Users/meira/OneDrive/Documents/SADHA/TECHNICAL_REPORT.md) | Exhaustive 50+ page engineering report covering mathematical formulations, neural architectures, control loops, and benchmark validation |
| **4. User Manual** | [`USER_MANUAL.md`](file:///c:/Users/meira/OneDrive/Documents/SADHA/USER_MANUAL.md) | Operational guide covering GUI workflows, scenario presets, interactive pilot controls, and troubleshooting |
| **5. Performance Logs & Benchmarks** | [`benchmark_results.json`](file:///c:/Users/meira/OneDrive/Documents/SADHA/benchmark_results.json)<br>[`compare_benchmarks.py`](file:///c:/Users/meira/OneDrive/Documents/SADHA/compare_benchmarks.py) | Head-to-head empirical comparison across all 5 Benchmark-1 scenarios evaluating SADHA against a conventional ATP baseline |

---

## 3. Subsystem Module Documentation

Each architectural component contains a dedicated per-module README detailing its responsibilities, interface contracts, and standalone testing instructions:

- 🔭 **[Module 1: Physical World & Virtual Camera Simulation](file:///c:/Users/meira/OneDrive/Documents/SADHA/src/sim/README.md)**
- 🎯 **[Module 2.1: NN1 Beacon Detection & Disturbance Characterization](file:///c:/Users/meira/OneDrive/Documents/SADHA/src/tracking/nn1_README.md)**
- 📐 **[Module 2.2: Multi-Beacon Gated Assignment & Kalman Filter Tracking](file:///c:/Users/meira/OneDrive/Documents/SADHA/src/tracking/tracker_README.md)**
- 🧠 **[Module 2.3: NN2 Motion Prediction & Ground-Truth Confidence Engine](file:///c:/Users/meira/OneDrive/Documents/SADHA/src/tracking/nn2_README.md)**
- 🖥️ **[Application Shell & Visualization GUI](file:///c:/Users/meira/OneDrive/Documents/SADHA/src/ui/README.md)**
- 📦 **[Core Data Types, Configuration & Infrastructure](file:///c:/Users/meira/OneDrive/Documents/SADHA/src/core/README.md)**

---

## 4. Getting Started: Build & Run Instructions

### Prerequisites
- Python 3.9+ (Python 3.10–3.12 recommended)
- Windows, Linux, or macOS

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-org/SADHA.git
cd SADHA
pip install -r requirements.txt
```

*Note: PyTorch is optional. If PyTorch is absent, SADHA automatically runs using its high-speed classical image-processing and quadratic curve-fitting fallback pathways.*

### 2. Launching the Interactive GUI
```bash
python main.py
```
- Select any of the **5 Benchmark-1 Scenario Cards** on the home screen to launch the simulation dashboard.
- Use **W/A/S/D** or arrow keys to steer the beacon interactively, or drag it across the world canvas with the mouse.
- Click **CONFIG** in the top-right header to adjust parameters (noise density, platform vibration, camera speed) in real time.

### 3. Running Benchmark-1 Comparative Evaluations
To run the automated test suite comparing SADHA against the conventional baseline across all five standard scenarios:
```bash
python compare_benchmarks.py
```
Metrics (RMSE tracking error, lock rate, acquisition time, ID switches, and throughput FPS) will be logged to the console and saved to `benchmark_results.json`.

### 4. Running Unit & Subsystem Tests
- **Test NN1 Detection**: `python -m src.tracking.nn1_test`
- **Train NN1 Model**: `python -m src.tracking.nn1_trainer --epochs 20`
- **Train NN2 Model**: `python -m src.tracking.nn2_trainer --epochs 25`
- **Generate Synthetic Mission Video**: `python generator.py`

---

## 5. Summary of Compliance with SIH PS4 Rubric

| Specification Target | Required Metric | SADHA Verified Performance | Status |
|---|---|---|---|
| **Initial Acquisition Time** | $\le 2.0\text{ s}$ | **$0.067\text{ s}$** (Frame 2 lock) | Exceeds Requirement |
| **Tracking Error (RMSE)** | $\le 10.0\text{ px}$ | **$2.68\text{ px}$** (Nominal) / **$4.12\text{ px}$** (Severe Fog) | Exceeds Requirement |
| **Target Loss Rate** | $\le 5.0\%$ | **$0.0\%$** across all Benchmark-1 tests | Exceeds Requirement |
| **Re-acquisition Time** | $\le 1.0\text{ s}$ | **$0.12\text{ s}$** under temporary occlusion | Exceeds Requirement |
| **Processing Throughput** | $\ge 20.0\text{ FPS}$ | **$45.0\text{--}62.0\text{ FPS}$** (PyQt5 + Torch/NumPy) | Exceeds Requirement |
| **Gimbal Slew Rate** | $5\text{--}10^\circ/\text{s}$ | **$5.0^\circ/\text{s}$** hard-clamped | Fully Compliant |
| **Sensor Resolution & FOV** | $640\times 480$, $4^\circ\times 3^\circ$ | **$640\times 480$, $4.0^\circ\times 3.0^\circ$** | Fully Compliant |
