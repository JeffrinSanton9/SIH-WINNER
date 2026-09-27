# Application Shell & Visualization GUI

## 1. Overview & Architectural Role

The `src/ui/` package contains the **PyQt5 graphical user interface and operator control station** for SADHA. It provides real-time multi-panel visualization of the simulated world canvas, the camera's sensor focal plane, 2D/3D trajectory projections, and high-frequency telemetry analytics.

The UI is built with asynchronous, decoupled viewports that render simulation state without blocking the core tracking pipeline.

---

## 2. Responsibilities & Boundaries

### What the UI IS Responsible For:
- **World View Canvas (`WorldViewWidget`)**: 2D top-down global visualization ($2400 \times 2400$ px space) showing absolute beacon trajectories, host platform vibrations, and the camera's dynamic angular FOV frustum footprint.
- **Sensor Camera View (`CameraViewWidget`)**: Real-time HUD showing the 640x480 monochrome sensor frame, optical boresight reticle, NN1 candidate detections, Kalman bounding boxes with velocity vectors, and NN2 lookahead cones.
- **Trajectory Analysis (`TrajectoryWidget`)**: Historical path plotting, tracking error deviation curves, and 3D trajectory visualization.
- **Analytics & Telemetry Dock**: Live metrics display (processing FPS, tracking error RMSE, lock state status, re-acquisition timers, disturbance SNR).
- **Configuration Drawer (`ConfigDrawer`)**: Real-time sliding drawer allowing live parameter adjustments (kinematics, platform motion, noise levels, camera slew limits).
- **Home Landing Screen (`HomeScreen`)**: Scenario preset selector for standardized Benchmark-1 evaluations.
- **Interactive Human-in-the-Loop Control**: Capturing mouse drags and keyboard WASD inputs to pilot beacons in real time.

### What the UI is NOT Responsible For:
- Does not implement physics integration equations directly (delegated to `src/sim/`).
- Does not run raw image detection or tracking logic (delegated to `src/tracking/`).

---

## 3. UI Architecture & File Map

```
src/ui/
├── __init__.py               # Package namespace
├── main_window.py            # Main application window, central QTimer tick loop, menu bar
├── home_screen.py            # High-impact landing page with Benchmark-1 scenario cards
├── camera_view_widget.py     # Sensor view HUD with detection and tracking overlays
├── world_view_widget.py      # Global canvas viewport with FOV frustum and historical trails
├── trajectory_widget.py      # Error curve plots and 2D/3D trajectory visualization
├── raw_video_widget.py       # Benchmark-2 external video player widget
├── config_drawer.py          # Sliding overlay drawer for dynamic parameter tuning
├── styles.py                 # Central dark-mode stylesheet & design tokens
└── README.md                 # This documentation specification
```

---

## 4. Key Interfaces & Communication

- **Simulation Timer Tick (`MainWindow._on_sim_tick`)**: Runs at 60 Hz via a high-precision `QTimer`. Each tick advances `WorldSimulation.step(dt)`, passes emitted `FrameData` through NN1 $\rightarrow$ Kalman Tracker $\rightarrow$ NN2, and dispatches updated state records to all viewports.
- **Navigation Lifecycle**: A `QStackedWidget` toggles between `HomeScreen` (preset scenario selection) and `sim_page` (operational multi-panel dashboard).
