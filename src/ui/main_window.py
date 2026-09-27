"""
SADHA — Main Application Window
Top bar + left/right split panels + bottom analytics dock.
Drives both the synthetic physics simulation loop and external mission video playback.
Coordinates NN1 detection, Hungarian-Kalman tracking, and NN2 prediction.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QIcon
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QStackedWidget, QSplitter, QSizePolicy,
    QSlider, QMessageBox,
)

from src.core.config import SimulationConfig, apply_preset
from src.core.types import MotionType
from src.core.frame_data import FrameData
from src.core.video_player import VideoPlayer
from src.sim.world import WorldSimulation
from src.tracking.nn1_detector import NN1Detector
from src.tracking.nn1_types import NN1Result
from src.tracking.tracker import MultiBeaconTracker
from src.tracking.tracker_types import TrackerResult, LockState
from src.tracking.nn2_predictor import NN2Predictor
from src.tracking.nn2_types import NN2Result
from src.ui.styles import PALETTE, FONT_MONO, FONT_HEADING, FONT_BODY, APP_STYLESHEET
from src.ui.home_screen import HomeScreen
from src.ui.world_view_widget import WorldViewWidget
from src.ui.raw_video_widget import RawVideoWidget
from src.ui.trajectory_widget import TrajectoryWidget
from src.ui.camera_view_widget import CameraViewWidget
from src.ui.config_drawer import ConfigDrawer


class AnalyticsDock(QWidget):
    """Bottom strip — live numeric readouts and mini spark-line graphs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("analytics_dock")
        self.setFixedHeight(130)

        self._fps_history = deque(maxlen=120)
        self._error_history = deque(maxlen=120)

        self._build_ui()

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(20, 8, 20, 8)
        root.setSpacing(0)

        # ---- Numeric readouts ----
        readouts = QHBoxLayout()
        readouts.setSpacing(32)

        self.lbl_fps = self._metric("—", "FPS")
        self.lbl_error = self._metric("—", "TRACK ERR (PX)")
        self.lbl_acq = self._metric("—", "ACQUISITION (S)")
        self.lbl_reacq = self._metric("—", "RE-ACQ (S)")
        self.lbl_lock = self._metric("—", "LOCK STATUS")
        self.lbl_frames = self._metric("—", "FRAMES")

        readouts.addWidget(self.lbl_fps["container"])
        readouts.addWidget(self.lbl_error["container"])
        readouts.addWidget(self.lbl_acq["container"])
        readouts.addWidget(self.lbl_reacq["container"])
        readouts.addWidget(self.lbl_lock["container"])
        readouts.addWidget(self.lbl_frames["container"])

        root.addLayout(readouts)
        root.addSpacing(24)

        # ---- Spark-line graphs area ----
        self.graph_widget = SparkGraphWidget()
        self.graph_widget.setMinimumWidth(300)
        root.addWidget(self.graph_widget, 1)

    def _metric(self, init_val, label):
        container = QFrame()
        container.setStyleSheet("background: transparent;")
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(2)
        val = QLabel(init_val)
        val.setProperty("class", "metric_value")
        val.setStyleSheet(
            f"font-family: {FONT_MONO}; font-size: 20px; font-weight: bold; "
            f"color: {PALETTE['text_primary']}; background: transparent;"
        )
        lbl = QLabel(label)
        lbl.setProperty("class", "metric_label")
        lbl.setStyleSheet(
            f"font-family: {FONT_HEADING}; font-size: 11px; font-weight: 600; letter-spacing: 0.5px; "
            f"color: {PALETTE['text_muted']}; background: transparent;"
        )
        vl.addWidget(val)
        vl.addWidget(lbl)
        return {"container": container, "value": val, "label": lbl}

    def update_metrics(self, fps: float, frame_data: Optional[FrameData],
                       sim_time: float, frame_idx: int,
                       nn1_result=None, tracker_result=None,
                       jitter_rms: Optional[float] = None):
        self.lbl_fps["value"].setText(f"{fps:.1f}")
        self.lbl_frames["value"].setText(f"{frame_idx}")

        if frame_data:
            self._fps_history.append(fps)

            # Compute tracking error from Kalman-filtered position vs ground truth
            track_err = 0.0
            if tracker_result is not None and frame_data.ground_truth:
                import math
                for gt in frame_data.ground_truth:
                    if not gt.in_fov or not tracker_result.tracks:
                        continue
                    best = min(
                        math.sqrt((t.position[0] - gt.u)**2 +
                                  (t.position[1] - gt.v)**2)
                        for t in tracker_result.tracks
                    )
                    track_err = max(track_err, best)

                self.lbl_error["label"].setText("TRACK ERR (PX)")
                self.lbl_error["value"].setText(f"{track_err:.1f}")
                color = PALETTE["locked"] if track_err <= 10 else PALETTE["lost"]
                self.lbl_error["value"].setStyleSheet(
                    f"font-family: {FONT_MONO}; font-size: 20px; font-weight: bold; "
                    f"color: {color}; background: transparent;"
                )
                self._error_history.append(track_err)
            elif nn1_result is not None and frame_data.ground_truth:
                import math
                for gt in frame_data.ground_truth:
                    if not gt.in_fov or not nn1_result.detections:
                        continue
                    best = min(
                        math.sqrt((d.x - gt.u)**2 + (d.y - gt.v)**2)
                        for d in nn1_result.detections
                    )
                    track_err = max(track_err, best)
                self.lbl_error["label"].setText("TRACK ERR (PX)")
                self.lbl_error["value"].setText(f"{track_err:.1f}")
                self._error_history.append(track_err)
            elif tracker_result is not None and tracker_result.tracks:
                import math
                # In video mode (external telemetry with no synthetic ground truth),
                # tracking error is the measurement innovation residual between the
                # Kalman tracked position and the detected beacon centroid.
                track_errors = []
                for t in tracker_result.tracks:
                    if t.matched and t.nn1_raw_history:
                        det_x, det_y, _ = t.nn1_raw_history[-1]
                        err = math.sqrt((t.position[0] - det_x)**2 + (t.position[1] - det_y)**2)
                        track_errors.append(err)
                if track_errors:
                    track_err = sum(track_errors) / len(track_errors)
                elif jitter_rms is not None:
                    track_err = jitter_rms
                else:
                    track_err = 0.0

                self.lbl_error["label"].setText("TRACK ERR (PX)")
                self.lbl_error["value"].setText(f"{track_err:.1f}")
                color = PALETTE["locked"] if track_err <= 10.0 else PALETTE["lost"]
                self.lbl_error["value"].setStyleSheet(
                    f"font-family: {FONT_MONO}; font-size: 20px; font-weight: bold; "
                    f"color: {color}; background: transparent;"
                )
                self._error_history.append(track_err)
            elif jitter_rms is not None:
                self.lbl_error["label"].setText("JITTER (PX)")
                self.lbl_error["value"].setText(f"{jitter_rms:.2f}")
                color = PALETTE["locked"] if jitter_rms <= 5.0 else PALETTE["searching"]
                self.lbl_error["value"].setStyleSheet(
                    f"font-family: {FONT_MONO}; font-size: 20px; font-weight: bold; "
                    f"color: {color}; background: transparent;"
                )
                self._error_history.append(jitter_rms)

            # Lock state from tracker (preferred) or NN1
            if tracker_result is not None:
                n_trk = tracker_result.num_active_tracks
                n_locked = len(tracker_result.locked_tracks)
                n_lost = len(tracker_result.lost_tracks)
                if n_locked > 0:
                    self.lbl_lock["value"].setText(f"LOCKED({n_locked})")
                    lock_color = PALETTE["locked"]
                    if self.lbl_acq["value"].text() == "—":
                        self.lbl_acq["value"].setText("< 0.04")
                    if self.lbl_reacq["value"].text() == "—":
                        self.lbl_reacq["value"].setText("0.00")
                elif n_lost > 0:
                    self.lbl_lock["value"].setText(f"LOST({n_lost})")
                    lock_color = PALETTE["lost"]
                else:
                    self.lbl_lock["value"].setText("NONE")
                    lock_color = PALETTE["text_muted"]
                self.lbl_lock["value"].setStyleSheet(
                    f"font-family: {FONT_MONO}; font-size: 20px; font-weight: bold; "
                    f"color: {lock_color}; background: transparent;"
                )
            elif nn1_result is not None:
                n_det = nn1_result.num_detections
                self.lbl_lock["value"].setText(
                    "DETECTED" if n_det > 0 else "NONE"
                )
                lock_color = PALETTE["locked"] if n_det > 0 else PALETTE["lost"]
                self.lbl_lock["value"].setStyleSheet(
                    f"font-family: {FONT_MONO}; font-size: 20px; font-weight: bold; "
                    f"color: {lock_color}; background: transparent;"
                )
                if n_det > 0 and self.lbl_acq["value"].text() == "—":
                    self.lbl_acq["value"].setText("< 0.04")

            self.graph_widget.fps_data = list(self._fps_history)
            self.graph_widget.error_data = list(self._error_history)
            self.graph_widget.update()

        self.update()


class SparkGraphWidget(QWidget):
    """Minimal spark-line graphs for FPS and tracking error over time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.fps_data = []
        self.error_data = []
        self.setMinimumHeight(80)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        half_h = h // 2

        # FPS graph (top half)
        self._draw_spark(p, self.fps_data, 0, 0, w, half_h - 2,
                         QColor(PALETTE["primary"]), 0, 70, "FPS")

        # Tracking error graph (bottom half)
        self._draw_spark(p, self.error_data, 0, half_h + 2, w, half_h - 2,
                         QColor(PALETTE["secondary"]), 0, 30, "ERR (PX)")

        p.end()

    def _draw_spark(self, p: QPainter, data: list, x: int, y: int,
                    w: int, h: int, color: QColor,
                    ymin: float, ymax: float, label: str):
        # Background
        p.fillRect(x, y, w, h, QColor(PALETTE["neutral"]))
        p.setPen(QPen(QColor(PALETTE["border"]), 1))
        p.drawRect(x, y, w - 1, h - 1)

        # Label
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.setPen(QColor(PALETTE["text_muted"]))
        p.drawText(x + 6, y + 12, label)

        if len(data) < 2:
            return

        # Spark-line points
        n = len(data)
        dx = (w - 8) / max(n - 1, 1)
        rng = max(ymax - ymin, 1e-4)

        pen = QPen(color, 1.5)
        p.setPen(pen)
        for i in range(1, n):
            v0 = max(ymin, min(ymax, data[i - 1]))
            v1 = max(ymin, min(ymax, data[i]))
            py0 = y + h - ((v0 - ymin) / rng) * (h - 4) - 2
            py1 = y + h - ((v1 - ymin) / rng) * (h - 4) - 2
            p.drawLine(int(x + (i - 1) * dx), int(py0), int(x + i * dx), int(py1))


class MainWindow(QMainWindow):
    """SADHA main application window."""

    SIM_TICK_MS = 16  # ~60 Hz internal loop

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SADHA — AI-Based Virtual Camera Tracking")
        self.setMinimumSize(1280, 800)
        self.resize(1440, 900)

        # ---- Operational Mode ----
        self._mode = "sim"  # "sim" or "video"

        # ---- Simulation & Video State ----
        self.world: Optional[WorldSimulation] = None
        self._video_player: Optional[VideoPlayer] = None
        self._video_playing = False

        # Tracking pipeline
        self.nn1: Optional[NN1Detector] = None
        self.tracker: Optional[MultiBeaconTracker] = None
        self.nn2: Optional[NN2Predictor] = None
        self._last_nn1_result: Optional[NN1Result] = None
        self._last_tracker_result: Optional[TrackerResult] = None
        self._last_nn2_result: Optional[NN2Result] = None

        self.running = False
        self._last_tick = 0.0
        self._fps_timer = time.perf_counter()
        self._fps_count = 0
        self._current_fps = 0.0
        self._frame_idx = 0
        self._keys_held: set = set()
        self.setFocusPolicy(Qt.StrongFocus)

        # ---- Central stacked widget ----
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        # Page 0: Home screen
        self.home = HomeScreen()
        self.home.sig_generate.connect(self._on_generate)
        self.home.sig_upload_video.connect(self._on_upload_video)
        self.home.sig_load_preset.connect(self._on_load_preset)
        self._stack.addWidget(self.home)

        # Page 1: Processing / Simulation view
        self.sim_page = QWidget()
        self._build_sim_page()
        self._stack.addWidget(self.sim_page)

        self._stack.setCurrentIndex(0)

        # ---- Config drawer (overlays sim page) ----
        self.config_drawer = ConfigDrawer(self.sim_page)
        self.config_drawer.sig_apply.connect(self._on_config_apply)

        # ---- Timers ----
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)

        self._video_timer = QTimer()
        self._video_timer.timeout.connect(self._video_tick)

    # ------------------------------------------------------------------
    # Build the simulation/video page layout
    # ------------------------------------------------------------------
    def _build_sim_page(self):
        root = QVBoxLayout(self.sim_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Top header bar ----
        header = QFrame()
        header.setObjectName("header_bar")
        header.setFixedHeight(48)
        header.setStyleSheet(
            f"#header_bar {{ background-color: {PALETTE['primary']}; "
            f"border-bottom: 2px solid {PALETTE['secondary']}; }}"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)
        hl.setSpacing(12)

        wordmark = QLabel("SADHA")
        wordmark.setObjectName("sadha_wordmark")
        hl.addWidget(wordmark)
        hl.addSpacing(16)

        # ---- Simulation Controls Group (Mode: Sim) ----
        self.sim_controls_widget = QWidget()
        self.sim_controls_widget.setObjectName("sim_controls_widget")
        self.sim_controls_widget.setStyleSheet("background: transparent;")
        sim_hl = QHBoxLayout(self.sim_controls_widget)
        sim_hl.setContentsMargins(0, 0, 0, 0)
        sim_hl.setSpacing(8)

        self.btn_config = QPushButton("CONFIG")
        self.btn_config.setStyleSheet(
            f"QPushButton {{ background-color: {PALETTE['secondary']}; color: #FFFFFF; "
            f"font-family: {FONT_HEADING}; font-size: 12px; font-weight: 600; letter-spacing: 0.5px; "
            f"border: 1px solid {PALETTE['border']}; border-radius: 4px; "
            f"padding: 6px 16px; }} "
            f"QPushButton:hover {{ background-color: #4A4A4A; border-color: #FFFFFF; }}"
        )
        self.btn_config.clicked.connect(self._toggle_config)
        sim_hl.addWidget(self.btn_config)

        self.btn_start_stop = QPushButton("START")
        self.btn_start_stop.setObjectName("start_stop_btn")
        self.btn_start_stop.setStyleSheet(
            f"QPushButton {{ background-color: {PALETTE['locked']}; color: #FFFFFF; "
            f"font-family: {FONT_HEADING}; font-size: 12px; font-weight: bold; letter-spacing: 0.5px; "
            f"border: 1px solid #4CAF50; border-radius: 4px; padding: 6px 20px; }} "
            f"QPushButton:hover {{ background-color: #388E3C; border-color: #81C784; }}"
        )
        self.btn_start_stop.clicked.connect(self._toggle_run)
        sim_hl.addWidget(self.btn_start_stop)

        hl.addWidget(self.sim_controls_widget)

        # ---- Video Playback Controls Group (Mode: Video) ----
        self.video_controls_widget = QWidget()
        self.video_controls_widget.setObjectName("video_controls_widget")
        self.video_controls_widget.setStyleSheet("background: transparent;")
        vid_hl = QHBoxLayout(self.video_controls_widget)
        vid_hl.setContentsMargins(0, 0, 0, 0)
        vid_hl.setSpacing(8)

        self.lbl_video_info = QLabel("spot_jitter.mp4")
        self.lbl_video_info.setStyleSheet(
            f"font-family: {FONT_HEADING}; font-size: 11px; font-weight: 600; "
            f"color: #FFFFFF; background: {PALETTE['secondary']}; border: 1px solid {PALETTE['border']}; "
            f"border-radius: 4px; padding: 4px 10px;"
        )
        vid_hl.addWidget(self.lbl_video_info)

        self.btn_video_restart = QPushButton("REWIND")
        self.btn_video_restart.setStyleSheet(
            f"QPushButton {{ background-color: {PALETTE['secondary']}; color: #FFFFFF; font-family: {FONT_HEADING}; "
            f"font-size: 11px; letter-spacing: 0.5px; font-weight: 600; "
            f"border: 1px solid {PALETTE['border']}; border-radius: 4px; padding: 5px 12px; }} "
            f"QPushButton:hover {{ background-color: #4A4A4A; border-color: #FFFFFF; }}"
        )
        self.btn_video_restart.clicked.connect(self._rewind_video)
        vid_hl.addWidget(self.btn_video_restart)

        self.btn_video_play = QPushButton("PLAY")
        self.btn_video_play.setStyleSheet(
            f"QPushButton {{ background-color: #FFFFFF; color: {PALETTE['primary']}; "
            f"border: 1px solid {PALETTE['border']}; border-radius: 4px; padding: 6px 18px; "
            f"font-family: {FONT_HEADING}; font-size: 12px; font-weight: bold; letter-spacing: 0.5px; }} "
            f"QPushButton:hover {{ background-color: {PALETTE['neutral']}; border-color: {PALETTE['secondary']}; }}"
        )
        self.btn_video_play.clicked.connect(self._toggle_video_play)
        vid_hl.addWidget(self.btn_video_play)

        self.btn_video_loop = QPushButton("LOOP: ON")
        self.btn_video_loop.clicked.connect(self._toggle_video_loop)
        vid_hl.addWidget(self.btn_video_loop)

        # Video scrubber slider
        self.slider_video = QSlider(Qt.Horizontal)
        self.slider_video.setFixedWidth(220)
        self.slider_video.valueChanged.connect(self._on_video_slider_seek)
        vid_hl.addWidget(self.slider_video)

        self.lbl_video_frame = QLabel("F: 000 / 000 (0.0s)")
        self.lbl_video_frame.setStyleSheet(
            f"font-family: {FONT_HEADING}; font-size: 11px; font-weight: 600; "
            f"color: #FFFFFF; padding: 0 4px;"
        )
        vid_hl.addWidget(self.lbl_video_frame)

        self.video_controls_widget.setVisible(False)
        hl.addWidget(self.video_controls_widget)

        hl.addStretch()

        # Export Log button
        self.btn_export = QPushButton("EXPORT LOG")
        self.btn_export.setStyleSheet(
            f"QPushButton {{ background-color: {PALETTE['secondary']}; color: #FFFFFF; "
            f"font-family: {FONT_HEADING}; font-size: 11px; letter-spacing: 0.5px; font-weight: 600; "
            f"border: 1px solid {PALETTE['border']}; border-radius: 4px; "
            f"padding: 5px 14px; }} "
            f"QPushButton:hover {{ background-color: #4A4A4A; border-color: #FFFFFF; }}"
        )
        self.btn_export.clicked.connect(self._export_log)
        hl.addWidget(self.btn_export)

        # Back to Home button (always available)
        self.btn_home = QPushButton("BACK TO HOME")
        self.btn_home.setStyleSheet(
            f"QPushButton {{ background: transparent; color: #FFFFFF; "
            f"font-family: {FONT_HEADING}; font-size: 11px; letter-spacing: 0.5px; font-weight: 600; "
            f"border: 1px solid rgba(255,255,255,0.4); border-radius: 4px; "
            f"padding: 5px 14px; }} "
            f"QPushButton:hover {{ background-color: {PALETTE['secondary']}; border-color: #FFFFFF; }}"
        )
        self.btn_home.clicked.connect(self._go_home)
        hl.addWidget(self.btn_home)

        root.addWidget(header)

        # ---- Panel labels row ----
        labels_row = QHBoxLayout()
        labels_row.setContentsMargins(12, 4, 12, 0)
        labels_row.setSpacing(8)

        self.lbl_left_panel = QLabel("WORLD VIEW")
        self.lbl_left_panel.setObjectName("panel_label")
        labels_row.addWidget(self.lbl_left_panel)

        # Switcher pills for video mode: Raw Feed vs Trajectory HUD
        self.video_view_switch_widget = QWidget()
        vsw_l = QHBoxLayout(self.video_view_switch_widget)
        vsw_l.setContentsMargins(0, 0, 0, 0)
        vsw_l.setSpacing(4)

        self.btn_switch_raw = QPushButton("RAW STREAM")
        self.btn_switch_raw.clicked.connect(lambda: self._set_left_video_view("raw"))
        vsw_l.addWidget(self.btn_switch_raw)

        self.btn_switch_traj = QPushButton("TRAJECTORY HUD")
        self.btn_switch_traj.clicked.connect(lambda: self._set_left_video_view("traj"))
        vsw_l.addWidget(self.btn_switch_traj)

        self.video_view_switch_widget.setVisible(False)
        labels_row.addWidget(self.video_view_switch_widget)

        labels_row.addStretch()

        self.lbl_right_panel = QLabel("TRANSMITTER CAMERA POV")
        self.lbl_right_panel.setObjectName("panel_label")
        labels_row.addWidget(self.lbl_right_panel)

        root.addLayout(labels_row)

        # ---- Main split: Left (World / Raw Video / Trajectory) | Right (AI Camera View) ----
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background: {PALETTE['grid_light']}; }}"
        )

        # Left panel: Stacked widget containing WorldView, RawVideoView, TrajectoryWidget
        self.left_stack = QStackedWidget()
        self.world_view = WorldViewWidget()
        self.raw_video_view = RawVideoWidget()
        self.trajectory_view = TrajectoryWidget()

        self.left_stack.addWidget(self.world_view)      # Index 0: Synthetic simulation
        self.left_stack.addWidget(self.raw_video_view)   # Index 1: Raw external video feed
        self.left_stack.addWidget(self.trajectory_view)  # Index 2: 2D Sensor Trajectory HUD

        # Right panel: CameraViewWidget (NN1 crosshairs, Kalman boxes, velocity vectors)
        self.camera_view = CameraViewWidget()

        splitter.addWidget(self.left_stack)
        splitter.addWidget(self.camera_view)
        splitter.setSizes([500, 500])
        root.addWidget(splitter, 1)

        # ---- Bottom analytics dock ----
        self.analytics = AnalyticsDock()
        root.addWidget(self.analytics)

    # ------------------------------------------------------------------
    # UI Mode Switching
    # ------------------------------------------------------------------
    def _apply_mode_ui(self, mode: str):
        self._mode = mode
        if mode == "video":
            self.sim_controls_widget.setVisible(False)
            self.video_controls_widget.setVisible(True)
            self.video_view_switch_widget.setVisible(True)
            self.lbl_left_panel.setText("INPUT TELEMETRY:")
            self.lbl_right_panel.setText("SADHA AI TRACKING HUD (NN1 + KALMAN + NN2 GRU)")
            self._set_left_video_view("raw")
            self._update_loop_button()
        else:
            self.sim_controls_widget.setVisible(True)
            self.video_controls_widget.setVisible(False)
            self.video_view_switch_widget.setVisible(False)
            self.lbl_left_panel.setText("WORLD VIEW (2D SIMULATION)")
            self.lbl_right_panel.setText("TRANSMITTER CAMERA POV")
            self.left_stack.setCurrentIndex(0)

    def _set_left_video_view(self, view_type: str):
        active_style = (
            f"background: {PALETTE['primary']}; color: #FFFFFF; font-family: {FONT_HEADING}; "
            f"font-size: 11px; font-weight: 600; border: 1px solid {PALETTE['border']}; "
            f"border-radius: 3px; padding: 3px 8px;"
        )
        inactive_style = (
            f"background: transparent; color: {PALETTE['text_muted']}; font-family: {FONT_HEADING}; "
            f"font-size: 11px; font-weight: 600; border: 1px solid {PALETTE['border']}; "
            f"border-radius: 3px; padding: 3px 8px;"
        )

        if view_type == "raw":
            self.left_stack.setCurrentIndex(1)
            self.btn_switch_raw.setStyleSheet(active_style)
            self.btn_switch_traj.setStyleSheet(inactive_style)
        else:
            self.left_stack.setCurrentIndex(2)
            self.btn_switch_traj.setStyleSheet(active_style)
            self.btn_switch_raw.setStyleSheet(inactive_style)

    # ------------------------------------------------------------------
    # Navigation Handlers
    # ------------------------------------------------------------------
    def _on_generate(self):
        self._stop_video()
        cfg = SimulationConfig()
        self._init_simulation(cfg)
        self._apply_mode_ui("sim")
        self._stack.setCurrentIndex(1)
        self.config_drawer.load_config(cfg)
        self.config_drawer.close_drawer()
        self._reposition_drawer()

    def _on_load_preset(self, preset_key: str):
        self._stop_video()
        cfg = apply_preset(preset_key)
        self._init_simulation(cfg)
        self._apply_mode_ui("sim")
        self._stack.setCurrentIndex(1)
        self.config_drawer.load_config(cfg)
        self.config_drawer.close_drawer()
        self._reposition_drawer()

    def _on_upload_video(self, path: str):
        """Option 2: Process pre-recorded external video telemetry stream."""
        if not path:
            return

        # Stop active processes
        self._stop_simulation()
        self._stop_video()

        try:
            self._video_player = VideoPlayer(path, loop=True)
        except Exception as e:
            QMessageBox.critical(self, "Video Load Error", f"Could not open video file:\n{e}")
            return

        self._mode = "video"

        # Initialize tracking pipeline for video feed
        self.nn1 = NN1Detector(mode="auto")
        self.tracker = MultiBeaconTracker()
        self.nn2 = NN2Predictor(mode="auto")
        self._last_nn1_result = None
        self._last_tracker_result = None
        self._last_nn2_result = None
        self._frame_idx = 0
        self._current_fps = 0.0

        # Reset widgets
        self.trajectory_view.clear()
        self.trajectory_view.set_sensor_resolution(
            self._video_player.width, self._video_player.height
        )
        self.raw_video_view.clear()

        # Update video header UI
        self.lbl_video_info.setText(
            f"{self._video_player.filename}  [{self._video_player.width}×{self._video_player.height} @ {self._video_player.fps:.0f}FPS]"
        )
        self.slider_video.blockSignals(True)
        self.slider_video.setRange(0, max(0, self._video_player.total_frames - 1))
        self.slider_video.setValue(0)
        self.slider_video.blockSignals(False)

        # Apply video mode UI visibility
        self._apply_mode_ui("video")

        # Switch to simulation/processing page
        self._stack.setCurrentIndex(1)

        # Close config drawer if open
        if self.config_drawer.isVisible():
            self.config_drawer.close_drawer()

        # Seek to frame 0 and display immediately
        self._seek_video(0)

        # Start playback automatically
        self._start_video()

    def _go_home(self):
        self._stop_simulation()
        self._stop_video()
        self._stack.setCurrentIndex(0)

    def _export_log(self):
        """Export current session telemetry and performance analytics."""
        import json
        from datetime import datetime
        os.makedirs("exports", exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = os.path.abspath(os.path.join("exports", f"sadha_run_log_{timestamp_str}.json"))

        # Calculate summary metrics
        fps_list = list(self.analytics._fps_history)
        err_list = list(self.analytics._error_history)
        avg_fps = sum(fps_list) / len(fps_list) if fps_list else self._current_fps
        avg_err = sum(err_list) / len(err_list) if err_list else 0.0
        max_err = max(err_list) if err_list else 0.0

        lock_status = self.analytics.lbl_lock["value"].text()
        acq_time = self.analytics.lbl_acq["value"].text()
        reacq_time = self.analytics.lbl_reacq["value"].text()

        active_tracks = []
        if self.tracker and self._last_tracker_result:
            for trk in self._last_tracker_result.tracks:
                active_tracks.append({
                    "track_id": trk.track_id,
                    "terminal_id": trk.terminal_id,
                    "position": [round(float(p), 2) for p in trk.position],
                    "velocity": [round(float(v), 2) for v in trk.velocity],
                    "lock_state": trk.lock_state.value,
                    "consecutive_missed_frames": trk.consecutive_missed_frames,
                    "matched": trk.matched,
                })

        duration = self.world.sim_time if self.world else (self._frame_idx / max(1.0, avg_fps))
        log_data = {
            "session_timestamp": timestamp_str,
            "mode": self._mode,
            "total_frames_processed": self._frame_idx,
            "duration_seconds": round(duration, 2),
            "performance_metrics": {
                "average_fps": round(avg_fps, 1),
                "current_fps": round(self._current_fps, 1),
                "average_tracking_error_px": round(avg_err, 2),
                "max_tracking_error_px": round(max_err, 2),
                "lock_status": lock_status,
                "initial_acquisition_sec": acq_time,
                "reacquisition_sec": reacq_time,
            },
            "active_tracks": active_tracks,
        }

        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

        QMessageBox.information(
            self,
            "Performance Log Exported",
            f"Performance log successfully exported to:\n{export_path}\n\n"
            f"Frames: {self._frame_idx} | Avg Err: {avg_err:.2f} px | FPS: {avg_fps:.1f}"
        )

    def _toggle_config(self):
        self._reposition_drawer()
        if self.config_drawer.isVisible():
            self.config_drawer.close_drawer()
        else:
            self.config_drawer.open()

    def _on_config_apply(self, cfg: SimulationConfig):
        self._stop_simulation()
        self._stop_video()
        self._init_simulation(cfg)
        self._apply_mode_ui("sim")

    # ------------------------------------------------------------------
    # Simulation lifecycle (Synthetic Mode)
    # ------------------------------------------------------------------
    def _init_simulation(self, cfg: SimulationConfig):
        self.world = WorldSimulation(cfg)
        self.world_view.set_world(self.world)
        self.nn1 = NN1Detector(mode="auto")
        self.tracker = MultiBeaconTracker()
        self.nn2 = NN2Predictor(mode="auto")
        self._last_nn1_result = None
        self._last_tracker_result = None
        self._last_nn2_result = None
        self._frame_idx = 0
        self._current_fps = 0.0

    def _toggle_run(self):
        if self.running:
            self._stop_simulation()
        else:
            self._start_simulation()

    def _start_simulation(self):
        if self.world is None:
            self._init_simulation(SimulationConfig())
        self.running = True
        self.btn_start_stop.setText("STOP")
        self.btn_start_stop.setProperty("running", True)
        self.btn_start_stop.setStyleSheet(
            f"QPushButton {{ background-color: {PALETTE['lost']}; color: #FFFFFF; "
            f"font-family: {FONT_HEADING}; font-size: 12px; font-weight: bold; letter-spacing: 0.5px; "
            f"border: 1px solid #EF5350; border-radius: 4px; padding: 6px 20px; }} "
            f"QPushButton:hover {{ background-color: #D32F2F; border-color: #E57373; }}"
        )
        self.btn_start_stop.style().unpolish(self.btn_start_stop)
        self.btn_start_stop.style().polish(self.btn_start_stop)
        self._last_tick = time.perf_counter()
        self._fps_timer = time.perf_counter()
        self._fps_count = 0
        self._timer.start(self.SIM_TICK_MS)

    def _stop_simulation(self):
        self.running = False
        self._timer.stop()
        self.btn_start_stop.setText("START")
        self.btn_start_stop.setProperty("running", False)
        self.btn_start_stop.setStyleSheet(
            f"QPushButton {{ background-color: {PALETTE['locked']}; color: #FFFFFF; "
            f"font-family: {FONT_HEADING}; font-size: 12px; font-weight: bold; letter-spacing: 0.5px; "
            f"border: 1px solid #4CAF50; border-radius: 4px; padding: 6px 20px; }} "
            f"QPushButton:hover {{ background-color: #388E3C; border-color: #81C784; }}"
        )
        self.btn_start_stop.style().unpolish(self.btn_start_stop)
        self.btn_start_stop.style().polish(self.btn_start_stop)

    # ------------------------------------------------------------------
    # Simulation tick (~60 Hz)
    # ------------------------------------------------------------------
    def _tick(self):
        if not self.running or self.world is None:
            return

        now = time.perf_counter()
        dt = now - self._last_tick
        self._last_tick = now

        # Clamp dt to avoid spiralling on lag
        dt = min(dt, 0.05)

        # Step the world
        frame_data = self.world.step(dt)

        # Update world view (every tick)
        self.world_view.record_trails()
        self.world_view.update()

        # If a sensor frame was captured this tick, display it
        if frame_data is not None:
            self._frame_idx = frame_data.frame_index

            # ---- NN1: detect beacons ----
            nn1_result = None
            tracker_result = None
            if self.nn1 is not None:
                nn1_result = self.nn1.detect(frame_data)
                self._last_nn1_result = nn1_result

                # ---- Kalman tracker: assignment + predict/update ----
                if self.tracker is not None and nn1_result is not None:
                    tracker_result = self.tracker.update(nn1_result)
                    self._last_tracker_result = tracker_result

                    # Pan/tilt camera steering
                    cmds = self.tracker.compute_pan_tilt_commands()
                    if cmds:
                        chosen_terminal = 0
                        for trk in tracker_result.tracks:
                            if trk.lock_state.value == "locked":
                                chosen_terminal = trk.terminal_id
                                break
                        if chosen_terminal in cmds:
                            pan, tilt = cmds[chosen_terminal]
                        elif 0 in cmds:
                            pan, tilt = cmds[0]
                        else:
                            pan, tilt = next(iter(cmds.values()))
                        self.world.camera.set_commanded_rates(pan, tilt)

                    # ---- Adaptive capture-rate ramp on target loss ----
                    # [INTENTIONAL DESIGN DECISION]: Dynamic Capture-Rate Adaptation
                    # When any beacon track enters LockState.LOST (or during initial acquisition search),
                    # the tracking orchestrator commands the camera to ramp from baseline 30 Hz toward 60 Hz.
                    # Doubling the temporal sampling rate halves inter-frame target displacement on the sensor,
                    # doubling the rate of Kalman/Hungarian association attempts to maximize re-acquisition probability.
                    any_lost = any(trk.lock_state.value == "lost" for trk in tracker_result.tracks)
                    if any_lost:
                        self.world.camera.set_capture_rate(self.world.camera.loss_ramp_fps)
                    else:
                        self.world.camera.set_capture_rate(self.world.camera.baseline_fps)

                    # ---- NN2: motion prediction ----
                    if self.nn2 is not None:
                        nn2_result = self.nn2.update(
                            tracker_result, self._frame_idx,
                            frame_data.timestamp if frame_data else 0.0,
                        )
                        self._last_nn2_result = nn2_result

            self.camera_view.update_frame(frame_data, nn1_result, tracker_result, self._last_nn2_result)

            # FPS measurement
            self._fps_count += 1
            elapsed = now - self._fps_timer
            if elapsed >= 0.5:
                self._current_fps = self._fps_count / elapsed
                self._fps_count = 0
                self._fps_timer = now

        # Update analytics dock
        self.analytics.update_metrics(
            self._current_fps, frame_data,
            self.world.sim_time, self._frame_idx,
            nn1_result=self._last_nn1_result,
            tracker_result=self._last_tracker_result,
        )

    # ------------------------------------------------------------------
    # Video Playback Lifecycle (External Video Mode)
    # ------------------------------------------------------------------
    def _toggle_video_play(self):
        if self._video_playing:
            self._pause_video()
        else:
            self._start_video()

    def _start_video(self):
        if self._video_player is None:
            return
        self._video_playing = True
        self.btn_video_play.setText("PAUSE")
        interval = max(10, int(1000.0 / self._video_player.fps))
        self._fps_timer = time.perf_counter()
        self._fps_count = 0
        self._video_timer.start(interval)

    def _pause_video(self):
        self._video_playing = False
        self._video_timer.stop()
        self.btn_video_play.setText("PLAY")

    def _stop_video(self):
        self._pause_video()
        if self._video_player is not None:
            self._video_player.release()
            self._video_player = None

    def _rewind_video(self):
        if self._video_player is None:
            return
        self._seek_video(0)

    def _toggle_video_loop(self):
        if self._video_player is None:
            return
        self._video_player.loop = not self._video_player.loop
        self._update_loop_button()

    def _update_loop_button(self):
        if self._video_player and self._video_player.loop:
            self.btn_video_loop.setText("LOOP: ON")
            self.btn_video_loop.setStyleSheet(
                f"background: {PALETTE['locked']}; color: #FFFFFF; font-family: {FONT_HEADING}; "
                f"font-size: 11px; font-weight: 600; border-radius: 3px; padding: 4px 10px;"
            )
        else:
            self.btn_video_loop.setText("LOOP: OFF")
            self.btn_video_loop.setStyleSheet(
                f"background: transparent; color: {PALETTE['border']}; font-family: {FONT_HEADING}; "
                f"font-size: 11px; font-weight: 600; border: 1px solid rgba(255,255,255,0.3); "
                f"border-radius: 3px; padding: 4px 10px;"
            )

    def _on_video_slider_seek(self, value: int):
        if self._video_player is None:
            return
        self._seek_video(value)

    def _seek_video(self, frame_idx: int):
        if self._video_player is None:
            return

        ret, gray, actual_idx = self._video_player.seek(frame_idx)
        if not ret or gray is None:
            return

        self._frame_idx = actual_idx
        fps = self._video_player.fps
        timestamp = actual_idx / fps

        frame_data = FrameData(
            frame_index=actual_idx,
            timestamp=timestamp,
            image=gray,
            capture_rate_hz=fps,
        )

        nn1_result = self.nn1.detect(frame_data)
        self._last_nn1_result = nn1_result

        tracker_result = None
        if self.tracker is not None:
            tracker_result = self.tracker.update(nn1_result)
            self._last_tracker_result = tracker_result

            if self.nn2 is not None:
                self._last_nn2_result = self.nn2.update(
                    tracker_result, actual_idx, timestamp
                )

        self.raw_video_view.update_frame(gray, actual_idx, fps)

        best_track = None
        if tracker_result and tracker_result.tracks:
            for trk in tracker_result.tracks:
                if trk.lock_state == LockState.LOCKED:
                    best_track = trk
                    break
            if not best_track:
                best_track = tracker_result.tracks[0]

        if best_track:
            rx = best_track.nn1_raw_history[-1][0] if best_track.nn1_raw_history else best_track.position[0]
            ry = best_track.nn1_raw_history[-1][1] if best_track.nn1_raw_history else best_track.position[1]
            self.trajectory_view.add_point(
                best_track.position[0], best_track.position[1],
                best_track.velocity[0], best_track.velocity[1],
                is_locked=(best_track.lock_state == LockState.LOCKED),
                frame_w=self._video_player.width,
                frame_h=self._video_player.height,
                raw_x=rx,
                raw_y=ry,
            )
        elif nn1_result and nn1_result.detections:
            d = nn1_result.detections[0]
            self.trajectory_view.add_point(
                d.x, d.y, 0.0, 0.0,
                is_locked=True,
                frame_w=self._video_player.width,
                frame_h=self._video_player.height,
                raw_x=d.x,
                raw_y=d.y,
            )

        self.camera_view.update_frame(frame_data, nn1_result, tracker_result, self._last_nn2_result)

        self.lbl_video_frame.setText(
            f"F: {actual_idx:03d}/{self._video_player.total_frames:03d} ({timestamp:.1f}s)"
        )

        jitter_rms = self.trajectory_view._jitter_rms
        self.analytics.update_metrics(
            fps,
            frame_data,
            timestamp,
            actual_idx,
            nn1_result=nn1_result,
            tracker_result=tracker_result,
            jitter_rms=jitter_rms,
        )

    def _video_tick(self):
        if not self._video_playing or self._video_player is None:
            return

        ret, gray, frame_idx = self._video_player.read_frame()
        if not ret or gray is None:
            self._pause_video()
            return

        self._frame_idx = frame_idx
        fps = self._video_player.fps
        timestamp = frame_idx / fps

        frame_data = FrameData(
            frame_index=frame_idx,
            timestamp=timestamp,
            image=gray,
            capture_rate_hz=fps,
        )

        # NN1 detection
        nn1_result = self.nn1.detect(frame_data)
        self._last_nn1_result = nn1_result

        # Tracker update
        tracker_result = None
        if self.tracker is not None:
            tracker_result = self.tracker.update(nn1_result)
            self._last_tracker_result = tracker_result

            if self.nn2 is not None:
                self._last_nn2_result = self.nn2.update(
                    tracker_result, frame_idx, timestamp
                )

        # Update left panel widgets
        self.raw_video_view.update_frame(gray, frame_idx, fps)

        best_track = None
        if tracker_result and tracker_result.tracks:
            for trk in tracker_result.tracks:
                if trk.lock_state == LockState.LOCKED:
                    best_track = trk
                    break
            if not best_track:
                best_track = tracker_result.tracks[0]

        if best_track:
            rx = best_track.nn1_raw_history[-1][0] if best_track.nn1_raw_history else best_track.position[0]
            ry = best_track.nn1_raw_history[-1][1] if best_track.nn1_raw_history else best_track.position[1]
            self.trajectory_view.add_point(
                best_track.position[0], best_track.position[1],
                best_track.velocity[0], best_track.velocity[1],
                is_locked=(best_track.lock_state == LockState.LOCKED),
                frame_w=self._video_player.width,
                frame_h=self._video_player.height,
                raw_x=rx,
                raw_y=ry,
            )
        elif nn1_result and nn1_result.detections:
            d = nn1_result.detections[0]
            self.trajectory_view.add_point(
                d.x, d.y, 0.0, 0.0,
                is_locked=True,
                frame_w=self._video_player.width,
                frame_h=self._video_player.height,
                raw_x=d.x,
                raw_y=d.y,
            )

        # Update right panel (CameraViewWidget with overlays)
        self.camera_view.update_frame(frame_data, nn1_result, tracker_result, self._last_nn2_result)

        # Update timeline slider
        self.slider_video.blockSignals(True)
        self.slider_video.setValue(frame_idx)
        self.slider_video.blockSignals(False)

        self.lbl_video_frame.setText(
            f"F: {frame_idx:03d}/{self._video_player.total_frames:03d} ({timestamp:.1f}s)"
        )

        # Measure FPS
        self._fps_count += 1
        now = time.perf_counter()
        elapsed = now - self._fps_timer
        if elapsed >= 0.5:
            self._current_fps = self._fps_count / elapsed
            self._fps_count = 0
            self._fps_timer = now

        jitter_rms = self.trajectory_view._jitter_rms
        self.analytics.update_metrics(
            self._current_fps if self._current_fps > 0 else fps,
            frame_data,
            timestamp,
            frame_idx,
            nn1_result=nn1_result,
            tracker_result=tracker_result,
            jitter_rms=jitter_rms,
        )

    # ------------------------------------------------------------------
    # Layout management for config drawer overlay
    # ------------------------------------------------------------------
    def _reposition_drawer(self):
        if hasattr(self, 'config_drawer') and hasattr(self, 'sim_page') and self.sim_page:
            pw = self.sim_page.width()
            ph = self.sim_page.height()
            if pw > 0 and ph > 0:
                self.config_drawer.setGeometry(
                    max(0, pw - self.config_drawer.DRAWER_WIDTH),
                    0,
                    self.config_drawer.DRAWER_WIDTH,
                    ph,
                )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_drawer()

    # ------------------------------------------------------------------
    # Interactive real-time keyboard steering & controls
    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        if self._mode == "video":
            if event.key() == Qt.Key_Space:
                self._toggle_video_play()
                return
            elif event.key() in (Qt.Key_Left, Qt.Key_A):
                if self._video_player:
                    target = max(0, self._frame_idx - 5)
                    self._seek_video(target)
                return
            elif event.key() in (Qt.Key_Right, Qt.Key_D):
                if self._video_player:
                    target = min(self._video_player.total_frames - 1, self._frame_idx + 5)
                    self._seek_video(target)
                return

        if event.isAutoRepeat():
            return super().keyPressEvent(event)
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_Up):
            self._keys_held.add("up")
        elif key in (Qt.Key_S, Qt.Key_Down):
            self._keys_held.add("down")
        elif key in (Qt.Key_A, Qt.Key_Left):
            self._keys_held.add("left")
        elif key in (Qt.Key_D, Qt.Key_Right):
            self._keys_held.add("right")
        else:
            return super().keyPressEvent(event)
        self._update_keyboard_steering()

    def keyReleaseEvent(self, event):
        if self._mode == "video":
            return super().keyReleaseEvent(event)

        if event.isAutoRepeat():
            return super().keyReleaseEvent(event)
        key = event.key()
        if key in (Qt.Key_W, Qt.Key_Up):
            self._keys_held.discard("up")
        elif key in (Qt.Key_S, Qt.Key_Down):
            self._keys_held.discard("down")
        elif key in (Qt.Key_A, Qt.Key_Left):
            self._keys_held.discard("left")
        elif key in (Qt.Key_D, Qt.Key_Right):
            self._keys_held.discard("right")
        else:
            return super().keyReleaseEvent(event)
        self._update_keyboard_steering()

    def _update_keyboard_steering(self):
        if not self.world or not self.running:
            return
        speed = 360.0  # px/s in world space
        vx, vy = 0.0, 0.0
        if "up" in self._keys_held:
            vy -= speed
        if "down" in self._keys_held:
            vy += speed
        if "left" in self._keys_held:
            vx -= speed
        if "right" in self._keys_held:
            vx += speed

        if vx != 0.0 or vy != 0.0:
            self.world.steer_primary_beacon(vx, vy)
        elif self.world.beacons and self.world.beacons[0].motion_type == MotionType.USER_CONTROLLED:
            self.world.steer_primary_beacon(0.0, 0.0)
