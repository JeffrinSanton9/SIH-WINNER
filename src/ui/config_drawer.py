"""
SADHA — Configuration Drawer (Right-Edge Overlay)
Floats over the existing layout — does NOT resize the main panels.
Contains every user-configurable parameter from the spec.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QGroupBox, QFormLayout,
)

from src.core.types import (
    MotionType, PlatformMotionType, TargetShape, AtmosphericCondition,
)
from src.core.config import SimulationConfig, PRESET_SCENARIOS, apply_preset
from src.ui.styles import PALETTE, FONT_MONO, FONT_HEADING, FONT_BODY


class ConfigDrawer(QWidget):
    """Right-edge configuration overlay panel."""

    sig_apply = pyqtSignal(object)   # emits SimulationConfig
    sig_close = pyqtSignal()

    DRAWER_WIDTH = 380

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("config_overlay")
        self.setFixedWidth(self.DRAWER_WIDTH)
        self._config = SimulationConfig()
        self._build_ui()
        self.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_config(self, cfg: SimulationConfig) -> None:
        """Populate all controls from a SimulationConfig."""
        self._config = cfg
        self._populate_from_config()

    def open(self):
        if self.parent():
            pw = self.parent().width()
            ph = self.parent().height()
            if pw > 0 and ph > 0:
                self.setGeometry(max(0, pw - self.DRAWER_WIDTH), 0, self.DRAWER_WIDTH, ph)
        self.show()
        self.raise_()

    def close_drawer(self):
        self.hide()
        self.sig_close.emit()

    # ------------------------------------------------------------------
    # Build UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Header ----
        hdr = QFrame()
        hdr.setFixedHeight(44)
        hdr.setStyleSheet(f"background-color: {PALETTE['primary']};")
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(14, 0, 8, 0)
        title = QLabel("CONFIGURATION")
        title.setStyleSheet(
            f"font-family: {FONT_HEADING}; font-size: 13px; font-weight: bold; "
            f"letter-spacing: 1px; color: #FFFFFF; background: transparent;"
        )
        hl.addWidget(title)
        hl.addStretch()
        close_btn = QPushButton("X")
        close_btn.setFixedSize(28, 28)
        close_btn.setStyleSheet(
            "background: transparent; color: #FFFFFF; font-size: 14px; font-weight: bold; border: none;"
        )
        close_btn.clicked.connect(self.close_drawer)
        hl.addWidget(close_btn)
        root.addWidget(hdr)

        # ---- Scrollable content ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet(f"background: {PALETTE['background']}; border: none;")
        content = QWidget()
        self._form = QVBoxLayout(content)
        self._form.setContentsMargins(14, 10, 14, 14)
        self._form.setSpacing(6)

        self._build_world_section()
        self._build_camera_section()
        self._build_target_section()
        self._build_platform_section()
        self._build_disturbance_section()
        self._build_preset_section()

        self._form.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # ---- Apply button ----
        apply_btn = QPushButton("APPLY CONFIGURATION")
        apply_btn.setFixedHeight(40)
        apply_btn.setStyleSheet(
            f"background-color: {PALETTE['primary']}; color: #FFFFFF; border: none; "
            f"font-family: {FONT_HEADING}; font-size: 13px; letter-spacing: 0.5px; font-weight: bold;"
        )
        apply_btn.clicked.connect(self._on_apply)
        root.addWidget(apply_btn)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------
    def _section(self, title: str) -> QFormLayout:
        lbl = QLabel(title)
        lbl.setObjectName("config_section_title")
        self._form.addWidget(lbl)
        fl = QFormLayout()
        fl.setLabelAlignment(Qt.AlignRight)
        fl.setSpacing(5)
        self._form.addLayout(fl)
        return fl

    def _build_world_section(self):
        fl = self._section("WORLD")
        self.sp_world_w = QSpinBox(); self.sp_world_w.setRange(2000, 8000); self.sp_world_w.setSingleStep(200)
        self.sp_world_h = QSpinBox(); self.sp_world_h.setRange(2000, 8000); self.sp_world_h.setSingleStep(200)
        fl.addRow("Width (px):", self.sp_world_w)
        fl.addRow("Height (px):", self.sp_world_h)

    def _build_camera_section(self):
        fl = self._section("CAMERA")
        self.sp_cam_w = QSpinBox(); self.sp_cam_w.setRange(320, 1920); self.sp_cam_w.setSingleStep(80)
        self.sp_cam_h = QSpinBox(); self.sp_cam_h.setRange(240, 1080); self.sp_cam_h.setSingleStep(60)
        self.sp_fov_h = QDoubleSpinBox(); self.sp_fov_h.setRange(1.0, 20.0); self.sp_fov_h.setSingleStep(0.5)
        self.sp_fov_v = QDoubleSpinBox(); self.sp_fov_v.setRange(1.0, 15.0); self.sp_fov_v.setSingleStep(0.5)
        self.sp_pan_spd = QDoubleSpinBox(); self.sp_pan_spd.setRange(5.0, 10.0); self.sp_pan_spd.setSingleStep(0.5)
        self.sp_tilt_spd = QDoubleSpinBox(); self.sp_tilt_spd.setRange(5.0, 10.0); self.sp_tilt_spd.setSingleStep(0.5)
        fl.addRow("Sensor W (px):", self.sp_cam_w)
        fl.addRow("Sensor H (px):", self.sp_cam_h)
        fl.addRow("FOV H (°):", self.sp_fov_h)
        fl.addRow("FOV V (°):", self.sp_fov_v)
        fl.addRow("Max Pan (°/s):", self.sp_pan_spd)
        fl.addRow("Max Tilt (°/s):", self.sp_tilt_spd)

    def _build_target_section(self):
        fl = self._section("TARGET")
        self.sp_tgt_count = QSpinBox(); self.sp_tgt_count.setRange(1, 5)
        self.cb_tgt_shape = QComboBox(); self.cb_tgt_shape.addItems([s.value for s in TargetShape])
        self.sp_tgt_size = QSpinBox(); self.sp_tgt_size.setRange(5, 20)
        self.cb_tgt_motion = QComboBox(); self.cb_tgt_motion.addItems([m.value for m in MotionType])
        self.sp_tgt_speed = QDoubleSpinBox(); self.sp_tgt_speed.setRange(0.5, 10.0); self.sp_tgt_speed.setSingleStep(0.5)
        self.sp_tgt_radius = QDoubleSpinBox(); self.sp_tgt_radius.setRange(50, 1000); self.sp_tgt_radius.setSingleStep(50)
        self.chk_tgt_loop = QCheckBox("Loop recorded path")
        fl.addRow("Count:", self.sp_tgt_count)
        fl.addRow("Shape:", self.cb_tgt_shape)
        fl.addRow("Size (px):", self.sp_tgt_size)
        fl.addRow("Motion:", self.cb_tgt_motion)
        fl.addRow("Speed:", self.sp_tgt_speed)
        fl.addRow("Radius (px):", self.sp_tgt_radius)
        fl.addRow("", self.chk_tgt_loop)

    def _build_platform_section(self):
        fl = self._section("PLATFORM MOTION")
        self.chk_plat_enabled = QCheckBox("Enable platform motion")
        self.cb_plat_type = QComboBox(); self.cb_plat_type.addItems([t.value for t in PlatformMotionType])
        self.sp_plat_disp = QDoubleSpinBox(); self.sp_plat_disp.setRange(0.0, 20.0); self.sp_plat_disp.setSingleStep(1.0)
        fl.addRow("", self.chk_plat_enabled)
        fl.addRow("Type:", self.cb_plat_type)
        fl.addRow("Max Displ (px/f):", self.sp_plat_disp)

    def _build_disturbance_section(self):
        fl = self._section("DISTURBANCES")
        self.chk_sp = QCheckBox("Salt & Pepper")
        self.sp_sp_dens = QDoubleSpinBox(); self.sp_sp_dens.setRange(0.0, 0.20); self.sp_sp_dens.setSingleStep(0.01); self.sp_sp_dens.setDecimals(3)
        self.chk_gauss = QCheckBox("Gaussian Noise")
        self.sp_gauss_std = QDoubleSpinBox(); self.sp_gauss_std.setRange(0.0, 50.0); self.sp_gauss_std.setSingleStep(1.0)
        self.chk_poisson = QCheckBox("Poisson Noise")
        self.chk_jitter = QCheckBox("Camera Jitter")
        self.sp_jitter_amp = QDoubleSpinBox(); self.sp_jitter_amp.setRange(0.0, 20.0); self.sp_jitter_amp.setSingleStep(1.0)
        self.cb_atmo = QComboBox(); self.cb_atmo.addItems([a.value for a in AtmosphericCondition])
        fl.addRow("", self.chk_sp)
        fl.addRow("  Density:", self.sp_sp_dens)
        fl.addRow("", self.chk_gauss)
        fl.addRow("  Std Dev:", self.sp_gauss_std)
        fl.addRow("", self.chk_poisson)
        fl.addRow("", self.chk_jitter)
        fl.addRow("  Amplitude:", self.sp_jitter_amp)
        fl.addRow("Atmospheric:", self.cb_atmo)

    def _build_preset_section(self):
        fl = self._section("PRESETS")
        self.cb_preset = QComboBox()
        self.cb_preset.addItem("— Select Preset —", "")
        for key, preset in PRESET_SCENARIOS.items():
            self.cb_preset.addItem(preset["name"], key)
        self.cb_preset.currentIndexChanged.connect(self._on_preset_selected)
        fl.addRow("Load:", self.cb_preset)

    # ------------------------------------------------------------------
    # Populate controls from config
    # ------------------------------------------------------------------
    def _populate_from_config(self):
        c = self._config
        self.sp_world_w.setValue(c.world.width)
        self.sp_world_h.setValue(c.world.height)
        self.sp_cam_w.setValue(c.camera.sensor_width)
        self.sp_cam_h.setValue(c.camera.sensor_height)
        self.sp_fov_h.setValue(c.camera.fov_h)
        self.sp_fov_v.setValue(c.camera.fov_v)
        self.sp_pan_spd.setValue(c.camera.max_pan_speed)
        self.sp_tilt_spd.setValue(c.camera.max_tilt_speed)
        self.sp_tgt_count.setValue(c.target.count)
        self.cb_tgt_shape.setCurrentText(c.target.shape.value)
        self.sp_tgt_size.setValue(c.target.size)
        self.cb_tgt_motion.setCurrentText(c.target.motion_type.value)
        self.sp_tgt_speed.setValue(c.target.speed)
        self.sp_tgt_radius.setValue(c.target.radius)
        self.chk_tgt_loop.setChecked(c.target.manual_loop)
        self.chk_plat_enabled.setChecked(c.platform.enabled)
        self.cb_plat_type.setCurrentText(c.platform.motion_type.value)
        self.sp_plat_disp.setValue(c.platform.max_displacement)
        d = c.disturbances
        self.chk_sp.setChecked(d.salt_pepper.enabled)
        self.sp_sp_dens.setValue(d.salt_pepper.density)
        self.chk_gauss.setChecked(d.gaussian.enabled)
        self.sp_gauss_std.setValue(d.gaussian.std_dev)
        self.chk_poisson.setChecked(d.poisson.enabled)
        self.chk_jitter.setChecked(d.camera_jitter.enabled)
        self.sp_jitter_amp.setValue(d.camera_jitter.amplitude)
        self.cb_atmo.setCurrentText(d.atmospheric.value)

    # ------------------------------------------------------------------
    # Read controls back into config
    # ------------------------------------------------------------------
    def _read_config(self) -> SimulationConfig:
        c = SimulationConfig()
        c.world.width = self.sp_world_w.value()
        c.world.height = self.sp_world_h.value()
        c.camera.sensor_width = self.sp_cam_w.value()
        c.camera.sensor_height = self.sp_cam_h.value()
        c.camera.fov_h = self.sp_fov_h.value()
        c.camera.fov_v = self.sp_fov_v.value()
        c.camera.max_pan_speed = self.sp_pan_spd.value()
        c.camera.max_tilt_speed = self.sp_tilt_spd.value()
        c.target.count = self.sp_tgt_count.value()
        c.target.shape = TargetShape(self.cb_tgt_shape.currentText())
        c.target.size = self.sp_tgt_size.value()
        c.target.motion_type = MotionType(self.cb_tgt_motion.currentText())
        c.target.speed = self.sp_tgt_speed.value()
        c.target.radius = self.sp_tgt_radius.value()
        c.target.manual_loop = self.chk_tgt_loop.isChecked()
        c.platform.enabled = self.chk_plat_enabled.isChecked()
        c.platform.motion_type = PlatformMotionType(self.cb_plat_type.currentText())
        c.platform.max_displacement = self.sp_plat_disp.value()
        c.disturbances.salt_pepper.enabled = self.chk_sp.isChecked()
        c.disturbances.salt_pepper.density = self.sp_sp_dens.value()
        c.disturbances.gaussian.enabled = self.chk_gauss.isChecked()
        c.disturbances.gaussian.std_dev = self.sp_gauss_std.value()
        c.disturbances.poisson.enabled = self.chk_poisson.isChecked()
        c.disturbances.camera_jitter.enabled = self.chk_jitter.isChecked()
        c.disturbances.camera_jitter.amplitude = self.sp_jitter_amp.value()
        c.disturbances.atmospheric = AtmosphericCondition(self.cb_atmo.currentText())
        return c

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_apply(self):
        cfg = self._read_config()
        self.sig_apply.emit(cfg)
        self.close_drawer()

    def _on_preset_selected(self, index):
        key = self.cb_preset.currentData()
        if key:
            cfg = apply_preset(key)
            self.load_config(cfg)
