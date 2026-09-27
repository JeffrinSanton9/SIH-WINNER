"""
SADHA — Home / Entry Screen
Three-option landing page: Simulate, Upload Video, or Load Preset.
Fully responsive and adaptive across all display resolutions and window sizes.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QListWidget, QListWidgetItem, QFrame,
    QScrollArea, QSizePolicy,
)

from src.core.config import PRESET_SCENARIOS
from src.ui.styles import PALETTE, FONT_MONO, FONT_HEADING, FONT_BODY


class HomeScreen(QWidget):
    """First screen the user sees — choose input source.
    
    Dynamically scales cards, typography, and margins as window size increases.
    """

    # Signals to the main window
    sig_generate = pyqtSignal()                   # option 1: virtual environment
    sig_upload_video = pyqtSignal(str)             # option 2: .mp4 path
    sig_load_preset = pyqtSignal(str)              # option 3: preset key

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("home_screen")
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ---- Header strip (fixed top bar) ----
        header = QFrame()
        header.setObjectName("header_bar")
        header.setFixedHeight(48)
        header.setStyleSheet(
            f"#header_bar {{ background-color: {PALETTE['primary']}; "
            f"border-bottom: 2px solid {PALETTE['secondary']}; }}"
        )
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)
        wordmark = QLabel("SADHA")
        wordmark.setObjectName("sadha_wordmark")
        hl.addWidget(wordmark)
        hl.addStretch()
        subtitle = QLabel("AI-BASED VIRTUAL CAMERA TRACKING · FSOC COARSE ALIGNMENT")
        subtitle.setStyleSheet(
            f"font-family: {FONT_HEADING}; font-size: 11px; font-weight: 600; letter-spacing: 1.5px; "
            f"color: #FFFFFF; padding-right: 18px; background: transparent;"
        )
        hl.addWidget(subtitle)
        root.addWidget(header)

        # ---- Scroll Area wrapping responsive content ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(f"background-color: {PALETTE['background']}; border: none;")

        container = QWidget()
        container.setObjectName("home_container")
        container.setStyleSheet(f"background-color: {PALETTE['background']};")

        self.container_layout = QVBoxLayout(container)
        self.container_layout.setContentsMargins(60, 32, 60, 36)
        self.container_layout.setSpacing(20)

        # ---- Header title block ----
        title_block = QVBoxLayout()
        title_block.setSpacing(6)

        title = QLabel("SELECT INPUT SOURCE")
        title.setStyleSheet(
            f"font-family: {FONT_HEADING}; font-size: 18px; font-weight: bold; "
            f"letter-spacing: 1.5px; color: #1E293B; background: transparent;"
        )
        title.setAlignment(Qt.AlignCenter)
        title_block.addWidget(title)

        tagline = QLabel(
            "Choose an operational mode to launch real-time simulation, "
            "feed external telemetry video, or execute standardized benchmark presets."
        )
        tagline.setStyleSheet(
            f"font-family: {FONT_BODY}; font-size: 13px; font-weight: normal; "
            f"color: {PALETTE['text_secondary']}; background: transparent;"
        )
        tagline.setAlignment(Qt.AlignCenter)
        title_block.addWidget(tagline)

        self.container_layout.addLayout(title_block)

        # ---- Responsive Cards Row ----
        self.cards_layout = QHBoxLayout()
        self.cards_layout.setSpacing(24)

        # Card 1: Generate Virtual Environment
        c1 = self._make_card(
            num="01",
            badge="SIMULATION ENGINE",
            title="GENERATE VIRTUAL ENVIRONMENT",
            desc="Configure synthetic world, camera optics, target kinematics, and disturbances. "
                 "Module 1 generates the scene while Module 2 performs real-time detection & homing.",
            features=[
                "Module 1 Physics & Synthetic Generator",
                "Real-time NN1 Heatmap Detection & Kalman Homing",
                "Interactive Mouse Drag & WASD Steering Controls",
                "Configurable Platform Jitter, Noise & Fog"
            ],
            btn_text="CONFIGURE & RUN SIMULATION",
        )
        c1["btn"].clicked.connect(self.sig_generate.emit)
        self.cards_layout.addWidget(c1["frame"], 1)

        # Card 2: Upload External Video
        c2 = self._make_card(
            num="02",
            badge="EXTERNAL FEED",
            title="UPLOAD EXTERNAL VIDEO",
            desc="Load a pre-recorded .mp4, .avi, or .mkv mission stream containing beacon movement. "
                 "Bypasses synthetic generation — Module 2 processes the feed frame-by-frame.",
            features=[
                "Direct Video Stream Ingestion (.mp4, .avi, .mkv)",
                "Bypasses Module 1 Synthetic Rendering",
                "Evaluates NN1 Sub-Pixel Accuracy on Real Video",
                "Tests Hungarian Gating on Empirical Trajectories"
            ],
            btn_text="SELECT VIDEO FILE",
        )
        c2["btn"].clicked.connect(self._on_upload)
        self.cards_layout.addWidget(c2["frame"], 1)

        # Card 3: Preset Benchmark Scenarios
        c3_frame = QFrame()
        c3_frame.setProperty("class", "home_option_card")
        c3_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        c3_frame.setMinimumSize(280, 420)
        c3_frame.setStyleSheet(
            f"background-color: #FFFFFF; border: 1px solid {PALETTE['grid_light']}; "
            f"border-radius: 8px; padding: 24px;"
        )
        c3l = QVBoxLayout(c3_frame)
        c3l.setContentsMargins(24, 24, 24, 24)
        c3l.setSpacing(12)

        # Card 3 top bar: Number + Badge
        c3_top = QHBoxLayout()
        num3 = QLabel("03")
        num3.setStyleSheet(
            f"font-family: {FONT_MONO}; font-size: 32px; font-weight: bold; "
            f"color: {PALETTE['primary']}; background: transparent;"
        )
        c3_top.addWidget(num3)
        c3_top.addStretch()

        badge3 = QLabel("BENCHMARK SUITE")
        badge3.setStyleSheet(
            f"font-family: {FONT_HEADING}; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; "
            f"color: {PALETTE['secondary']}; background: {PALETTE['neutral']}; "
            f"border: 1px solid {PALETTE['border']}; border-radius: 4px; padding: 4px 8px;"
        )
        c3_top.addWidget(badge3)
        c3l.addLayout(c3_top)

        t3 = QLabel("PRESET CONFIGURATIONS")
        t3.setStyleSheet(
            f"font-family: {FONT_HEADING}; font-size: 15px; font-weight: bold; "
            f"letter-spacing: 0.5px; color: #0F172A; background: transparent;"
        )
        c3l.addWidget(t3)

        d3 = QLabel("Select a standardized benchmark scenario from the SIH PS4 evaluation suite:")
        d3.setStyleSheet(
            f"font-family: {FONT_BODY}; font-size: 13px; font-weight: normal; color: #334155; "
            f"line-height: 1.4; background: transparent;"
        )
        d3.setWordWrap(True)
        c3l.addWidget(d3)

        # Preset selection list
        self.preset_list = QListWidget()
        self.preset_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preset_list.setMinimumHeight(150)
        self.preset_list.setWordWrap(True)
        self.preset_list.setSpacing(6)
        self.preset_list.setStyleSheet(
            f"font-family: {FONT_BODY}; font-size: 12px; font-weight: normal; color: {PALETTE['text_primary']}; "
            f"border: 1px solid {PALETTE['border']}; border-radius: 6px; "
            f"background: #FFFFFF; padding: 6px;"
        )

        for key, preset in PRESET_SCENARIOS.items():
            item_text = f"{preset['name']}\n   {preset['description']}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, key)
            self.preset_list.addItem(item)

        if self.preset_list.count() > 0:
            self.preset_list.setCurrentRow(0)

        self.preset_list.itemDoubleClicked.connect(lambda: self._on_preset())
        c3l.addWidget(self.preset_list, 1)

        btn3 = QPushButton("LOAD BENCHMARK PRESET")
        btn3.setFixedHeight(44)
        btn3.setStyleSheet(
            f"background-color: {PALETTE['primary']}; color: #FFFFFF; "
            f"border: none; border-radius: 6px; padding: 10px 20px; "
            f"font-family: {FONT_HEADING}; font-size: 13px; letter-spacing: 0.5px; font-weight: bold;"
        )
        btn3.clicked.connect(self._on_preset)
        c3l.addWidget(btn3)

        self.cards_layout.addWidget(c3_frame, 1)

        # Stretch factor 1 allows cards layout to scale with window height
        self.container_layout.addLayout(self.cards_layout, 1)

        # ---- Footer info strip ----
        footer = QLabel("SADHA v1.0 · SMART ADAPTIVE DISTURBANCE-AWARE HYBRID ACQUISITION · SIH PS4")
        footer.setStyleSheet(
            f"font-family: {FONT_HEADING}; font-size: 11px; font-weight: 600; "
            f"color: {PALETTE['text_muted']}; letter-spacing: 1px; padding-top: 10px; background: transparent;"
        )
        footer.setAlignment(Qt.AlignCenter)
        self.container_layout.addWidget(footer)

        scroll.setWidget(container)
        root.addWidget(scroll, 1)

    # ------------------------------------------------------------------
    # Dynamic Scaling & Adaptive Margins
    # ------------------------------------------------------------------
    def resizeEvent(self, event):
        """Adaptively scale margins, spacing, and layout proportions as window grows."""
        super().resizeEvent(event)
        w = self.width()
        h = self.height()

        # Responsive horizontal margins: proportional breathing room
        margin_x = max(36, min(140, int(w * 0.055)))
        margin_y = max(20, min(50, int(h * 0.045)))
        spacing = max(16, min(32, int(w * 0.016)))

        self.container_layout.setContentsMargins(margin_x, margin_y, margin_x, margin_y)
        self.cards_layout.setSpacing(spacing)

    # ------------------------------------------------------------------
    # Card Builder
    # ------------------------------------------------------------------
    def _make_card(self, num: str, badge: str, title: str, desc: str,
                   features: list[str], btn_text: str) -> dict:
        frame = QFrame()
        frame.setProperty("class", "home_option_card")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        frame.setMinimumSize(280, 420)
        frame.setStyleSheet(
            f"background-color: #FFFFFF; border: 1px solid {PALETTE['grid_light']}; "
            f"border-radius: 8px; padding: 24px;"
        )

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        # Top bar: Number + Badge
        top_bar = QHBoxLayout()
        n = QLabel(num)
        n.setStyleSheet(
            f"font-family: {FONT_MONO}; font-size: 32px; font-weight: bold; "
            f"color: {PALETTE['primary']}; background: transparent;"
        )
        top_bar.addWidget(n)
        top_bar.addStretch()

        b = QLabel(badge)
        b.setStyleSheet(
            f"font-family: {FONT_HEADING}; font-size: 10px; font-weight: 600; letter-spacing: 0.5px; "
            f"color: {PALETTE['secondary']}; background: {PALETTE['neutral']}; "
            f"border: 1px solid {PALETTE['border']}; border-radius: 4px; padding: 4px 8px;"
        )
        top_bar.addWidget(b)
        layout.addLayout(top_bar)

        # Title
        t = QLabel(title)
        t.setStyleSheet(
            f"font-family: {FONT_HEADING}; font-size: 15px; font-weight: bold; "
            f"letter-spacing: 0.5px; color: #0F172A; background: transparent;"
        )
        t.setWordWrap(True)
        layout.addWidget(t)

        # Description
        d = QLabel(desc)
        d.setStyleSheet(
            f"font-family: {FONT_BODY}; font-size: 13px; font-weight: normal; color: #334155; "
            f"line-height: 1.4; background: transparent;"
        )
        d.setWordWrap(True)
        layout.addWidget(d)

        # Feature bullet points
        feat_box = QVBoxLayout()
        feat_box.setSpacing(6)
        for feat in features:
            flbl = QLabel(f"-  {feat}")
            flbl.setStyleSheet(
                f"font-family: {FONT_BODY}; font-size: 12px; font-weight: normal; color: {PALETTE['text_secondary']}; "
                f"background: transparent;"
            )
            flbl.setWordWrap(True)
            feat_box.addWidget(flbl)
        layout.addLayout(feat_box)

        # Expanding vertical space pushes the button to the bottom
        layout.addStretch(1)

        # Action Button
        btn = QPushButton(btn_text)
        btn.setFixedHeight(44)
        btn.setStyleSheet(
            f"background-color: {PALETTE['primary']}; color: #FFFFFF; "
            f"border: none; border-radius: 6px; padding: 10px 20px; "
            f"font-family: {FONT_HEADING}; font-size: 13px; letter-spacing: 0.5px; font-weight: bold;"
        )
        layout.addWidget(btn)

        return {"frame": frame, "btn": btn}

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv)"
        )
        if path:
            self.sig_upload_video.emit(path)

    def _on_preset(self):
        item = self.preset_list.currentItem()
        if item:
            key = item.data(Qt.UserRole)
            self.sig_load_preset.emit(key)
