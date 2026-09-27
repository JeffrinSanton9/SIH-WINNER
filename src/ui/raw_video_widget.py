"""
SADHA — Raw Video Feed Widget
Displays the unprocessed raw external video stream with optical boresight reference.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import QPainter, QImage, QPixmap, QColor, QPen, QFont, QBrush
from PyQt5.QtWidgets import QWidget

from src.ui.styles import PALETTE


class RawVideoWidget(QWidget):
    """Displays raw, unprocessed external video frames."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("raw_video_panel")
        self.setMinimumSize(400, 300)
        self._pixmap: Optional[QPixmap] = None
        self._frame_idx = 0
        self._fps = 30.0
        self._res_w = 640
        self._res_h = 480

    def update_frame(self, image: np.ndarray, frame_idx: int = 0, fps: float = 30.0):
        """Update display with monochrome uint8 image (H, W)."""
        h, w = image.shape[:2]
        self._res_w = w
        self._res_h = h
        self._frame_idx = frame_idx
        self._fps = fps

        qimg = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def clear(self):
        self._pixmap = None
        self._frame_idx = 0
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Dark surface background
        p.fillRect(self.rect(), QColor(PALETTE["dark_surface"]))

        ww, wh = self.width(), self.height()

        if self._pixmap is None:
            p.setFont(QFont("Segoe UI", 12, QFont.Bold))
            p.setPen(QColor(PALETTE["text_muted"]))
            p.drawText(QRectF(0, 0, ww, wh), Qt.AlignCenter, "AWAITING EXTERNAL VIDEO FEED...")
            p.end()
            return

        # Maintain aspect ratio
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale = min(ww / pw, wh / ph)
        sw, sh = int(pw * scale), int(ph * scale)
        ox = (ww - sw) // 2
        oy = (wh - sh) // 2

        # Draw raw frame
        p.drawPixmap(ox, oy, sw, sh, self._pixmap)

        # Subtle sensor boundary
        p.setPen(QPen(QColor(PALETTE["dark_grid"]), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRect(ox, oy, sw - 1, sh - 1)

        # Optical center crosshairs
        cx, cy = ox + sw // 2, oy + sh // 2
        p.setPen(QPen(QColor(181, 181, 181, 70), 1, Qt.DashLine))
        p.drawLine(cx - 20, cy, cx + 20, cy)
        p.drawLine(cx, cy - 20, cx, cy + 20)

        # Top-left telemetry badge
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.setPen(QColor(PALETTE["dark_text"]))
        p.drawText(ox + 10, oy + 20, f"RAW FEED · F{self._frame_idx:05d} · {self._fps:.0f}Hz")
        p.drawText(ox + 10, oy + 36, f"RESOLUTION: {self._res_w} × {self._res_h}")

        # Bottom-left badge: UNPROCESSED INPUT
        bx = ox + 10
        by = oy + sh - 26
        p.setBrush(QBrush(QColor(PALETTE["primary"])))
        p.setPen(QPen(QColor(PALETTE["border"]), 1))
        p.drawRoundedRect(bx, by, 160, 18, 3, 3)
        p.setPen(QColor("#FFFFFF"))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(bx + 8, by + 13, "UNPROCESSED INPUT")

        p.end()
