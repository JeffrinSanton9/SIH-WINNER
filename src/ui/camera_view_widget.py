"""
SADHA — Camera View Widget (Right Panel)
Displays the monochrome transmitter camera sensor output.
Overlays:
  • NN1 detection crosshairs (green)
  • Kalman tracker: track ID, velocity vector, lock state badge
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QImage, QPixmap, QColor, QPen, QFont,
    QBrush, QRadialGradient, QPolygonF,
)
from PyQt5.QtWidgets import QWidget

from src.core.frame_data import FrameData
from src.tracking.nn1_types import NN1Result
from src.tracking.tracker_types import TrackerResult, LockState
from src.tracking.nn2_types import NN2Result
from src.ui.styles import PALETTE


class CameraViewWidget(QWidget):
    """Right-panel transmitter camera POV with NN1 + Kalman + NN2 overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("camera_panel")
        self.setMinimumSize(400, 300)
        self._pixmap: Optional[QPixmap] = None
        self._frame_data: Optional[FrameData] = None
        self._nn1_result: Optional[NN1Result] = None
        self._tracker_result: Optional[TrackerResult] = None
        self._nn2_result: Optional[NN2Result] = None
        self._show_detections = True

    def update_frame(
        self,
        frame_data: FrameData,
        nn1_result: Optional[NN1Result] = None,
        tracker_result: Optional[TrackerResult] = None,
        nn2_result: Optional[NN2Result] = None,
    ) -> None:
        """Receive frame data, NN1 detections, tracker output, and NN2 predictions."""
        self._frame_data = frame_data
        self._nn1_result = nn1_result
        self._tracker_result = tracker_result
        self._nn2_result = nn2_result
        img = frame_data.image
        h, w = img.shape[:2]

        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Background
        p.fillRect(self.rect(), QColor(PALETTE["dark_surface"]))

        if self._pixmap is None:
            self._draw_placeholder(p)
            p.end()
            return

        # Scale sensor frame to fill widget maintaining aspect ratio
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        sw, sh = int(pw * scale), int(ph * scale)
        ox = (ww - sw) // 2
        oy = (wh - sh) // 2
        p.drawPixmap(ox, oy, sw, sh, self._pixmap)

        # ---- Minimal HUD overlays ----
        hud_font = QFont("Segoe UI", 10, QFont.Bold)
        p.setFont(hud_font)

        # Centre crosshair (optical axis reference)
        cx, cy = ww // 2, wh // 2
        ch_pen = QPen(QColor(181, 181, 181, 90), 1, Qt.DashLine)
        p.setPen(ch_pen)
        p.drawLine(cx - 20, cy, cx + 20, cy)
        p.drawLine(cx, cy - 20, cx, cy + 20)

        # Frame index + capture rate
        if self._frame_data:
            fd = self._frame_data
            p.setPen(QPen(QColor(PALETTE.get("dark_text", "#E2EDF2"))))
            p.drawText(ox + 8, oy + 18,
                       f"F{fd.frame_index:05d}  {fd.capture_rate_hz:.0f}Hz")
            p.drawText(ox + 8, oy + 34,
                       f"PAN {fd.camera_pan_deg:+.2f}  TILT {fd.camera_tilt_deg:+.2f}")

        # ---- NN1 Detection Overlays ----
        if self._show_detections and self._nn1_result is not None:
            self._draw_detections(p, ox, oy, scale)

        # ---- Kalman Tracker Overlays ----
        if self._tracker_result is not None:
            self._draw_tracks(p, ox, oy, scale)

        # ---- NN2 Motion Prediction Overlays ----
        if self._nn2_result is not None and self._nn2_result.predictions:
            self._draw_nn2_predictions(p, ox, oy, scale)

        # Sensor border frame
        border_pen = QPen(QColor(PALETTE["dark_grid"]), 1)
        p.setPen(border_pen)
        p.setBrush(Qt.NoBrush)
        p.drawRect(ox, oy, sw - 1, sh - 1)

        # Track count badge (bottom-right)
        n_det = 0
        n_trk = 0
        has_nn2 = False
        if self._nn1_result is not None:
            n_det = self._nn1_result.num_detections
        if self._tracker_result is not None:
            n_trk = self._tracker_result.num_active_tracks
        if self._nn2_result is not None and self._nn2_result.predictions:
            has_nn2 = True

        badge_color = QColor(PALETTE["locked"]) if n_det > 0 else QColor(PALETTE["lost"])
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(badge_color))
        bw = 150 if has_nn2 else 103
        bx = ox + sw - (bw + 7)
        by = oy + sh - 24
        p.drawRoundedRect(bx, by, bw, 18, 3, 3)
        p.setPen(QPen(QColor("#FFFFFF")))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        text = f"DET:{n_det} TRK:{n_trk} NN2:ACT" if has_nn2 else f"DET:{n_det} TRK:{n_trk}"
        p.drawText(bx + 5, by + 13, text)

        p.end()

    def _draw_nn2_predictions(self, p: QPainter, ox: int, oy: int, scale: float):
        """Draw NN2 GRU future trajectory predictions and lookahead reticles."""
        pred_color = QColor("#00E5FF")  # High-visibility cyan for AI prediction

        for pred in self._nn2_result.predictions:
            px = ox + pred.predicted_position[0] * scale
            py = oy + pred.predicted_position[1] * scale

            # Connect current track position to NN2 prediction with dashed projection line
            if self._tracker_result is not None:
                for ts in self._tracker_result.tracks:
                    if ts.track_id == pred.track_id:
                        cur_x = ox + ts.position[0] * scale
                        cur_y = oy + ts.position[1] * scale
                        proj_pen = QPen(pred_color, 1.2, Qt.DashLine)
                        p.setPen(proj_pen)
                        p.drawLine(QPointF(cur_x, cur_y), QPointF(px, py))
                        break

            # Draw NN2 Prediction Diamond Reticle
            d_size = 7.0
            diamond = QPolygonF([
                QPointF(px, py - d_size),
                QPointF(px + d_size, py),
                QPointF(px, py + d_size),
                QPointF(px - d_size, py),
            ])
            p.setPen(QPen(pred_color, 1.5))
            p.setBrush(QBrush(QColor(0, 229, 255, 40)))
            p.drawPolygon(diamond)

            # Central prediction point
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(pred_color))
            p.drawEllipse(QPointF(px, py), 2.0, 2.0)

            # Label: NN2 (+1F)
            p.setFont(QFont("Segoe UI", 9, QFont.Bold))
            p.setPen(QPen(pred_color))
            lookahead = max(1, pred.target_frame_index - pred.frame_index)
            p.drawText(int(px + 10), int(py - 4), f"NN2 (+{lookahead}F)")

    def _draw_detections(self, p: QPainter, ox: int, oy: int, scale: float):
        """Draw NN1 detection crosshair markers."""
        for det in self._nn1_result.detections:
            wx = ox + det.x * scale
            wy = oy + det.y * scale

            # Outer glow ring
            glow = QRadialGradient(wx, wy, 18)
            glow.setColorAt(0, QColor(90, 143, 107, 100))
            glow.setColorAt(1, QColor(90, 143, 107, 0))
            p.setBrush(QBrush(glow))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(wx, wy), 18, 18)

            # Crosshair marker
            marker_pen = QPen(QColor(PALETTE["locked"]), 1.5)
            p.setPen(marker_pen)
            arm = 10
            p.drawLine(int(wx - arm), int(wy), int(wx - 4), int(wy))
            p.drawLine(int(wx + 4), int(wy), int(wx + arm), int(wy))
            p.drawLine(int(wx), int(wy - arm), int(wx), int(wy - 4))
            p.drawLine(int(wx), int(wy + 4), int(wx), int(wy + arm))

            # Centre dot
            p.setBrush(QBrush(QColor(PALETTE["locked"])))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(wx, wy), 2.5, 2.5)

    def _draw_tracks(self, p: QPainter, ox: int, oy: int, scale: float):
        """Draw Kalman tracker overlays: track ID, velocity vector, lock state."""
        for ts in self._tracker_result.tracks:
            px = ox + ts.position[0] * scale
            py = ox + ts.position[1] * scale
            # Fix: oy not ox for y
            py = oy + ts.position[1] * scale

            # Colour by lock state
            if ts.lock_state == LockState.LOCKED:
                color = QColor(PALETTE["locked"])
            elif ts.lock_state == LockState.LOST:
                color = QColor(PALETTE["lost"])
            else:
                color = QColor(PALETTE["searching"])

            # Velocity vector
            vx, vy = ts.velocity
            speed = math.sqrt(vx * vx + vy * vy)
            if speed > 0.5:
                vec_scale = scale * 0.15
                vx_px = vx * vec_scale
                vy_px = vy * vec_scale
                # Clamp vector length
                vec_len = math.sqrt(vx_px ** 2 + vy_px ** 2)
                if vec_len > 40:
                    vx_px *= 40 / vec_len
                    vy_px *= 40 / vec_len

                vel_pen = QPen(color, 1.5)
                p.setPen(vel_pen)
                p.drawLine(int(px), int(py), int(px + vx_px), int(py + vy_px))
                # Arrowhead
                angle = math.atan2(vy_px, vx_px)
                a1 = angle + 2.5
                a2 = angle - 2.5
                ah = 6
                p.drawLine(
                    int(px + vx_px), int(py + vy_px),
                    int(px + vx_px - ah * math.cos(a1)),
                    int(py + vy_px - ah * math.sin(a1)),
                )
                p.drawLine(
                    int(px + vx_px), int(py + vy_px),
                    int(px + vx_px - ah * math.cos(a2)),
                    int(py + vy_px - ah * math.sin(a2)),
                )

            # Track ID label
            p.setPen(QPen(color))
            p.setFont(QFont("Segoe UI", 9, QFont.Bold))
            p.drawText(int(px + 14), int(py - 10),
                       f"T{ts.track_id}")

            # Lock state tag
            state_str = ts.lock_state.value.upper()[:3]
            p.setFont(QFont("Segoe UI", 8))
            p.drawText(int(px + 14), int(py + 2), state_str)

            # Predicted position marker (small dot)
            ppx = ox + ts.predicted_position[0] * scale
            ppy = oy + ts.predicted_position[1] * scale
            pred_pen = QPen(color, 1, Qt.DotLine)
            p.setPen(pred_pen)
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(QPointF(ppx, ppy), 4, 4)

    def _draw_placeholder(self, p: QPainter):
        p.setPen(QPen(QColor(PALETTE["text_muted"])))
        font = QFont("Segoe UI", 12)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignCenter, "SENSOR VIEW — AWAITING FEED")
