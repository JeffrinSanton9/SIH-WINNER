"""
SADHA — 2D Sensor Trajectory & Jitter Dispersion HUD Widget
Visualizes empirical beacon motion trajectory, centroid dispersion,
and optical jitter stability across the 2D camera sensor plane.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont,
    QRadialGradient, QLinearGradient, QPolygonF,
)
from PyQt5.QtWidgets import QWidget

from src.ui.styles import PALETTE, FONT_MONO, FONT_HEADING


class TrajectoryWidget(QWidget):
    """Real-time 2D Sensor Coordinate Trajectory & Jitter Dispersion HUD."""

    MAX_HISTORY = 400

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("trajectory_panel")
        self.setMinimumSize(400, 300)

        # History: list of (x, y, vx, vy, is_locked)
        self._history: Deque[Tuple[float, float, float, float, bool]] = deque(maxlen=self.MAX_HISTORY)

        self._frame_w = 640
        self._frame_h = 480
        self._current_x = 0.0
        self._current_y = 0.0
        self._current_vx = 0.0
        self._current_vy = 0.0
        self._is_locked = False
        self._has_data = False

        # Jitter statistics
        self._mean_x = 0.0
        self._mean_y = 0.0
        self._jitter_rms = 0.0
        self._jitter_std_x = 0.0
        self._jitter_std_y = 0.0

    def clear(self):
        self._history.clear()
        self._has_data = False
        self._current_x = 0.0
        self._current_y = 0.0
        self._current_vx = 0.0
        self._current_vy = 0.0
        self._is_locked = False
        self._mean_x = 0.0
        self._mean_y = 0.0
        self._jitter_rms = 0.0
        self.update()

    def set_sensor_resolution(self, width: int, height: int):
        self._frame_w = max(1, width)
        self._frame_h = max(1, height)
        self.update()

    def add_point(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        is_locked: bool = True,
        frame_w: Optional[int] = None,
        frame_h: Optional[int] = None,
        raw_x: Optional[float] = None,
        raw_y: Optional[float] = None,
    ):
        if frame_w:
            self._frame_w = frame_w
        if frame_h:
            self._frame_h = frame_h

        self._current_x = x
        self._current_y = y
        self._current_vx = vx
        self._current_vy = vy
        self._is_locked = is_locked
        self._has_data = True

        rx = raw_x if raw_x is not None else x
        ry = raw_y if raw_y is not None else y
        self._history.append((x, y, vx, vy, is_locked, rx, ry))
        self._update_statistics()
        self.update()

    def _update_statistics(self):
        n = len(self._history)
        if n < 2:
            self._jitter_rms = 0.0
            self._jitter_std_x = 0.0
            self._jitter_std_y = 0.0
            self._mean_x = self._current_x
            self._mean_y = self._current_y
            return

        recent = list(self._history)[-min(n, 120):]
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        k = len(xs)

        self._mean_x = sum(xs) / k
        self._mean_y = sum(ys) / k

        # Check if raw detection centroids were passed (tuple len >= 7)
        residuals_x = []
        residuals_y = []
        for p in recent:
            if len(p) >= 7 and p[5] is not None and p[6] is not None:
                residuals_x.append(p[5] - p[0])
                residuals_y.append(p[6] - p[1])

        # If raw measurement residuals exist and have variation, use them
        has_residuals = len(residuals_x) >= 2 and any(abs(rx) > 1e-4 for rx in residuals_x)
        if has_residuals:
            var_x = sum(rx ** 2 for rx in residuals_x) / len(residuals_x)
            var_y = sum(ry ** 2 for ry in residuals_y) / len(residuals_y)
        else:
            # Kinematic velocity: if target is nearly stationary, use coordinate variance around mean
            spd = math.sqrt(self._current_vx ** 2 + self._current_vy ** 2)
            if spd < 1.0:
                var_x = sum((x - self._mean_x) ** 2 for x in xs) / k
                var_y = sum((y - self._mean_y) ** 2 for y in ys) / k
            else:
                # Target is moving across the sensor plane: compute high-frequency acceleration jitter (2nd difference)
                # This isolates optical jitter from macroscopic flight trajectory.
                diffs_x = [xs[i] - 2 * xs[i - 1] + xs[i - 2] for i in range(2, k)]
                diffs_y = [ys[i] - 2 * ys[i - 1] + ys[i - 2] for i in range(2, k)]
                if diffs_x:
                    var_x = (sum(dx ** 2 for dx in diffs_x) / len(diffs_x)) / 6.0
                    var_y = (sum(dy ** 2 for dy in diffs_y) / len(diffs_y)) / 6.0
                else:
                    var_x = 0.0
                    var_y = 0.0

        self._jitter_std_x = math.sqrt(var_x)
        self._jitter_std_y = math.sqrt(var_y)
        self._jitter_rms = math.sqrt(var_x + var_y)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Background
        p.fillRect(self.rect(), QColor(PALETTE["dark_surface"]))

        ww, wh = self.width(), self.height()
        margin = 30
        avail_w = max(10, ww - margin * 2)
        avail_h = max(10, wh - margin * 2)

        # Coordinate mapping: Sensor (0..frame_w, 0..frame_h) -> Widget
        scale = min(avail_w / self._frame_w, avail_h / self._frame_h)
        sw = self._frame_w * scale
        sh = self._frame_h * scale
        ox = (ww - sw) / 2.0
        oy = (wh - sh) / 2.0

        # Draw Sensor Boundary & Grid
        self._draw_grid(p, ox, oy, sw, sh, scale)

        if not self._has_data:
            self._draw_no_data(p, ww, wh)
            p.end()
            return

        # Draw Jitter Dispersion Ellipses around mean
        self._draw_dispersion(p, ox, oy, scale)

        # Draw Trajectory Trail
        self._draw_trail(p, ox, oy, scale)

        # Draw Current Beacon Target Position
        self._draw_target(p, ox, oy, scale)

        # Draw Telemetry HUD Overlay
        self._draw_telemetry_hud(p, ww, wh)

        p.end()

    def _draw_grid(self, p: QPainter, ox: float, oy: float, sw: float, sh: float, scale: float):
        # Sensor outer boundary
        rect = QRectF(ox, oy, sw, sh)
        p.setPen(QPen(QColor(PALETTE["dark_grid"]), 1.5))
        p.setBrush(QColor("#0D1117"))
        p.drawRect(rect)

        # Subtle sub-grid lines
        grid_pen = QPen(QColor(60, 60, 60, 80), 1, Qt.DotLine)
        p.setPen(grid_pen)

        cols = 8
        rows = 6
        for c in range(1, cols):
            gx = ox + (sw / cols) * c
            p.drawLine(QPointF(gx, oy), QPointF(gx, oy + sh))

        for r in range(1, rows):
            gy = oy + (sh / rows) * r
            p.drawLine(QPointF(ox, gy), QPointF(ox + sw, gy))

        # Optical boresight center crosshairs (sensor center)
        cx = ox + sw / 2.0
        cy = oy + sh / 2.0
        ch_pen = QPen(QColor(PALETTE["silver"]), 1, Qt.DashLine)
        p.setPen(ch_pen)
        p.drawLine(QPointF(cx - 30, cy), QPointF(cx + 30, cy))
        p.drawLine(QPointF(cx, cy - 30), QPointF(cx, cy + 30))

        # Center label
        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor(181, 181, 181, 140))
        p.drawText(int(cx + 4), int(cy - 6), "BORESIGHT (0,0)")

        # Sensor boundary dimensions label
        p.drawText(int(ox + 6), int(oy + sh - 8), f"SENSOR PLANE: {self._frame_w} × {self._frame_h} PX")

    def _draw_dispersion(self, p: QPainter, ox: float, oy: float, scale: float):
        if len(self._history) < 5 or self._jitter_rms <= 0.05:
            return

        # For moving targets, center jitter dispersion around the current beacon position
        spd = math.sqrt(self._current_vx ** 2 + self._current_vy ** 2)
        if spd > 1.5:
            center_x = self._current_x
            center_y = self._current_y
        else:
            center_x = self._mean_x
            center_y = self._mean_y

        mx = ox + center_x * scale
        my = oy + center_y * scale

        # 1-sigma jitter circle / ellipse (scaled)
        rx_1 = max(4.0, self._jitter_std_x * scale * 2.0)
        ry_1 = max(4.0, self._jitter_std_y * scale * 2.0)

        p.setBrush(QBrush(QColor(46, 125, 50, 30)))
        p.setPen(QPen(QColor(46, 125, 50, 140), 1, Qt.DashLine))
        p.drawEllipse(QPointF(mx, my), rx_1, ry_1)

        # Center cross
        p.setPen(QPen(QColor(181, 181, 181, 180), 1))
        p.drawLine(QPointF(mx - 6, my), QPointF(mx + 6, my))
        p.drawLine(QPointF(mx, my - 6), QPointF(mx, my + 6))

    def _draw_trail(self, p: QPainter, ox: float, oy: float, scale: float):
        pts = list(self._history)
        n = len(pts)
        if n < 2:
            return

        for i in range(1, n):
            x1, y1, _, _, _ = pts[i - 1]
            x2, y2, _, _, _ = pts[i]

            sx1 = ox + x1 * scale
            sy1 = oy + y1 * scale
            sx2 = ox + x2 * scale
            sy2 = oy + y2 * scale

            alpha = int(30 + (220 * (i / n)))
            pen = QPen(QColor(46, 125, 50, alpha), 1.8)
            p.setPen(pen)
            p.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

            # Faint scatter point
            if i % 3 == 0:
                p.setBrush(QBrush(QColor(181, 181, 181, alpha // 2)))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(sx2, sy2), 1.5, 1.5)

    def _draw_target(self, p: QPainter, ox: float, oy: float, scale: float):
        tx = ox + self._current_x * scale
        ty = oy + self._current_y * scale

        # Outer glow
        glow = QRadialGradient(tx, ty, 22)
        glow_color = QColor(46, 125, 50) if self._is_locked else QColor(198, 40, 40)
        glow.setColorAt(0, QColor(glow_color.red(), glow_color.green(), glow_color.blue(), 120))
        glow.setColorAt(1, QColor(glow_color.red(), glow_color.green(), glow_color.blue(), 0))
        p.setBrush(QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(tx, ty), 22, 22)

        # Concentric reticle rings
        reticle_color = QColor(PALETTE["locked"]) if self._is_locked else QColor(PALETTE["lost"])
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(reticle_color, 1.5))
        p.drawEllipse(QPointF(tx, ty), 10, 10)

        # Center dot
        p.setBrush(QBrush(reticle_color))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(tx, ty), 3.0, 3.0)

        # Velocity vector
        spd = math.sqrt(self._current_vx ** 2 + self._current_vy ** 2)
        if spd > 0.5:
            v_len = min(40.0, spd * 1.5)
            vx_norm = (self._current_vx / spd) * v_len
            vy_norm = (self._current_vy / spd) * v_len
            p.setPen(QPen(QColor("#FFFFFF"), 1.5))
            p.drawLine(QPointF(tx, ty), QPointF(tx + vx_norm, ty + vy_norm))

    def _draw_telemetry_hud(self, p: QPainter, ww: int, wh: int):
        # HUD readout box in top-left
        hud_w = 210
        hud_h = 100
        hud_x = 16
        hud_y = 16

        p.setBrush(QBrush(QColor(19, 19, 19, 210)))
        p.setPen(QPen(QColor(PALETTE["border"]), 1))
        p.drawRoundedRect(hud_x, hud_y, hud_w, hud_h, 4, 4)

        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.setPen(QColor(PALETTE["dark_text"]))
        p.drawText(hud_x + 10, hud_y + 18, "BEACON MOTION TELEMETRY")

        p.setPen(QPen(QColor(PALETTE["grid_light"]), 1))
        p.drawLine(hud_x + 8, hud_y + 24, hud_x + hud_w - 8, hud_y + 24)

        p.setFont(QFont("Segoe UI", 9))
        p.setPen(QColor("#FFFFFF"))
        p.drawText(hud_x + 10, hud_y + 40, f"POS U,V:  {self._current_x:6.1f}, {self._current_y:6.1f} px")
        p.drawText(hud_x + 10, hud_y + 56, f"VEL VX,VY:{self._current_vx:+6.1f}, {self._current_vy:+6.1f} px/s")
        p.drawText(hud_x + 10, hud_y + 72, f"JITTER σ: {self._jitter_rms:6.2f} px (RMS)")

        lock_str = "STATE:   LOCKED" if self._is_locked else "STATE:   SEARCHING"
        lock_col = QColor(PALETTE["locked"]) if self._is_locked else QColor(PALETTE["lost"])
        p.setPen(lock_col)
        p.drawText(hud_x + 10, hud_y + 88, lock_str)

    def _draw_no_data(self, p: QPainter, ww: int, wh: int):
        p.setFont(QFont("Segoe UI", 12, QFont.Bold))
        p.setPen(QColor(PALETTE["text_muted"]))
        p.drawText(QRectF(0, 0, ww, wh), Qt.AlignCenter, "AWAITING BEACON TELEMETRY STREAM...")
