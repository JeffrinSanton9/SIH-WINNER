"""
SADHA — World View Widget (Left Panel)
Top-down mission-control display of the full virtual environment:
  grid, beacons with motion trails, camera frustum footprint, platform indicator.
Rendered via QPainter for maximum visual quality and HUD fidelity.
Panda3D scene graph integration point for future 3D perspective mode.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Optional

from PyQt5.QtCore import Qt, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QRadialGradient,
    QLinearGradient, QFont, QPolygonF,
)
from PyQt5.QtWidgets import QWidget

from src.sim.world import WorldSimulation
from src.ui.styles import PALETTE


class WorldViewWidget(QWidget):
    """Left-panel overhead world view with mission-control HUD aesthetic."""

    MAX_TRAIL_POINTS = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("world_panel")
        self.setMinimumSize(400, 300)
        self.world: Optional[WorldSimulation] = None

        # Beacon motion trails: beacon_id → deque of (x, y)
        self._trails: dict = {}

        # Mouse-drag path recording support
        self._is_recording = False

    def set_world(self, world: WorldSimulation):
        self.world = world
        self._trails = {b.id: deque(maxlen=self.MAX_TRAIL_POINTS) for b in world.beacons}

    def record_trails(self):
        """Call each frame to append current beacon positions to trails."""
        if self.world is None:
            return
        for b in self.world.beacons:
            trail = self._trails.get(b.id)
            if trail is None:
                trail = deque(maxlen=self.MAX_TRAIL_POINTS)
                self._trails[b.id] = trail
            trail.append((b.x, b.y))

    # ------------------------------------------------------------------
    # Coordinate mapping: world → widget
    # ------------------------------------------------------------------
    def _world_to_widget(self, wx, wy):
        if self.world is None:
            return 0, 0
        ww, wh = self.world.world_w, self.world.world_h
        margin = 12
        usable_w = self.width() - 2 * margin
        usable_h = self.height() - 2 * margin
        scale = min(usable_w / ww, usable_h / wh)
        ox = margin + (usable_w - ww * scale) * 0.5
        oy = margin + (usable_h - wh * scale) * 0.5
        return ox + wx * scale, oy + wy * scale

    def _world_scale(self):
        if self.world is None:
            return 1.0
        ww, wh = self.world.world_w, self.world.world_h
        margin = 12
        usable_w = self.width() - 2 * margin
        usable_h = self.height() - 2 * margin
        return min(usable_w / ww, usable_h / wh)

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Background — deep navy
        dark = QColor(PALETTE["dark_surface"])
        p.fillRect(self.rect(), dark)

        if self.world is None:
            self._draw_placeholder(p)
            p.end()
            return

        ww, wh = self.world.world_w, self.world.world_h
        scale = self._world_scale()

        # ---- Grid lines ----
        grid_pen = QPen(QColor(PALETTE["dark_grid"]), 1)
        p.setPen(grid_pen)
        spacing = self.world.cfg.world.width // 12
        for gx in range(0, ww + 1, spacing):
            x1, y1 = self._world_to_widget(gx, 0)
            x2, y2 = self._world_to_widget(gx, wh)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))
        for gy in range(0, wh + 1, spacing):
            x1, y1 = self._world_to_widget(0, gy)
            x2, y2 = self._world_to_widget(ww, gy)
            p.drawLine(int(x1), int(y1), int(x2), int(y2))

        # Centre crosshair
        cx, cy = self._world_to_widget(ww / 2, wh / 2)
        chpen = QPen(QColor(PALETTE["dark_accent"]), 1, Qt.DashLine)
        p.setPen(chpen)
        p.drawLine(int(cx), int(cy - 20), int(cx), int(cy + 20))
        p.drawLine(int(cx - 20), int(cy), int(cx + 20), int(cy))

        # ---- Camera frustum footprint ----
        corners = self.world.camera.frustum_corners()
        poly = QPolygonF([QPointF(*self._world_to_widget(c[0], c[1])) for c in corners])

        # Fill
        frustum_fill = QColor(181, 181, 181, 30)
        p.setBrush(QBrush(frustum_fill))
        frustum_pen = QPen(QColor(PALETTE["frustum_edge"]), 1.5)
        p.setPen(frustum_pen)
        p.drawPolygon(poly)
        p.setBrush(Qt.NoBrush)

        # Camera aim dot
        ax, ay = self._world_to_widget(self.world.camera.aim_x, self.world.camera.aim_y)
        cam_dot = QRadialGradient(ax, ay, 6)
        cam_dot.setColorAt(0, QColor(PALETTE["primary"]))
        cam_dot.setColorAt(1, QColor(PALETTE["secondary"]))
        p.setBrush(QBrush(cam_dot))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QPointF(ax, ay), 4, 4)

        # ---- Platform offset indicator ----
        if self.world.platform.enabled:
            plat_cx = ww / 2 + self.world.platform.offset_x
            plat_cy = wh / 2 + self.world.platform.offset_y
            px, py = self._world_to_widget(plat_cx, plat_cy)
            plat_pen = QPen(QColor(PALETTE["text_muted"]), 1, Qt.DotLine)
            p.setPen(plat_pen)
            p.drawLine(int(cx), int(cy), int(px), int(py))
            p.setBrush(QBrush(QColor(PALETTE["text_muted"])))
            p.setPen(Qt.NoPen)
            p.drawRect(int(px) - 3, int(py) - 3, 6, 6)

        # ---- Beacon trails ----
        trail_pen = QPen(QColor(PALETTE["beacon_glow"]), 1.2)
        trail_pen.setStyle(Qt.SolidLine)
        for bid, trail in self._trails.items():
            if len(trail) < 2:
                continue
            pts = list(trail)
            for i in range(1, len(pts)):
                alpha = int(40 + 120 * (i / len(pts)))
                c = QColor(PALETTE["beacon_glow"])
                c.setAlpha(alpha)
                trail_pen.setColor(c)
                p.setPen(trail_pen)
                x1, y1 = self._world_to_widget(pts[i - 1][0], pts[i - 1][1])
                x2, y2 = self._world_to_widget(pts[i][0], pts[i][1])
                p.drawLine(int(x1), int(y1), int(x2), int(y2))

        # ---- Beacon markers ----
        for b in self.world.beacons:
            bx, by = self._world_to_widget(b.x, b.y)

            # Outer glow
            glow = QRadialGradient(bx, by, 14)
            glow.setColorAt(0, QColor(255, 255, 255, 180))
            glow.setColorAt(1, QColor(181, 181, 181, 0))
            p.setBrush(QBrush(glow))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(bx, by), 14, 14)

            # Core dot
            p.setBrush(QBrush(QColor("#FFFFFF")))
            p.drawEllipse(QPointF(bx, by), 4, 4)

            # Label
            p.setPen(QPen(QColor(PALETTE["beacon_glow"])))
            font = QFont("Segoe UI", 9)
            p.setFont(font)
            p.drawText(int(bx) + 10, int(by) - 6, f"B{b.id}")

        # ---- HUD overlay text ----
        p.setPen(QPen(QColor(PALETTE.get("dark_text", "#E2EDF2"))))
        hud_font = QFont("Segoe UI", 10, QFont.Bold)
        p.setFont(hud_font)
        p.drawText(14, 20, f"WORLD {ww}×{wh}")
        p.drawText(14, 36, f"PAN {self.world.camera.pan:+.2f}°  TILT {self.world.camera.tilt:+.2f}°")
        p.drawText(14, 52, f"CAP {self.world.camera.current_fps:.0f} Hz")

        # Interactive Pilot Status Banner
        is_user_controlled = (
            self.world.beacons and
            self.world.beacons[0].motion_type.value == "user_controlled"
        )
        if is_user_controlled:
            p.setPen(QPen(QColor(PALETTE["searching"])))
            p.drawText(14, 68, "MANUAL PILOT ACTIVE [DRAG MOUSE / WASD TO STEER]")
        else:
            p.setPen(QPen(QColor(PALETTE.get("dark_text", "#E2EDF2"))))
            p.drawText(14, 68, "INTERACTIVE: CLICK/DRAG IN WORLD OR WASD TO PILOT")

        p.end()

    def _draw_placeholder(self, p: QPainter):
        p.setPen(QPen(QColor(PALETTE["text_muted"])))
        font = QFont("Segoe UI", 12)
        p.setFont(font)
        p.drawText(self.rect(), Qt.AlignCenter, "WORLD VIEW — NO SIMULATION ACTIVE")

    # ------------------------------------------------------------------
    # Mouse interaction: Live Drag & Waypoint Path Recording
    # ------------------------------------------------------------------
    def _widget_to_world(self, pos) -> tuple:
        """Convert widget pixel coordinates to world coordinates (wx, wy)."""
        if self.world is None:
            return (0.0, 0.0)
        ww, wh = self.world.world_w, self.world.world_h
        scale = self._world_scale()
        margin = 12
        usable_w = self.width() - 2 * margin
        usable_h = self.height() - 2 * margin
        ox = margin + (usable_w - ww * scale) * 0.5
        oy = margin + (usable_h - wh * scale) * 0.5
        wx = (pos.x() - ox) / scale
        wy = (pos.y() - oy) / scale
        return (max(0.0, min(float(ww), float(wx))), max(0.0, min(float(wh), float(wy))))

    def mousePressEvent(self, event):
        if self.world and event.button() == Qt.LeftButton:
            if self._is_recording:
                self._record_point(event.pos())
            else:
                wx, wy = self._widget_to_world(event.pos())
                self.world.drag_primary_beacon(wx, wy)
                self._is_dragging_beacon = True
                self.update()

    def mouseMoveEvent(self, event):
        if self.world and (event.buttons() & Qt.LeftButton):
            if self._is_recording:
                self._record_point(event.pos())
            elif getattr(self, "_is_dragging_beacon", False):
                wx, wy = self._widget_to_world(event.pos())
                self.world.drag_primary_beacon(wx, wy)
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging_beacon = False

    def _record_point(self, pos):
        """Convert widget coords to normalized 0–1 world coords and relay."""
        if self.world is None:
            return
        ww, wh = self.world.world_w, self.world.world_h
        wx, wy = self._widget_to_world(pos)
        nx = max(0, min(1, wx / ww))
        ny = max(0, min(1, wy / wh))
        self.world.add_path_waypoint(nx, ny)
