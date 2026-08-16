import math
import time

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget


def bearing_to_screen(angle_deg: float, radius: float, cx: float, cy: float):
    """Compass angle (0 = North, clockwise) to screen (x, y)."""
    rad = math.radians(angle_deg)
    return cx + radius * math.sin(rad), cy - radius * math.cos(rad)


class PolarRadarWidget(QWidget):
    """Compass radar: plots sample RSSI vs. compass bearing, fox-hunter style.

    Mirrors RogueOS's polar radar screen: 0 degrees at top (North), RSSI maps
    to radius, fading green dots for historical samples, a red needle for the
    estimated emitter bearing, and an optional white line for live heading.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 280)
        self.points = []  # (bearing_deg, rssi_db, timestamp)
        self.rssi_min = -90.0
        self.rssi_max = -30.0
        self.emitter_bearing = None  # deg or None
        self.emitter_distance_m = None  # m or None
        self.heading = None  # live heading deg or None
        self._bg = QColor("#0d1117")

    def set_data(
        self,
        points,
        rssi_range=None,
        emitter_bearing=None,
        emitter_distance_m=None,
        heading=None,
    ):
        self.points = list(points)
        if rssi_range:
            self.rssi_min, self.rssi_max = rssi_range
        else:
            rssi = [p[1] for p in self.points] or [self.rssi_min]
            self.rssi_min = min(rssi)
            self.rssi_max = max(rssi)
            if self.rssi_max - self.rssi_min < 1.0:
                self.rssi_max = self.rssi_min + 1.0
        self.emitter_bearing = emitter_bearing
        self.emitter_distance_m = emitter_distance_m
        self.heading = heading
        self.update()

    def clear(self):
        self.points = []
        self.emitter_bearing = None
        self.emitter_distance_m = None
        self.update()

    def _plot_rect(self):
        w = self.width()
        h = self.height()
        side = max(40, min(w, h) - 26)
        left = (w - side) / 2
        top = (h - side) / 2
        return QRectF(left, top, side, side)

    def _rssi_color(self, rssi):
        t = 0.0
        span = self.rssi_max - self.rssi_min
        if span > 0:
            t = max(0.0, min(1.0, (rssi - self.rssi_min) / span))
        # green -> yellow -> red
        return QColor(
            int(60 + 195 * t), int(200 - 160 * t), int(90 - 60 * t)
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self._bg)

        rect = self._plot_rect()
        cx = rect.center().x()
        cy = rect.center().y()
        R = rect.width() / 2.0

        now = time.time()

        # --- grid rings + radial lines --------------------------------
        ring_pen = QPen(QColor(46, 54, 66, 200))
        ring_pen.setWidthF(1.0)
        tick_pen = QPen(QColor(90, 105, 125, 220))
        tick_pen.setWidthF(1.2)

        for frac in (0.25, 0.5, 0.75, 1.0):
            painter.setPen(ring_pen)
            painter.drawEllipse(QRectF(cx - R * frac, cy - R * frac, 2 * R * frac, 2 * R * frac))
        painter.setPen(tick_pen)
        for deg in range(0, 360, 30):
            x0, y0 = bearing_to_screen(deg, R, cx, cy)
            x1, y1 = bearing_to_screen(deg, R - 8, cx, cy)
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        # --- cardinal labels -------------------------------------------
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSize(max(8, font.pointSize()))
        painter.setFont(font)
        painter.setPen(QColor(170, 185, 200))
        label_r = R + 12
        for deg, text in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
            x, y = bearing_to_screen(deg, label_r, cx, cy)
            painter.drawText(
                QRectF(x - 14, y - 10, 28, 20),
                Qt.AlignmentFlag.AlignCenter,
                text,
            )

        # --- sample dots -----------------------------------------------
        if self.points:
            newest = max(p[2] for p in self.points)
            for bearing, rssi, ts in self.points:
                age = max(0.0, newest - ts)
                fade = max(60, int(255 - min(age, 240.0) * 0.8))
                color = self._rssi_color(rssi)
                color.setAlpha(fade)
                span = self.rssi_max - self.rssi_min
                frac = max(0.0, min(1.0, (rssi - self.rssi_min) / span)) if span > 0 else 0.5
                radius = max(2.5, R * 0.9 * (0.15 + 0.85 * frac))
                x, y = bearing_to_screen(bearing, radius, cx, cy)
                painter.setPen(QPen(color, 1.0))
                painter.setBrush(color)
                painter.drawEllipse(QRectF(x - 3, y - 3, 6, 6))

        # --- emitter needle ----------------------------------------------
        if self.emitter_bearing is not None:
            needle_pen = QPen(QColor("#e63232"))
            needle_pen.setWidthF(2.5)
            painter.setPen(needle_pen)
            x0, y0 = bearing_to_screen(self.emitter_bearing, R * 0.06, cx, cy)
            x1, y1 = bearing_to_screen(self.emitter_bearing, R * 0.92, cx, cy)
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))
            # arrow head
            ax, ay = bearing_to_screen(self.emitter_bearing + 160, R * 0.14, x1, y1)
            bx, by = bearing_to_screen(self.emitter_bearing - 160, R * 0.14, x1, y1)
            painter.setBrush(QColor("#e63232"))
            painter.drawPolygon(QPolygonF([QPointF(x1, y1), QPointF(ax, ay), QPointF(bx, by)]))
            if self.emitter_distance_m is not None:
                painter.setPen(QColor("#e63232"))
                painter.drawText(
                    QRectF(cx - R + 4, cy + R * 0.1, 2 * R - 8, 16),
                    Qt.AlignmentFlag.AlignHCenter,
                    "{0:03.0f} deg · {1:.0f} m".format(
                        self.emitter_bearing, self.emitter_distance_m
                    ),
                )

        # --- live heading (white) -----------------------------------------
        if self.heading is not None:
            head_pen = QPen(QColor(255, 255, 255, 220))
            head_pen.setWidthF(1.5)
            painter.setPen(head_pen)
            x0, y0 = bearing_to_screen(self.heading, R * 0.1, cx, cy)
            x1, y1 = bearing_to_screen(self.heading, R * 0.98, cx, cy)
            painter.drawLine(QPointF(x0, y0), QPointF(x1, y1))

        # --- center ---------------------------------------------------------
        painter.setPen(QPen(QColor("#2f7ff0"), 1.5))
        painter.setBrush(QColor("#2f7ff0"))
        painter.drawEllipse(QRectF(cx - 3, cy - 3, 6, 6))

        # --- RSSI gradient legend ---------------------------------------------
        lw = max(40, int(R * 1.4))
        gx = cx - lw / 2
        gy = rect.bottom() + 8
        for i in range(lw):
            t = i / max(1, lw - 1)
            color = self._rssi_color(self.rssi_min + t * (self.rssi_max - self.rssi_min))
            painter.setPen(QPen(color, 2.0))
            painter.drawLine(QPointF(gx + i, gy), QPointF(gx + i, gy + 5))
        painter.setPen(QColor(140, 152, 165))
        font.setPointSize(max(7, font.pointSize() - 1))
        painter.setFont(font)
        painter.drawText(
            QRectF(cx - lw / 2 - 34, gy - 1, 30, 12),
            Qt.AlignmentFlag.AlignRight,
            "{0:.0f}".format(self.rssi_min),
        )
        painter.drawText(
            QRectF(cx + lw / 2 + 4, gy - 1, 34, 12),
            Qt.AlignmentFlag.AlignLeft,
            "{0:.0f} dB".format(self.rssi_max),
        )
        painter.end()

    def sizeHint(self):
        return QSize(340, 340)

    def minimumSizeHint(self):
        return QSize(240, 240)
