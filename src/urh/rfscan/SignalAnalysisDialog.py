from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class SpectrumWidget(QWidget):
    """Minimal dependency-free FFT spectrum plot (dB vs. absolute frequency)."""

    GRID_GRANULARITY_DB = 10.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._freqs_hz = None
        self._mag_db = None
        self._center_hz = None
        self.setMinimumHeight(240)

    def set_analysis(self, analysis):
        if analysis is not None and analysis.get("mag_db") is not None:
            self._freqs_hz = analysis["freqs_hz"]
            self._mag_db = analysis["mag_db"]
            self._center_hz = analysis.get("center_freq", 0.0)
        else:
            self._freqs_hz = None
            self._mag_db = None
            self._center_hz = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(16, 16, 20))
        w, h = self.width(), self.height()
        margin = 8

        painter.setPen(QPen(QColor(120, 120, 120), 1))
        font = QFont(painter.font())
        font.setPointSize(8)
        painter.setFont(font)

        if self._freqs_hz is None or self._mag_db is None:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No spectrum available")
            return

        x0, x1 = float(self._freqs_hz[0]), float(self._freqs_hz[-1])
        y0, y1 = float(self._mag_db.min()), float(self._mag_db.max())
        y0 = min(y0, y1 - 20.0)

        def to_px(freq_hz, db):
            px = margin + (freq_hz - x0) / (x1 - x0) * (w - 2 * margin)
            py = h - margin - (db - y0) / (y1 - y0) * (h - 2 * margin)
            return px, py

        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for db in range(int(y0 // self.GRID_GRANULARITY_DB) * int(self.GRID_GRANULARITY_DB),
                        int(y1), int(self.GRID_GRANULARITY_DB)):
            _, py = to_px(x0, float(db))
            painter.drawLine(int(margin), int(py), int(w - margin), int(py))
            painter.setPen(QColor(180, 180, 180))
            painter.drawText(int(margin) + 2, int(py) - 2, "{0:.0f}".format(db))
            painter.setPen(QPen(QColor(60, 60, 60), 1))

        if self._center_hz:
            cx, cy0 = to_px(float(self._center_hz), y0)
            _, cy1 = to_px(float(self._center_hz), y1)
            painter.setPen(QPen(QColor(80, 140, 80), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(cx), int(cy0), int(cx), int(cy1))
            painter.setPen(QColor(140, 200, 140))
            painter.drawText(int(cx) + 2, int(cy0) + 8, "{0:.3f} MHz".format(self._center_hz / 1e6))

        mag = self._mag_db
        n = len(mag)
        step = max(1, n // max(1, w))
        pts = []
        for i in range(0, n, step):
            px, py = to_px(float(self._freqs_hz[i]), float(mag[i]))
            pts.append((int(px), int(py)))
        if pts:
            painter.setPen(QPen(QColor(70, 170, 255), 1))
            for j in range(1, len(pts)):
                painter.drawLine(pts[j - 1][0], pts[j - 1][1], pts[j][0], pts[j][1])

        painter.setPen(QColor(180, 180, 180))
        tx, ty = to_px(x0, y0)
        painter.drawText(int(margin), h - 3, "{0:.1f} MHz".format(x0 / 1e6))
        tx2, _ = to_px(x1, y0)
        painter.drawText(int(w - 2 * margin - 45), h - 3, "{0:.1f} MHz".format(x1 / 1e6))


class SignalAnalysisDialog(QDialog):
    """Shows the spectral dissection of one captured sample."""

    def __init__(self, sample, analysis, parent=None):
        super().__init__(parent)
        self.sample = sample
        self.analysis = analysis
        self.setWindowTitle(
            "Signal dissection - {0:.3f} MHz".format(sample["freq"] / 1e6)
        )
        self.setMinimumSize(720, 460)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        self.ui_spectrum = SpectrumWidget()
        self.ui_spectrum.set_analysis(self.analysis)
        layout.addWidget(self.ui_spectrum, 1)

        a = self.analysis
        if a is not None and a.get("noise_floor_db") is not None:
            info = (
                "Center {0:.3f} MHz | FFT {1} | noise floor {2:.1f} dB | "
                "bandwidth {3:.0f} kHz | {4} peaks | {5:.6f}, {6:.6f} @ {7}"
            ).format(
                a.get("center_freq", 0.0) / 1e6,
                a.get("n_fft", 0),
                a.get("noise_floor_db", 0.0),
                a.get("bandwidth_hz", 0.0) / 1e3,
                a.get("n_peaks", 0),
                self.sample["lat"],
                self.sample["lon"],
                self.sample.get("time_str", ""),
            )
        else:
            info = "No spectrum data captured for this sample."
        layout.addWidget(QLabel(info))

        self.ui_peaks = QTableWidget(0, 4)
        self.ui_peaks.setHorizontalHeaderLabels(
            ["Freq (MHz)", "dB", "dB above floor", "Width (kHz)"]
        )
        self.ui_peaks.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        peaks = a.get("peaks", []) if a else []
        for p in peaks:
            row = self.ui_peaks.rowCount()
            self.ui_peaks.insertRow(row)
            self.ui_peaks.setItem(row, 0, QTableWidgetItem("{0:.5f}".format(p["freq_mhz"])))
            self.ui_peaks.setItem(row, 1, QTableWidgetItem("{0:.1f}".format(p["db"])))
            self.ui_peaks.setItem(
                row, 2, QTableWidgetItem("{0:.1f}".format(p["db_above_floor"]))
            )
            self.ui_peaks.setItem(row, 3, QTableWidgetItem("{0:.1f}".format(p["width_hz"] / 1e3)))
        layout.addWidget(self.ui_peaks, 2)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self.ui_peaks.setFocus()
