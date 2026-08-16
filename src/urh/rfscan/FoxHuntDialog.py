import math
import os
import time
from datetime import datetime

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from urh.rfscan.Forensics import build_case, write_case
from urh.rfscan.Geolocator import (
    estimate_confidence,
    haversine_m,
    trilaterate,
    weighted_centroid,
)
from urh.rfscan.PolarRadarWidget import PolarRadarWidget
from urh.util.Logger import logger


def compass_bearing(lat1, lon1, lat2, lon2) -> float:
    """Initial compass bearing (0 = North, clockwise) from point 1 to point 2."""
    r = math.radians
    dlon = r(lon2 - lon1)
    y = math.sin(dlon) * math.cos(r(lat2))
    x = math.cos(r(lat1)) * math.sin(r(lat2)) - math.sin(r(lat1)) * math.cos(
        r(lat2)
    ) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


class FoxHuntDialog(QDialog):
    """Live fox-hunting dashboard: bearing radar + emitter estimate + export.

    Non-modal; refreshes itself from the owning RF exploration tab so the radar
    tracks new samples and the emitter estimate as you walk.
    """

    REFRESH_MS = 500

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.setWindowTitle("Fox-hunt radar")
        self.resize(760, 520)

        layout = QVBoxLayout(self)

        self.ui_lblTitle = QLabel("Fox-hunt")
        self.ui_lblTitle.setStyleSheet("font-weight:bold; font-size:15px;")
        layout.addWidget(self.ui_lblTitle)

        body = QHBoxLayout()
        self.radar = PolarRadarWidget(self)
        body.addWidget(self.radar, 3)

        panel = QFrame(self)
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        form = QFormLayout(panel)
        self.ui_lblBearing = QLabel("-")
        self.ui_lblDistance = QLabel("-")
        self.ui_lblCentroid = QLabel("-")
        self.ui_lblP0 = QLabel("-")
        self.ui_lblN = QLabel("-")
        self.ui_lblRms = QLabel("-")
        self.ui_lblConfidence = QLabel("-")
        self.ui_lblSamples = QLabel("-")
        self.ui_lblHeading = QLabel("-")
        form.addRow("Emitter bearing:", self.ui_lblBearing)
        form.addRow("Distance:", self.ui_lblDistance)
        form.addRow("Centroid:", self.ui_lblCentroid)
        form.addRow("Power P0:", self.ui_lblP0)
        form.addRow("Path loss n:", self.ui_lblN)
        form.addRow("RMS:", self.ui_lblRms)
        form.addRow("Confidence:", self.ui_lblConfidence)
        form.addRow("Samples shown:", self.ui_lblSamples)
        form.addRow("Heading:", self.ui_lblHeading)
        body.addWidget(panel, 1)
        layout.addLayout(body, 1)

        buttons = QHBoxLayout()
        self.ui_btnExport = QPushButton("Export forensic report (JSON+HTML)")
        self.ui_btnPng = QPushButton("Save radar image")
        self.ui_btnClose = QPushButton("Close")
        buttons.addWidget(self.ui_btnExport)
        buttons.addWidget(self.ui_btnPng)
        buttons.addStretch(1)
        buttons.addWidget(self.ui_btnClose)
        layout.addLayout(buttons)

        self.ui_btnExport.clicked.connect(self._export_report)
        self.ui_btnPng.clicked.connect(self._export_png)
        self.ui_btnClose.clicked.connect(self.close)

        self._timer = QTimer(self)
        self._timer.setInterval(self.REFRESH_MS)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ------------------------------------------------------------------ data

    def _samples(self):
        if self.controller is None:
            return []
        return self.controller._filtered_samples()

    def _origin(self):
        """Origin for bearing plots: current GPS fix, else survey centroid."""
        provider = self.controller.gps_provider if self.controller else None
        if provider is not None and provider.has_fix():
            return provider.position[0], provider.position[1]
        samples = self._samples()
        if samples:
            coords = [(s["lat"], s["lon"], s["rssi"]) for s in samples]
            cen = weighted_centroid(coords)
            if cen is not None:
                return cen[0], cen[1]
        return None

    def _estimate(self):
        samples = self._samples()
        if len(samples) < 3:
            return None
        coords = [(s["lat"], s["lon"], s["rssi"]) for s in samples]
        return trilaterate(coords)

    # ----------------------------------------------------------------- refresh

    def _refresh(self):
        samples = self._samples()
        origin = self._origin()
        estimate = self._estimate()

        points = []
        for s in samples:
            if origin is None:
                continue
            brg = compass_bearing(origin[0], origin[1], s["lat"], s["lon"])
            points.append((brg, s["rssi"], s.get("ts", time.time())))

        emitter_bearing = None
        emitter_dist = None
        if estimate is not None and origin is not None:
            emitter_bearing = compass_bearing(
                origin[0], origin[1], estimate[0], estimate[1]
            )
            emitter_dist = haversine_m(origin[0], origin[1], estimate[0], estimate[1])

        rssi_range = None
        if points:
            rssi_range = (min(p[1] for p in points), max(p[1] for p in points))

        self.radar.set_data(
            points,
            rssi_range=rssi_range,
            emitter_bearing=emitter_bearing,
            emitter_distance_m=emitter_dist,
            heading=None,
        )

        if estimate is not None:
            self.ui_lblBearing.setText(
                "{0:.0f} deg".format(emitter_bearing)
                if emitter_bearing is not None
                else "-"
            )
            self.ui_lblDistance.setText(
                "{0:.0f} m".format(emitter_dist) if emitter_dist is not None else "-"
            )
            self.ui_lblP0.setText("{0:.1f} dB".format(estimate[2]))
            self.ui_lblN.setText("{0:.2f}".format(estimate[3]))
            self.ui_lblRms.setText("{0:.2f} dB".format(estimate[4]))
            self.ui_lblConfidence.setText(estimate_confidence(estimate[4]))
        else:
            self.ui_lblBearing.setText("-")
            self.ui_lblDistance.setText("-")
            self.ui_lblP0.setText("-")
            self.ui_lblN.setText("-")
            self.ui_lblRms.setText("-")
            self.ui_lblConfidence.setText("-")

        if origin is not None:
            self.ui_lblCentroid.setText("{0:.6f}, {1:.6f}".format(origin[0], origin[1]))
        else:
            self.ui_lblCentroid.setText("-")
        self.ui_lblSamples.setText(str(len(samples)))
        self.ui_lblHeading.setText("n/a")

    # ----------------------------------------------------------------- export

    def _export_report(self):
        samples = self._samples()
        if not samples:
            logger.warning("No samples to export")
            return
        estimate = self._estimate()
        origin = self._origin()
        meta = {
            "frequency_filter_mhz": None,
            "case_note": "",
        }
        case = build_case(samples, estimate, origin=origin, meta=meta)
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        base, _ = QFileDialog.getSaveFileName(
            self,
            "Export forensic report",
            os.path.expanduser(os.path.join("~", "RF_case_{0}".format(stamp))),
            "Report files (*.json *.html)",
        )
        if not base:
            return
        try:
            json_path, html_path = write_case(base, case)
            logger.info("Forensic report written: {0}, {1}".format(json_path, html_path))
            self.ui_lblTitle.setText("Fox-hunt - report saved")
        except Exception as e:
            logger.error("Report export failed: {0}".format(e))

    def _export_png(self):
        if self.radar.width() <= 0:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save radar image",
            os.path.expanduser(os.path.join("~", "radar.png")),
            "PNG image (*.png)",
        )
        if not path:
            return
        self.radar.grab().save(path)
        logger.info("Radar image saved: {0}".format(path))

    def closeEvent(self, event):
        self._timer.stop()
        super().closeEvent(event)
