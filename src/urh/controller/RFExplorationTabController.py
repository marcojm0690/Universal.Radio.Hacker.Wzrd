import json
import os
import threading
import time
from datetime import datetime

from PyQt6.QtCore import QObject, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from urh import settings
from urh.gps.GpsProvider import create_gps_provider
from urh.rfscan.Geolocator import (
    estimate_confidence,
    haversine_m,
    trilaterate,
    weighted_centroid,
)
from urh.rfscan.RssiScanner import RssiScanner
from urh.util.Logger import logger

MAP_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "map", "map.html")
)

SAMPLE_SPACING_KEY = "rfexplore_spacing"
DWELL_SECONDS_KEY = "rfexplore_dwell"


class MapBridge(QObject):
    """Bridge between the Leaflet map (JS) and Python via QWebChannel."""

    map_ready = pyqtSignal()
    map_clicked = pyqtSignal(float, float)

    @pyqtSlot()
    def mapReady(self):
        self.map_ready.emit()

    @pyqtSlot(float, float)
    def mapClicked(self, lat, lon):
        self.map_clicked.emit(lat, lon)

    @pyqtSlot(str)
    def reportError(self, message):
        logger.warning("Map JS error: {0}".format(message))


class RfSampleWorker(QObject):
    """Drives RSSI sampling for all configured frequencies.

    The sampling loop runs in a daemon thread. On each tick it checks the
    current GPS position; if the receiver moved more than `spacing` meters
    since the last sample (or has dwelled for `dwell` seconds) it retunes to
    every frequency, settles, averages the RSSI and emits one sample per
    frequency. Sample data is delivered via the thread-safe `sample_ready`
    signal.
    """

    sample_ready = pyqtSignal(float, float, float, float)  # freq_hz, lat, lon, rssi

    def __init__(self, scanner: RssiScanner, gps_provider, frequencies, parent=None):
        super().__init__(parent)
        self.scanner = scanner
        self.gps_provider = gps_provider
        self.frequencies = frequencies
        self.spacing_m = 6.0
        self.dwell_s = 10.0
        self._running = False
        self._last_pos = None
        self._last_sample_time = 0.0
        self._thread = None

    def configure(self, spacing_m: float, dwell_s: float):
        self.spacing_m = max(1.0, spacing_m)
        self.dwell_s = max(2.0, dwell_s)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._last_pos = None
        self._last_sample_time = 0.0
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(3.0)
        self._thread = None

    def run(self):
        while self._running:
            try:
                provider = self.gps_provider
                pos = provider.position if provider is not None else None
                if pos is None or not provider.has_fix():
                    time.sleep(0.5)
                    continue

                now = time.time()
                moved = (
                    self._last_pos is None
                    or haversine_m(pos[0], pos[1], self._last_pos[0], self._last_pos[1])
                    >= self.spacing_m
                )
                dwelled = now - self._last_sample_time >= self.dwell_s
                if not (moved or dwelled):
                    time.sleep(0.5)
                    continue

                self.take_samples(pos)
                self._last_pos = pos
                self._last_sample_time = time.time()
            except Exception as e:
                logger.error("RfSampleWorker loop error: {0}".format(e))
                time.sleep(1.0)

    def take_samples(self, pos):
        """Measure RSSI at every frequency (may block; call off the GUI thread)."""
        lat, lon = pos[0], pos[1]
        for freq in self.frequencies:
            try:
                self.scanner.set_frequency(freq)
                self.scanner.clear_history()
                time.sleep(0.4)
                rssi = self.scanner.average_rssi(0.6)
                self.sample_ready.emit(freq, lat, lon, rssi)
            except Exception as e:
                logger.error("RfSampleWorker sample error: {0}".format(e))


class RFExplorationTabController(QWidget):
    DEFAULT_FREQUENCIES = "433.920, 868.000"
    SAMPLE_RATES = [250000, 1000000, 2048000]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.samples = []  # list of dicts: freq, lat, lon, rssi, ts
        self.scanner = RssiScanner(parent=self)
        self.gps_provider = None
        self.worker = None
        self.bridge = None
        self.ui_map_view = None
        self._map_ready = False

        self._build_ui()
        self._load_settings()
        self._init_gps()
        self._init_map()

        self.ui_btnStartStop.clicked.connect(self.on_start_stop)
        self.ui_btnSampleNow.clicked.connect(self.take_manual_sample)
        self.ui_btnClear.clicked.connect(self.clear_samples)
        self.ui_cbFreqDisplay.currentIndexChanged.connect(self.refresh_map)
        self.scanner.rssi_updated.connect(self._on_rssi)
        self.scanner.device_error.connect(self._on_device_error)

        self.gps_timer = QTimer(self)
        self.gps_timer.setInterval(3000)
        self.gps_timer.timeout.connect(self._update_gps_marker)
        self.gps_timer.start()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        layout = QVBoxLayout(self)

        controls = QGridLayout()
        controls.addWidget(QLabel("Frequencies (MHz):"), 0, 0)
        self.ui_edtFreq = QLineEdit()
        self.ui_edtFreq.setPlaceholderText("e.g. 433.920, 868.000")
        controls.addWidget(self.ui_edtFreq, 0, 1)

        controls.addWidget(QLabel("Sample rate:"), 0, 2)
        self.ui_cbSampleRate = QComboBox()
        for sr in self.SAMPLE_RATES:
            self.ui_cbSampleRate.addItem("{0:.3f} MS/s".format(sr / 1e6), sr)
        controls.addWidget(self.ui_cbSampleRate, 0, 3)

        controls.addWidget(QLabel("Gain (x0.1 dB):"), 0, 4)
        self.ui_spinGain = QSpinBox()
        self.ui_spinGain.setRange(0, 99)
        self.ui_spinGain.setValue(20)
        controls.addWidget(self.ui_spinGain, 0, 5)

        self.ui_btnStartStop = QPushButton("Start")
        controls.addWidget(self.ui_btnStartStop, 0, 6)

        self.ui_btnSampleNow = QPushButton("Sample now")
        controls.addWidget(self.ui_btnSampleNow, 0, 7)

        self.ui_btnClear = QPushButton("Clear")
        controls.addWidget(self.ui_btnClear, 0, 8)

        self.ui_chkAuto = QCheckBox("Auto-sample while moving")
        self.ui_chkAuto.setChecked(True)
        controls.addWidget(self.ui_chkAuto, 1, 0, 1, 2)

        controls.addWidget(QLabel("Spacing (m):"), 1, 2)
        self.ui_spinSpacing = QDoubleSpinBox()
        self.ui_spinSpacing.setRange(1, 200)
        self.ui_spinSpacing.setValue(6.0)
        self.ui_spinSpacing.setDecimals(1)
        controls.addWidget(self.ui_spinSpacing, 1, 3)

        controls.addWidget(QLabel("Dwell (s):"), 1, 4)
        self.ui_spinDwell = QSpinBox()
        self.ui_spinDwell.setRange(2, 600)
        self.ui_spinDwell.setValue(10)
        controls.addWidget(self.ui_spinDwell, 1, 5)

        controls.addWidget(QLabel("Show:"), 2, 0)
        self.ui_cbFreqDisplay = QComboBox()
        controls.addWidget(self.ui_cbFreqDisplay, 2, 1)

        self.ui_lblGps = QLabel("GPS: searching...")
        controls.addWidget(self.ui_lblGps, 2, 2, 1, 3)

        self.ui_lblLive = QLabel("")
        controls.addWidget(self.ui_lblLive, 2, 5, 1, 4)

        layout.addLayout(controls)

        splitter = QSplitter()

        self.ui_map_frame = QFrame()
        map_layout = QVBoxLayout(self.ui_map_frame)
        map_layout.setContentsMargins(0, 0, 0, 0)
        self.ui_lblMapFallback = QLabel("Map unavailable")
        self.ui_lblMapFallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
        map_layout.addWidget(self.ui_lblMapFallback)
        splitter.addWidget(self.ui_map_frame)

        bottom = QSplitter()
        bottom.setOrientation(Qt.Orientation.Horizontal)

        estimate_group = QGroupBox("Emitter estimate")
        form = QFormLayout(estimate_group)
        self.ui_lblCentroid = QLabel("-")
        self.ui_lblTrilat = QLabel("-")
        self.ui_lblP0 = QLabel("-")
        self.ui_lblN = QLabel("-")
        self.ui_lblRms = QLabel("-")
        self.ui_lblConfidence = QLabel("-")
        form.addRow("Centroid:", self.ui_lblCentroid)
        form.addRow("Trilateration:", self.ui_lblTrilat)
        form.addRow("Power P0:", self.ui_lblP0)
        form.addRow("Path loss n:", self.ui_lblN)
        form.addRow("RMS:", self.ui_lblRms)
        form.addRow("Confidence:", self.ui_lblConfidence)
        bottom.addWidget(estimate_group)

        self.ui_table = QTableWidget(0, 5)
        self.ui_table.setHorizontalHeaderLabels(["Freq (MHz)", "Lat", "Lon", "RSSI", "Time"])
        self.ui_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bottom.addWidget(self.ui_table)
        bottom.setSizes([300, 600])

        splitter.addWidget(bottom)
        splitter.setSizes([600, 300])
        splitter.setOrientation(Qt.Orientation.Vertical)
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------ settings

    def _load_settings(self):
        self.ui_edtFreq.setText(
            settings.read("rfexplore_frequencies", self.DEFAULT_FREQUENCIES, str)
        )
        srate = settings.read("rfexplore_sample_rate", 1000000, int)
        idx = self.ui_cbSampleRate.findData(srate)
        self.ui_cbSampleRate.setCurrentIndex(max(0, idx))
        self.ui_spinGain.setValue(settings.read("rfexplore_gain", 20, int))
        self.ui_chkAuto.setChecked(settings.read("rfexplore_auto_sample", True, bool))
        self.ui_spinSpacing.setValue(settings.read(SAMPLE_SPACING_KEY, 6.0, float))
        self.ui_spinDwell.setValue(settings.read(DWELL_SECONDS_KEY, 10, int))

    def _save_settings(self):
        settings.write("rfexplore_frequencies", self.ui_edtFreq.text().strip())
        settings.write("rfexplore_sample_rate", self.ui_cbSampleRate.currentData())
        settings.write("rfexplore_gain", self.ui_spinGain.value())
        settings.write("rfexplore_auto_sample", self.ui_chkAuto.isChecked())
        settings.write(SAMPLE_SPACING_KEY, self.ui_spinSpacing.value())
        settings.write(DWELL_SECONDS_KEY, self.ui_spinDwell.value())

    # ----------------------------------------------------------------- GPS

    def _init_gps(self):
        try:
            self.gps_provider = create_gps_provider(require_usb=False)
        except Exception as e:
            logger.error("GPS init failed: {0}".format(e))
            self.gps_provider = None
        self._update_gps_label()

    def _gps_summary(self):
        p = self.gps_provider
        if p is None:
            return "GPS: unavailable"
        kind = type(p).__name__
        if p.error:
            return "GPS ({0}): error - {1}".format(kind, p.error)
        if p.has_fix():
            pos = p.position
            return "GPS ({0}): {1:.6f}, {2:.6f} | fix {3} | sats {4}".format(
                kind, pos[0], pos[1], p.fix_quality, p.num_satellites
            )
        return "GPS ({0}): searching for fix...".format(kind)

    def _update_gps_label(self):
        self.ui_lblGps.setText(self._gps_summary())

    def _update_gps_marker(self):
        self._update_gps_label()
        p = self.gps_provider
        if (
            p is not None
            and p.has_fix()
            and self._map_ready
            and self.ui_map_view is not None
        ):
            lat, lon = p.position[0], p.position[1]
            self._run_js(
                "WZRD.setGps({0}, {1}, {2});".format(repr(lat), repr(lon), repr(p.hdop))
            )

    # ----------------------------------------------------------------- map

    def _init_map(self):
        try:
            from PyQt6.QtWebChannel import QWebChannel
            from PyQt6.QtWebEngineCore import QWebEngineSettings
            from PyQt6.QtWebEngineWidgets import QWebEngineView

            view = QWebEngineView()
            page_settings = view.settings()
            page_settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
            )
            page_settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
            )
            self.bridge = MapBridge(self)
            self.bridge.map_ready.connect(self._on_map_ready)
            self.bridge.map_clicked.connect(self._on_map_clicked)
            channel = QWebChannel(self)
            channel.registerObject("rfmap", self.bridge)
            view.page().setWebChannel(channel)
            view.setUrl(QUrl.fromLocalFile(MAP_FILE))
            self.ui_map_frame.layout().removeWidget(self.ui_lblMapFallback)
            self.ui_lblMapFallback.hide()
            self.ui_map_frame.layout().addWidget(view)
            self.ui_map_view = view
        except ImportError as e:
            logger.warning("WebEngine not available, map disabled: {0}".format(e))

    def _run_js(self, script: str):
        if self.ui_map_view is not None:
            try:
                self.ui_map_view.page().runJavaScript(script)
            except Exception as e:
                logger.debug("runJavaScript failed: {0}".format(e))

    @pyqtSlot()
    def _on_map_ready(self):
        self._map_ready = True
        logger.info("RF map loaded")

    @pyqtSlot(float, float)
    def _on_map_clicked(self, lat, lon):
        logger.info("Map clicked: {0}, {1}".format(lat, lon))

    # ------------------------------------------------------------ scanning

    def _frequencies(self):
        raw = self.ui_edtFreq.text().strip()
        freqs = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                freqs.append(float(part) * 1e6)
            except ValueError:
                continue
        return freqs

    def _refresh_freq_display(self):
        self.ui_cbFreqDisplay.blockSignals(True)
        current = self.ui_cbFreqDisplay.currentData()
        self.ui_cbFreqDisplay.clear()
        freqs = sorted({s["freq"] for s in self.samples})
        for f in freqs:
            self.ui_cbFreqDisplay.addItem("{0:.3f} MHz".format(f / 1e6), f)
        if freqs:
            self.ui_cbFreqDisplay.addItem("All", None)
            if current is None:
                self.ui_cbFreqDisplay.setCurrentIndex(self.ui_cbFreqDisplay.count() - 1)
            else:
                idx = self.ui_cbFreqDisplay.findData(current)
                self.ui_cbFreqDisplay.setCurrentIndex(max(0, idx))
        self.ui_cbFreqDisplay.blockSignals(False)

    def _current_filter_freq(self):
        return self.ui_cbFreqDisplay.currentData()  # None means all

    def on_start_stop(self):
        if self.worker is not None and self.worker._thread is not None and self.worker._thread.is_alive():
            self.stop_scanning()
        else:
            self.start_scanning()

    def start_scanning(self):
        freqs = self._frequencies()
        if not freqs:
            logger.warning("No valid frequencies configured")
            return
        srate = self.ui_cbSampleRate.currentData()
        gain = self.ui_spinGain.value()

        self.ui_btnStartStop.setEnabled(False)
        self.scanner.start(freqs[0], srate, gain)
        self.ui_btnStartStop.setEnabled(True)
        self.ui_btnStartStop.setText("Stop")

        self.worker = RfSampleWorker(self.scanner, self.gps_provider, freqs, parent=self)
        self.worker.sample_ready.connect(self._on_sample)
        self.worker.configure(self.ui_spinSpacing.value(), self.ui_spinDwell.value())
        if self.ui_chkAuto.isChecked():
            self.worker.start()
        self._refresh_freq_display()
        self._save_settings()
        logger.info(
            "RF exploration started: freqs={0} Hz, srate={1}, gain={2}".format(
                freqs, srate, gain
            )
        )

    def stop_scanning(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker = None
        self.scanner.stop()
        self.ui_btnStartStop.setText("Start")
        logger.info("RF exploration stopped")

    def take_manual_sample(self):
        if self.worker is None:
            self.start_scanning()
        if self.worker is None or self.scanner is None or not self.scanner.is_running:
            return
        pos = self.gps_provider.position if self.gps_provider is not None else None
        if pos is None or not self.gps_provider.has_fix():
            logger.warning("No GPS fix, cannot sample")
            return
        t = threading.Thread(
            target=self.worker.take_samples, args=(pos,), daemon=True
        )
        t.start()

    def _on_sample(self, freq, lat, lon, rssi):
        sample = {
            "freq": freq,
            "lat": lat,
            "lon": lon,
            "rssi": rssi,
            "ts": time.time(),
        }
        self.samples.append(sample)
        self._append_table_row(sample)
        self._refresh_freq_display()
        self.refresh_map()

    def _append_table_row(self, s):
        row = self.ui_table.rowCount()
        self.ui_table.insertRow(row)
        self.ui_table.setItem(row, 0, QTableWidgetItem("{0:.3f}".format(s["freq"] / 1e6)))
        self.ui_table.setItem(row, 1, QTableWidgetItem("{0:.6f}".format(s["lat"])))
        self.ui_table.setItem(row, 2, QTableWidgetItem("{0:.6f}".format(s["lon"])))
        self.ui_table.setItem(row, 3, QTableWidgetItem("{0:.1f}".format(s["rssi"])))
        self.ui_table.setItem(
            row,
            4,
            QTableWidgetItem(datetime.fromtimestamp(s["ts"]).strftime("%H:%M:%S")),
        )
        if self.ui_table.rowCount() > 500:
            self.ui_table.removeRow(0)

    def clear_samples(self):
        self.samples.clear()
        self.ui_table.setRowCount(0)
        self._refresh_freq_display()
        self.refresh_map()

    # ------------------------------------------------------------- display

    def _on_rssi(self, rssi, ts):
        self.ui_lblLive.setText("RSSI {0:.1f} dB".format(rssi))

    def _on_device_error(self, msg):
        logger.error("RF scanner error: {0}".format(msg))
        self.ui_lblGps.setText("Scanner error: {0}".format(msg))

    def _filtered_samples(self):
        filt = self._current_filter_freq()
        if filt is None:
            return list(self.samples)
        return [s for s in self.samples if s["freq"] == filt]

    def refresh_map(self):
        if not self._map_ready:
            return
        samples = self._filtered_samples()
        js = [[i, s["lat"], s["lon"], s["rssi"]] for i, s in enumerate(samples)]
        self._run_js("WZRD.upsertSamples({0});".format(json.dumps(js)))

        est = self._estimate()
        if est is not None:
            est_js = {
                "lat": est[0],
                "lon": est[1],
                "p0": est[2],
                "n": est[3],
                "rms": est[4],
                "confidence": estimate_confidence(est[4]),
                "method": "trilateration",
            }
            self._run_js("WZRD.setEstimate({0});".format(json.dumps(est_js)))
            self.ui_lblTrilat.setText("{0:.6f}, {1:.6f}".format(est[0], est[1]))
            self.ui_lblP0.setText("{0:.1f} dB".format(est[2]))
            self.ui_lblN.setText("{0:.2f}".format(est[3]))
            self.ui_lblRms.setText("{0:.2f} dB".format(est[4]))
            self.ui_lblConfidence.setText(estimate_confidence(est[4]))
        else:
            self._run_js("WZRD.setEstimate(null);")
            self.ui_lblTrilat.setText("-")
            self.ui_lblP0.setText("-")
            self.ui_lblN.setText("-")
            self.ui_lblRms.setText("-")
            self.ui_lblConfidence.setText("-")

        cen = weighted_centroid(samples) if len(samples) >= 2 else None
        if cen is not None:
            self.ui_lblCentroid.setText("{0:.6f}, {1:.6f}".format(cen[0], cen[1]))
        else:
            self.ui_lblCentroid.setText("-")

    def _estimate(self):
        samples = self._filtered_samples()
        if len(samples) < 3:
            return None
        coords = [(s["lat"], s["lon"], s["rssi"]) for s in samples]
        return trilaterate(coords)

    def closeEvent(self, event):
        self.stop_scanning()
        if self.gps_provider is not None:
            try:
                self.gps_provider.stop()
            except Exception:
                pass
        self._save_settings()
        super().closeEvent(event)
