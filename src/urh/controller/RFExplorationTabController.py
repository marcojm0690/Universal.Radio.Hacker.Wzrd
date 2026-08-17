import json
import os
import threading
import time
from datetime import datetime

import numpy as np

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
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QMessageBox,
)

from urh import settings
from urh.gps.GpsProvider import (
    GPS_SOURCE_AUTO,
    GPS_SOURCE_CORELOCATION,
    GPS_SOURCE_OFF,
    GPS_SOURCE_USB,
    core_location_available,
    create_gps_provider,
    detect_usb_ports,
)
from urh.rfscan.Geolocator import (
    estimate_confidence,
    haversine_m,
    trilaterate,
    weighted_centroid,
)
from urh.rfscan.FoxHuntDialog import FoxHuntDialog
from urh.rfscan.RssiScanner import RssiScanner
from urh.rfscan.SignalAnalysisDialog import SignalAnalysisDialog
from urh.rfscan.SignalAnalyzer import analyze_signal, peaks_summary
from urh.util.Logger import logger

MAP_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "map", "map.html")
)

SAMPLE_SPACING_KEY = "rfexplore_spacing"
DWELL_SECONDS_KEY = "rfexplore_dwell"
CSV_ENABLED_KEY = "rfexplore_csv_log"
CSV_DIR = os.path.expanduser(os.path.join("~", ".config", "urh", "rfexplore_logs"))
CAPTURE_DIR = os.path.join(CSV_DIR, "bursts")
CSV_HEADER = (
    "date,time,freq_mhz,lat,lon,rssi_db,n_peaks,top_freq_mhz,bandwidth_khz,"
    "noise_floor_db,summary"
)


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

    sample_ready = pyqtSignal(float, float, float, float, object)  # freq, lat, lon, rssi, analysis

    def __init__(self, scanner: RssiScanner, gps_provider, frequencies, parent=None):
        super().__init__(parent)
        self.scanner = scanner
        self.gps_provider = gps_provider
        self.frequencies = frequencies
        self.spacing_m = 6.0
        self.dwell_s = 10.0
        self._running = False
        self._cancelled = False
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
        self._cancelled = False
        self._last_pos = None
        self._last_sample_time = 0.0
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._cancelled = True
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
        """Measure RSSI at every frequency (may block; call off the GUI thread).

        Retuning a streaming RTL-SDR on macOS wedges the R82xx tuner
        (r82xx_read: i2c wr failed), so the device is reopened per frequency.
        """
        lat, lon = pos[0], pos[1]
        for freq in self.frequencies:
            if self._cancelled:
                break
            try:
                self._open_at(freq)
                attempts = 0
                while (
                    not self.scanner.data_received
                    and attempts < 3
                    and not self._cancelled
                ):
                    logger.warning(
                        "RfSampleWorker: no IQ data at {0:.3f} MHz, reopening".format(
                            freq / 1e6
                        )
                    )
                    time.sleep(1.0)
                    self._open_at(freq)
                    attempts += 1
                rssi = self.scanner.average_rssi(0.8)
                analysis = None
                if self.scanner.data_received:
                    snap = self.scanner.snapshot(2 ** 18)
                    if snap is not None:
                        try:
                            analysis = analyze_signal(
                                snap,
                                freq,
                                self.scanner.sample_rate,
                                fft_averages=4,
                            )
                            signal_rssi = analysis.get("signal_rssi_db")
                            if signal_rssi is not None:
                                # Narrowband RSSI of the dominant peak: excludes
                                # broadband noise so it scales with the emitter's
                                # distance instead of reading the noise floor.
                                rssi = signal_rssi
                        except Exception as e:
                            logger.error(
                                "Signal analysis failed at {0:.3f} MHz: {1}".format(
                                    freq / 1e6, e
                                )
                            )
                self.sample_ready.emit(freq, lat, lon, rssi, analysis)
            except Exception as e:
                logger.error("RfSampleWorker sample error: {0}".format(e))

    def _open_at(self, freq):
        if self.scanner.is_running:
            self.scanner.stop()
            time.sleep(0.3)
        self.scanner.start(
            freq,
            self.scanner.sample_rate,
            self.scanner.gain,
            device_number=self.scanner.device_number,
        )
        time.sleep(0.8)


class RFExplorationTabController(QWidget):
    DEFAULT_FREQUENCIES = "433.920, 868.000, 915.000"
    FREQUENCY_PRESETS = [
        "433.920, 868.000, 915.000",
        "433.920",
        "868.000",
        "915.000",
        "315.000",
        "403.000",
        "868.000, 915.000",
        "2400.000",
    ]
    SAMPLE_RATES = [250000, 1000000, 2048000]

    burst_captured = pyqtSignal(str, float)  # path, sample_rate

    def __init__(self, main_controller=None, parent=None):
        super().__init__(parent)
        self.main_controller = main_controller
        self.samples = []  # list of dicts: freq, lat, lon, rssi, ts
        self.scanner = RssiScanner(parent=self)
        self.gps_provider = None
        self.worker = None
        self.bridge = None
        self.ui_map_view = None
        self._map_ready = False
        self._devices = []
        self._csv_file = None
        self._fox_hunt_dialog = None

        self._build_ui()
        self._refresh_devices()
        self._load_settings()
        self._init_gps()
        self._init_map()

        self.ui_btnStartStop.clicked.connect(self.on_start_stop)
        self.ui_btnSampleNow.clicked.connect(self.take_manual_sample)
        self.ui_btnClear.clicked.connect(self.clear_samples)
        self.ui_btnFitMap.clicked.connect(self.fit_map)
        self.ui_cbFreqDisplay.currentIndexChanged.connect(self.refresh_map)
        self.ui_cbGpsSource.currentIndexChanged.connect(self.on_gps_source_changed)
        self.ui_btnDetectGps.clicked.connect(self.on_detect_gps)
        self.ui_btnRefreshDevice.clicked.connect(self._refresh_devices)
        self.ui_btnOpenLog.clicked.connect(self.open_csv_log)
        self.ui_btnFoxHunt.clicked.connect(self.open_fox_hunt)
        self.ui_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.scanner.rssi_updated.connect(self._on_rssi)
        self.scanner.device_error.connect(self._on_device_error)
        self.burst_captured.connect(self._on_burst_captured)

        self.gps_timer = QTimer(self)
        self.gps_timer.setInterval(3000)
        self.gps_timer.timeout.connect(self._update_gps_marker)
        self.gps_timer.start()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        layout = QVBoxLayout(self)

        controls = QGridLayout()
        controls.addWidget(QLabel("Frequencies (MHz):"), 0, 0)
        self.ui_cbFreq = QComboBox()
        self.ui_cbFreq.setEditable(True)
        self.ui_cbFreq.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.ui_cbFreq.addItems(self.FREQUENCY_PRESETS)
        self.ui_cbFreq.setToolTip(
            "Pick a frequency preset, or type your own comma-separated list "
            "in MHz (e.g. 433.920, 868.000)."
        )
        controls.addWidget(self.ui_cbFreq, 0, 1)

        controls.addWidget(QLabel("Sample rate:"), 0, 2)
        self.ui_cbSampleRate = QComboBox()
        for sr in self.SAMPLE_RATES:
            self.ui_cbSampleRate.addItem("{0:.3f} MS/s".format(sr / 1e6), sr)
        controls.addWidget(self.ui_cbSampleRate, 0, 3)

        controls.addWidget(QLabel("Gain (0.1 dB):"), 0, 4)
        self.ui_spinGain = QSpinBox()
        self.ui_spinGain.setRange(0, 500)
        self.ui_spinGain.setValue(300)
        self.ui_spinGain.setToolTip(
            "RTL-SDR gain in 0.1 dB steps (e.g. 300 = 30.0 dB). Max is ~49.6 dB.\n"
            "Raise it to pick up weak / far-away signals; back off if strong\n"
            "near-field signals saturate (clipping) the 8-bit ADC."
        )
        controls.addWidget(self.ui_spinGain, 0, 5)

        self.ui_btnStartStop = QPushButton("Start")
        controls.addWidget(self.ui_btnStartStop, 0, 6)

        self.ui_btnSampleNow = QPushButton("Sample now")
        controls.addWidget(self.ui_btnSampleNow, 0, 7)

        self.ui_btnClear = QPushButton("Clear")
        controls.addWidget(self.ui_btnClear, 0, 8)

        self.ui_btnFitMap = QPushButton("Fit map")
        controls.addWidget(self.ui_btnFitMap, 0, 9)

        self.ui_chkAuto = QCheckBox("Auto-sample while moving")
        self.ui_chkAuto.setChecked(True)
        controls.addWidget(self.ui_chkAuto, 1, 3, 1, 2)

        controls.addWidget(QLabel("Device:"), 1, 0)
        self.ui_cbDevice = QComboBox()
        controls.addWidget(self.ui_cbDevice, 1, 1)
        self.ui_btnRefreshDevice = QPushButton("Refresh")
        controls.addWidget(self.ui_btnRefreshDevice, 1, 2)

        controls.addWidget(QLabel("Spacing (m):"), 1, 5)
        self.ui_spinSpacing = QDoubleSpinBox()
        self.ui_spinSpacing.setRange(1, 200)
        self.ui_spinSpacing.setValue(6.0)
        self.ui_spinSpacing.setDecimals(1)
        controls.addWidget(self.ui_spinSpacing, 1, 6)

        controls.addWidget(QLabel("Dwell (s):"), 1, 7)
        self.ui_spinDwell = QSpinBox()
        self.ui_spinDwell.setRange(2, 600)
        self.ui_spinDwell.setValue(10)
        controls.addWidget(self.ui_spinDwell, 1, 8)

        controls.addWidget(QLabel("Show:"), 2, 0)
        self.ui_cbFreqDisplay = QComboBox()
        controls.addWidget(self.ui_cbFreqDisplay, 2, 1)

        controls.addWidget(QLabel("GPS source:"), 2, 2)
        self.ui_cbGpsSource = QComboBox()
        self.ui_cbGpsSource.addItem("Auto (USB preferred)", GPS_SOURCE_AUTO)
        self.ui_cbGpsSource.addItem("USB GNSS", GPS_SOURCE_USB)
        self.ui_cbGpsSource.addItem("macOS location", GPS_SOURCE_CORELOCATION)
        self.ui_cbGpsSource.addItem("Off", GPS_SOURCE_OFF)
        controls.addWidget(self.ui_cbGpsSource, 2, 3)

        self.ui_btnDetectGps = QPushButton("Detect")
        controls.addWidget(self.ui_btnDetectGps, 2, 4)

        self.ui_lblGps = QLabel("GPS: searching...")
        controls.addWidget(self.ui_lblGps, 2, 5, 1, 2)

        self.ui_lblLive = QLabel("")
        controls.addWidget(self.ui_lblLive, 2, 7, 1, 2)

        self.ui_chkCsv = QCheckBox("Log to CSV")
        controls.addWidget(self.ui_chkCsv, 3, 0, 1, 2)
        self.ui_btnOpenLog = QPushButton("Open log")
        controls.addWidget(self.ui_btnOpenLog, 3, 2)

        self.ui_btnFoxHunt = QPushButton("Fox-hunt")
        self.ui_btnFoxHunt.setToolTip(
            "Open the fox-hunting dashboard: bearing radar, emitter bearing/distance,\n"
            "and forensic report export."
        )
        controls.addWidget(self.ui_btnFoxHunt, 3, 3)

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

        self.ui_table = QTableWidget(0, 7)
        self.ui_table.setHorizontalHeaderLabels(
            ["Freq (MHz)", "Lat", "Lon", "RSSI", "Date", "Time", "Analysis"]
        )
        self.ui_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bottom.addWidget(self.ui_table)
        bottom.setSizes([300, 600])

        splitter.addWidget(bottom)
        splitter.setSizes([600, 300])
        splitter.setOrientation(Qt.Orientation.Vertical)
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------ settings

    def _load_settings(self):
        self.ui_cbFreq.setCurrentText(
            settings.read("rfexplore_frequencies", self.DEFAULT_FREQUENCIES, str)
        )
        srate = settings.read("rfexplore_sample_rate", 1000000, int)
        idx = self.ui_cbSampleRate.findData(srate)
        self.ui_cbSampleRate.setCurrentIndex(max(0, idx))
        self.ui_spinGain.setValue(settings.read("rfexplore_gain", 300, int))
        self.ui_chkAuto.setChecked(settings.read("rfexplore_auto_sample", True, bool))
        self.ui_spinSpacing.setValue(settings.read(SAMPLE_SPACING_KEY, 6.0, float))
        self.ui_spinDwell.setValue(settings.read(DWELL_SECONDS_KEY, 10, int))
        self.ui_chkCsv.setChecked(settings.read(CSV_ENABLED_KEY, False, bool))
        saved_device = settings.read("rfexplore_sdr_device", -1, int)
        didx = self.ui_cbDevice.findData(saved_device)
        if didx >= 0:
            self.ui_cbDevice.setCurrentIndex(didx)

    def _save_settings(self):
        settings.write("rfexplore_frequencies", self.ui_cbFreq.currentText().strip())
        settings.write("rfexplore_sample_rate", self.ui_cbSampleRate.currentData())
        settings.write("rfexplore_gain", self.ui_spinGain.value())
        settings.write("rfexplore_auto_sample", self.ui_chkAuto.isChecked())
        settings.write(SAMPLE_SPACING_KEY, self.ui_spinSpacing.value())
        settings.write(DWELL_SECONDS_KEY, self.ui_spinDwell.value())
        settings.write(CSV_ENABLED_KEY, self.ui_chkCsv.isChecked())
        settings.write(
            "rfexplore_gps_source", self.ui_cbGpsSource.currentData()
            if self.ui_cbGpsSource.count() else GPS_SOURCE_AUTO
        )
        if self.ui_cbDevice.currentData() is not None:
            settings.write("rfexplore_sdr_device", self.ui_cbDevice.currentData())

    # --------------------------------------------------------------- Device

    def _refresh_devices(self):
        devices = RssiScanner.detect_devices()
        self._devices = devices
        self.ui_cbDevice.blockSignals(True)
        self.ui_cbDevice.clear()
        for info in devices:
            self.ui_cbDevice.addItem(RssiScanner.device_label(info), info["index"])
        if not devices:
            self.ui_cbDevice.addItem("No RTL-SDR detected")
        self.ui_cbDevice.blockSignals(False)
        saved_device = settings.read("rfexplore_sdr_device", -1, int)
        didx = self.ui_cbDevice.findData(saved_device)
        if didx >= 0:
            self.ui_cbDevice.setCurrentIndex(didx)
        logger.info(
            "RTL-SDR devices: {0} detected".format(len(devices))
        )

    # ----------------------------------------------------------------- GPS

    def _init_gps(self):
        source = settings.read("rfexplore_gps_source", GPS_SOURCE_AUTO, str)
        self._apply_gps_source(source)

    def _apply_gps_source(self, source):
        self._stop_gps()
        self.gps_provider = None
        if source == GPS_SOURCE_OFF:
            self.ui_lblGps.setText("GPS: disabled")
            self._update_gps_marker()
            return
        try:
            if source == GPS_SOURCE_CORELOCATION:
                self.gps_provider = create_gps_provider(source=GPS_SOURCE_CORELOCATION)
            elif source == GPS_SOURCE_USB:
                self.gps_provider = create_gps_provider(source=GPS_SOURCE_USB)
            else:
                self.gps_provider = create_gps_provider(source=GPS_SOURCE_AUTO)
        except Exception as e:
            logger.error("GPS init failed: {0}".format(e))
            self.gps_provider = None
        self._update_gps_label()
        self._update_gps_marker()

    def _stop_gps(self):
        if self.gps_provider is not None:
            try:
                self.gps_provider.stop()
            except Exception:
                pass
            self.gps_provider = None

    def on_gps_source_changed(self):
        source = self.ui_cbGpsSource.currentData()
        settings.write("rfexplore_gps_source", source)
        self._apply_gps_source(source)

    def on_detect_gps(self):
        ports = detect_usb_ports()
        summary = []
        if ports:
            summary.append("USB GNSS: {0}".format(", ".join(ports)))
        else:
            summary.append("USB GNSS: none detected")
        if core_location_available():
            summary.append("macOS location: available")
        else:
            summary.append("macOS location: not available (PyObjC missing)")
        self.ui_lblGps.setText("Detected - " + " | ".join(summary))
        self._update_gps_label()

    def _gps_summary(self):
        p = self.gps_provider
        if p is None:
            return "GPS: no provider"
        kind = type(p).__name__
        if p.has_fix():
            pos = p.position
            return "GPS ({0}): {1:.6f}, {2:.6f} | fix {3} | sats {4}".format(
                kind, pos[0], pos[1], p.fix_quality, p.num_satellites
            )
        if p.error:
            return "GPS ({0}): {1}".format(kind, p.error)
        if p.status_message:
            return "GPS ({0}): {1}".format(kind, p.status_message)
        if kind == "CoreLocationGpsProvider":
            label = p.AUTHORIZATION_LABELS.get(p.authorization_status, "?")
            return "GPS (CoreLocation): waiting - authorization {0}".format(label)
        return "GPS ({0}): waiting for fix...".format(kind)

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
            logger.info("GPS pos: {0:.6f}, {1:.6f}".format(lat, lon))
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
        raw = self.ui_cbFreq.currentText().strip()
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
        device_number = self.ui_cbDevice.currentData()
        if device_number is None or not self._devices:
            QMessageBox.warning(
                self,
                "No RTL-SDR detected",
                "No RTL-SDR device is currently visible to macOS.\n\n"
                "Please check the USB connection (unplug and re-plug the dongle)\n"
                "and click Refresh.",
            )
            return

        self.ui_btnStartStop.setEnabled(False)
        self.scanner.start(freqs[0], srate, gain, device_number=device_number)
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

    def _on_sample(self, freq, lat, lon, rssi, analysis):
        dt = datetime.fromtimestamp(time.time())
        sample = {
            "freq": freq,
            "lat": lat,
            "lon": lon,
            "rssi": rssi,
            "ts": dt.timestamp(),
            "date_str": dt.strftime("%Y-%m-%d"),
            "time_str": dt.strftime("%H:%M:%S"),
            "analysis": analysis,
        }
        self.samples.append(sample)
        logger.info(
            "Sample: {0:.3f} MHz @ {1:.6f}, {2:.6f} rssi={3:.1f} {4}".format(
                freq / 1e6,
                lat,
                lon,
                rssi,
                peaks_summary(analysis) if analysis is not None else "",
            )
        )
        self._append_table_row(sample)
        self._write_csv(sample)
        self._refresh_freq_display()
        self.refresh_map()

    def _append_table_row(self, s):
        row = self.ui_table.rowCount()
        self.ui_table.insertRow(row)
        self.ui_table.setItem(row, 0, QTableWidgetItem("{0:.3f}".format(s["freq"] / 1e6)))
        self.ui_table.setItem(row, 1, QTableWidgetItem("{0:.6f}".format(s["lat"])))
        self.ui_table.setItem(row, 2, QTableWidgetItem("{0:.6f}".format(s["lon"])))
        self.ui_table.setItem(row, 3, QTableWidgetItem("{0:.1f}".format(s["rssi"])))
        self.ui_table.setItem(row, 4, QTableWidgetItem(s["date_str"]))
        self.ui_table.setItem(row, 5, QTableWidgetItem(s["time_str"]))
        self.ui_table.setItem(
            row,
            6,
            QTableWidgetItem(peaks_summary(s["analysis"]) if s["analysis"] is not None else "-"),
        )
        if self.ui_table.rowCount() > 500:
            self.ui_table.removeRow(0)

    def _on_cell_double_clicked(self, row, _column):
        if row < len(self.samples):
            self._open_analysis(self.samples[row])

    def _open_analysis(self, sample):
        dialog = SignalAnalysisDialog(
            sample,
            sample.get("analysis"),
            parent=self,
            capture_cb=lambda: self.start_burst_capture(sample),
        )
        self.burst_captured.connect(dialog.on_burst_captured)
        dialog.exec()
        try:
            self.burst_captured.disconnect(dialog.on_burst_captured)
        except (TypeError, RuntimeError):
            pass

    # ------------------------------------------------------ burst capture

    def start_burst_capture(self, sample):
        """Capture a continuous IQ burst at this sample's frequency.

        Retunes the SDR (pausing any active scan), records ~1 s of IQ, saves
        it as a URH-compatible .complex16s file and opens it in the Analysis
        tab for time-domain / spectrum / demodulator work. Runs off-thread.
        """
        freq = float(sample["freq"])
        srate = self.scanner.sample_rate or self.ui_cbSampleRate.currentData()
        gain = (
            self.scanner.gain
            if self.scanner.gain is not None
            else self.ui_spinGain.value()
        )
        device_number = self.scanner.device_number
        threading.Thread(
            target=self._capture_burst_worker,
            args=(freq, srate, gain, device_number),
            daemon=True,
        ).start()

    def _capture_burst_worker(self, freq, srate, gain, device_number):
        was_scanning = (
            self.worker is not None
            and self.worker._thread is not None
            and self.worker._thread.is_alive()
        )
        try:
            if was_scanning:
                self.worker.stop()
            if self.scanner.is_running:
                self.scanner.stop()
            self.scanner.start(freq, srate, gain, device_number=device_number)

            deadline = time.time() + 6.0
            while not self.scanner.data_received and time.time() < deadline:
                time.sleep(0.2)
            time.sleep(1.5)  # let the rolling buffer fill with a fresh window

            snap = self.scanner.snapshot(2 ** 18)
            self.scanner.stop()

            if snap is None or len(snap) < 1024:
                logger.error("Burst capture: no IQ data at {0:.3f} MHz".format(freq / 1e6))
                self.burst_captured.emit("", 0.0)
                return

            os.makedirs(CAPTURE_DIR, exist_ok=True)
            path = os.path.join(
                CAPTURE_DIR,
                "burst_{0:.3f}MHz_{1}.complex16s".format(
                    freq / 1e6, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                ),
            )
            snap.astype(np.int8).tofile(path)
            logger.info(
                "Burst captured: {0} samples @ {1} Hz -> {2}".format(
                    len(snap), srate, path
                )
            )
            self.burst_captured.emit(path, float(srate))
        except Exception as e:
            logger.error("Burst capture failed: {0}".format(e))
            self.burst_captured.emit("", 0.0)
        finally:
            if was_scanning:
                self.worker.start()

    @pyqtSlot(str, float)
    def _on_burst_captured(self, path, sample_rate):
        if path and self.main_controller is not None:
            self.main_controller.add_signalfile(
                path, enforce_sample_rate=sample_rate
            )
            self.main_controller.ui.tabWidget.setCurrentWidget(
                self.main_controller.ui.tab_interpretation
            )

    def open_fox_hunt(self):
        if self._fox_hunt_dialog is None or not self._fox_hunt_dialog.isVisible():
            self._fox_hunt_dialog = FoxHuntDialog(self, parent=self)
            self._fox_hunt_dialog.show()
        else:
            self._fox_hunt_dialog.raise_()
            self._fox_hunt_dialog.activateWindow()

    # ------------------------------------------------------------------- CSV

    def _csv_path(self):
        if self._csv_file is None:
            os.makedirs(CSV_DIR, exist_ok=True)
            self._csv_file = os.path.join(
                CSV_DIR,
                "RF_{0}.csv".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S")),
            )
            with open(self._csv_file, "w") as f:
                f.write(CSV_HEADER + "\n")
            logger.info("CSV log started: {0}".format(self._csv_file))
        return self._csv_file

    def _write_csv(self, s):
        if not self.ui_chkCsv.isChecked():
            return
        try:
            a = s["analysis"]
            if a is not None and a.get("peaks"):
                top = a["peaks"][0]
                n_peaks = a.get("n_peaks", 0)
                top_freq = top["freq_mhz"]
                bw = a.get("bandwidth_hz", 0.0) / 1e3
                floor = a.get("noise_floor_db", 0.0)
                summary = peaks_summary(a)
            else:
                n_peaks = 0
                top_freq = 0.0
                bw = 0.0
                floor = 0.0
                summary = "no peaks"
            line = "{0},{1},{2:.3f},{3:.6f},{4:.6f},{5:.1f},{6},{7:.5f},{8:.1f},{9:.1f},{10}".format(
                s["date_str"],
                s["time_str"],
                s["freq"] / 1e6,
                s["lat"],
                s["lon"],
                s["rssi"],
                n_peaks,
                top_freq,
                bw,
                floor,
                summary,
            )
            with open(self._csv_path(), "a") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.error("CSV write failed: {0}".format(e))

    def open_csv_log(self):
        import subprocess

        if self._csv_file is None:
            os.makedirs(CSV_DIR, exist_ok=True)
            path = CSV_DIR
        else:
            path = self._csv_file
        subprocess.Popen(["open", "-R", path])

    def clear_samples(self):
        self.samples.clear()
        self.ui_table.setRowCount(0)
        self._refresh_freq_display()
        self.refresh_map()

    def fit_map(self):
        self._run_js("WZRD.fitAll();")

    # ------------------------------------------------------------- display

    def _on_rssi(self, rssi, ts):
        self.ui_lblLive.setText("RSSI {0:.1f} dB".format(rssi))

    def _on_device_error(self, msg):
        logger.error("RF scanner error: {0}".format(msg))
        self.ui_lblLive.setText("Scanner error: {0}".format(msg))

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
            conf = estimate_confidence(est[4])
            est_js = {
                "lat": est[0],
                "lon": est[1],
                "p0": est[2],
                "n": est[3],
                "rms": est[4],
                "confidence": conf,
                "method": "trilateration",
                "radius_m": {"high": 20, "medium": 80, "low": 300}.get(conf, 300),
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

        cen = None
        if len(samples) >= 2:
            coords = [(s["lat"], s["lon"], s["rssi"]) for s in samples]
            cen = weighted_centroid(coords)
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
        self._stop_gps()
        self._save_settings()
        super().closeEvent(event)
