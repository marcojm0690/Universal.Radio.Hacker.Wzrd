import json
import os
import threading
import time
from datetime import datetime

import numpy as np

from PyQt6.QtCore import QObject, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot, QPropertyAnimation
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QAbstractItemView,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
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
from urh.rfscan.SignalAnalysisDialog import SignalAnalysisDialog, DEFAULT_RADIOLLM_URL
from urh.rfscan import RadioLLMClient
from urh.rfscan.WaterfallWidget import WaterfallWidget, COLS as WF_COLS
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
    "noise_floor_db,saturated,summary"
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
        # Optional controller-provided reopen(freq)->bool; used instead of raw
        # stop/start so wedged dongles are detected instead of hammered.
        self.reopen_fn = None

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
                if not self._open_at(freq):
                    logger.warning(
                        "RfSampleWorker: dongle not delivering data at {0:.3f} MHz "
                        "-- skipping remaining frequencies this cycle".format(freq / 1e6)
                    )
                    break
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

    def _open_at(self, freq) -> bool:
        """(Re)open the scanner at `freq`. Returns True if IQ data flows."""
        if self.reopen_fn is not None:
            return bool(self.reopen_fn(freq))
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
        return True


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
    waterfall_denoise_done = pyqtSignal(str, str, object)  # out_path, msg, burst|None
    _live_msg = pyqtSignal(str)  # thread-safe ui_lblLive updates
    _monitor_open_done = pyqtSignal(bool)
    _scan_open_done = pyqtSignal(bool)

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
        # Dongle health tracking: rapid unclean stop/start cycles can wedge the
        # USB interface (all later opens fail with -1 until physical replug).
        self._open_fail_streak = 0
        self._dongle_dead = False
        self._last_reopen_ts = 0.0

        self._build_ui()
        self._refresh_devices()
        self._load_settings()
        self._init_gps()
        self._init_map()

        self.ui_btnStartStop.clicked.connect(self.on_start_stop)
        self.ui_btnMonitor.clicked.connect(self.on_monitor_toggle)
        self.ui_btnSampleNow.clicked.connect(self.take_manual_sample)
        self.ui_btnClear.clicked.connect(self.clear_samples)
        self.ui_btnFitMap.clicked.connect(self.fit_map)
        self.ui_cbFreqDisplay.currentIndexChanged.connect(self.refresh_map)
        self.ui_cbGpsSource.currentIndexChanged.connect(self.on_gps_source_changed)
        self.ui_btnDetectGps.clicked.connect(self.on_detect_gps)
        self.ui_btnRefreshDevice.clicked.connect(self._refresh_devices)
        self.ui_btnOptions.toggled.connect(self._toggle_options)
        self.ui_btnOpenLog.clicked.connect(self.open_csv_log)
        self.ui_btnFoxHunt.clicked.connect(self.open_fox_hunt)
        self.ui_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.scanner.rssi_updated.connect(self._on_rssi)
        self.scanner.device_error.connect(self._on_device_error)
        self.burst_captured.connect(self._on_burst_captured)
        self.waterfall_denoise_done.connect(self._on_waterfall_denoise_done)
        self._live_msg.connect(self.ui_lblLive.setText)
        self._monitor_open_done.connect(self._on_monitor_open_done)
        self._scan_open_done.connect(self._on_scan_open_done)

        self.gps_timer = QTimer(self)
        self.gps_timer.setInterval(3000)
        self.gps_timer.timeout.connect(self._update_gps_marker)
        self.gps_timer.start()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._compose_status)
        self._status_timer.start()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---------------------------------------------------------- receiver
        recv = QGroupBox("Receiver")
        rg = QGridLayout(recv)
        rg.addWidget(QLabel("Frequencies (MHz):"), 0, 0)
        self.ui_cbFreq = QComboBox()
        self.ui_cbFreq.setEditable(True)
        self.ui_cbFreq.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.ui_cbFreq.addItems(self.FREQUENCY_PRESETS)
        self.ui_cbFreq.setToolTip(
            "Pick a frequency preset, or type your own comma-separated list "
            "in MHz (e.g. 433.920, 868.000)."
        )
        rg.addWidget(self.ui_cbFreq, 0, 1)

        rg.addWidget(QLabel("Sample rate:"), 0, 2)
        self.ui_cbSampleRate = QComboBox()
        for sr in self.SAMPLE_RATES:
            self.ui_cbSampleRate.addItem("{0:.3f} MS/s".format(sr / 1e6), sr)
        rg.addWidget(self.ui_cbSampleRate, 0, 3)

        rg.addWidget(QLabel("Gain (0.1 dB):"), 0, 4)
        self.ui_spinGain = QSpinBox()
        self.ui_spinGain.setRange(0, 500)
        self.ui_spinGain.setValue(300)
        self.ui_spinGain.setToolTip(
            "RTL-SDR gain in 0.1 dB steps (e.g. 300 = 30.0 dB). Max is ~49.6 dB.\n"
            "Raise it to pick up weak / far-away signals; back off if strong\n"
            "near-field signals saturate (clipping) the 8-bit ADC."
        )
        rg.addWidget(self.ui_spinGain, 0, 5)

        rg.addWidget(QLabel("Device:"), 1, 0)
        self.ui_cbDevice = QComboBox()
        rg.addWidget(self.ui_cbDevice, 1, 1)
        self.ui_btnRefreshDevice = QPushButton("Refresh")
        self.ui_btnRefreshDevice.setToolTip("Re-scan USB after unplugging/replugging a dongle.")
        rg.addWidget(self.ui_btnRefreshDevice, 1, 2)

        self.ui_btnStartStop = QPushButton("Start")
        self.ui_btnStartStop.setToolTip(
            "Hop scan: cycles through each frequency measuring RSSI\n"
            "(fox-hunt sampling). Waterfall follows the hops.")
        rg.addWidget(self.ui_btnStartStop, 0, 7)

        self.ui_btnMonitor = QPushButton("Monitor")
        self.ui_btnMonitor.setToolTip(
            "Wideband monitor: tunes ONCE to the first frequency and never hops.\n"
            "Every transmitter inside the sample-rate span shows up in the\n"
            "waterfall below, auto-tagged. Click a tag to denoise it."
        )
        self.ui_btnMonitor.setCheckable(True)
        rg.addWidget(self.ui_btnMonitor, 0, 8)

        # --------------------------------------------------- fox-hunt sampling
        fox = QGroupBox("Fox-hunt sampling")
        fg = QGridLayout(fox)
        self.ui_chkAuto = QCheckBox("Auto-sample while moving")
        self.ui_chkAuto.setChecked(True)
        fg.addWidget(self.ui_chkAuto, 0, 0, 1, 2)

        fg.addWidget(QLabel("Spacing (m):"), 0, 2)
        self.ui_spinSpacing = QDoubleSpinBox()
        self.ui_spinSpacing.setRange(1, 200)
        self.ui_spinSpacing.setValue(6.0)
        self.ui_spinSpacing.setDecimals(1)
        fg.addWidget(self.ui_spinSpacing, 0, 3)

        fg.addWidget(QLabel("Dwell (s):"), 0, 4)
        self.ui_spinDwell = QSpinBox()
        self.ui_spinDwell.setRange(2, 600)
        self.ui_spinDwell.setValue(10)
        fg.addWidget(self.ui_spinDwell, 0, 5)

        self.ui_btnSampleNow = QPushButton("Sample now")
        fg.addWidget(self.ui_btnSampleNow, 1, 0)

        self.ui_chkCsv = QCheckBox("Log to CSV")
        fg.addWidget(self.ui_chkCsv, 1, 1, 1, 2)
        self.ui_btnOpenLog = QPushButton("Open log")
        fg.addWidget(self.ui_btnOpenLog, 1, 3)

        # --------------------------------------------------------- map & GPS
        mgps = QGroupBox("Map && GPS")
        gg = QGridLayout(mgps)

        gg.addWidget(QLabel("Show:"), 0, 0)
        self.ui_cbFreqDisplay = QComboBox()
        gg.addWidget(self.ui_cbFreqDisplay, 0, 1)

        gg.addWidget(QLabel("GPS source:"), 0, 2)
        self.ui_cbGpsSource = QComboBox()
        self.ui_cbGpsSource.addItem("Auto (USB preferred)", GPS_SOURCE_AUTO)
        self.ui_cbGpsSource.addItem("USB GNSS", GPS_SOURCE_USB)
        self.ui_cbGpsSource.addItem("macOS location", GPS_SOURCE_CORELOCATION)
        self.ui_cbGpsSource.addItem("Off", GPS_SOURCE_OFF)
        gg.addWidget(self.ui_cbGpsSource, 0, 3)

        self.ui_btnDetectGps = QPushButton("Detect")
        gg.addWidget(self.ui_btnDetectGps, 0, 4)

        self.ui_lblGps = QLabel("GPS: searching...")
        self.ui_lblGps.hide()  # consolidated into the status line

        self.ui_btnFoxHunt = QPushButton("Fox-hunt dashboard")
        self.ui_btnFoxHunt.setToolTip(
            "Open the fox-hunting dashboard: bearing radar, emitter bearing/distance,\n"
            "and forensic report export."
        )
        gg.addWidget(self.ui_btnFoxHunt, 1, 0, 1, 2)

        self.ui_btnFitMap = QPushButton("Fit map")
        gg.addWidget(self.ui_btnFitMap, 1, 2)

        self.ui_btnClear = QPushButton("Clear samples")
        gg.addWidget(self.ui_btnClear, 1, 3)

        rows = QVBoxLayout()
        rows.addWidget(recv)

        # Collapsible options drawer: fox-hunt + Map&GPS hidden by default so
        # the waterfall gets maximum space; the Options button slides it open.
        self._options_drawer = QWidget()
        drawer_lay = QHBoxLayout(self._options_drawer)
        drawer_lay.setContentsMargins(0, 0, 0, 0)
        drawer_lay.addWidget(fox)
        drawer_lay.addWidget(mgps)
        rows.addWidget(self._options_drawer)
        self._options_drawer.setVisible(False)
        self._options_drawer.setMaximumHeight(0)

        status_row = QHBoxLayout()
        self.ui_btnOptions = QPushButton("Options ▾")
        self.ui_btnOptions.setCheckable(True)
        self.ui_btnOptions.setFlat(True)
        self.ui_btnOptions.setToolTip(
            "Slide out fox-hunt sampling and Map & GPS controls.\n"
            "Keep them hidden to give the waterfall more space.")
        status_row.addWidget(self.ui_btnOptions)
        # Single consolidated status line: clock · GPS · live state.
        # ui_lblLive / ui_lblGps stay alive as hidden text holders so all
        # existing writers keep working; the 1 s composer merges them here.
        self.ui_lblStatus = QLabel("")
        self.ui_lblStatus.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: #d8dee9;")
        self.ui_lblStatus.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        status_row.addWidget(self.ui_lblStatus, 1)
        self.ui_lblLive = QLabel("")
        self.ui_lblLive.hide()
        self._status_clock = ""
        self.ui_lblSat = QLabel("")
        self.ui_lblSat.setStyleSheet("color: #e06c6c; font-weight: bold;")
        self.ui_lblSat.hide()
        status_row.addWidget(self.ui_lblSat)
        rows.addLayout(status_row)

        layout.addLayout(rows)

        splitter = QSplitter()

        self.waterfall = WaterfallWidget()
        self.waterfall.burst_selected.connect(self._on_waterfall_burst)
        self.waterfall.burst_finalized.connect(self._append_burst_row)
        splitter.addWidget(self.waterfall)

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

        # ONE unified events table: waterfall bursts + fox-hunt RSSI samples.
        self.ui_table = QTableWidget(0, 5)
        self.ui_table.setHorizontalHeaderLabels(
            ["Time", "Freq MHz", "Kind", "Detail", "GPS"]
        )
        self.ui_table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch)
        self.ui_table.verticalHeader().setVisible(False)
        self.ui_table.setSortingEnabled(True)
        self.ui_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.ui_table.setToolTip(
            "Every event this session: Burst rows open the RadioLLM menu,\n"
            "RSSI sample rows open the signal analysis dialog (double-click).")
        bottom.addWidget(self.ui_table)
        bottom.setSizes([300, 600])

        splitter.addWidget(bottom)
        splitter.setSizes([260, 560, 300])
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
        if settings.read("rfexplore_options_visible", False, bool):
            self.ui_btnOptions.setChecked(True)  # triggers the slide-out

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
    def _compose_status(self):
        """Merge clock + GPS + live state into the single bold status line."""
        parts = [time.strftime("%H:%M:%S")]
        gps = self.ui_lblGps.text().strip()
        if gps and "searching" not in gps.lower():
            parts.append(gps.replace("GPS:", "").strip())
        live = self.ui_lblLive.text().strip()
        if live:
            parts.append(live)
        self.ui_lblStatus.setText("   ·   ".join(parts))

    def _toggle_options(self, checked):
        """Slide the fox-hunt / Map&GPS drawer in and out."""
        d = self._options_drawer
        self.ui_btnOptions.setText("Hide options ▴" if checked else "Options ▾")
        anim = getattr(self, "_options_anim", None)
        if anim is not None and anim.state() == QPropertyAnimation.State.Running:
            anim.stop()
        anim = QPropertyAnimation(d, b"maximumHeight", self)
        anim.setDuration(180)
        if checked:
            d.setVisible(True)
            target = max(d.sizeHint().height(), 10)
            d.setMaximumHeight(0)
            anim.setStartValue(0)
            anim.setEndValue(target)
            anim.finished.connect(lambda: d.setMaximumHeight(16777215))
        else:
            anim.setStartValue(d.height())
            anim.setEndValue(0)
            anim.finished.connect(lambda: d.setVisible(False))
        self._options_anim = anim
        anim.start()
        settings.write("rfexplore_options_visible", bool(checked))

    def _refresh_devices(self):
        # A refresh usually follows a replug, so clear wedged-dongle state.
        self._dongle_dead = False
        self._open_fail_streak = 0
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

    # ------------------------------------------------------------ wideband monitor

    def on_monitor_toggle(self):
        if self.ui_btnMonitor.isChecked():
            self.start_monitor()
        else:
            self.stop_monitor()

    def start_monitor(self):
        """Tune ONCE to the first frequency; waterfall watches the whole span."""
        if getattr(self, "_opening", False):
            return
        self._dongle_dead = False
        self._open_fail_streak = 0
        freqs = self._frequencies()
        if not freqs:
            QMessageBox.information(self, "Monitor", "Enter at least one frequency (MHz).")
            self.ui_btnMonitor.setChecked(False)
            return
        srate = self.ui_cbSampleRate.currentData()
        gain = self.ui_spinGain.value()
        device_number = self.ui_cbDevice.currentData()
        if device_number is None or not self._devices:
            QMessageBox.warning(self, "No RTL-SDR detected",
                                "No RTL-SDR device is currently visible to macOS.")
            self.ui_btnMonitor.setChecked(False)
            return
        # stop any hopping scan first
        if self.worker is not None and self.worker._thread is not None \
                and self.worker._thread.is_alive():
            self.worker.stop()
        if self.scanner.is_running:
            self.scanner.remove_spectrum_sink(self.waterfall.on_iq)
            self.scanner.stop()

        center = freqs[0]
        self.waterfall.reset()
        self.waterfall.set_stream(center, srate)
        self.scanner.add_spectrum_sink(self.waterfall.on_iq)
        self._opening = True
        self.ui_btnMonitor.setEnabled(False)
        self.ui_btnStartStop.setEnabled(False)
        self._live_msg.emit("Opening dongle at {0:.4f} MHz ...".format(center / 1e6))
        threading.Thread(
            target=lambda: self._monitor_open_done.emit(
                self._restart_scanner(center, srate=srate, gain=gain,
                                      device_number=device_number)),
            daemon=True,
        ).start()
        self.ui_btnStartStop.setText("Start")

    @pyqtSlot(bool)
    def _on_monitor_open_done(self, ok):
        self._opening = False
        self.ui_btnMonitor.setEnabled(True)
        self.ui_btnStartStop.setEnabled(True)
        if not ok:
            self.ui_btnMonitor.setChecked(False)
            if hasattr(self, "_monitor_timer"):
                self._monitor_timer.stop()
            if not self._dongle_dead:
                self._live_msg.emit(
                    "No IQ from dongle - replug USB, press Refresh, try again")
            return
        center = self.scanner.frequency
        srate = self.scanner.sample_rate or 1
        self._last_rssi_ts = time.time()
        if not hasattr(self, "_monitor_timer"):
            self._monitor_timer = QTimer(self)
            self._monitor_timer.timeout.connect(self._monitor_watchdog)
        self._monitor_timer.start(3000)
        self.ui_lblLive.setText(
            "Monitoring {:.4f} MHz ±{:.2f} MHz — click any burst tag to denoise".format(
                center / 1e6, srate / 2e6))
        logger.info("Wideband monitor started at {0} Hz, {1} Sps".format(center, srate))

    def stop_monitor(self):
        if hasattr(self, "_monitor_timer"):
            self._monitor_timer.stop()
        self.scanner.remove_spectrum_sink(self.waterfall.on_iq)
        if self.scanner.is_running:
            self.scanner.stop()
        self.ui_lblLive.setText("Monitor stopped")
        logger.info("Wideband monitor stopped")

    def start_scanning(self):
        if getattr(self, "_opening", False):
            return
        self._dongle_dead = False
        self._open_fail_streak = 0
        freqs = self._frequencies()
        if not freqs:
            logger.warning("No valid frequencies configured")
            return
        if self.ui_btnMonitor.isChecked():
            self.ui_btnMonitor.setChecked(False)
            self.scanner.remove_spectrum_sink(self.waterfall.on_iq)
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

        self._opening = True
        self.ui_btnStartStop.setEnabled(False)
        self.waterfall.reset()
        self.waterfall.set_stream(freqs[0], srate)
        self.scanner.add_spectrum_sink(self.waterfall.on_iq)
        self._live_msg.emit("Opening dongle at {0:.4f} MHz ...".format(freqs[0] / 1e6))
        threading.Thread(
            target=lambda: self._scan_open_done.emit(
                self._restart_scanner(freqs[0], srate=srate, gain=gain,
                                      device_number=device_number)),
            daemon=True,
        ).start()

    @pyqtSlot(bool)
    def _on_scan_open_done(self, ok):
        self._opening = False
        self.ui_btnStartStop.setEnabled(True)
        if not ok:
            self.scanner.remove_spectrum_sink(self.waterfall.on_iq)
            if not self._dongle_dead:
                self._live_msg.emit(
                    "No IQ from dongle - replug USB, press Refresh, try again")
            return
        self.ui_btnStartStop.setText("Stop")
        freqs = self._frequencies()
        self.worker = RfSampleWorker(self.scanner, self.gps_provider, freqs, parent=self)
        self.worker.reopen_fn = self._restart_scanner
        self.worker.sample_ready.connect(self._on_sample)
        self.worker.configure(self.ui_spinSpacing.value(), self.ui_spinDwell.value())
        if self.ui_chkAuto.isChecked():
            self.worker.start()
        self._refresh_freq_display()
        self._save_settings()
        logger.info(
            "RF exploration started: freqs={0} Hz, srate={1}, gain={2}".format(
                freqs, self.scanner.sample_rate, self.scanner.gain
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
        saturated = bool(analysis is not None and analysis.get("saturated"))
        sample = {
            "freq": freq,
            "lat": lat,
            "lon": lon,
            "rssi": rssi,
            "ts": dt.timestamp(),
            "date_str": dt.strftime("%Y-%m-%d"),
            "time_str": dt.strftime("%H:%M:%S"),
            "analysis": analysis,
            "saturated": saturated,
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
        if saturated:
            clip = analysis.get("clip_ratio", 0.0)
            self.ui_lblSat.setText(
                "SATURATED ({0:.1f}% clipped) - lower the gain!".format(clip * 100.0)
            )
            self.ui_lblSat.show()
        else:
            self.ui_lblSat.hide()
        self._append_table_row(sample)
        self._write_csv(sample)
        self._refresh_freq_display()
        self.refresh_map()

    def _append_table_row(self, s):
        """Add one fox-hunt RSSI sample to the unified events table."""
        row = self.ui_table.rowCount()
        self.ui_table.insertRow(row)
        vals = [s["time_str"], s["freq"] / 1e6, "RSSI",
                "{0:.1f} dB {1}".format(s["rssi"], peaks_summary(s["analysis"])).strip(),
                "{0:.5f}, {1:.5f}".format(s["lat"], s["lon"])]
        for cidx, v in enumerate(vals):
            it = QTableWidgetItem("{0:.4f}".format(v) if cidx == 1 else str(v))
            if cidx == 1:
                it.setData(Qt.ItemDataRole.DisplayRole, float(v))
            it.setData(Qt.ItemDataRole.UserRole, ("sample", s))
            self.ui_table.setItem(row, cidx, it)
        if s.get("saturated"):
            red = QColor(90, 30, 30)
            for col in range(self.ui_table.columnCount()):
                item = self.ui_table.item(row, col)
                if item is not None:
                    item.setBackground(red)
        if self.ui_table.rowCount() > 500:
            self.ui_table.removeRow(0)

    def _infer_burst_kind(self, burst):
        """Cheap local inference from the buffered IQ: pulse structure +
        amplitude-vs-frequency behaviour -> human-readable hint."""
        try:
            iq, sr = self.waterfall.extract_burst_iq(burst)
        except Exception:
            return ""
        if iq is None or iq.size < 4096 or sr <= 0:
            return ""
        x = np.asarray(iq, dtype=np.complex64)
        env = np.abs(x)
        mean_env = float(env.mean()) or 1e-9
        cv = float(env.std() / mean_env)
        hints = []
        if cv < 0.22:
            # near-constant envelope -> continuous carrier; look at frequency
            ph = np.unwrap(np.angle(x.astype(np.complex128)))
            fdev = float(np.std(np.diff(ph) * sr / (2 * np.pi)))
            if fdev > sr * 0.02:
                hints.append("continuous, wideband FM/FSK (~{0:.0f} kHz dev)".format(
                    fdev / 1e3))
            elif fdev > sr * 0.002:
                hints.append("continuous, narrow FM/PM (~{0:.1f} kHz dev)".format(
                    fdev / 1e3))
            else:
                hints.append("continuous carrier (CW)")
            return " · ".join(hints)
        lo = float(np.percentile(env, 20))
        hi = float(np.percentile(env, 90))
        thr = lo + 0.35 * (hi - lo)
        hot = env > thr
        duty = float(hot.mean())
        edges = np.diff(hot.astype(np.int8))
        n_pulses = int((edges == 1).sum())
        pw_ms = (float(hot.sum()) / max(n_pulses, 1)) / sr * 1e3
        gap_ms = (x.size - float(hot.sum())) / max(n_pulses, 1) / sr * 1e3
        if n_pulses >= 4 and duty < 0.7:
            hints.append("OOK-style, {0} pulses ~{1:.1f}ms on/{2:.1f}ms off".format(
                n_pulses, pw_ms, gap_ms))
            if 0.2 < pw_ms < 6.0 and 0.2 < gap_ms < 6.0 and abs(pw_ms - gap_ms) < 2.5:
                hints.append("PWM-like coding")
        else:
            hints.append("short burst")
        return " · ".join(hints)

    def _append_burst_row(self, b):
        """Add one waterfall burst to the unified events table."""
        row = self.ui_table.rowCount()
        self.ui_table.insertRow(row)
        band = (b.get("label", "burst").split("  ", 1) + ["burst"])[1]
        kind = self._infer_burst_kind(b)
        b["detail"] = "{0} · ~{1:.0f} kHz · SNR {2:.0f} dB".format(
            band, b.get("bw", 0.0) / 1e3, b.get("snr", 0.0))
        if kind:
            b["detail"] += " · " + kind
        vals = [b.get("time_str", ""), b.get("fc", 0.0) / 1e6, "Burst",
                b["detail"], "-"]
        for cidx, v in enumerate(vals):
            it = QTableWidgetItem("{0:.4f}".format(v) if cidx == 1 else str(v))
            if cidx == 1:
                it.setData(Qt.ItemDataRole.DisplayRole, float(v))
            it.setData(Qt.ItemDataRole.UserRole, ("burst", b))
            self.ui_table.setItem(row, cidx, it)
        if self.ui_table.rowCount() > 500:
            self.ui_table.removeRow(0)

    def _on_cell_double_clicked(self, row, _column):
        item = self.ui_table.item(row, 0)
        payload = item.data(Qt.ItemDataRole.UserRole) if item else None
        if not payload:
            return
        kind, obj = payload
        if kind == "sample":
            self._open_analysis(obj)
        elif kind == "burst":
            self._on_waterfall_burst(obj)

    def _open_analysis(self, sample):
        dialog = SignalAnalysisDialog(
            sample,
            sample.get("analysis"),
            parent=self,
            capture_cb=lambda: self.start_burst_capture(sample),
            bursts_dir=CAPTURE_DIR,
            open_file_cb=self._open_in_urh,
        )
        self.burst_captured.connect(dialog.on_burst_captured)
        dialog.exec()
        try:
            self.burst_captured.disconnect(dialog.on_burst_captured)
        except (TypeError, RuntimeError):
            pass

    def _open_in_urh(self, path):
        """Open an IQ file in the URH Analysis tab (used for denoised bursts)."""
        if self.main_controller is None:
            logger.warning("No main controller; cannot open {0}".format(path))
            return
        self.main_controller.add_signalfile(path)
        self.ui.tab_interpretation.set_active_message(-1)
        idx = self.main_controller.signal_tab_controller.currentIndex()
        self.ui.tab_interpretation.set_active_message(idx)

    # ------------------------------------------- waterfall AI tagging/denoise

    def _on_waterfall_burst(self, burst):
        menu = QMenu(self)
        label = burst.get("label", "burst")
        fc = self.waterfall.center_freq - self.waterfall.sample_rate / 2 + \
            (burst["lo"] + (burst["hi"] - burst["lo"]) / 2 + .5) / \
            WF_COLS * self.waterfall.sample_rate
        title = menu.addAction("{0}".format(label))
        title.setEnabled(False)
        act_detail = menu.addAction(burst.get("detail", ""))
        act_detail.setEnabled(False)
        origin = self._burst_origin(fc, burst)
        if origin:
            act_org = menu.addAction(origin)
            act_org.setEnabled(False)
        menu.addSeparator()
        act_denoise = menu.addAction("Denoise this burst (RadioLLM)")
        act_save = menu.addAction("Save raw slice only")
        chosen = menu.exec(self.cursor().pos())
        if chosen == act_denoise:
            self._denoise_waterfall_burst(burst, open_result=True, origin=origin)
        elif chosen == act_save:
            self._denoise_waterfall_burst(burst, open_result=False, origin=origin)

    def _burst_origin(self, fc, burst):
        """Human-readable origin line: frequency, time, GPS of this capture."""
        parts = ["{0:.4f} MHz".format(fc / 1e6),
                 datetime.now().strftime("%H:%M:%S")]
        provider = getattr(self, "gps_provider", None)
        pos = provider.position if provider is not None else None
        if pos is not None and provider.has_fix():
            parts.append("GPS {0:.5f},{1:.5f}".format(pos[0], pos[1]))
        bw = burst.get("hi", 0) - burst.get("lo", 0)
        if bw > 0:
            parts.append("BW ~{0:.0f} kHz".format(
                bw / WF_COLS * (self.waterfall.sample_rate / 1e3)))
        return " · ".join(parts)

    def _denoise_waterfall_burst(self, burst, open_result=True, origin=""):
        iq, sr = self.waterfall.extract_burst_iq(burst)
        if iq is None or iq.size < 4096:
            QMessageBox.information(self, "RadioLLM",
                                    "Not enough buffered IQ for that burst yet.\n"
                                    "Wait a moment and click again.")
            return
        fc = self.waterfall.center_freq - self.waterfall.sample_rate / 2 + \
            (burst["lo"] + (burst["hi"] - burst["lo"]) / 2 + .5) / \
            WF_COLS * self.waterfall.sample_rate
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%H%M%S")
        path = os.path.join(CAPTURE_DIR,
                            "wf_{:.3f}MHz_{}.complex16s".format(fc / 1e6, stamp))
        RadioLLMClient.save_iq(path, np.asarray(iq, dtype=np.complex64))

        url = settings.read("radiollm_url", DEFAULT_RADIOLLM_URL, str)
        label = burst.get("label", "")

        def work():
            try:
                out, stats = RadioLLMClient.denoise(
                    path, url=url, sample_rate=float(sr), center_freq=float(fc))
                msg = "{0} -> {1} ({2}s)".format(label, out, stats.get("total_s"))
                if origin:
                    msg = "[{0}] {1}".format(origin, msg)
                self.waterfall_denoise_done.emit(
                    out if open_result else "", msg, burst)
            except Exception as e:
                logger.exception("waterfall denoise failed")
                self.waterfall_denoise_done.emit("", "error: {0}".format(e), burst)

        threading.Thread(target=work, daemon=True).start()
        self.ui_lblLive.setText("RadioLLM: denoising {0} ...".format(label))

    @pyqtSlot(str, str, object)
    def _on_waterfall_denoise_done(self, out_path, message, burst=None):
        self.ui_lblLive.setText("RadioLLM: {0}".format(message))
        if burst is not None and out_path:
            # Upgrade the burst's band guess with the model's verdict.
            verdict = "RadioLLM: {0}".format(
                os.path.basename(out_path).rsplit(".", 1)[0])
            for ln in message.splitlines():
                if "->" in ln:
                    verdict = "RadioLLM: {0}".format(ln.split("->", 1)[1].strip())
                    break
            burst["detail"] = verdict
            self._refresh_burst_row(burst)
        if out_path and os.path.isfile(out_path):
            self._open_in_urh(out_path)

    def _refresh_burst_row(self, burst):
        """Update the Detail cell of the events-table row bound to `burst`."""
        for r in range(self.ui_table.rowCount()):
            it = self.ui_table.item(r, 0)
            payload = it.data(Qt.ItemDataRole.UserRole) if it else None
            if payload and payload[0] == "burst" and payload[1] is burst:
                det = self.ui_table.item(r, 3)
                if det is not None:
                    det.setText(burst.get("detail", ""))
                return

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
                summary = (
                    "saturated (clipped)"
                    if a is not None and a.get("saturated")
                    else "no peaks"
                )
            line = "{0},{1},{2:.3f},{3:.6f},{4:.6f},{5:.1f},{6},{7:.5f},{8:.1f},{9:.1f},{10},{11}".format(
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
                1 if s.get("saturated") else 0,
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
        self._last_rssi_ts = ts
        self.ui_lblLive.setText("RSSI {0:.1f} dB".format(rssi))

    def _restart_scanner(self, freq, wait_s: float = 2.0,
                         srate=None, gain=None, device_number=None) -> bool:
        """Stop+start the scanner and verify IQ actually flows.

        Safe to call from any thread. Detects the wedged-dongle state (open
        succeeds but no samples arrive) after 2 consecutive failures and stops
        hammering the device; recovery requires a physical replug + Refresh or
        a fresh Monitor press.
        """
        if self._dongle_dead:
            return False
        if self.scanner.is_running:
            self.scanner.stop()
            time.sleep(0.3)
        try:
            self.scanner.start(
                freq,
                srate or getattr(self.scanner, "sample_rate", None) or 1_024_000,
                gain if gain is not None else getattr(self.scanner, "gain", 40),
                device_number=(self.scanner.device_number
                               if device_number is None else device_number),
            )
        except Exception as e:
            logger.error("Scanner restart failed at {0:.3f} MHz: {1}".format(freq / 1e6, e))
            return self._note_open_result(False)
        t0 = time.time()
        while time.time() - t0 < wait_s and not self.scanner.data_received:
            time.sleep(0.1)
        return self._note_open_result(bool(self.scanner.data_received))

    def _note_open_result(self, ok: bool) -> bool:
        if ok:
            if self._dongle_dead or self._open_fail_streak:
                logger.info("Dongle recovered after {0} failed open(s)".format(
                    self._open_fail_streak))
            self._open_fail_streak = 0
            self._dongle_dead = False
            return True
        self._open_fail_streak += 1
        if self._open_fail_streak >= 2 and not self._dongle_dead:
            self._dongle_dead = True
            logger.warning("Dongle marked dead: {0} consecutive opens with no data. "
                           "Unplug/replug the USB dongle, then press Refresh.".format(
                               self._open_fail_streak))
            self._live_msg.emit("Dongle wedged - unplug/replug USB, then press Refresh")
        return False

    def _monitor_watchdog(self):
        """Reopen the dongle if IQ stops flowing while monitoring."""
        if not self.ui_btnMonitor.isChecked() or not self.scanner.is_running:
            return
        if self._dongle_dead:
            return
        now = time.time()
        if now - getattr(self, "_last_rssi_ts", 0) > 6.0 \
                and now - self._last_reopen_ts > 8.0:
            freq = self.scanner.frequency
            srate = self.scanner.sample_rate
            gain = self.scanner.gain
            dev = self.scanner.device_number
            self._last_reopen_ts = now
            logger.warning("Monitor watchdog: no IQ for >6 s, reopening device at "
                           "{0:.3f} MHz".format(freq / 1e6))
            self._live_msg.emit("Watchdog: no IQ - reopening dongle ...")

            def _reopen():
                ok = self._restart_scanner(freq)
                if ok:
                    self._last_rssi_ts = time.time()
                elif not self._dongle_dead:
                    self._live_msg.emit("Watchdog reopen failed - still no IQ")

            threading.Thread(target=_reopen, daemon=True).start()

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
