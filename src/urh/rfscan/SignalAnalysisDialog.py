import json
import threading
import urllib.request

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from urh import settings
from urh.util.Logger import logger

DEFAULT_LMSTUDIO_URL = "http://localhost:1234"


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

    ai_result = pyqtSignal(str, bool)

    def __init__(self, sample, analysis, parent=None):
        super().__init__(parent)
        self.sample = sample
        self.analysis = analysis
        self.setWindowTitle(
            "Signal dissection - {0:.3f} MHz".format(sample["freq"] / 1e6)
        )
        self.setMinimumSize(760, 560)
        self.ai_result.connect(self._on_ai_result)
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
                "signal RSSI {3} | bandwidth {4:.0f} kHz | {5} peaks | "
                "{6:.6f}, {7:.6f} @ {8}"
            ).format(
                a.get("center_freq", 0.0) / 1e6,
                a.get("n_fft", 0),
                a.get("noise_floor_db", 0.0),
                "{0:.1f} dB".format(a["signal_rssi_db"])
                if a.get("signal_rssi_db") is not None
                else "n/a",
                a.get("bandwidth_hz", 0.0) / 1e3,
                a.get("n_peaks", 0),
                self.sample["lat"],
                self.sample["lon"],
                self.sample.get("time_str", ""),
            )
        else:
            info = "No spectrum data captured for this sample."
        layout.addWidget(QLabel(info))

        self.ui_peaks = QTableWidget(0, 5)
        self.ui_peaks.setHorizontalHeaderLabels(
            ["Freq (MHz)", "dB", "dB above floor", "Width (kHz)", "Signal RSSI (dB)"]
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
            self.ui_peaks.setItem(
                row, 4, QTableWidgetItem("{0:.1f}".format(p["signal_rssi_db"]))
            )
        layout.addWidget(self.ui_peaks, 2)

        ai_row = QHBoxLayout()
        ai_row.addWidget(QLabel("LM Studio:"))
        self.ui_ai_url = QLineEdit(
            settings.read("ai_lmstudio_url", DEFAULT_LMSTUDIO_URL, str)
        )
        self.ui_ai_url.setPlaceholderText(DEFAULT_LMSTUDIO_URL)
        self.ui_ai_url.setMinimumWidth(240)
        ai_row.addWidget(self.ui_ai_url, 1)
        self.ui_btn_ai = QPushButton("Analyze with AI")
        self.ui_btn_ai.setToolTip(
            "Send this signal's dissection to a local LM Studio model "
            "(OpenAI-compatible API at the URL above)."
        )
        self.ui_btn_ai.clicked.connect(self._start_ai_analysis)
        ai_row.addWidget(self.ui_btn_ai)
        layout.addLayout(ai_row)

        self.ui_ai_output = QPlainTextEdit()
        self.ui_ai_output.setReadOnly(True)
        self.ui_ai_output.setMaximumHeight(170)
        self.ui_ai_output.setPlaceholderText(
            'Click "Analyze with AI" to have LM Studio interpret this signal.'
        )
        layout.addWidget(self.ui_ai_output, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        self.ui_peaks.setFocus()

    # ------------------------------------------------------------- AI analysis

    def _start_ai_analysis(self):
        url = (self.ui_ai_url.text() or DEFAULT_LMSTUDIO_URL).strip()
        settings.write("ai_lmstudio_url", url)
        self.ui_btn_ai.setEnabled(False)
        self.ui_ai_output.setPlainText(
            "Analyzing with LM Studio at {0} ...".format(url)
        )
        fingerprint = self._build_fingerprint()
        threading.Thread(
            target=self._run_ai_worker, args=(url, fingerprint), daemon=True
        ).start()

    def _run_ai_worker(self, url, fingerprint):
        try:
            logger.debug("AI: querying LM Studio at {0}".format(url))
            reply = self._query_lmstudio(url, fingerprint)
            logger.debug("AI: reply received ({0} chars)".format(len(reply)))
            self.ai_result.emit(reply, False)
        except Exception as e:
            logger.error("AI analysis failed: {0}".format(e))
            self.ai_result.emit("AI analysis failed: {0}".format(e), True)

    def _on_ai_result(self, text, is_error):
        self.ui_btn_ai.setEnabled(True)
        if is_error:
            self.ui_ai_output.setStyleSheet("color: #e06c6c;")
            try:
                self.sample["ai_error"] = text
            except TypeError:
                pass
        else:
            self.ui_ai_output.setStyleSheet("")
            try:
                self.sample["ai_analysis"] = text
            except TypeError:
                pass
        self.ui_ai_output.setPlainText(text or "(empty reply from LM Studio)")

    def _build_fingerprint(self) -> str:
        a = self.analysis
        s = self.sample
        lines = [
            "Sample: {0:.3f} MHz at {1:.6f}, {2:.6f} @ {3}".format(
                s["freq"] / 1e6, s["lat"], s["lon"], s.get("time_str", "")
            ),
            "Sample RSSI: {0:.1f} dB".format(s["rssi"]),
        ]
        if a is None:
            lines.append("No spectrum analysis available for this sample.")
            return "\n".join(lines)
        lines.extend(
            [
                "Sample rate: {0} Hz".format(a.get("sample_rate", 0)),
                "FFT size: {0}".format(a.get("n_fft", 0)),
                "Noise floor: {0:.1f} dB".format(a.get("noise_floor_db", 0.0)),
            ]
        )
        sig = a.get("signal_rssi_db")
        lines.append(
            "Signal RSSI (narrowband): {0:.1f} dB".format(sig)
            if sig is not None
            else "Signal RSSI (narrowband): n/a (no peak detected)"
        )
        lines.append(
            "Occupied bandwidth: {0:.1f} kHz".format(a.get("bandwidth_hz", 0.0) / 1e3)
        )
        lines.append("Detected peaks: {0}".format(a.get("n_peaks", 0)))
        for p in a.get("peaks", []):
            lines.append(
                "  {0:.5f} MHz | {1:.1f} dB ({2:.1f} dB above floor) | "
                "width {3:.1f} kHz | signal {4:.1f} dB".format(
                    p["freq_mhz"],
                    p["db"],
                    p["db_above_floor"],
                    p["width_hz"] / 1e3,
                    p["signal_rssi_db"],
                )
            )
        return "\n".join(lines)

    @staticmethod
    def _query_lmstudio(url: str, fingerprint: str) -> str:
        model = "local-model"
        try:
            with urllib.request.urlopen(url + "/v1/models", timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data") or []
            if models and models[0].get("id"):
                model = models[0]["id"]
            logger.debug("AI: using LM Studio model '{0}'".format(model))
        except Exception as e:
            logger.debug("AI: model discovery failed: {0}".format(e))

        base_messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert in radio frequency (RF) signal analysis "
                    "and software-defined radio (SDR). You interpret spectral "
                    "survey data to identify the likely type of wireless signals."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here is a spectrum analysis captured with Universal Radio "
                    "Hacker (URH), an SDR tool.\n\n"
                    + fingerprint
                    + "\n\nPlease interpret this signal: what kind of "
                    "transmission might it be (e.g. ISM-band OOK/FSK/LoRa-like, "
                    "telemetry, remote control, noise), is the dominant peak a "
                    "genuine emitter or an artifact of the receiver, and what "
                    "should we look for next? Be concise."
                ),
            },
        ]

        def post(payload) -> dict:
            req = urllib.request.Request(
                url + "/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                return json.loads(resp.read().decode("utf-8"))

        def extract_content(data) -> str:
            try:
                message = data["choices"][0]["message"]
            except (KeyError, IndexError):
                raise RuntimeError(
                    "Unexpected LM Studio response: {0}".format(str(data)[:2000])
                )
            return (message.get("content") or "").strip()

        def extract_reasoning(data) -> str:
            try:
                message = data["choices"][0]["message"]
            except (KeyError, IndexError):
                return ""
            return (message.get("reasoning_content") or "").strip()

        payload = {
            "model": model,
            "messages": base_messages,
            "temperature": 0.3,
            "max_tokens": 1200,
            "stream": False,
        }
        data = post(payload)
        logger.debug("AI: raw LM Studio response: {0}".format(str(data)[:4000]))
        reply = extract_content(data)
        if reply:
            return reply

        # Qwen3-style reasoning models sometimes burn max_tokens on thinking and
        # emit an empty `content`. Retry with chain-of-thought disabled.
        logger.warning(
            "AI: empty reply from {0}, retrying with thinking disabled".format(model)
        )
        no_think = dict(payload)
        no_think["enable_thinking"] = False
        no_think["chat_template_kwargs"] = {"enable_thinking": False}
        reply = extract_content(post(no_think))
        if reply:
            return reply

        # Last resort: surface the model's reasoning so the user sees something.
        reasoning = extract_reasoning(data)
        if reasoning:
            return (
                reasoning
                + "\n\n[The model produced no final answer; showing its "
                "reasoning instead.]"
            )
        raise RuntimeError(
            "LM Studio returned an empty reply (model: {0}).".format(model)
        )
