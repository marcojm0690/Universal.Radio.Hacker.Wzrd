import time
import threading
import collections

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRectF
from PyQt6.QtGui import QImage, QPainter, QColor, QFont, QPen, QBrush, QPixmap, QPainterPath, QLinearGradient
from PyQt6.QtWidgets import QWidget, QComboBox, QDoubleSpinBox, QCheckBox, QPushButton, QHBoxLayout, QVBoxLayout, QLabel

FFT_SIZE = 1024
FFT_STEP = 512
COLS = 512
ROWS = 384
RAW_KEEP_S = 10.0
PEND_MAX = 24          # max queued raw chunks before dropping oldest
BURST_HISTORY_MAX = 300

GUT_L, GUT_R, GUT_B = 40, 34, 18   # time labels | dB legend | freq ticks
SPEC_H = 54


def _lut_from_stops(stops):
    lut = np.zeros((256, 3), dtype=np.uint8)
    xs = np.linspace(0.0, 1.0, 256)
    for ch in range(3):
        lut[:, ch] = np.interp(xs, [s[0] for s in stops],
                               [s[1][ch] for s in stops]).astype(np.uint8)
    return lut


def _turbo_lut():
    return _lut_from_stops([
        (0.00, (48, 18, 59)), (0.125, (70, 107, 227)), (0.25, (42, 170, 227)),
        (0.375, (33, 212, 168)), (0.5, (129, 239, 82)), (0.625, (222, 227, 44)),
        (0.75, (250, 164, 27)), (0.875, (232, 84, 15)), (1.00, (190, 20, 20)),
    ])


def _viridis_lut():
    return _lut_from_stops([
        (0.00, (68, 1, 84)), (0.25, (59, 82, 139)), (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)), (1.00, (253, 231, 37)),
    ])


def _inferno_lut():
    return _lut_from_stops([
        (0.00, (0, 0, 4)), (0.25, (87, 16, 110)), (0.50, (188, 55, 84)),
        (0.75, (249, 142, 9)), (1.00, (252, 255, 164)),
    ])


LUTS = {"Turbo": _turbo_lut(), "Viridis": _viridis_lut(), "Inferno": _inferno_lut()}
LUT = LUTS["Turbo"]   # legacy alias


def classify_signal(f_center_hz: float, bw_hz: float):
    fmhz = f_center_hz / 1e6
    bwk = bw_hz / 1e3

    def bw_tag():
        if bwk < 12:
            return "narrowband FM/FSK"
        if bwk < 40:
            return "wide FM data"
        if bwk > 1500:
            return "wideband (WiFi/video?)"
        return "~100 kHz data link"

    table = [
        (88.0, 108.0, "FM broadcast", "radio station"),
        (156.0, 162.4, "Marine VHF", "ship radio"),
        (162.0, 174.0, "VHF public safety", "police/fire/utility"),
        (285.0, 325.0, "Remote ~315 MHz", "car key fob / TPMS"),
        (330.0, 391.0, "Garage / Gate", "garage-door opener / gate remote"),
        (406.0, 406.1, "COSPAS-SARSAT", "emergency beacon"),
        (433.05, 434.79, "ISM 433 MHz", "key fob / sensor / telemetry"),
        (446.0, 446.2, "PMR446", "walkie-talkie"),
        (454.0, 471.0, "UHF biz / pager", "FRS-GMRS / POCSAG pager"),
        (863.0, 870.0, "ISM 868 MHz", "LoRa / alarm sensor"),
        (902.0, 928.0, "ISM 915 MHz", "LoRa / utility meter"),
        (1090.0, 1090.1, "ADS-B", "aircraft transponder"),
        (1574.0, 1576.5, "GPS L1", "satellite navigation"),
        (1880.0, 1900.0, "DECT", "cordless phone"),
        (2400.0, 2483.5, "2.4 GHz ISM", "WiFi/BT/Zigbee/drone RC"),
        (5150.0, 5350.0, "5 GHz WiFi", "WiFi/WiFi Direct"),
        (5725.0, 5875.0, "5.8 GHz ISM", "WiFi / FPV video"),
    ]
    for lo, hi, band, srcs in table:
        if lo <= fmhz <= hi:
            return band, srcs
    if 118.0 <= fmhz <= 136.9:
        return "Airband AM", "aircraft/atc"
    return "{:.3f} MHz".format(fmhz), bw_tag()


def _fmt_time(t):
    lt = time.localtime(t)
    return "{:02d}:{:02d}:{:02d}".format(lt.tm_hour, lt.tm_min, lt.tm_sec)


class _Canvas(QWidget):
    """Spectrogram drawing surface: spectrum strip + waterfall + axes + bursts."""
    burst_selected = pyqtSignal(dict)
    rows_ready = pyqtSignal(object, object)   # pooled dB rows, row timestamps
    burst_finalized = pyqtSignal(dict)
    tune_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)

        self.center_freq = 433.92e6
        self.sample_rate = 2.4e6

        # display state
        self.palette_name = "Turbo"
        self.db_floor = -95.0
        self.db_range = 70.0
        self.auto_scale = True
        self.zx0, self.zx1 = 0.0, 1.0     # horizontal view window (fractions)
        self._pan_x0 = None
        self.flip_rows = False            # newest-on-top display mode
        self._focus = None
        self._focus_ts = 0.0

        self._wf = np.full((ROWS, COLS), -120.0, dtype=np.float32)
        self._img_rows = 0
        self._row_times = collections.deque(maxlen=ROWS)
        self._spec_line = None
        self._pend = collections.deque()
        self._pend_lock = threading.Lock()
        self._carry = np.zeros((0,), dtype=np.complex64)
        self._raw = collections.deque()
        self._raw_lock = threading.Lock()
        self._raw_seconds = 0.0
        self._noise_floor = np.full(COLS, -80.0, dtype=np.float32)
        self._active = []
        self._bursts = collections.deque(maxlen=BURST_HISTORY_MAX)
        self._hover = QPoint(-1, -1)
        self._pixmap = None
        self._dbg_calls = 0
        self._dbg_rows = 0
        self.rows_ready.connect(self._on_rows)

        self._worker = threading.Thread(target=self._consume, daemon=True)
        self._worker.start()

    # ------------------------------------------------------------- streaming

    def reset(self):
        with self._pend_lock:
            self._pend.clear()
        self._carry = np.zeros((0,), dtype=np.complex64)
        with self._raw_lock:
            self._raw.clear()
            self._raw_seconds = 0.0
        self._wf[:] = -120.0
        self._img_rows = 0
        self._row_times.clear()
        self._active.clear()
        self._bursts.clear()
        self._noise_floor[:] = -80.0
        self._floor_seeded = False
        self._spec_line = None
        self._pixmap = None
        self.update()

    def on_iq(self, iq: np.ndarray, freq_hz=None):
        """Thread-safe entry point (called from the scanner thread)."""
        self._dbg_calls += 1
        iq = self._to_complex(iq)
        if iq.size:
            with self._pend_lock:
                self._pend.append(iq)
                while len(self._pend) > PEND_MAX:
                    self._pend.popleft()

    @staticmethod
    def _to_complex(a) -> np.ndarray:
        a = np.asarray(a)
        if a.dtype.kind == "c":
            return a.astype(np.complex64, copy=False)
        a = a.astype(np.float32) / 128.0
        if a.ndim == 2 and a.shape[1] >= 2:
            return (a[:, 0] + 1j * a[:, 1]).astype(np.complex64)
        n = (a.size // 2) * 2
        return (a[0:n:2] + 1j * a[1:n:2]).astype(np.complex64)

    # ------------------------------------------------------------ DSP thread

    def _consume(self):
        """Background thread: chunk -> FFT rows -> hand to GUI."""
        win = np.hanning(FFT_SIZE).astype(np.complex64)
        edges = np.linspace(0, FFT_SIZE, COLS + 1).astype(int)
        while True:
            try:
                with self._pend_lock:
                    chunks = []
                    while self._pend and len(chunks) < 8:
                        chunks.append(self._pend.popleft())
                if not chunks:
                    time.sleep(0.02)
                    continue

                joined = np.concatenate([self._carry] + chunks)
                now = time.time()
                # Keep EVERY chunk of the batch in the raw ring (staggered
                # timestamps); previously only the last one survived, which
                # starved burst extraction under bursty arrival.
                durs = [c.shape[0] / max(self.sample_rate, 1.0) for c in chunks]
                total = sum(durs)
                acc = 0.0
                with self._raw_lock:
                    for ch, d in zip(chunks, durs):
                        acc += d
                        self._raw.append((now - (total - acc), ch))
                    self._raw_seconds += total
                    while self._raw_seconds > RAW_KEEP_S and len(self._raw) > 1:
                        _, c = self._raw.popleft()
                        self._raw_seconds -= c.shape[0] / max(self.sample_rate, 1.0)

                n_rows = (len(joined) - FFT_SIZE) // FFT_STEP + 1
                if n_rows <= 0:
                    self._carry = joined
                    continue
                idx = np.arange(FFT_SIZE)[None, :] + FFT_STEP * np.arange(n_rows)[:, None]
                frames = joined[idx]
                spec = np.fft.fftshift(np.fft.fft(frames * win[None, :], axis=1), axes=1)
                db = 20.0 * np.log10(np.abs(spec) + 1e-9).astype(np.float32)
                pooled = np.stack([
                    db[:, edges[i]:max(edges[i + 1], edges[i] + 1)].max(axis=1)
                    for i in range(COLS)], axis=1)
                dt_row = FFT_STEP / self.sample_rate
                times = [now - (n_rows - 1 - r) * dt_row for r in range(n_rows)]
                self._dbg_rows += n_rows
                self.rows_ready.emit(pooled, times)
                self._spec_line = pooled[-1]
            except Exception as e:
                import logging
                logging.getLogger(__name__).exception("waterfall dsp error: %s", e)
                self._carry = np.zeros((0,), dtype=np.complex64)
                time.sleep(0.2)

    def _on_rows(self, pooled, times):
        finalized = []
        for r in range(pooled.shape[0]):
            self._wf[:-1] = self._wf[1:]
            self._wf[-1] = pooled[r]
            self._row_times.append(times[r])
            self._img_rows = min(self._img_rows + 1, ROWS)
            finalized.extend(self._update_floor_and_bursts(pooled[r]))
        for b in finalized:
            self.burst_finalized.emit(b)
        self._render()

    # ------------------------------------------------------- burst tracking

    def _update_floor_and_bursts(self, row_db):
        if not getattr(self, "_floor_seeded", False):
            # Seed the floor from the first real row so the fixed -80 dB
            # prior doesn't flag the whole band as "hot" at stream start.
            self._noise_floor = row_db.astype(np.float32).copy()
            self._floor_seeded = True
            return []
        self._noise_floor = 0.98 * self._noise_floor + 0.02 * row_db
        hot = row_db > self._noise_floor + 8.0
        spans = []
        c = 0
        while c < COLS:
            if hot[c]:
                c0 = c
                while c < COLS and hot[c]:
                    c += 1
                if c - c0 >= 2:
                    spans.append((c0, c - 1))
            else:
                c += 1
        unmatched = list(range(len(spans)))
        for tr in self._active:
            best, best_d = None, 1e9
            for i in unmatched:
                lo, hi = spans[i]
                d = max(lo - tr["hi"], tr["lo"] - hi, 0)
                if d < best_d and d <= 6:
                    best, best_d = i, d
            if best is not None:
                lo, hi = spans[best]
                tr["lo"], tr["hi"] = min(tr["lo"], lo), max(tr["hi"], hi)
                tr["peak"] = max(tr.get("peak", -120.0), float(row_db[lo:hi + 1].max()))
                tr["miss"] = 0
                unmatched.remove(best)
            else:
                tr["miss"] += 1
        for i in unmatched:
            lo, hi = spans[i]
            self._active.append({"lo": lo, "hi": hi, "miss": 0,
                                 "t0": self._row_times[-1] if self._row_times else time.time(),
                                 "peak": float(row_db[lo:hi + 1].max())})
        done = [tr for tr in self._active if tr["miss"] >= 8]
        out = []
        for tr in done:
            if self._row_times:
                tr["t1"] = self._row_times[-1]
                self._annotate(tr)
                self._bursts.append(tr)
                out.append(tr)
        self._active[:] = [tr for tr in self._active if tr["miss"] < 8]
        return out

    def _annotate(self, b):
        fc = self.center_freq - self.sample_rate / 2 + \
            (b["lo"] + (b["hi"] - b["lo"]) / 2 + .5) / COLS * self.sample_rate
        bw = (b["hi"] - b["lo"] + 1) / COLS * self.sample_rate
        snr = float(b.get("peak", 0) - self._noise_floor[b["lo"]:b["hi"] + 1].mean())
        band, srcs = classify_signal(fc, bw)
        b["fc"] = fc
        b["bw"] = bw
        b["snr"] = snr
        b["label"] = "{:.3f} MHz  {}".format(fc / 1e6, band)
        b["detail"] = "~{:.0f} kHz · SNR {:.0f} dB · likely {}".format(bw / 1e3, snr, srcs)
        b["time_str"] = _fmt_time(b.get("t0") or time.time())

    def extract_burst_iq(self, burst):
        with self._raw_lock:
            snapshot = list(self._raw)
        t0 = burst.get("t0")
        if t0 is None or not snapshot:
            return None, self.sample_rate
        t1 = burst.get("t1") or (snapshot[-1][0])
        parts = []
        for t_end, chunk in snapshot:
            dur = chunk.shape[0] / max(self.sample_rate, 1.0)
            t_beg = t_end - dur
            if t_end >= t0 - 0.15 and t_beg <= t1 + 0.15:
                parts.append(chunk)
        if not parts:
            return None, self.sample_rate
        return np.concatenate(parts), self.sample_rate

    # -------------------------------------------------------------- painting

    def _render(self):
        lut = LUTS[self.palette_name]
        if self.auto_scale:
            floor = float(np.median(self._noise_floor)) - 4.0
            rng = max(float(self._noise_floor.max()) - floor + 14.0, 30.0)
            self.db_floor, self.db_range = floor, rng
        norm = np.clip((self._wf - self.db_floor) / max(self.db_range, 1.0), 0.0, 1.0)
        if self.flip_rows:
            norm = norm[::-1]
        rgb = lut[(norm * 255).astype(np.uint8)]
        h, w = rgb.shape[:2]
        img = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    # ---- geometry helpers

    def _plot_rect(self):
        return self.rect().adjusted(GUT_L, 6, -GUT_R, -(GUT_B + 1))

    def _spec_rect(self):
        r = self._plot_rect()
        return QRectF(r.left(), r.top(), r.width(), SPEC_H)

    def _wf_rect(self):
        r = self._plot_rect()
        return QRectF(r.left(), r.top() + SPEC_H + 6,
                      r.width(), max(r.height() - SPEC_H - 6, 40))

    def _zoom_w(self):
        return max(self.zx1 - self.zx0, 1e-6)

    def _clamp_zoom(self):
        w = self._zoom_w()
        if w > 1.0:
            self.zx0, self.zx1 = 0.0, 1.0
        self.zx0 = min(max(self.zx0, 0.0), 1.0 - w)
        self.zx1 = self.zx0 + w

    def _col_to_x(self, col_f, r):
        u = col_f / COLS
        return r.left() + (u - self.zx0) / self._zoom_w() * r.width()

    def _x_to_col(self, x, r):
        return (self.zx0 + (x - r.left()) / max(r.width(), 1) * self._zoom_w()) * COLS

    # ---- paint

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(8, 10, 14))
        plot = self._plot_rect()
        spec_r = self._spec_rect()
        wf_r = self._wf_rect()

        self._draw_spectrum(p, spec_r, wf_r)
        self._draw_waterfall(p, wf_r)
        self._draw_time_axis(p, wf_r)
        self._draw_freq_axis(p, wf_r)
        self._draw_db_legend(p, wf_r)
        self._draw_bursts(p, wf_r)
        self._draw_hover(p, wf_r)
        p.end()

    def _visible_cols(self):
        c0 = int(self.zx0 * COLS)
        c1 = min(int(np.ceil(self.zx1 * COLS)), COLS)
        return max(c0, 0), max(c1, c0 + 1)

    def _draw_spectrum(self, p, spec_r, wf_r):
        p.fillRect(spec_r, QColor(12, 16, 22))
        p.setPen(QColor(70, 76, 88))
        p.drawLine(int(wf_r.left()), int(spec_r.bottom()),
                   int(wf_r.right()), int(spec_r.bottom()))
        if self._spec_line is None:
            return
        c0, c1 = self._visible_cols()
        base = float(np.median(self._noise_floor)) if self.auto_scale else self.db_floor
        lo = base - 4.0
        seg = self._spec_line[c0:c1]
        span = max(float(seg.max()) - lo, 10.0)
        path = QPainterPath()
        n = c1 - c0
        for i in range(n):
            x = wf_r.left() + i * wf_r.width() / max(n - 1, 1)
            y = spec_r.bottom() - (float(seg[i]) - lo) / span * spec_r.height()
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        p.setPen(QPen(QColor(120, 220, 160), 1.4))
        p.drawPath(path)

    def _draw_waterfall(self, p, wf_r):
        if self._pixmap is not None and self._img_rows > 0:
            p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            sx0 = int(self.zx0 * COLS)
            sx1 = int(np.ceil(self.zx1 * COLS))
            src = self._pixmap.copy(sx0, ROWS - self._img_rows,
                                    max(sx1 - sx0, 1), self._img_rows)
            p.drawPixmap(QRectF(wf_r), src, QRectF(src.rect()))
        else:
            p.setPen(QColor(90, 100, 115))
            p.drawText(wf_r, Qt.AlignmentFlag.AlignCenter,
                       "Waterfall idle — press Start or Monitor")
        p.setPen(QPen(QColor(60, 66, 78), 1))
        p.drawRect(wf_r)

    def _draw_time_axis(self, p, r):
        if len(self._row_times) < 2:
            return
        t_new, t_old = self._row_times[-1], self._row_times[0]
        t_span = max(t_new - t_old, 1e-6)

        def ty(t):
            frac = (t_new - t) / t_span
            if self.flip_rows:
                return r.top() + frac * r.height()
            return r.bottom() - frac * r.height()

        step = next((s for s in (5, 10, 15, 30, 60, 120, 300) if t_span / s <= 6), 300)
        p.setFont(QFont("Menlo", 9))
        tt = np.floor(t_old / step) * step
        while tt <= t_new:
            if tt >= t_old:
                y = ty(tt)
                p.setPen(QPen(QColor(255, 255, 255, 16), 1))
                p.drawLine(int(r.left()), int(y), int(r.right()), int(y))
                p.setPen(QColor(120, 126, 138))
                p.fillRect(2, int(y) - 7, GUT_L - 6, 13, QColor(8, 10, 14, 170))
                p.drawText(3, int(y) + 4, _fmt_time(tt))
            tt += step

    def _draw_freq_axis(self, p, r):
        span_mhz = (self._zoom_w()) * self.sample_rate / 1e6
        left_f = self.center_freq - self.sample_rate / 2 + self.zx0 * self.sample_rate
        step = next((s for s in (0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0)
                     if span_mhz / s <= 10), 10.0)
        p.setFont(QFont("Menlo", 9))
        f = np.floor(left_f / (step * 1e6)) * step * 1e6
        while f <= left_f + span_mhz * 1e6:
            u = (f - (self.center_freq - self.sample_rate / 2)) / self.sample_rate
            x = r.left() + (u - self.zx0) / self._zoom_w() * r.width()
            if r.left() - 1 <= x <= r.right() + 1:
                p.setPen(QColor(120, 126, 138))
                p.drawLine(int(x), int(r.bottom()), int(x), int(r.bottom()) + GUT_B - 12)
                label = "{:g}".format(round(f / 1e6, 4))
                tw = p.fontMetrics().horizontalAdvance(label)
                p.drawText(int(x - tw / 2), int(r.bottom()) + GUT_B - 2, label)
            f += step * 1e6

    def _draw_db_legend(self, p, r):
        lut = LUTS[self.palette_name]
        bx = r.right() + 6
        grad = QLinearGradient(bx, r.top(), bx, r.bottom())
        for i in range(0, 256, 8):
            u = 1.0 - i / 255.0
            c = QColor(int(lut[i][0]), int(lut[i][1]), int(lut[i][2]))
            grad.setColorAt(u, c)
        p.fillRect(QRectF(bx, r.top(), 10, r.height()), QBrush(grad))
        p.setPen(QPen(QColor(60, 66, 78), 1))
        p.drawRect(QRectF(bx, r.top(), 10, r.height()))
        p.setFont(QFont("Menlo", 9))
        p.setPen(QColor(150, 156, 168))
        top = self.db_floor + self.db_range
        p.drawText(int(bx + 13), int(r.top()) + 10, "{:.0f}".format(top))
        p.drawText(int(bx + 13), int(r.bottom()) - 2, "{:.0f}".format(self.db_floor))

    def _burst_rect(self, b, r):
        if len(self._row_times) < 2 or b.get("t0") is None:
            return None
        t_new, t_old = self._row_times[-1], self._row_times[0]
        t_span = max(t_new - t_old, 1e-6)

        def ty(t):
            return r.bottom() - ((t_new - t) / t_span) * r.height()

        t1 = b.get("t1")
        if t1 is None:
            t1 = t_new
        y1 = ty(min(t1, t_new))
        y0 = ty(max(b["t0"], t_old))
        x0 = self._col_to_x(b["lo"], r)
        x1 = self._col_to_x(b["hi"] + 1, r)
        rect = QRectF(x0, y1, max(x1 - x0, 3.0), max(y0 - y1, 3.0))
        if rect.right() < r.left() or rect.left() > r.right():
            return None
        return rect.intersected(r)

    def _draw_bursts(self, p, r):
        p.setFont(QFont("Menlo", 10, QFont.Weight.Bold))
        items = list(self._bursts) + [dict(tr, t1=None) for tr in self._active]
        for b in items:
            rect = self._burst_rect(b, r)
            if rect is None:
                continue
            if "label" not in b:
                self._annotate(b)
            focused = (b is self._focus) and (time.time() - self._focus_ts < 4.0)
            color = QColor(120, 200, 255, 240) if focused else QColor(255, 208, 80, 230)
            fillc = QColor(120, 200, 255, 40) if focused else QColor(255, 208, 80, 26)
            p.setPen(QPen(color, 2.2 if focused else 1.6))
            p.setBrush(QBrush(fillc))
            p.drawRect(rect)
            tw = p.fontMetrics().horizontalAdvance(b["label"]) + 8
            lx = int(rect.x())
            ly = int(rect.y()) - 16
            if ly < 2:
                ly = int(rect.y()) + 2
            p.fillRect(lx, ly, tw, 15, QColor(20, 24, 30, 210))
            p.setPen(color)
            p.drawText(lx + 4, ly + 11, b["label"])
            p.setBrush(Qt.BrushStyle.NoBrush)

    def _draw_hover(self, p, r):
        hx, hy = self._hover.x(), self._hover.y()
        if hx < 0 or not (r.left() <= hx <= r.right() and r.top() <= hy <= r.bottom()):
            return
        p.setPen(QPen(QColor(200, 205, 215, 90), 1, Qt.PenStyle.DashLine))
        p.drawLine(int(r.left()), hy, int(r.right()), hy)
        p.drawLine(hx, int(r.top()), hx, int(r.bottom()))
        col = self._x_to_col(hx, r)
        fx = self.center_freq - self.sample_rate / 2 + col / COLS * self.sample_rate
        txt = "{:.4f} MHz".format(fx / 1e6)
        if len(self._row_times) >= 2:
            t_new, t_old = self._row_times[-1], self._row_times[0]
            frac = (r.bottom() - hy) / max(r.height(), 1)
            t = t_old + frac * (t_new - t_old)
            txt += "  " + _fmt_time(t)
        p.setPen(QColor(210, 214, 225))
        p.drawText(hx + 8, hy - 6, txt)

    # ------------------------------------------------------------- interaction

    def wheelEvent(self, e):
        r = self._wf_rect()
        fx = (self._x_to_col(e.position().x(), r) / COLS)
        fx = min(max(fx, 0.0), 1.0)
        factor = 1.25 ** (-e.angleDelta().y() / 120.0)
        w = min(max(self._zoom_w() * factor, 0.02), 1.0)
        self.zx0 = fx - (fx - self.zx0) * (w / self._zoom_w())
        self.zx1 = self.zx0 + w
        self._clamp_zoom()
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.zx0, self.zx1 = 0.0, 1.0
        self.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._pan_x0 = e.position().x()
        hit = self._hit_burst(e.position().toPoint())
        if hit:
            self.burst_selected.emit(hit)

    def mouseMoveEvent(self, e):
        self._hover = e.position().toPoint()
        if self._pan_x0 is not None:
            r = self._wf_rect()
            dx = e.position().x() - self._pan_x0
            if abs(dx) > 0:
                d = -(dx / max(r.width(), 1)) * self._zoom_w()
                self.zx0 += d
                self.zx1 += d
                self._clamp_zoom()
                self._pan_x0 = e.position().x()
        hit = self._hit_burst(self._hover)
        self.setToolTip("{0}\n{1}".format(hit.get("label", ""), hit.get("detail", ""))
                        if hit else "")
        self.update()

    def mouseReleaseEvent(self, e):
        self._pan_x0 = None

    def leaveEvent(self, e):
        self._hover = QPoint(-1, -1)
        self._pan_x0 = None
        self.update()

    def _hit_burst(self, pt):
        r = self._wf_rect()
        for b in reversed(list(self._bursts)):
            rect = self._burst_rect(b, r)
            if rect is not None and rect.contains(pt.x(), pt.y()):
                return b
        return None

    def focus_burst(self, b):
        self._focus = b
        self._focus_ts = time.time()
        self.update()


class WaterfallWidget(QWidget):
    """Composite: display controls + spectrogram canvas + burst history table."""

    burst_selected = pyqtSignal(dict)
    burst_finalized = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        # ---------------------------------------------------------- controls
        ctl = QHBoxLayout()
        ctl.setSpacing(6)
        self.ui_palette = QComboBox()
        self.ui_palette.addItems(list(LUTS.keys()))
        self.ui_palette.setToolTip("Waterfall color palette")
        ctl.addWidget(QLabel("Palette:"), 0)
        ctl.addWidget(self.ui_palette, 0)

        self.ui_floor = QDoubleSpinBox()
        self.ui_floor.setRange(-160.0, 0.0)
        self.ui_floor.setValue(-95.0)
        self.ui_floor.setSuffix(" dB")
        self.ui_floor.setToolTip("Display lower bound (dB)")
        self.ui_range = QDoubleSpinBox()
        self.ui_range.setRange(5.0, 140.0)
        self.ui_range.setValue(70.0)
        self.ui_range.setSuffix(" dB")
        self.ui_range.setToolTip("Display range above the floor (dB)")
        self.ui_auto = QCheckBox("Auto")
        self.ui_auto.setChecked(True)
        self.ui_auto.setToolTip("Track the noise floor automatically")
        self.ui_reset_zoom = QPushButton("Reset zoom")
        self.ui_reset_zoom.setToolTip("Restore the full span (or double-click the waterfall)")
        for wdg in (QLabel("Floor:"), self.ui_floor, QLabel("Range:"),
                    self.ui_range, self.ui_auto, self.ui_reset_zoom):
            ctl.addWidget(wdg, 0)
        ctl.addStretch(1)
        lay.addLayout(ctl)

        # ------------------------------------------------------------ canvas
        self.canvas = _Canvas()
        self.canvas.burst_selected.connect(self.burst_selected)
        self.canvas.burst_finalized.connect(self.burst_finalized)
        lay.addWidget(self.canvas, 1)

        self.ui_palette.currentTextChanged.connect(self._on_palette)
        self.ui_auto.toggled.connect(self._on_auto)
        self.ui_floor.valueChanged.connect(self._on_manual_scale)
        self.ui_range.valueChanged.connect(self._on_manual_scale)
        self.ui_reset_zoom.clicked.connect(self._reset_zoom)

    # ------------------------------------------------------ delegated API
    # Everything the controller touches lives here and forwards to canvas.

    @property
    def center_freq(self):
        return self.canvas.center_freq

    @center_freq.setter
    def center_freq(self, v):
        self.canvas.center_freq = v

    @property
    def sample_rate(self):
        return self.canvas.sample_rate

    @sample_rate.setter
    def sample_rate(self, v):
        self.canvas.sample_rate = v

    def set_stream(self, freq_hz: float, sample_rate: float):
        self.canvas.center_freq = float(freq_hz)
        self.canvas.sample_rate = max(float(sample_rate), 1.0)
        self.canvas.update()

    def reset(self):
        self.canvas.reset()

    def on_iq(self, iq, freq_hz=None):
        self.canvas.on_iq(iq)

    def extract_burst_iq(self, burst):
        return self.canvas.extract_burst_iq(burst)

    @property
    def _bursts(self):
        return self.canvas._bursts

    # ------------------------------------------------------------ slots

    def _on_palette(self, name):
        self.canvas.palette_name = name
        self.canvas._render()

    def _on_auto(self, on):
        self.canvas.auto_scale = bool(on)
        self.ui_floor.setEnabled(not on)
        self.ui_range.setEnabled(not on)
        if not on:
            self.ui_floor.setValue(self.canvas.db_floor)
            self.ui_range.setValue(self.canvas.db_range)
        self.canvas._render()

    def _on_manual_scale(self):
        self.canvas.db_floor = self.ui_floor.value()
        self.canvas.db_range = self.ui_range.value()
        self.canvas._render()

    def _sync_spinners(self):
        self.ui_floor.blockSignals(True)
        self.ui_range.blockSignals(True)
        self.ui_floor.setValue(self.canvas.db_floor)
        self.ui_range.setValue(self.canvas.db_range)
        self.ui_floor.blockSignals(False)
        self.ui_range.blockSignals(False)

    def _reset_zoom(self):
        self.canvas.zx0, self.canvas.zx1 = 0.0, 1.0
        self.canvas.update()
