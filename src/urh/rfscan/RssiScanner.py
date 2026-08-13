import threading
import time
from collections import deque

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from urh.dev.BackendHandler import BackendHandler
from urh.dev.VirtualDevice import Mode, VirtualDevice
from urh.util.Logger import logger


class RssiScanner(QObject):
    """Continuously measures the received signal power (RSSI) on one frequency.

    Keeps the SDR open and computes the average power of each new IQ chunk,
    so relative RSSI measurements stay consistent within a session.
    """

    rssi_updated = pyqtSignal(float, float)  # rssi_db, timestamp
    device_error = pyqtSignal(str)
    state_changed = pyqtSignal(bool)  # running

    def __init__(self, backend_handler: BackendHandler = None, parent=None):
        super().__init__(parent)
        self.backend_handler = backend_handler or BackendHandler()
        self.device = None
        self._read_thread = None
        self._running = False
        self._old_index = 0
        self.frequency = None
        self.sample_rate = None
        self.gain = None
        self.last_rssi = -200.0
        self._rssi_history = deque(maxlen=400)
        self._history_lock = threading.Lock()

    def clear_history(self):
        with self._history_lock:
            self._rssi_history.clear()

    def average_rssi(self, window: float = 1.0) -> float:
        """Average RSSI over the last `window` seconds (or last value if too few)."""
        with self._history_lock:
            if not self._rssi_history:
                return self.last_rssi
            now = time.time()
            recent = [r for ts, r in self._rssi_history if now - ts <= window]
            if not recent:
                return self.last_rssi
            return float(np.mean(recent))

    @property
    def is_running(self) -> bool:
        return self._running

    def set_frequency(self, frequency: float):
        self.frequency = frequency
        if self.device is not None:
            try:
                self.device.frequency = frequency
                logger.info("RssiScanner: retuned to {0} Hz".format(frequency))
            except Exception as e:
                logger.error("RssiScanner: retune failed: {0}".format(e))
                self.device_error.emit(str(e))

    def start(self, frequency: float, sample_rate: int, gain: int):
        if self._running:
            return
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.gain = gain
        self._old_index = 0

        self.device = VirtualDevice(
            self.backend_handler,
            "RTL-SDR",
            Mode.receive,
            freq=frequency,
            sample_rate=sample_rate,
            gain=gain,
            resume_on_full_receive_buffer=True,
            raw_mode=True,
        )
        self.device.start()
        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        self.state_changed.emit(True)
        logger.info(
            "RssiScanner: started at {0} Hz, {1} Sps, gain {2}".format(
                frequency, sample_rate, gain
            )
        )

    def stop(self):
        if not self._running:
            return
        self._running = False
        try:
            if self.device is not None:
                self.device.stop("Stopping RSSI scanner")
        except Exception as e:
            logger.error("RssiScanner: stop error: {0}".format(e))
        if self._read_thread is not None and self._read_thread.is_alive():
            self._read_thread.join(2.0)
        self._read_thread = None
        self.state_changed.emit(False)
        logger.info("RssiScanner: stopped")

    def _read_loop(self):
        while self._running:
            try:
                data = self.device.data if self.device is not None else None
                if data is None:
                    time.sleep(0.05)
                    continue
                current = self.device.current_index
                if current == self._old_index:
                    time.sleep(0.05)
                    continue
                if current > self._old_index:
                    chunk = data[self._old_index : current]
                else:
                    chunk = np.concatenate((data[self._old_index:], data[:current]))
                self._old_index = current

                if len(chunk) > 0:
                    rssi = self._compute_rssi(chunk)
                    with self._history_lock:
                        self.last_rssi = rssi
                        self._rssi_history.append((time.time(), rssi))
                    self.rssi_updated.emit(rssi, time.time())
            except Exception as e:
                logger.error("RssiScanner read error: {0}".format(e))
                self.device_error.emit(str(e))
                time.sleep(0.5)

    @staticmethod
    def _compute_rssi(iq) -> float:
        power = iq.real.astype(np.float64) ** 2 + iq.imag.astype(np.float64) ** 2
        mean_power = float(np.mean(power))
        if mean_power <= 0:
            return -200.0
        return 10.0 * np.log10(mean_power)
