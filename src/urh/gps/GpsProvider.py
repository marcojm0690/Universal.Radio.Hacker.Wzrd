import threading
import time
from datetime import datetime

from urh.util.Logger import logger

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None


def _parse_lat_lon(value: str, hemi: str) -> float:
    """Convert NMEA DDMM.MMMM (+ hemisphere) to decimal degrees."""
    try:
        degrees = int(float(value) // 100)
        minutes = float(value) - degrees * 100
        result = degrees + minutes / 60.0
        if hemi in ("S", "W"):
            result = -result
        return result
    except (ValueError, TypeError):
        return 0.0


class GpsProvider(object):
    """Base class for GPS position providers."""

    def __init__(self):
        self.position = None  # (lat, lon, alt)
        self.fix_quality = 0
        self.num_satellites = 0
        self.hdop = 0.0
        self.last_update = 0.0
        self.error = None
        self.running = False

    def start(self):
        raise NotImplementedError

    def stop(self):
        self.running = False

    def has_fix(self) -> bool:
        return self.position is not None and self.fix_quality > 0


class NmeaGpsProvider(GpsProvider, threading.Thread):
    """Reads NMEA sentences (GGA/RMC) from a USB GNSS over a serial port."""

    BAUDRATES = (9600, 38400, 115200)

    def __init__(self, port: str = None, parent=None):
        GpsProvider.__init__(self)
        threading.Thread.__init__(self, daemon=True)
        self.port = port or self.detect_port()
        self.parent = parent
        self._serial = None

    @staticmethod
    def detect_port():
        if serial is None:
            return None
        import glob

        candidates = [
            "/dev/cu.usbserial-*",
            "/dev/cu.usbmodem*",
            "/dev/cu.PL2303*",
            "/dev/cu.ESP32*",
            "/dev/tty.usbserial-*",
            "/dev/tty.usbmodem*",
        ]
        for pattern in candidates:
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[0]
        return None

    def open_serial(self):
        if serial is None:
            raise RuntimeError("pyserial is not installed")
        if self.port is None:
            raise RuntimeError("No USB GNSS device found")
        last_error = None
        for baud in self.BAUDRATES:
            try:
                ser = serial.Serial(self.port, baud, timeout=0.5)
                ser.flushInput()
                self._serial = ser
                logger.info("NMEA GPS: opened {0} @ {1} baud".format(self.port, baud))
                return
            except (OSError, serial.SerialException) as e:
                last_error = e
        raise RuntimeError(str(last_error))

    def run(self):
        try:
            self.open_serial()
        except Exception as e:
            self.error = str(e)
            logger.error("NMEA GPS failed: {0}".format(e))
            return
        self.running = True
        buffer = ""
        while self.running:
            try:
                data = self._serial.read(256)
                if not data:
                    continue
                buffer += data.decode("ascii", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line.startswith("$") or "*" not in line:
                        continue
                    self.parse_line(line)
            except (OSError, serial.SerialException) as e:
                self.error = str(e)
                logger.error("NMEA GPS read error: {0}".format(e))
                time.sleep(1.0)

    def parse_line(self, line: str):
        try:
            fields = line.split(",")
            if fields[0].endswith("GGA"):
                # GGA,utc,lat,lat_h,lon,lon_h,fix,sats,hdop,alt,M,...
                if len(fields) < 10:
                    return
                fix = int(float(fields[6]))
                if fix <= 0:
                    return
                lat = _parse_lat_lon(fields[2], fields[3])
                lon = _parse_lat_lon(fields[4], fields[5])
                alt = float(fields[9]) if fields[9] else 0.0
                self.position = (lat, lon, alt)
                self.fix_quality = fix
                self.num_satellites = int(float(fields[7])) if fields[7] else 0
                self.hdop = float(fields[8]) if fields[8] else 0.0
                self.last_update = time.time()
            elif fields[0].endswith("RMC"):
                # RMC,utc,status,lat,lat_h,lon,lon_h,speed,course,date,...
                if len(fields) < 7 or fields[2] not in ("A", "V"):
                    return
                if fields[2] == "V":
                    return
                lat = _parse_lat_lon(fields[3], fields[4])
                lon = _parse_lat_lon(fields[5], fields[6])
                alt = self.position[2] if self.position else 0.0
                self.position = (lat, lon, alt)
                self.fix_quality = self.fix_quality or 1
                self.last_update = time.time()
        except (ValueError, IndexError) as e:
            logger.debug("NMEA parse error: {0}".format(e))

    def stop(self):
        self.running = False
        try:
            if self._serial is not None:
                self._serial.close()
        except OSError:
            pass
        super().stop()


class CoreLocationGpsProvider(GpsProvider):
    """macOS Location Services fallback (no GPS hardware needed).

    Uses CoreLocation via PyObjC. Accuracy is Wi-Fi/network based (~100 m - 1 km),
    but it works on any Mac. The OS may show a location authorization prompt.
    """

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self._manager = None
        self._delegate = None

    def start(self):
        try:
            import CoreLocation
            from Foundation import NSObject
        except ImportError as e:
            self.error = "PyObjC/CoreLocation not available: {0}".format(e)
            logger.error(self.error)
            return

        provider = self

        class _Delegate(NSObject):
            def locationManager_didUpdateLocations_(self, manager, locations):
                loc = locations.lastObject()
                if loc is None:
                    return
                coord = loc.coordinate()
                provider.position = (
                    coord.latitude,
                    coord.longitude,
                    loc.altitude(),
                )
                provider.fix_quality = 1
                provider.num_satellites = 0
                provider.hdop = 0.0
                provider.last_update = time.time()

            def locationManager_didFailWithError_(self, manager, error):
                provider.error = str(error.localizedDescription())
                logger.warning("CoreLocation error: {0}".format(provider.error))

            def locationManagerDidChangeAuthorization_(self, manager):
                status = manager.authorizationStatus()
                logger.info("CoreLocation authorization status: {0}".format(status))

        self._delegate = _Delegate.alloc().init()
        self._manager = CoreLocation.CLLocationManager.alloc().init()
        self._manager.setDelegate_(self._delegate)
        self._manager.setDesiredAccuracy_(
            CoreLocation.kCLLocationAccuracyNearestTenMeters
        )
        self._manager.requestWhenInUseAuthorization()
        if hasattr(self._manager, "requestLocation"):
            self._manager.requestLocation()
        self._manager.startUpdatingLocation()
        self.running = True
        logger.info("CoreLocation GPS started")

    def stop(self):
        try:
            if self._manager is not None:
                self._manager.stopUpdatingLocation()
        except Exception:
            pass
        self.running = False
        super().stop()


def create_gps_provider(require_usb: bool = True):
    """Return the best available GPS provider (USB GNSS preferred).

    With require_usb=False, falls back to the CoreLocation provider. Returns
    None if no provider could be started.
    """
    usb = NmeaGpsProvider()
    if usb.port is not None:
        try:
            usb.open_serial()
            usb.start()
            logger.info("Using USB GNSS on {0}".format(usb.port))
            return usb
        except Exception as e:
            logger.warning(
                "USB GNSS not usable ({0}), falling back to CoreLocation".format(e)
            )
    if require_usb:
        return None
    try:
        cl = CoreLocationGpsProvider()
        cl.start()
        if cl.error:
            logger.warning("CoreLocation GPS not usable: {0}".format(cl.error))
            return None
        return cl
    except Exception as e:
        logger.warning("CoreLocation GPS init failed: {0}".format(e))
        return None
