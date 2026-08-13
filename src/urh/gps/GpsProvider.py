import threading
import time
from datetime import datetime

from urh.util.Logger import logger

try:
    import serial
except ImportError:  # pragma: no cover
    serial = None


USB_PORT_PATTERNS = (
    "/dev/cu.usbserial-*",
    "/dev/cu.usbmodem*",
    "/dev/cu.PL2303*",
    "/dev/cu.ESP32*",
    "/dev/cu.SLAB_USBtoUART*",
    "/dev/cu.wchusbserial*",
    "/dev/tty.usbserial-*",
    "/dev/tty.usbmodem*",
)

GPS_SOURCE_AUTO = "auto"
GPS_SOURCE_USB = "usb"
GPS_SOURCE_CORELOCATION = "corelocation"
GPS_SOURCE_OFF = "off"


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


def detect_usb_ports():
    """Return all serial ports that look like USB GNSS receivers (sorted, dedup)."""
    if serial is None:
        return []
    import glob

    ports = []
    for pattern in USB_PORT_PATTERNS:
        for match in glob.glob(pattern):
            if match not in ports:
                ports.append(match)
    return sorted(ports)


def core_location_available() -> bool:
    try:
        import CoreLocation  # noqa: F401
        import Foundation  # noqa: F401

        return True
    except ImportError:
        return False


class GpsProvider(object):
    """Base class for GPS position providers."""

    def __init__(self):
        self.position = None  # (lat, lon, alt)
        self.fix_quality = 0
        self.num_satellites = 0
        self.hdop = 0.0
        self.last_update = 0.0
        self.error = None
        self.status_message = None
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
        ports = detect_usb_ports()
        return ports[0] if ports else None

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
                self.error = None
                self.status_message = None
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

    Uses CoreLocation via PyObjC. Accuracy is Wi-Fi/network based (~100 m - 1 km).
    Requires location permission; a bare Python process may need the parent app
    (Terminal/iTerm) to have location access under System Settings > Privacy &
    Security > Location Services.
    """

    AUTHORIZATION_LABELS = {
        0: "not determined",
        1: "restricted",
        2: "denied",
        3: "authorized always",
        4: "authorized in use",
    }

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self._manager = None
        self._delegate = None
        self.authorization_status = 0

    def _set_error_from_code(self, code):
        # kCLErrorLocationUnknown (0) is transient: keep waiting.
        if code == 0:
            self.status_message = "Waiting for location fix..."
            return
        if code == 1:
            self.error = (
                "Location access denied. Enable it in System Settings > "
                "Privacy & Security > Location Services."
            )
            return
        if code == 2:
            self.error = "Location unavailable: no network."
            return
        self.status_message = "Location error (code {0})".format(code)

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
                provider.error = None
                provider.status_message = None

            def locationManager_didFailWithError_(self, manager, error):
                try:
                    code = int(error.code())
                except Exception:
                    code = -1
                provider._set_error_from_code(code)
                logger.warning(
                    "CoreLocation error code {0}: {1}".format(
                        code, error.localizedDescription()
                    )
                )

            def locationManagerDidChangeAuthorization_(self, manager):
                status = int(manager.authorizationStatus())
                provider.authorization_status = status
                logger.info(
                    "CoreLocation authorization status: {0} ({1})".format(
                        status, provider.AUTHORIZATION_LABELS.get(status, "?")
                    )
                )
                if status == 0 and hasattr(manager, "requestWhenInUseAuthorization"):
                    manager.requestWhenInUseAuthorization()

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


def create_gps_provider(source=GPS_SOURCE_AUTO, port: str = None, parent=None):
    """Create and start a GPS provider.

    :param source: "auto" (USB preferred, CoreLocation fallback), "usb", "corelocation"
    :param port: explicit serial port for USB providers (optional)
    :return: a started GpsProvider or None if none could be started
    """
    if source == GPS_SOURCE_CORELOCATION:
        return _start_core_location()

    ports = [port] if port else detect_usb_ports()
    if ports:
        chosen = port or ports[0]
        usb = NmeaGpsProvider(port=chosen, parent=parent)
        try:
            usb.open_serial()
            usb.start()
            logger.info("Using USB GNSS on {0}".format(chosen))
            return usb
        except Exception as e:
            logger.warning("USB GNSS on {0} not usable: {1}".format(chosen, e))
            if source == GPS_SOURCE_USB:
                return None

    if source == GPS_SOURCE_USB:
        return None
    return _start_core_location()


def _start_core_location(parent=None):
    if not core_location_available():
        logger.warning("CoreLocation not available")
        return None
    try:
        cl = CoreLocationGpsProvider(parent=parent)
        cl.start()
        if cl.error:
            logger.warning("CoreLocation GPS not usable: {0}".format(cl.error))
            return cl  # still return it so the UI can show the reason
        return cl
    except Exception as e:
        logger.warning("CoreLocation GPS init failed: {0}".format(e))
        return None
