import os
import glob
import time
import threading

import cv2
import numpy as np

from ..utils.exceptions import CameraError, GPIOError


# BASLER CAMERA
class BaslerCamera:
    """Basler pypylon InstantCamera source. Used when CAMERA='camera' (USE_CAMERA=true)."""

    def __init__(self, serial: str = "", exposure_us: int = 8000,
                 retry_delay: float = 0.2, retries: int = 2,
                 warmup_frames: int = 5):
        self._serial      = serial
        self._exposure_us = exposure_us
        self._retry_delay = retry_delay
        self._retries     = retries
        self._warmup_frames = warmup_frames

        self._camera = None
        try:
            from pypylon import pylon
            self._pylon = pylon
        except ImportError:
            raise CameraError("pypylon not installed — cannot use CAMERA='camera'")

    def open(self):
        try:
            pylon = self._pylon
            tl    = pylon.TlFactory.GetInstance()
            devs  = tl.EnumerateDevices()
            device = None
            for d in devs:
                if not self._serial or d.GetSerialNumber() == self._serial:
                    device = tl.CreateDevice(d)
                    break
            if device is None:
                raise CameraError(
                    f"Basler camera not found (serial='{self._serial}')")
            self._camera = pylon.InstantCamera(device)
            self._camera.Open()
            self._camera.ExposureAuto.SetValue("Off")
            try:
                self._camera.ExposureTimeAbs.SetValue(float(self._exposure_us))
            except Exception:
                self._camera.ExposureTime.SetValue(float(self._exposure_us))
            self._camera.PixelFormat.SetValue("Mono8")
            self._camera.TriggerSelector.SetValue("FrameStart")
            self._camera.TriggerMode.SetValue("On")
            self._camera.TriggerSource.SetValue("Software")
            self._camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            print(f"[Camera] Opened (software trigger). Exposure={self._exposure_us} µs")
        except CameraError:
            raise
        except Exception as e:
            raise CameraError(f"Camera open failed: {e}")

    def grab(self) -> np.ndarray:
        """Return BGR ndarray or raise CameraError."""
        for attempt in range(self._retries + 1):
            try:
                img = self._grab_basler()
                if img is not None:
                    return img
            except CameraError:
                raise
            except Exception as e:
                if attempt < self._retries:
                    time.sleep(self._retry_delay)
                else:
                    raise CameraError(f"Grab failed after retries: {e}")
        raise CameraError("Grab returned None after retries")

    def _grab_basler(self) -> np.ndarray:
        if self._camera is None:
            raise CameraError("Camera not open")
        self._camera.ExecuteSoftwareTrigger()
        result = self._camera.RetrieveResult(
            5000, self._pylon.TimeoutHandling_ThrowException)
        try:
            if result.GrabSucceeded():
                gray = result.GetArray().copy()
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            raise CameraError(f"Grab failed: code {result.ErrorCode}")
        finally:
            result.Release()

    def grab_first(self) -> np.ndarray:
        return self.grab()

    def warmup(self):
        for _ in range(self._warmup_frames):
            try:
                self._grab_basler()
            except Exception:
                print(f"[Camera] Warmup failed")
        print(f"[Camera] Warmup done ({self._warmup_frames} frames).")

    def set_exposure(self, us: int):
        self._exposure_us = int(us)
        if self._camera and self._camera.IsOpen():
            try:
                self._camera.ExposureTimeAbs.SetValue(float(us))
            except Exception as e:
                print(f"[Camera] Exposure set error: {e}")

    def close(self):
        if self._camera:
            try:
                self._camera.StopGrabbing()
                self._camera.Close()
            except Exception:
                pass
            self._camera = None

    def is_open(self) -> bool:
        return self._camera is not None and self._camera.IsOpen()

    def has_more(self) -> bool:
        return True

    def reset(self):
        pass


# CELL-CON
class CellCon:
    """
    Serial interface to the Cell-con lot tracker.
    Protocol: send 'LA\\r\\n' → receive 'LS,<lot_number>[,…]'
    get_lot() returns lot string or '' on any error/timeout.
    """
    BAUD    = 38400
    TIMEOUT = 1.0
    RETRIES = 5

    def __init__(self, port: str = "/dev/ttyUSB0"):
        self._port = port

    def get_lot(self) -> str:
        try:
            import serial as _serial
            with _serial.Serial(
                    port=self._port, baudrate=self.BAUD,
                    parity=_serial.PARITY_NONE,
                    stopbits=_serial.STOPBITS_ONE,
                    bytesize=_serial.EIGHTBITS,
                    timeout=self.TIMEOUT) as ser:
                ser.write(b"LA\r\n")
                for _ in range(self.RETRIES):
                    line = ser.readline().decode("utf-8", errors="ignore").strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if parts[0] == "LS" and len(parts) >= 2:
                        lot = parts[1].strip()
                        print(f"[CellCon] Lot received: {lot}")
                        return lot
        except Exception as e:
            print(f"[CellCon] Error: {e} — check USB at {self._port}")
        return ""


# PORT DETECTION
def _detect_ports(usb_id_hint: str = "") -> dict:
    """
    Returns {"lighting": path|None, "cellcon": path|None}.
    Lighting: resolved from /dev/serial/by-id/ by matching usb_id_hint.
    CellCon:  LA\\r\\n probe on remaining ttyUSB ports (3 retries, 0.5 s each).
    """
    result = {"lighting": None, "cellcon": None}

    # Lighting: claimed first by hardware identity
    if usb_id_hint:
        for link in glob.glob("/dev/serial/by-id/*"):
            if usb_id_hint in os.path.basename(link):
                result["lighting"] = os.path.realpath(link)
                break

    # CellCon: probe remaining ports
    candidates = sorted(p for p in glob.glob("/dev/ttyUSB*")
                        if p != result["lighting"])
    for port in candidates:
        try:
            import serial as _serial
            with _serial.Serial(port, 38400,
                                parity=_serial.PARITY_NONE,
                                stopbits=_serial.STOPBITS_ONE,
                                bytesize=_serial.EIGHTBITS,
                                timeout=0.5) as s:
                s.write(b"LA\r\n")
                for _ in range(3):
                    line = s.readline().decode("utf-8", errors="ignore").strip()
                    if line.startswith("LS"):
                        result["cellcon"] = port
                        break
        except Exception:
            pass
        if result["cellcon"]:
            break

    light_str = result["lighting"] or "NOT FOUND"
    cell_str  = result["cellcon"]  or "NOT FOUND"
    print(f"[Ports] Lighting → {light_str}")
    print(f"[Ports] CellCon  → {cell_str}")
    return result


# LIGHTING CONTROLLER
class LightingController:
    """Serial ring-light controller (RS232 over USB-Prolific, IFWFOCR01 protocol)."""
    BAUD = 38400

    def __init__(self, enabled: bool, port: str, value: int = 100):
        self._enabled       = enabled
        self._ser           = None
        self._controller_ok = False
        if not enabled:
            print("[Lighting] Disabled.")
            return
        try:
            import serial as _serial
            self._ser = _serial.Serial(
                port=port, baudrate=self.BAUD,
                parity=_serial.PARITY_NONE,
                stopbits=_serial.STOPBITS_ONE,
                bytesize=_serial.EIGHTBITS,
                timeout=1)
            # Probe: send off command and read ACK to confirm controller is alive
            self._ser.reset_input_buffer()
            self._ser.write(b"@00L0007C\r\n")
            resp = self._ser.read(32)
            self._controller_ok = bool(resp)
            if self._controller_ok:
                print(f"[Lighting] Port {port} OK — controller responding.")
            else:
                print(f"[Lighting] Port {port} open but no response from controller ⚠")
        except Exception as e:
            print(f"[Lighting] Port error: {e}")
            self._enabled = False

    @property
    def controller_ok(self) -> bool:
        return self._controller_ok

    def probe(self, timeout_s: float = 0.2) -> bool:
        """Send off command and read ACK. Returns True if controller responds."""
        if not self._enabled or self._ser is None:
            return False
        try:
            prev_timeout = self._ser.timeout
            self._ser.timeout = timeout_s
            self._ser.reset_input_buffer()
            self._ser.write(b"@00L0007C\r\n")
            resp = self._ser.read(32)
            self._ser.timeout = prev_timeout
            self._controller_ok = bool(resp)
        except Exception:
            self._controller_ok = False
        return self._controller_ok

    def _send(self, data: bytes):
        if not self._enabled or self._ser is None:
            return
        self._ser.write(data)

    @staticmethod
    def _brightness_cmd(value: int) -> bytes:
        value = max(0, min(255, value))
        body  = f"@00F{value:03}00"
        chk   = sum(body.encode("ascii")) & 0xFF
        return (body + f"{chk:02X}\r\n").encode("ascii")

    def set_brightness(self, value: int):
        self._send(self._brightness_cmd(value))

    def on(self):
        self._send(b"@00L1007D\r\n")

    def off(self):
        self._send(b"@00L0007C\r\n")

    def close(self):
        if self._ser:
            try:
                self._ser.close()
            except Exception:
                pass


# RASPBERRY IO
class RaspberryIO:
    """
    BCM-mode GPIO handler.
    Falls back to mock logging when IO=False or RPi.GPIO unavailable.

    Pins
    ----
    START_PIN (IN, active HIGH 10 ms pulse) — machine signals ready for one shot
    BUSY_PIN  (OUT, HIGH during full inspection + retry)
    END_PIN   (OUT, normally HIGH; pulses LOW 40 ms after inspection done)
    INSPEC_STAGE (OUT, normally HIGH; LOW = both ICs pass, HIGH = any fail)

    Mock trigger
    ------------
    In mock mode wait_for_start() blocks until trigger() is called from the UI.
    """

    _END_PIN_PULSE_SEC = 0.040 #ENDING SIGNAL PULSE duration (40 ms LOW)
    _GPIO_PRE_END_SEC = 0.010  #PIN STAGE PULSE, DEBOUNCE GAPS

    def __init__(self, io_enabled: bool = True,
                 start_pin: int = 17, busy_pin: int = 23,
                 end_pin: int = 18, inspec_stage_pin: int = 24):
        self._gpio_ok          = False
        self._GPIO             = None
        self._start_pin        = start_pin
        self._busy_pin         = busy_pin
        self._end_pin          = end_pin
        self._inspec_stage_pin = inspec_stage_pin
        self._mock_trigger     = threading.Event()

        if not io_enabled:
            print("[IO] IO=False — mock mode (manual trigger).")
            return

        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._start_pin,        GPIO.IN,  pull_up_down=GPIO.PUD_DOWN)  # active HIGH
            GPIO.setup(self._busy_pin,         GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self._end_pin,          GPIO.OUT, initial=GPIO.HIGH)            # idle HIGH
            GPIO.setup(self._inspec_stage_pin, GPIO.OUT, initial=GPIO.HIGH)            # idle HIGH
            self._gpio_ok = True
            print("[IO] GPIO initialised (BCM mode).")
        except Exception as e:
            raise GPIOError(f"GPIO init failed: {e}")

    def is_initialised(self) -> bool:
        return self._gpio_ok

    def _out(self, pin: int, high: bool, pin_name: str = ""):
        if self._gpio_ok:
            self._GPIO.output(pin, self._GPIO.HIGH if high else self._GPIO.LOW)
        else:
            print(f"[IO MOCK] {pin_name or pin} → {'HIGH' if high else 'LOW'}")

    # ── outputs ────────────────────────────────────────────────────────────────

    def set_busy(self, v: bool):
        self._out(self._busy_pin, v, "BUSY_PIN")

    def set_inspec_stage(self, high: bool):
        """HIGH = NG / idle; LOW = both ICs pass."""
        self._out(self._inspec_stage_pin, high, "INSPEC_STAGE")

    def pulse_end_pin(self):
        """Pulse END_PIN LOW for 40 ms. Blocking — call from worker thread only."""
        self._out(self._end_pin, False, "END_PIN")
        time.sleep(self._END_PIN_PULSE_SEC)
        self._out(self._end_pin, True, "END_PIN")

    def clear_outputs(self):
        """Restore all outputs to idle state."""
        self._out(self._busy_pin,         False, "BUSY_PIN")          # LOW
        self._out(self._inspec_stage_pin, True,  "INSPEC_STAGE")      # HIGH (idle)
        self._out(self._end_pin,          True,  "END_PIN")            # HIGH (idle)

    # ── inputs / blocking waits ─────────────────────────────────────────────────

    def trigger(self):
        """Inject a mock START pulse (mock mode only). Called from UI thread."""
        if not self._gpio_ok:
            self._mock_trigger.set()

    def wait_for_start(self, stop_flag_fn, timeout_s: float = 0.0) -> bool | None:
        """Block until START_PIN RISING edge or stop_flag_fn() returns True.
        In mock mode, blocks until trigger() is called from the UI.

        Returns True=started, False=stopped, None=timed out (real GPIO only).
        """
        if not self._gpio_ok:
            while not stop_flag_fn():
                if self._mock_trigger.wait(timeout=0.02):
                    self._mock_trigger.clear()
                    print("[IO MOCK] START_PIN HIGH pulse (manual trigger)")
                    return True
            return False
        GPIO = self._GPIO
        deadline = (time.monotonic() + timeout_s) if timeout_s > 0 else None
        while not stop_flag_fn():
            if GPIO.wait_for_edge(self._start_pin, GPIO.RISING, timeout=20) is not None:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return None
        return False

    def drain_start_pin(self, timeout_ms: int = 500):
        """Discard a stale START_PIN HIGH after resume (wait until idle LOW)."""
        if not self._gpio_ok:
            self._mock_trigger.clear()
            print("[IO MOCK] drain_start_pin (mock trigger cleared)")
            return
        GPIO = self._GPIO
        if GPIO.input(self._start_pin) == GPIO.HIGH:
            GPIO.wait_for_edge(self._start_pin, GPIO.FALLING, timeout=timeout_ms)

    def cleanup(self):
        if self._gpio_ok:
            try:
                self._GPIO.cleanup()
            except Exception:
                pass
