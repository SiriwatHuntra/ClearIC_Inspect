import threading
import time

from ..utils.exceptions import GPIOError


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
            GPIO.setup(self._start_pin,        GPIO.IN,  pull_up_down=GPIO.PUD_DOWN)
            GPIO.setup(self._busy_pin,         GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self._end_pin,          GPIO.OUT, initial=GPIO.HIGH)
            GPIO.setup(self._inspec_stage_pin, GPIO.OUT, initial=GPIO.HIGH)
            self._gpio_ok = True
            print("[IO] GPIO initialised (BCM mode).")
        except Exception as e:
            raise GPIOError(f"GPIO init failed: {e}")

    def _out(self, pin: int, high: bool, pin_name: str = ""):
        if self._gpio_ok:
            self._GPIO.output(pin, self._GPIO.HIGH if high else self._GPIO.LOW)
        else:
            print(f"[IO MOCK] {pin_name or pin} → {'HIGH' if high else 'LOW'}")

    def set_busy(self, v: bool):
        self._out(self._busy_pin, v, "BUSY_PIN")

    def set_inspec_stage(self, high: bool):
        """HIGH = NG / idle; LOW = both ICs pass."""
        self._out(self._inspec_stage_pin, high, "INSPEC_STAGE")

    def pulse_end_pin(self):
        """Pulse END_PIN LOW for 40 ms. Blocking — call from worker thread only."""
        self._out(self._end_pin, False, "END_PIN")
        time.sleep(0.040)
        self._out(self._end_pin, True, "END_PIN")

    def clear_outputs(self):
        """Restore all outputs to idle state."""
        self._out(self._busy_pin,         False, "BUSY_PIN")
        self._out(self._inspec_stage_pin, True,  "INSPEC_STAGE")
        self._out(self._end_pin,          True,  "END_PIN")

    def trigger(self):
        """Inject a mock START pulse (mock mode only). Called from UI thread."""
        if not self._gpio_ok:
            self._mock_trigger.set()

    def wait_for_start(self, stop_flag_fn) -> bool:
        """Block until START_PIN RISING edge or stop_flag_fn() returns True.
        In mock mode, blocks until trigger() is called from the UI."""
        if not self._gpio_ok:
            while not stop_flag_fn():
                if self._mock_trigger.wait(timeout=0.02):
                    self._mock_trigger.clear()
                    print("[IO MOCK] START_PIN HIGH pulse (manual trigger)")
                    return True
            return False
        GPIO = self._GPIO
        while not stop_flag_fn():
            if GPIO.wait_for_edge(self._start_pin, GPIO.RISING, timeout=20) is not None:
                return True
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
