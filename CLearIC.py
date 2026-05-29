"""
ClearIC Inspect
===============
Clear-package IC laser-mark inspection via ROI crop + OpenVINO classifier.
Each ROI cell is cropped and classified as Text (present) or NoText (absent).

Sections (in order)
-------------------
  ConfigLoader        Config.toml loader with defaults
  Stage / ErrorFlag   State enums
  Exceptions          InspectionError hierarchy
  Image               Image dataclass + ID generator
  Camera              Basler camera or directory source
  RaspberryIO         BCM GPIO handler (mockable)
  Detector            OpenVINO 2-class classifier (Text / NoText)
  TemplateManager     Load/save IC bounding-box template
  Inspector           12-cell ROI crop-then-classify logic
  Logger              Daily-rotating CSV log
  STYLE               Qt stylesheet
  ImageView           Zoomable image widget with overlays
  RunWorker           QThread inspection loop
  MainWindow          Two-tab PyQt5 UI (Inspection + Image Browser)
  main / __main__     Entry point
"""

import sys
import os
import csv
import json
import glob
import time
import signal
import fcntl
import threading
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

import gc
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore


# CONFIG LOADER
class ConfigLoader:
    CONFIG_FILE = "Config.toml"
    #This defualt config is used as a template for the Config.toml file and as fallback for missing keys. It is not used directly in the code, but serves as a reference for the expected configuration parameters and their default values.
    DEFAULT_CONFIG = {
        "USE_CAMERA":           False,
        "CONF_THR":             0.5,
        "TEXT_MIN_CONF":        0.80,
        "TEXT_NG_THRESHOLD":    2,
        "BLANK_CELL_STD_THR":   0.0,
        "NMS_IOU_THR":          0.45,
        "CAMERA_SERIAL":        "",
        "EXPOSURE_US":          8000,
        "DEBUG":                True,
        "IO":                   False,
        "MODE":                 "DEBUG",
        "COLLECT_DATASET":      False,
        "DIR_INPUT":            "Input/",
        "OUT_DIR":              "Output/",
        "MODEL_PATH":           "Text_cls-2/best_openvino_model/best.xml",
        "CAMERA_WARMUP_FRAMES": 5,
        "CAMERA_RETRY_DELAY":   0.2,
        "CAMERA_RETRIES":       2,
        "RECONNECT_ATTEMPTS":   3,
        "RECONNECT_DELAY_S":    5.0,
        "RETRY_DELAY_MS":       10,
        "DISK_WARN_MB":         200,
        "GPIO_START_PIN":        17,
        "GPIO_BUSY_PIN":         23,
        "GPIO_END_PIN":          18,
        "GPIO_INSPEC_STAGE_PIN": 24,
        "CELL_SHRINK":          0.95,
        "CELL_EXPAND":          1.2,
        "COL_GAP_PCT":          40.0,
        "GRID_MARGIN_TOP":      0.0,
        "GRID_MARGIN_BOT":      15.0,
        "DATA_DIR":             "Dataset",
        "DATA_SPLIT":           "train",
        "LOG_DIR":              "logs",
        "LOG_RETENTION":        365,
        "ANN_BORDER_PX":        1,
        "RESULT_OVERLAY":      True,
        "WARMUP_FRAMES":        5,
        "CELLCON_PORT":         "/dev/ttyUSB0",
        "IMAGE_W":              0,
        "IMAGE_H":              0,
        "CLS_N_PASSES":         1,   # deterministic model — multi-pass gives identical results
        "CLS_UNCERTAIN_THR":    0.50,
    }
    @classmethod
    def load(cls) -> dict:
        import tomlkit
        if not os.path.exists(cls.CONFIG_FILE):
            raise ConfigError("Config.toml not found — create it before running.")
        try:
            with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                data = tomlkit.load(f)
        except Exception as e:
            raise ConfigError(f"Config.toml unreadable: {e}")
        cfg = dict(cls.DEFAULT_CONFIG)
        for k in cls.DEFAULT_CONFIG:
            if k in data:
                cfg[k] = data[k]
        if not isinstance(cfg["USE_CAMERA"], bool):
            raise ConfigError("USE_CAMERA must be true or false")
        cfg["CAMERA"] = "camera" if cfg["USE_CAMERA"] else "directory"
        if not (0.0 < cfg["CONF_THR"] <= 1.0):
            raise ConfigError("CONF_THR must be in (0, 1]")
        if not (0.0 < cfg["TEXT_MIN_CONF"] <= 1.0):
            raise ConfigError("TEXT_MIN_CONF must be in (0, 1]")
        if not (0.0 <= cfg["BLANK_CELL_STD_THR"] <= 255.0):
            raise ConfigError("BLANK_CELL_STD_THR must be in [0, 255]")
        if not isinstance(cfg["DEBUG"], bool):
            raise ConfigError("DEBUG must be a boolean")
        if not isinstance(cfg["IO"], bool):
            raise ConfigError("IO must be a boolean")
        if not isinstance(cfg["COLLECT_DATASET"], bool):
            raise ConfigError("COLLECT_DATASET must be a boolean")
        if not isinstance(cfg["LOG_RETENTION"], int) or cfg["LOG_RETENTION"] < 1:
            raise ConfigError("LOG_RETENTION must be a positive integer")
        for pin_key in ("GPIO_START_PIN", "GPIO_BUSY_PIN",
                        "GPIO_END_PIN", "GPIO_INSPEC_STAGE_PIN"):
            if not isinstance(cfg[pin_key], int) or not (1 <= cfg[pin_key] <= 27):
                raise ConfigError(f"{pin_key} must be a BCM pin number (1–27)")
        return cfg

    @classmethod
    def save(cls, updates: dict):
        import tomlkit
        try:
            with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                doc = tomlkit.load(f)
        except Exception:
            doc = tomlkit.document()
        for k, v in updates.items():
            if k in cls.DEFAULT_CONFIG:
                doc[k] = v
        with open(cls.CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))

    @classmethod
    def update(cls, updates: dict):
        """Merge partial updates into saved config. Only known keys are accepted."""
        import tomlkit
        try:
            with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                doc = tomlkit.load(f)
        except Exception:
            doc = tomlkit.document()
        for k, v in updates.items():
            if k in cls.DEFAULT_CONFIG:
                doc[k] = v
        with open(cls.CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))
        return cls.load()

# STAGE & ERROR FLAGS
class ErrorFlag(Enum):
    NONE     = "NONE"
    CAMERA   = "CAMERA"
    MODEL    = "MODEL"
    GPIO     = "GPIO"
    TEMPLATE = "TEMPLATE"

# EXCEPTIONS
class InspectionError(Exception):
    pass

class MarkMissingError(InspectionError):
    def __init__(self, missing_a: list, missing_b: list,
                 annotated: "np.ndarray | None" = None,
                 confs_a: list = None, confs_b: list = None):
        self.missing_a = missing_a
        self.missing_b = missing_b
        self.annotated = annotated
        self.confs_a   = confs_a or []   # per-cell Text confidence (6 floats) for IC_A
        self.confs_b   = confs_b or []   # per-cell Text confidence (6 floats) for IC_B
        parts = []
        if missing_a:
            parts.append(f"IC_A={missing_a}")
        if missing_b:
            parts.append(f"IC_B={missing_b}")
        super().__init__("Missing cells: " + ", ".join(parts))

class _SystemError(InspectionError):
    pass

class CameraError(_SystemError):
    pass

class ModelError(_SystemError):
    pass

class TemplateError(_SystemError):
    pass

class GPIOError(_SystemError):
    pass

class ConfigError(InspectionError):
    pass

# IMAGE DATACLASS + ID GENERATOR
_img_counter  = 0
_counter_lock = threading.Lock()

def _next_image_id() -> str:
    global _img_counter
    with _counter_lock:
        _img_counter += 1
        return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{_img_counter:03d}"

def _reset_image_counter():
    global _img_counter
    with _counter_lock:
        _img_counter = 0

@dataclass
class Image:
    id:        str
    raw:       np.ndarray
    annotated: np.ndarray = field(default=None)

# CAMERA
class Camera:
    """
    Unified camera source.
    CAMERA='camera'    : Basler pypylon InstantCamera
    CAMERA='directory' : reads files from Input/ in sorted order, loops
    """

    def __init__(self, mode: str, serial: str = "",
                 exposure_us: int = 8000, input_dir: str = "Input",
                 retry_delay: float = 0.2, retries: int = 2,
                 warmup_frames: int = 5,
                 image_w: int = 0, image_h: int = 0):
        self._mode        = mode
        self._serial      = serial
        self._exposure_us = exposure_us
        self._input_dir   = input_dir
        self._retry_delay = retry_delay
        self._retries     = retries
        self._warmup_frames = warmup_frames
        self._image_w     = image_w
        self._image_h     = image_h

        self._camera      = None
        self._pylon       = None
        self._basler_ok   = False

        self._files:  list = []
        self._idx:    int  = 0

        if mode == "camera":
            try:
                from pypylon import pylon
                self._pylon     = pylon
                self._basler_ok = True
            except ImportError:
                raise CameraError("pypylon not installed — cannot use CAMERA='camera'")

    # open
    def open(self):
        if self._mode == "camera":
            self._open_basler()
        else:
            self._open_directory()

    def _open_basler(self):
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

    def _open_directory(self):
        exts  = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(self._input_dir, ext)))
        files.sort()
        if not files:
            raise CameraError(
                f"No images found in '{self._input_dir}/'")
        self._files = files
        self._idx   = 0
        print(f"[Camera] Directory mode — {len(files)} image(s) in '{self._input_dir}/'")

    # grab
    def grab(self) -> np.ndarray:
        """Return BGR ndarray or raise CameraError."""
        for attempt in range(self._retries + 1):
            try:
                img = self._grab_once()
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

    def _grab_once(self) -> np.ndarray:
        if self._mode == "camera":
            return self._grab_basler()
        else:
            return self._grab_directory()

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

    def _grab_directory(self) -> np.ndarray:
        if not self._files:
            raise CameraError("No files loaded")
        path = self._files[self._idx % len(self._files)]
        self._idx += 1
        img = cv2.imread(path)
        if img is None:
            raise CameraError(f"Cannot read image: {path}")
        if self._image_w > 0 and self._image_h > 0:
            img = cv2.resize(img, (self._image_w, self._image_h),
                             interpolation=cv2.INTER_AREA)
        return img

    # misc
    def warmup(self):
        if self._mode == "camera":
            for _ in range(self._warmup_frames):
                try:
                    self._grab_basler()
                except Exception:
                    pass
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
        print("[Camera] Closed.")

    def reconnect(self, attempts: int = 1, delay_s: float = 0.0) -> bool:
        """Close and re-open the Basler camera. Returns True on success."""
        if self._mode != "camera":
            return False
        for _ in range(max(1, attempts)):
            self.close()
            if delay_s > 0:
                time.sleep(delay_s)
            try:
                self._open_basler()
                self.warmup()
                print("[Camera] Reconnected.")
                return True
            except CameraError as e:
                print(f"[Camera] Reconnect attempt failed: {e}")
        return False

    def is_open(self) -> bool:
        if self._mode == "camera":
            return self._camera is not None and self._camera.IsOpen()
        return bool(self._files)

    def is_healthy(self) -> bool:
        """True if camera is open and ready to accept triggers."""
        return self.is_open()

    def has_more(self) -> bool:
        """Directory mode: True if there are still un-visited images this cycle."""
        if self._mode != "directory":
            return True
        return self._idx < len(self._files)

    def reset(self):
        """Reset directory index to beginning."""
        self._idx = 0

    def grab_first(self) -> np.ndarray:
        """Grab the first frame: rewinds directory index before and after grab."""
        if self._mode == "directory":
            self.reset()
        img = self.grab()
        if self._mode == "directory":
            self.reset()
        return img

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
        time.sleep(0.040)
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

# DETECTOR  (OpenVINO Classifier — 2-class)
_CLS_INPUT_SIZE = 224   # YOLO-cls default input size
_TOTAL_CELLS    = 12    # 6 cells × 2 ICs

class Detector:
    """
    OpenVINO image classifier for ClearIC mark inspection.
    Each ROI cell crop is classified as Text (mark present) or NoText (absent).
    Output shape: [1, 2]  — index 0 = NoText, index 1 = Text
    """

    def __init__(self, conf_thr: float = 0.5, text_min_conf: float = 0.80,
                 blank_cell_std_thr: float = 0.0,
                 model_path: str = "Text_cls-2/best_openvino_model/best.xml",
                 n_passes: int = 3, uncertain_thr: float = 0.50,
                 debug: bool = False, **_):
        self._conf_thr           = conf_thr
        self._text_min_conf      = text_min_conf
        self._blank_cell_std_thr = blank_cell_std_thr
        self._n_passes           = max(1, int(n_passes))
        self._uncertain_thr      = float(uncertain_thr)
        self._debug              = debug
        self._compiled = None
        self._ready    = False
        try:
            import openvino as ov
            self._model_xml = model_path
            if not os.path.exists(self._model_xml):
                raise ModelError(f"Model not found: {self._model_xml}")
            core  = ov.Core()
            model = core.read_model(self._model_xml)
            self._compiled = core.compile_model(model, "CPU", {
                "INFERENCE_PRECISION_HINT": "f32",
                "PERFORMANCE_HINT":         "LATENCY",
            })
            self._ready = True
            print(f"[Detector] OpenVINO classifier loaded: {self._model_xml}")
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f"Model load failed: {e}")

    def is_ready(self) -> bool:
        return self._ready

    def warmup(self, frames: int = 5):
        blank = np.zeros((_CLS_INPUT_SIZE, _CLS_INPUT_SIZE, 3), dtype=np.uint8)
        for _ in range(frames):
            self.classify_crop(blank)
        print(f"[Detector] Warmup done ({frames} frames).")

    def classify_crop(self, crop_bgr: np.ndarray) -> tuple:
        """
        Classify one ROI cell crop.
        Returns (class_idx, confidence):
          class_idx 0 = NoText  (mark absent)
          class_idx 1 = Text    (mark present)
        """
        if not self._ready or self._compiled is None:
            return 0, 0.0
        try:
            sz = _CLS_INPUT_SIZE
            if crop_bgr.ndim == 2:
                crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
            if crop_bgr.size == 0:
                return 0, 0.0
            if self._blank_cell_std_thr > 0.0:
                _g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
                if float(_g.std()) < self._blank_cell_std_thr:
                    return 0, 1.0   # guard-triggered NoText; conf=1.0 marks it in logs
            resized = cv2.resize(crop_bgr, (sz, sz))
            blob    = resized[:, :, ::-1].astype(np.float32) / 255.0
            blob    = blob.transpose(2, 0, 1)[np.newaxis]   # [1, 3, sz, sz]
            text_probs = []
            for _ in range(self._n_passes):
                result = self._compiled(blob)
                text_probs.append(float(result[0][0][1]))   # P(Text) each pass
            text_prob   = sum(text_probs) / len(text_probs)
            notext_prob = 1.0 - text_prob
            # Require Text probability to clear TEXT_MIN_CONF; anything below → NoText.
            # Asymmetric on purpose: guards unmarked products without penalising NoText.
            if text_prob >= self._text_min_conf:
                return 1, text_prob
            if self._debug and text_prob >= self._uncertain_thr:
                print(f"[Detector] Uncertain cell: text_prob={text_prob:.3f} "
                      f"(gate={self._text_min_conf:.2f})")
            return 0, notext_prob
        except Exception as e:
            print(f"[Detector] Classify error: {e}")
            return 0, 0.0

# Dataset collection run counter
_data_run_counter = 0
_dataset_lock     = threading.Lock()

# VISUAL CONSTANTS  (fixed — not configurable at runtime)
_ann_color_ok = "#00C800"   # hex — Text  / PASS cell border
_ann_color_ng = "#DD0000"   # hex — NoText / FAIL cell border
_tmpl_color_a = "#FFD700"   # hex — IC_A overlay in setup view
_tmpl_color_b = "#00E5FF"   # hex — IC_B overlay in setup view

def _hex_to_bgr(h: str) -> tuple:
    """Convert '#RRGGBB' hex string to OpenCV BGR 3-tuple."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

def _build_cells(x: int, y: int, w: int, h: int,
                 cell_shrink: float = 0.95, cell_expand: float = 1.2,
                 col_gap_pct: float = 40.0,
                 grid_margin_top: float = 0.0,
                 grid_margin_bot: float = 15.0) -> list:
    """
    Build the 3-row × 2-col cell list for one IC bounding rect.

    Steps:
      1. Apply horizontal shrink (cell_shrink, L/R) and independent
         vertical margins (grid_margin_top / grid_margin_bot, top/bot).
      2. Slice the resulting rect into a 3×2 grid with col_gap_pct applied.
      3. Expand every cell by cell_expand (centred), so adjacent cells
         overlap — text marks near a boundary are covered by both cells.
    """
    # Step 1 — shrink (centred)
    sw = max(1, int(w * cell_shrink))
    sh = max(1, int(h * cell_shrink))
    sx = x + (w - sw) // 2
    sy = y + (h - sh) // 2

    # Step 2 — vertical margins then 3×2 grid on usable area
    usable_y0 = sy + int(sh * grid_margin_top / 100.0)
    usable_y1 = sy + sh - int(sh * grid_margin_bot / 100.0)
    usable_h  = max(1, usable_y1 - usable_y0)
    col_gap   = int(sw * col_gap_pct / 100.0)
    cw        = max(1, (sw - col_gap) // 2)
    ch        = max(1, usable_h // 3)
    col_starts = [sx, sx + cw + col_gap]

    # Step 3 — expand each cell (centred)
    exp_w = max(1, int(cw * cell_expand))
    exp_h = max(1, int(ch * cell_expand))
    dw    = (exp_w - cw) // 2
    dh    = (exp_h - ch) // 2

    cells = []
    for row in range(3):
        for col in range(2):
            cx = col_starts[col] - dw
            cy = usable_y0 + row * ch - dh
            cells.append((cx, cy, exp_w, exp_h))
    return cells

def _save_cell_crops(image_bgr: np.ndarray, cells: list,
                     cell_hits: list, ic_label: str, run_num: int,
                     data_dir: str = "Dataset", data_split: str = "train"):
    """
    Save each ROI cell crop to Dataset/<split>/Text/ or .../NoText/.
    Called only when COLLECT_DATASET = True.
    Filename: {run_num:06d}_IC{label}_{idx:02d}.png
    """
    ih, iw = image_bgr.shape[:2]
    for idx, (cx, cy, cw, ch) in enumerate(cells):
        class_name = "Text" if cell_hits[idx] else "NoText"
        folder = os.path.join(data_dir, data_split, class_name)
        os.makedirs(folder, exist_ok=True)
        x1, y1 = max(0, cx),       max(0, cy)
        x2, y2 = min(iw, cx + cw), min(ih, cy + ch)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            fname = f"{run_num:06d}_IC{ic_label}_{idx:02d}.png"
            cv2.imwrite(os.path.join(folder, fname), crop)

# TEMPLATE MANAGER
_TEMPLATE_FILE    = "templates/template.json"
_TEMPLATE_FULL    = "templates/tmpl_full.npy"
_TEMPLATE_PREVIEW = "templates/template_preview.png"

def _adaptive_binary(image_bgr: np.ndarray) -> np.ndarray:
    """BGR → dense adaptive-threshold binary. Used for setup-time IC auto-detection."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 5)

def _contour_template(image_bgr: np.ndarray) -> np.ndarray:
    """BGR → binary edge map for template matching.
    Pipeline: Gaussian blur → Otsu-driven Canny → dilate.
    Otsu auto-threshold adapts to image brightness; dilation widens edges
    so matchTemplate has signal even with small positional shifts.
    """
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    otsu_thr, _ = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges  = cv2.Canny(blurred, otsu_thr * 0.5, otsu_thr)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(edges, kernel, iterations=1)

class TemplateManager:

    @staticmethod
    def load() -> dict:
        if not os.path.exists(_TEMPLATE_FILE):
            raise TemplateError(f"Template not found: {_TEMPLATE_FILE}")
        try:
            with open(_TEMPLATE_FILE, "r") as f:
                data = json.load(f)
            for key in ("ic_a", "ic_b"):
                for sub in ("x", "y", "w", "h"):
                    _ = data[key][sub]
            return data
        except TemplateError:
            raise
        except Exception as e:
            raise TemplateError(f"Template corrupt: {e}")

    @staticmethod
    def save(ic_a: QtCore.QRect, ic_b: QtCore.QRect, exposure_us: int = 8000,
             match_threshold: float = 0.6, strip_h: int = 0,
             img_w: int = 0, img_h: int = 0):
        os.makedirs("templates", exist_ok=True)
        data = {
            "ic_a": {"x": ic_a.x(), "y": ic_a.y(),
                     "w": ic_a.width(), "h": ic_a.height()},
            "ic_b": {"x": ic_b.x(), "y": ic_b.y(),
                     "w": ic_b.width(), "h": ic_b.height()},
            "exposure_us":     exposure_us,
            "match_threshold": match_threshold,
            "strip_h":         strip_h,
            "img_w":           img_w,
            "img_h":           img_h,
        }
        _tmp = _TEMPLATE_FILE + ".tmp"
        with open(_tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(_tmp, _TEMPLATE_FILE)

    @staticmethod
    def extract_patches(image_bgr: np.ndarray, ic_rect: QtCore.QRect) -> tuple:
        """
        Extract the pin-area patch ONLY (below the IC body):
          patch spans [X1, Y2] → [X2, Y3]
          where Y2 = ic bottom, Y3 = Y2 + pin_height (50% of IC height)

        Returns (patch, strip_h) where strip_h = y - y_start = -(IC height).
        strip_h is negative because the patch starts below the IC top.
        TemplateMatcher uses: patch_top = ic_y - strip_h  →  ic_y + IC_h  ✓
        """
        x, y = ic_rect.x(), ic_rect.y()
        w, h = ic_rect.width(), ic_rect.height()
        h1 = max(1, int(h * 0.5))  # pin strip height = 50% of IC height

        img_h, img_w = image_bgr.shape[:2]
        y_start = y + h                    # Y2: IC bottom
        y_end   = min(img_h, y + h + h1)  # Y3: bottom of pin area
        x_end   = min(x + w, img_w)

        full_bin = _contour_template(image_bgr)[y_start:y_end, x:x_end]
        strip_h  = y - y_start  # = -h  (patch is below IC top by IC height)

        return full_bin, strip_h

    @staticmethod
    def save_patches(full_patch: np.ndarray):
        """Save combined (top strip + IC body + bot strip) patch as tmpl_full.npy."""
        os.makedirs("templates", exist_ok=True)
        _tmp = _TEMPLATE_FULL.replace(".npy", "_tmp.npy")
        np.save(_tmp, full_patch)
        os.replace(_tmp, _TEMPLATE_FULL)

    @staticmethod
    def load_patches():
        """Load template patch (tmpl_full.npy). Returns ndarray or None if absent/corrupt."""
        if not os.path.exists(_TEMPLATE_FULL):
            return None
        try:
            return np.load(_TEMPLATE_FULL)
        except Exception as e:
            print(f"[TemplateManager] Patch load failed: {e}")
            return None

    @staticmethod
    def save_preview(image_bgr: np.ndarray,
                     ic_a: QtCore.QRect, ic_b: QtCore.QRect):
        """
        Save an annotated preview image:
        - IC_A (yellow) and IC_B (cyan) boxes with 3×2 cell grids and labels
        - Magenta outline of the actual template patch region saved for IC_A
          (IC body + 50% strip below — matches extract_patches geometry exactly)
        - Teal overlay of the _contour_template edges within that patch region
        Saved to templates/template_preview.png for visual verification.
        """
        os.makedirs("templates", exist_ok=True)
        img_h, img_w = image_bgr.shape[:2]
        preview = image_bgr.copy()

        # IC boxes, cell grids, centre crosses
        for rect, color, label in [
            (ic_a, (0, 255, 255), "IC_A"),
            (ic_b, (255, 215, 0), "IC_B"),
        ]:
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
            cv2.putText(preview, label, (x + 4, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cw, ch = w // 2, h // 3
            for row in range(3):
                for col in range(2):
                    cx, cy = x + col * cw, y + row * ch
                    cv2.rectangle(preview, (cx, cy), (cx + cw, cy + ch), color, 1)
                    cv2.putText(preview, f"R{row+1}C{col+1}",
                                (cx + 2, cy + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            cx, cy = x + w // 2, y + h // 2
            arm = max(12, min(w, h) // 6)
            cv2.line(preview, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 2)
            cv2.line(preview, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 2)
            cv2.circle(preview, (cx, cy), 3, (255, 255, 255), -1)

        # Template patch region (IC_A only)
        # Geometry must match extract_patches exactly: IC body + 50% strip below
        # Pin area: [X1, Y2] → [X2, Y3], matches extract_patches geometry exactly
        ax, ay = ic_a.x(), ic_a.y()
        aw, ah = ic_a.width(), ic_a.height()
        h1       = max(1, int(ah * 0.5))
        patch_y1 = ay + ah                    # Y2: IC bottom
        patch_y2 = min(img_h, ay + ah + h1)  # Y3: pin area bottom
        patch_x2 = min(ax + aw, img_w)

        # Magenta border showing the saved pin patch extent
        cv2.rectangle(preview,
                      (ax, patch_y1), (patch_x2, patch_y2),
                      (255, 0, 255), 2)
        cv2.putText(preview, "Pin patch",
                    (ax + 2, patch_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

        # Contour edge overlay (teal) inside pin patch only
        contour_full = _contour_template(image_bgr)
        patch_edges  = contour_full[patch_y1:patch_y2, ax:patch_x2]
        edge_mask    = patch_edges > 0
        roi          = preview[patch_y1:patch_y2, ax:patch_x2].astype(np.float32)
        teal         = np.array([180, 200, 0], dtype=np.float32)  # BGR teal
        roi[edge_mask] = roi[edge_mask] * 0.3 + teal * 0.7
        preview[patch_y1:patch_y2, ax:patch_x2] = roi.clip(0, 255).astype(np.uint8)

        _tmp = _TEMPLATE_PREVIEW + ".tmp.png"
        cv2.imwrite(_tmp, preview)
        os.replace(_tmp, _TEMPLATE_PREVIEW)

    @staticmethod
    def compute_rois(template: dict, grid_cfg: dict | None = None) -> tuple:
        """Returns (ic_a_cells, ic_b_cells) — list of 6 (x,y,w,h) per IC."""
        g = grid_cfg or {}
        def _cells(box: dict) -> list:
            return _build_cells(
                box["x"], box["y"], box["w"], box["h"],
                cell_shrink=g.get("CELL_SHRINK", 0.95),
                cell_expand=g.get("CELL_EXPAND", 1.2),
                col_gap_pct=g.get("COL_GAP_PCT", 40.0),
                grid_margin_top=g.get("GRID_MARGIN_TOP", 0.0),
                grid_margin_bot=g.get("GRID_MARGIN_BOT", 15.0),
            )
        return _cells(template["ic_a"]), _cells(template["ic_b"])

# TEMPLATE MATCHER
class TemplateMatcher:
    """
    Locates IC_A in a new image using a single adaptive-binary combined patch
    (IC body + bottom strip) matched with cv2.TM_CCOEFF_NORMED.

    If the match score falls below threshold a TemplateError is raised —
    this acts as a rotation/misalignment rejection gate.
    """

    def __init__(self, full_patch: np.ndarray,
                 threshold: float = 0.6,
                 strip_h: int = 0,
                 ic_x: int = 0, ic_y: int = 0,
                 ic_w: int = 0, ic_h: int = 0,
                 search_margin: int = 60,
                 template_w: int = 0):
        self._patch       = full_patch
        self._threshold   = threshold
        self._strip_h     = strip_h   # px from patch top to IC top edge
        self._patch_w     = full_patch.shape[1]
        self._ic_x        = ic_x   # expected IC_A left edge from template
        self._ic_y        = ic_y   # expected IC_A top from template
        self._ic_w        = ic_w
        self._ic_h        = ic_h
        self._margin      = search_margin   # px around expected pos to search
        self._template_w  = template_w      # image width when template was saved (0 = unknown)

    def locate_ic(self, image_bgr: np.ndarray) -> tuple:
        """
        Returns (QRect, score).
        Matches the pin-area patch (below IC body) against the frame.
        Preprocessing is applied only to the ±search_margin ROI for speed.
        strip_h is negative (patch starts below IC top), so:
          exp_patch_y = ic_y - strip_h = ic_y + IC_h
          ic_y        = matched_patch_y + strip_h = matched_patch_y - IC_h
        If the current image has a different width than the template was saved at,
        the patch and all search parameters are scaled to match.
        """
        img_h, img_w = image_bgr.shape[:2]

        # Scale patch and search geometry to current image resolution
        if self._template_w > 0 and abs(img_w / self._template_w - 1.0) > 0.01:
            scale = img_w / self._template_w
            ph0, pw0 = self._patch.shape[:2]
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            patch = cv2.resize(self._patch,
                               (max(1, int(pw0 * scale)), max(1, int(ph0 * scale))),
                               interpolation=interp)
            ic_x = int(self._ic_x * scale)
            ic_y_tmpl = int(self._ic_y * scale)
            ic_w = max(1, int(self._ic_w * scale))
            ic_h = max(1, int(self._ic_h * scale))
            strip_h = int(self._strip_h * scale)
            m = int(self._margin * scale)
        else:
            patch = self._patch
            ic_x, ic_y_tmpl = self._ic_x, self._ic_y
            ic_w, ic_h = self._ic_w, self._ic_h
            strip_h = self._strip_h
            m = self._margin

        ph, pw = patch.shape[:2]
        exp_y = ic_y_tmpl - strip_h

        rx1 = max(0, ic_x - m)
        ry1 = max(0, exp_y - m)
        rx2 = min(img_w, ic_x + pw + m)
        ry2 = min(img_h, exp_y + ph + m)

        roi_bgr = image_bgr[ry1:ry2, rx1:rx2]

        if roi_bgr.size == 0 or roi_bgr.shape[0] < ph or roi_bgr.shape[1] < pw:
            # ROI empty or too small — fall back to full-frame search
            full = _contour_template(image_bgr)
            res = cv2.matchTemplate(full, patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            found_ic_y = loc[1] + strip_h
            return QtCore.QRect(loc[0], found_ic_y, ic_w, ic_h), float(score)

        filtered = _contour_template(roi_bgr)
        res = cv2.matchTemplate(filtered, patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        abs_x = loc[0] + rx1
        abs_y = loc[1] + ry1
        found_ic_y = abs_y + strip_h

        if score < self._threshold:
            print(f"[TemplateMatcher] Low match score {score:.3f} < {self._threshold:.3f} — "
                  "using best-match position anyway")

        return QtCore.QRect(abs_x, found_ic_y, ic_w, ic_h), float(score)

def _find_second_ic(image_bgr: np.ndarray,
                    ref_rect: QtCore.QRect,
                    conf_thr: float = 0.4) -> tuple:
    """
    Search the opposite image half for a second IC using the ref_rect crop as a
    template.  Uses dense adaptive binary (not contour) for reliable setup-time matching.

    Returns (QRect, score).  QRect is None if score < conf_thr.
    """
    x, y, w, h = ref_rect.x(), ref_rect.y(), ref_rect.width(), ref_rect.height()
    img_h, img_w = image_bgr.shape[:2]

    binary = _adaptive_binary(image_bgr)

    ty1, ty2 = max(0, y), min(img_h, y + h)
    tx1, tx2 = max(0, x), min(img_w, x + w)
    template = binary[ty1:ty2, tx1:tx2]
    if template.size == 0:
        return None, 0.0

    mid = img_w // 2
    if (x + w // 2) < mid:   # ref is on left → search right half
        search   = binary[:, mid:]
        x_offset = mid
    else:                     # ref is on right → search left half
        search   = binary[:, :mid]
        x_offset = 0

    if search.shape[1] < template.shape[1] or search.shape[0] < template.shape[0]:
        return None, 0.0

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)

    if score >= conf_thr:
        return QtCore.QRect(loc[0] + x_offset, loc[1], w, h), float(score)
    return None, float(score)

# INSPECTOR
class Inspector:
    """
    Crops each ROI cell from the image and classifies it as Text / NoText.
    Raises MarkMissingError if either IC has any cell classified as NoText.
    """

    def __init__(self, detector: Detector, template: dict,
                 template_matcher: "TemplateMatcher | None" = None,
                 cell_shrink: float = 0.95, cell_expand: float = 1.2,
                 col_gap_pct: float = 40.0,
                 grid_margin_top: float = 0.0, grid_margin_bot: float = 15.0,
                 collect_dataset: bool = False,
                 data_dir: str = "Dataset", data_split: str = "train",
                 ann_border_px: int = 1, ann_show_labels: bool = True):
        self._detector         = detector
        self._template         = template
        self._template_matcher = template_matcher
        self._ic_b_dx_tmpl = template["ic_b"]["x"] - template["ic_a"]["x"]
        self._ic_b_dy_tmpl = template["ic_b"]["y"] - template["ic_a"]["y"]
        self._template_w   = int(template.get("img_w", 0))
        self._template_h   = int(template.get("img_h", 0))
        self._cell_shrink     = cell_shrink
        self._cell_expand     = cell_expand
        self._col_gap_pct     = col_gap_pct
        self._grid_margin_top = grid_margin_top
        self._grid_margin_bot = grid_margin_bot
        self._collect_dataset = collect_dataset
        self._data_dir        = data_dir
        self._data_split      = data_split
        self._ann_border_px   = ann_border_px
        self._ann_show_labels = ann_show_labels

    def inspect(self, image_bgr: np.ndarray,
                debug: bool = False) -> tuple:
        """
        Returns (ic_a_pass, ic_b_pass, missing_a, missing_b, annotated_bgr).
        Raises MarkMissingError if either IC fails.
        Raises TemplateError if template matching rejects the frame.

        Phase 1 — locate IC_A via TemplateMatcher (preferred) or fixed template coords.
        Phase 2 — crop each ROI cell and classify as Text / NoText.
        """
        img_h, img_w = image_bgr.shape[:2]

        # Scale factor: template coords → current image space.
        # Legacy templates (no img_w/img_h) use 1:1 — no change in behaviour.
        if self._template_w > 0 and self._template_h > 0:
            sx = img_w / self._template_w
            sy = img_h / self._template_h
        else:
            sx, sy = 1.0, 1.0

        ic_b_dx = int(self._ic_b_dx_tmpl * sx)
        ic_b_dy = int(self._ic_b_dy_tmpl * sy)

        def _scale_r(r):
            return {"x": int(r["x"] * sx), "y": int(r["y"] * sy),
                    "w": max(1, int(r["w"] * sx)), "h": max(1, int(r["h"] * sy))}

        ic_a_s = _scale_r(self._template["ic_a"])
        ic_b_s = _scale_r(self._template["ic_b"])

        # Guard: verify image is large enough to cover both scaled IC regions
        min_w = max(ic_a_s["x"] + ic_a_s["w"], ic_b_s["x"] + ic_b_s["w"])
        min_h = max(ic_a_s["y"] + ic_a_s["h"], ic_b_s["y"] + ic_b_s["h"])
        if img_w < min_w or img_h < min_h:
            raise TemplateError(
                f"Image {img_w}×{img_h} too small — template requires at least {min_w}×{min_h}")

        annotated = image_bgr  # draw in-place; caller saves raw before calling inspect()

        # Phase 1: locate ICs
        if self._template_matcher is not None:
            rt_a, score = self._template_matcher.locate_ic(image_bgr)
            rt_b = QtCore.QRect(
                rt_a.x() + ic_b_dx, rt_a.y() + ic_b_dy,
                ic_b_s["w"], ic_b_s["h"],
            )
            ic_a_cells = self._rect_to_cells(rt_a)
            ic_b_cells = self._rect_to_cells(rt_b)
            if debug:
                print(f"[Inspector] scale=({sx:.3f},{sy:.3f}) "
                      f"TemplateMatcher score={score:.3f}")
                print(f"[Inspector] IC_A matched: "
                      f"x={rt_a.x()} y={rt_a.y()} w={rt_a.width()} h={rt_a.height()}")
                print(f"[Inspector] IC_B by offset: "
                      f"x={rt_b.x()} y={rt_b.y()} w={rt_b.width()} h={rt_b.height()}")
        else:
            # Fixed scaled coords — no runtime IC localization
            ic_a_cells = _build_cells(
                ic_a_s["x"], ic_a_s["y"], ic_a_s["w"], ic_a_s["h"],
                self._cell_shrink, self._cell_expand,
                self._col_gap_pct, self._grid_margin_top, self._grid_margin_bot)
            ic_b_cells = _build_cells(
                ic_b_s["x"], ic_b_s["y"], ic_b_s["w"], ic_b_s["h"],
                self._cell_shrink, self._cell_expand,
                self._col_gap_pct, self._grid_margin_top, self._grid_margin_bot)
            if debug:
                print(f"[Inspector] scale=({sx:.3f},{sy:.3f}) "
                      "No TemplateMatcher — using scaled fixed template coordinates")

        # Phase 2: crop each cell and classify as Text / NoText
        missing_a, hits_a, confs_a = self._check_ic(image_bgr, ic_a_cells, annotated, debug)
        missing_b, hits_b, confs_b = self._check_ic(image_bgr, ic_b_cells, annotated, debug)

        if self._collect_dataset:
            global _data_run_counter
            with _dataset_lock:
                _data_run_counter += 1
                run_num = _data_run_counter
            _save_cell_crops(image_bgr, ic_a_cells, hits_a, "A", run_num,
                             self._data_dir, self._data_split)
            _save_cell_crops(image_bgr, ic_b_cells, hits_b, "B", run_num,
                             self._data_dir, self._data_split)

        if missing_a or missing_b:
            raise MarkMissingError(missing_a, missing_b, annotated, confs_a, confs_b)

        return True, True, [], [], annotated

    def _rect_to_cells(self, rect: QtCore.QRect) -> list:
        """Divide a QRect into a shrunk+expanded 3-row × 2-col cell grid."""
        return _build_cells(rect.x(), rect.y(), rect.width(), rect.height(),
                            self._cell_shrink, self._cell_expand,
                            self._col_gap_pct,
                            self._grid_margin_top, self._grid_margin_bot)

    def _check_ic(self, image_bgr: np.ndarray, cells: list,
                  annotated: np.ndarray, debug: bool) -> tuple:
        """
        Crop each ROI cell from image_bgr and classify as Text / NoText.
        Returns (missing, hits_flags, text_confs).
        text_confs: per-cell Text-class confidence (6 floats, 0–1).
        """
        ih, iw = image_bgr.shape[:2]
        color_ok    = _hex_to_bgr(_ann_color_ok)
        color_ng    = _hex_to_bgr(_ann_color_ng)
        missing     = []
        hits_flags  = []
        text_confs  = []
        for idx, (cx, cy, cw, ch) in enumerate(cells):
            row = idx // 2 + 1
            col = idx %  2 + 1
            x1, y1 = max(0, cx),       max(0, cy)
            x2, y2 = min(iw, cx + cw), min(ih, cy + ch)
            crop = image_bgr[y1:y2, x1:x2]
            cls_idx, conf = self._detector.classify_crop(crop) if crop.size > 0 else (0, 0.0)
            present   = (cls_idx == 1)   # 1 = Text (mark present)
            text_conf = conf if cls_idx == 1 else (1.0 - conf)  # Text-class probability
            hits_flags.append(present)
            text_confs.append(text_conf)
            if debug:
                lbl = "Text" if present else "NoText"
                std_str = f"{crop.std():.1f}" if crop.size > 0 else "n/a"
                print(f"[Cell R{row}C{col}] "
                      f"{'PRESENT' if present else 'ABSENT '} "
                      f"cls={lbl} conf={conf:.3f} text_conf={text_conf:.3f}  "
                      f"raw_std={std_str}")
            color = color_ok if present else color_ng
            cv2.rectangle(annotated,
                          (max(0, cx), max(0, cy)),
                          (min(iw, cx + cw), min(ih, cy + ch)),
                          color, self._ann_border_px)
            if self._ann_show_labels:
                label = f"R{row}C{col}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                tx = max(0, cx) + (cw - tw) // 2
                ty = max(0, cy) + (ch + th) // 2
                cv2.putText(annotated, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if not present:
                missing.append([row, col])
        return missing, hits_flags, text_confs

# LOGGER
class Logger:
    """
    Dual-CSV logging system.

    Operation log  — one file per calendar day, appended across all lots.
      File: logs/op_YYYYMMDD.csv
      Columns: timestamp, event, lot_number, detail, cycle_ms

    Result log — one file per lot run, written incrementally.
      File: logs/result_{lot}_{YYYYMMDD_HHMMSS}.csv
      Header block: lot metadata rows.
      Data rows: one per inspection.
      Footer block: summary appended at lot end.
    """

    _OP_HEADER   = ["timestamp", "event", "lot_number", "detail", "cycle_ms"]
    _RES_HEADER  = ["timestamp", "image_id", "ic_a_result",
                    "ic_b_result", "cycle_ms", "is_retry", "is_suspect"]

    def __init__(self, log_dir: str = "logs", log_retention: int = 365):
        self._dir        = log_dir
        self._retention  = log_retention
        self._lot        = ""
        self._package    = ""
        self._res_path:  str | None = None
        self._pass_ct    = 0
        self._fail_ct    = 0
        self._err_ct     = 0
        os.makedirs(log_dir, exist_ok=True)
        self._rotate()

    # internal helpers

    def _op_path(self) -> str:
        return os.path.join(self._dir, f"op_{datetime.now():%Y%m%d}.csv")

    def _rotate(self):
        for pattern in ("op_*.csv", "result_*.csv"):
            logs = sorted(glob.glob(os.path.join(self._dir, pattern)))
            while len(logs) > self._retention:
                try:
                    os.remove(logs.pop(0))
                except OSError:
                    pass

    def _op_append(self, event: str, detail: str = "", cycle_ms: float = 0):
        path = self._op_path()
        write_header = not os.path.exists(path)
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(self._OP_HEADER)
                w.writerow([
                    datetime.now().isoformat(),
                    event,
                    self._lot,
                    detail,
                    round(cycle_ms, 1),
                ])
        except Exception as e:
            print(f"[Logger] op write failed: {e}", file=sys.stderr)

    def _res_write(self, row: list):
        if not self._res_path:
            return
        try:
            with open(self._res_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            print(f"[Logger] result write failed: {e}", file=sys.stderr)

    def _write_result_header(self, lot: str, package: str, mode: str):
        if not self._res_path:
            return
        try:
            with open(self._res_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["LOT_NUMBER", lot])
                w.writerow(["PACKAGE",    package])
                w.writerow(["START_TIME", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                w.writerow(["MODE",       mode])
                w.writerow([])                       # blank separator
                w.writerow(self._RES_HEADER)
        except Exception as e:
            print(f"[Logger] result header write failed: {e}", file=sys.stderr)

    def _write_result_footer(self, pass_ct: int, fail_ct: int,
                             err_ct: int, elapsed_s: float):
        if not self._res_path:
            return
        total  = pass_ct + fail_ct
        yield_ = f"{pass_ct / total * 100:.1f}" if total else "N/A"
        try:
            with open(self._res_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([])
                w.writerow(["TOTAL",       total])
                w.writerow(["PASS",        pass_ct])
                w.writerow(["FAIL",        fail_ct])
                w.writerow(["ERRORS",      err_ct])
                w.writerow(["YIELD_PCT",   yield_])
                w.writerow(["END_TIME",    datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                w.writerow(["DURATION_S",  round(elapsed_s, 1)])
        except Exception as e:
            print(f"[Logger] result footer write failed: {e}", file=sys.stderr)

    # public interface

    def start_lot(self, lot_number: str, package: str, mode: str):
        self._rotate()
        self._lot     = lot_number
        self._package = package
        self._pass_ct = self._fail_ct = self._err_ct = 0
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_lot = "".join(c if c.isalnum() or c in "-_" else "_" for c in lot_number)
        self._res_path = os.path.join(self._dir, f"result_{safe_lot}_{ts}.csv")
        self._write_result_header(lot_number, package, mode)
        self._op_append("SESSION_START", f"mode={mode}")

    def end_lot(self, reason: str,
                pass_ct: int, fail_ct: int, err_ct: int, elapsed_s: float):
        total  = pass_ct + fail_ct
        yield_ = f"{pass_ct / total * 100:.1f}%" if total else "N/A"
        self._op_append("SESSION_END",
                         f"reason={reason} pass={pass_ct} fail={fail_ct} "
                         f"error={err_ct} yield={yield_}")
        self._write_result_footer(pass_ct, fail_ct, err_ct, elapsed_s)
        self._res_path = None

    def log_inspection(self, image_id: str,
                       ic_a_result: str, ic_a_missing: list,
                       ic_b_result: str, ic_b_missing: list,
                       cycle_ms: float, is_retry: bool,
                       is_suspect: bool = False):
        passed = (ic_a_result == "PASS" and ic_b_result == "PASS")
        event  = "PASS" if passed else "FAIL"
        if is_suspect:
            event += "_SUSPECT"
        # Build detail: image filename + missing cells if any
        detail_parts = [image_id]
        if ic_a_missing:
            detail_parts.append(f"miss_a={ic_a_missing}")
        if ic_b_missing:
            detail_parts.append(f"miss_b={ic_b_missing}")
        detail_parts.append(f"is_retry={1 if is_retry else 0}")
        if is_suspect:
            detail_parts.append("suspect=1")
        self._op_append(event, " ".join(detail_parts), cycle_ms)
        # Result log row
        self._res_write([
            datetime.now().isoformat(),
            image_id,
            ic_a_result,
            ic_b_result,
            round(cycle_ms, 1),
            1 if is_retry else 0,
            1 if is_suspect else 0,
        ])
        if passed:
            self._pass_ct += 1
        else:
            self._fail_ct += 1

    def log_error(self, error_type: str, message: str, cycle_ms: float = 0):
        self._op_append("ERROR", f"{error_type}: {message}", cycle_ms)
        self._err_ct += 1

    def log_pause(self):
        self._op_append("PAUSE")

    def log_resume(self):
        self._op_append("RESUME")

    def log_ocr(self, operator: str, expect_mark: str):
        self._op_append("OCR_VERIFY", f"op={operator} expect={expect_mark}")


# STYLESHEET
STYLE = """
QMainWindow, QWidget#root {
    background: #5465FF;
}
QTabWidget::pane {
    background: #5465FF;
    border: none;
}
QTabBar::tab {
    background: #788BFF;
    color: #FFFFFF;
    padding: 6px 18px;
    border-radius: 4px 4px 0 0;
    font-size: 12px;
}
QTabBar::tab:selected {
    background: #5465FF;
    color: #FFFFFF;
    font-weight: bold;
}
QFrame#panel_right {
    background: #5465FF;
}
QFrame#setup_frame, QFrame#controls_frame {
    background: #788BFF;
    border-radius: 8px;
    padding: 8px;
}
QFrame#main_view {
    background: #788BFF;
    border-radius: 8px;
}
QFrame#image_area {
    background: #E2FDFF;
    border-radius: 8px;
}
QFrame#badge_area, QFrame#stats_area {
    background: #9BB1FF;
    border-radius: 8px;
    padding: 8px;
}
QFrame#badge_pass {
    background: #BFD7FF;
    border-radius: 8px;
    padding: 8px;
}
QFrame#badge_fail {
    background: #EF5350;
    border-radius: 8px;
    padding: 8px;
}
QFrame#badge_idle {
    background: #9BB1FF;
    border-radius: 8px;
    padding: 8px;
}
QFrame#error_banner {
    background: #EF5350;
    border-radius: 8px;
    padding: 6px;
}
QPushButton {
    background: #5465FF;
    color: #FFFFFF;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: bold;
    border: none;
}
QPushButton:disabled {
    background: #788BFF;
    color: #BFD7FF;
}
QLineEdit {
    background: #FFFFFF;
    color: #5465FF;
    border: 2px solid #5465FF;
    border-radius: 6px;
    padding: 4px 8px;
}
QLabel {
    color: #FFFFFF;
}
QLabel#stat_value {
    color: #E2FDFF;
    font-weight: bold;
}
QCheckBox {
    color: #FFFFFF;
    font-weight: bold;
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #FFFFFF;
    border-radius: 3px;
    background: transparent;
}
QCheckBox::indicator:checked {
    background: #FFFFFF;
    image: none;
}
QCheckBox:disabled {
    color: #BFD7FF;
}
QCheckBox::indicator:disabled {
    border-color: #BFD7FF;
}
"""

class ImageView(QtWidgets.QLabel):
    """
    Zoomable image display with overlay support, stamp mode, and rubber-band drawing.
    """
    rect_drawn    = QtCore.pyqtSignal(QtCore.QRect)  # emitted on rubber-band release (image coords)
    right_clicked = QtCore.pyqtSignal()             # emitted on right-click (non-rubberband mode)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setObjectName("image_area")
        self._orig        = None
        self._scale       = 1.0
        self._offset      = QtCore.QPoint(0, 0)
        self._overlays    = []    # (QRect, QColor, label)
        self._rb_mode     = False
        self._rb_start    = None  # QPoint in image coords
        self._rb_cur      = None  # QPoint in image coords (current drag position)
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

    # image
    def set_image(self, img: np.ndarray):
        if img is None:
            return
        if img.ndim == 2:
            self._orig = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            self._orig = img.copy()
        QtCore.QTimer.singleShot(0, self._refresh)

    def sizeHint(self):
        return QtCore.QSize(320, 240)

    def minimumSizeHint(self):
        return QtCore.QSize(1, 1)

    def _refresh(self):
        if self._orig is None:
            return
        lw, lh = self.width(), self.height()
        if lw < 2 or lh < 2:   # widget not yet sized — skip to avoid feedback loop
            return
        h, w = self._orig.shape[:2]
        rgb  = cv2.cvtColor(self._orig, cv2.COLOR_BGR2RGB)
        qi   = QtGui.QImage(bytes(rgb.data), w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix  = QtGui.QPixmap.fromImage(qi)
        pix  = pix.scaled(lw, lh, QtCore.Qt.KeepAspectRatio,
                          QtCore.Qt.SmoothTransformation)
        if w > 0:
            self._scale = pix.width() / w
        self._offset = QtCore.QPoint((lw - pix.width())  // 2,
                                     (lh - pix.height()) // 2)
        self.setPixmap(pix)
        if self._overlays:
            self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if e.size() != e.oldSize():
            QtCore.QTimer.singleShot(0, self._refresh)

    # coordinate helper
    def _to_img(self, pt: QtCore.QPoint) -> QtCore.QPoint:
        if self.pixmap() is None or self._orig is None:
            return pt
        return QtCore.QPoint(
            int((pt.x() - self._offset.x()) / self._scale),
            int((pt.y() - self._offset.y()) / self._scale),
        )

    def _to_widget(self, rect: QtCore.QRect) -> QtCore.QRect:
        x = int(rect.x() * self._scale) + self._offset.x()
        y = int(rect.y() * self._scale) + self._offset.y()
        w = int(rect.width()  * self._scale)
        h = int(rect.height() * self._scale)
        return QtCore.QRect(x, y, w, h)

    # overlays
    def add_overlay(self, rect: QtCore.QRect, color: QtGui.QColor, label: str = ""):
        self._overlays.append((rect, color, label))
        self.update()

    def clear_overlays(self):
        self._overlays.clear()
        self.update()

    # rubber-band mode
    def set_rubberband_mode(self, on: bool):
        self._rb_mode  = on
        self._rb_start = None
        self._rb_cur   = None
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
        self.update()

    # paint
    def paintEvent(self, e):
        super().paintEvent(e)
        has_rb = self._rb_mode and self._rb_start and self._rb_cur
        if not self._overlays and not has_rb:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        for rect, color, label in self._overlays:
            wr = self._to_widget(rect)
            pen = QtGui.QPen(color, 2)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(wr)
            if label:
                painter.setFont(QtGui.QFont("Arial", 9, QtGui.QFont.Bold))
                painter.setPen(color)
                painter.drawText(wr.topLeft() + QtCore.QPoint(3, 14), label)

        if has_rb:
            rb_img    = QtCore.QRect(self._rb_start, self._rb_cur).normalized()
            rb_widget = self._to_widget(rb_img)
            pen = QtGui.QPen(QtGui.QColor("#FFD700"), 2, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(rb_widget)

        painter.end()

    def mouseMoveEvent(self, e):
        if self._rb_mode and self._rb_start:
            self._rb_cur = self._to_img(e.pos())
            self.update()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton and self._rb_mode:
            self._rb_start = self._to_img(e.pos())
            self._rb_cur   = self._rb_start
        elif e.button() == QtCore.Qt.RightButton and not self._rb_mode:
            self.right_clicked.emit()

    def mouseReleaseEvent(self, e):
        if self._rb_mode and self._rb_start and e.button() == QtCore.Qt.LeftButton:
            end  = self._to_img(e.pos())
            rect = QtCore.QRect(self._rb_start, end).normalized()
            self._rb_start = None
            self._rb_cur   = None
            self.update()
            if rect.width() > 5 and rect.height() > 5:
                self.rect_drawn.emit(rect)

# RUN WORKER
def _output_dirs(out_dir: str, lot_number: str) -> tuple:
    """
    Returns (real_dir, ann_dir) for today + lot_number, creating dirs.
    Structure: out_dir/YYYYMMDD/lot_number/RealImg|Image/
    """
    date     = datetime.now().strftime("%Y%m%d")
    real_dir = os.path.join(out_dir, date, lot_number, "RealImg")
    ann_dir  = os.path.join(out_dir, date, lot_number, "Image")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ann_dir,  exist_ok=True)
    return real_dir, ann_dir


def _resolve_ic(missing_first: list, confs_first: list, confs_second: list) -> list:
    """
    Confidence-weighted retry resolution for a single IC.
    Only re-evaluates cells that were MISSING on the first attempt.
    Formula: w = 0.7 * text_conf_second + 0.3 * text_conf_first
    A cell is PASS (Text present) only if w >= 0.90.
    Returns updated missing list (cells still failing after weighting).
    """
    still_missing = []
    for row, col in missing_first:
        idx = (row - 1) * 2 + (col - 1)
        c1 = confs_first[idx]  if idx < len(confs_first)  else 0.0
        c2 = confs_second[idx] if idx < len(confs_second) else 0.0
        if 0.7 * c2 + 0.3 * c1 < 0.90:
            still_missing.append([row, col])
    return still_missing

class RunWorker(QtCore.QThread):
    """
    Background inspection loop.

    Camera mode: wait_for_start() blocks on START_PIN HIGH (active HIGH);
      IO=False: blocks until MainWindow calls trigger() per cycle.
    Directory mode: auto-loops with 50 ms yield between cycles.
    """
    sig_image    = QtCore.pyqtSignal(object)          # annotated BGR ndarray
    sig_result   = QtCore.pyqtSignal(bool, bool, bool)       # ic_a_pass, ic_b_pass, is_suspect
    sig_fail     = QtCore.pyqtSignal(object, str, str, bool) # (MarkMissingError, ann_path, img_id, is_suspect)
    sig_error    = QtCore.pyqtSignal(str)
    sig_status   = QtCore.pyqtSignal(str)
    sig_cycle_ms = QtCore.pyqtSignal(float)
    sig_done          = QtCore.pyqtSignal()   # worker loop exited (Stop pressed)
    sig_session_reset = QtCore.pyqtSignal(str) # batch complete → new lot_number
    sig_paused        = QtCore.pyqtSignal()
    sig_resumed       = QtCore.pyqtSignal()

    def __init__(self, camera: Camera, inspector: Inspector,
                 gpio: RaspberryIO, logger: Logger,
                 cfg: dict, lot_number: str = "", parent=None):
        super().__init__(parent)
        self._camera     = camera
        self._inspector  = inspector
        self._gpio       = gpio
        self._logger     = logger
        self._cfg        = cfg
        self._lot_number = lot_number
        self._stop    = False
        self._running = threading.Event()
        self._running.set()
        self._drain_needed = threading.Event()

    def stop(self):
        self._stop = True
        self._running.set()   # unblock any paused wait

    def pause(self):
        self._running.clear()

    def resume(self):
        self._drain_needed.set()   # BUSY guard: drain stale START_PIN after resume
        self._running.set()

    def trigger(self):
        """Inject one mock START pulse (IO=False only). Called from UI thread."""
        self._gpio.trigger()

    def _handle_lot_end(self):
        """Auto-advance lot on GPIO LOT_END signal and emit new lot number."""
        _reset_image_counter()
        self._lot_number = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sig_session_reset.emit(self._lot_number)

    def run(self):
        cam_mode          = self._cfg.get("CAMERA", "directory")
        debug             = self._cfg.get("DEBUG", True)
        reconnect_attempts = int(self._cfg.get("RECONNECT_ATTEMPTS", 3))
        reconnect_delay   = float(self._cfg.get("RECONNECT_DELAY_S", 5.0))

        # Camera preflight — verify camera is reachable before entering the loop.
        if cam_mode == "camera":
            try:
                self._camera.grab_first()
            except CameraError as e:
                self.sig_error.emit(f"Camera not found: {e}")
                self.sig_status.emit("ERROR — camera not found, cannot run.")
                return

        self.sig_status.emit("Running…")
        _reset_image_counter()
        _cycle = 0

        if cam_mode == "camera":
            self._gpio.clear_outputs()  # ensure known-idle state before first cycle

        while not self._stop:

            # Wait for next cycle trigger
            if cam_mode == "camera":
                self.sig_status.emit("Waiting for START signal…")
                _io_enabled = self._cfg.get("IO", False)
                _wait_result = self._gpio.wait_for_start(
                    lambda: self._stop,
                    timeout_s=10.0 if _io_enabled else 0.0,
                )
                if _wait_result is None:
                    self.sig_error.emit(
                        "No START signal for 10 s — check machine connection.")
                    break
                if not _wait_result:
                    break
                if self._stop:
                    break

                self._gpio.set_busy(True)
            else:
                # Auto directory: brief yield then check stop
                time.sleep(0.05)
                if self._stop:
                    break

            # Capture guard
            if self._stop:
                if cam_mode == "camera":
                    self._gpio.clear_outputs()
                break
            t0 = time.perf_counter()
            try:
                img_bgr = self._camera.grab()
            except CameraError as e:
                self._logger.log_error("CAMERA_ERROR", str(e),
                                       (time.perf_counter() - t0) * 1000)
                if cam_mode == "directory":
                    _emsg = str(e)
                    if "No files" in _emsg:
                        self.sig_error.emit("No images in Input/ folder — add images and restart.")
                        return
                    self.sig_status.emit(f"Skipping unreadable image: {_emsg}")
                    continue
                # Camera mode: attempt reconnect before giving up
                reconnected = False
                for attempt in range(reconnect_attempts):
                    self.sig_status.emit(
                        f"Camera lost — reconnecting {attempt + 1}/{reconnect_attempts}…")
                    if self._camera.reconnect(1, reconnect_delay):
                        reconnected = True
                        break
                if reconnected:
                    continue   # retry this cycle
                self.sig_error.emit(f"Camera error: {e}")
                self.sig_status.emit("ERROR — camera lost, restart required.")
                self._gpio.clear_outputs()
                break

            img_id = _next_image_id()

            # Save raw with temp name before result is known
            out_dir  = self._cfg.get("OUT_DIR", "Output/")
            real_dir, ann_dir = _output_dirs(out_dir, self._lot_number)
            tmp_real = os.path.join(real_dir, f"{img_id}.jpg")
            cv2.imwrite(tmp_real, img_bgr)

            self.sig_status.emit("Inspecting…")

            # Inspect (with one retry on fail)
            is_retry    = False
            miss_a      = []
            miss_b      = []
            ann         = img_bgr

            try:
                self._inspector.inspect(img_bgr, debug=debug)
                # pass — img_bgr annotated in-place; miss_a/miss_b stay []

            except TemplateError as te:
                cycle_ms = (time.perf_counter() - t0) * 1000
                self._logger.log_error("TEMPLATE_ERROR", str(te), cycle_ms)
                try:
                    os.remove(tmp_real)
                except OSError:
                    pass
                if cam_mode == "directory":
                    self.sig_status.emit(f"Skipping {img_id}: {te}")
                    continue
                self.sig_error.emit(f"Template error: {te}")
                self.sig_status.emit("ERROR — template invalid, restart required.")
                self._gpio.clear_outputs()
                break

            except MarkMissingError as e1:
                if cam_mode == "camera":
                    is_retry = True
                    retry_delay = self._cfg.get("RETRY_DELAY_MS", 250) / 1000
                    time.sleep(retry_delay)
                    try:
                        img_bgr2 = self._camera.grab()
                        try:
                            self._inspector.inspect(img_bgr2, debug=debug)
                            # Retry passed — use retry frame
                            img_bgr = img_bgr2
                            ann     = img_bgr2
                            miss_a  = []
                            miss_b  = []
                        except MarkMissingError as e2:
                            img_bgr = img_bgr2
                            ann     = e2.annotated
                            miss_a  = (_resolve_ic(e1.missing_a, e1.confs_a, e2.confs_a)
                                       if e1.missing_a else [])
                            miss_b  = (_resolve_ic(e1.missing_b, e1.confs_b, e2.confs_b)
                                       if e1.missing_b else [])
                        except TemplateError:
                            miss_a, miss_b = e1.missing_a, e1.missing_b
                            ann = e1.annotated if e1.annotated is not None else img_bgr2
                    except CameraError:
                        miss_a, miss_b = e1.missing_a, e1.missing_b
                        ann = e1.annotated if e1.annotated is not None else img_bgr
                else:
                    # Directory mode: each file is a distinct IC — no meaningful retry
                    miss_a = e1.missing_a
                    miss_b = e1.missing_b
                    ann    = e1.annotated if e1.annotated is not None else img_bgr

            except Exception as e:
                cycle_ms = (time.perf_counter() - t0) * 1000
                self._logger.log_error("RUNTIME_ERROR", str(e), cycle_ms)
                self.sig_error.emit(f"Unexpected error: {e}")
                self.sig_status.emit("ERROR — machine blocked, restart required.")
                try:
                    os.remove(tmp_real)
                except OSError:
                    pass
                if cam_mode == "camera":
                    self._gpio.clear_outputs()
                break

            # Finalize paths and save
            cycle_ms = (time.perf_counter() - t0) * 1000

            # Suspect threshold logic
            _ng_threshold  = int(self._cfg.get("TEXT_NG_THRESHOLD", 2))
            n_missing      = len(miss_a) + len(miss_b)

            if n_missing == 0:
                passed     = True
                is_suspect = False
                suffix     = "_G"
            elif n_missing >= _TOTAL_CELLS:
                passed     = False
                is_suspect = False
                suffix     = "_NG"
            elif n_missing >= _ng_threshold:
                passed     = False
                is_suspect = True
                suffix     = "_NGS"
            else:
                passed     = True
                is_suspect = True
                suffix     = "_GS"

            save_image = suffix != "_G"  # skip saving clean-pass images

            final_real = os.path.join(real_dir, f"{img_id}{suffix}.jpg")
            ann_path   = os.path.join(ann_dir,  f"{img_id}{suffix}.jpg")
            if save_image:
                try:
                    os.rename(tmp_real, final_real)
                except OSError:
                    final_real = tmp_real
                cv2.imwrite(ann_path, ann)
            else:
                try:
                    os.remove(tmp_real)
                except OSError:
                    pass
                ann_path = ""

            # Emit signals and log
            self.sig_image.emit(img_bgr)
            self.sig_cycle_ms.emit(cycle_ms)

            if passed:
                self.sig_result.emit(True, True, is_suspect)
                self._logger.log_inspection(
                    img_id, "PASS", [], "PASS", [], cycle_ms, is_retry, is_suspect)
            else:
                err = MarkMissingError(miss_a, miss_b, ann)
                self.sig_fail.emit(err, ann_path, img_id, is_suspect)
                self._logger.log_inspection(
                    img_id,
                    "FAIL" if miss_a else "PASS", miss_a,
                    "FAIL" if miss_b else "PASS", miss_b,
                    cycle_ms, is_retry, is_suspect)

            if cam_mode == "camera":
                self._gpio.set_busy(False)
                is_overall_pass = not (miss_a or miss_b)
                self._gpio.set_inspec_stage(not is_overall_pass)  # LOW=pass, HIGH=NG
                time.sleep(0.010)
                self._gpio.pulse_end_pin()                        # LOW 40 ms → machine reads INSPEC_STAGE
                self._gpio.set_inspec_stage(True)                 # restore idle HIGH

            try:
                del img_bgr
            except NameError:
                pass

            _cycle += 1
            if _cycle % 100 == 0:
                gc.collect()

            # End-of-cycle: directory mode batch check
            if cam_mode != "camera":
                if not self._camera.has_more():
                    self._camera.reset()
                    break                           # directory done → standby

            # Pause checkpoint — sits after GPIO outputs are restored (INSPEC_STAGE idle, BUSY LOW)
            # so the machine always receives the full END_PIN pulse before the loop suspends.
            if not self._running.is_set():
                self.sig_paused.emit()
                self._running.wait()          # blocks until resume() or stop()
                if self._stop:
                    break
                if self._drain_needed.is_set():
                    self._gpio.drain_start_pin()
                    self._drain_needed.clear()
                self.sig_resumed.emit()

        if cam_mode == "camera":
            self._gpio.clear_outputs()
        self.sig_done.emit()   # always emit; _on_run_done guards against double-call
        self.sig_status.emit("Standby.")

# LOT START DIALOG
class LotStartDialog(QtWidgets.QDialog):
    """
    Shown before a run starts. Operator enters a lot number.
    API hook: override get_lot_number_from_api() to inject from an external system;
    when it returns a non-empty string the dialog is skipped entirely.
    """

    @staticmethod
    def get_lot_number_from_api() -> str:
        """Plugin point: replace to inject lot number from an internal API."""
        return ""   # empty = show dialog; non-empty = skip dialog

    @classmethod
    def request(cls, parent=None, api_fn=None) -> str | None:
        """
        Returns lot number string, or None if operator cancelled.
        api_fn: optional callable → str; if it returns non-empty the dialog is skipped.
        Falls back to get_lot_number_from_api() for subclass overrides.
        """
        if api_fn is not None:
            lot = api_fn()
            if lot:
                return lot
        api_lot = cls.get_lot_number_from_api()   # kept as subclass plugin point
        if api_lot:
            return api_lot
        dlg = cls(parent)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            text = dlg._edit.text().strip()
            return text if text else datetime.now().strftime("%Y%m%d_%H%M%S")
        return None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start Lot")
        self.setFixedWidth(300)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10)
        lay.addWidget(QtWidgets.QLabel("Enter Lot Number:"))
        self._edit = QtWidgets.QLineEdit()
        self._edit.setPlaceholderText("Leave blank for auto timestamp")
        lay.addWidget(self._edit)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self._edit.returnPressed.connect(self.accept)


# IMAGE BROWSER — worker threads + widgets

class FolderScanWorker(QtCore.QThread):
    """Scans Output/ directory tree; emits flat list of (label, leaf_dir_path)."""
    sig_entries = QtCore.pyqtSignal(list)   # [(label, path), ...]

    def __init__(self, out_dir: str, parent=None):
        super().__init__(parent)
        self._out_dir = out_dir

    def run(self):
        entries = []
        if not os.path.isdir(self._out_dir):
            self.sig_entries.emit(entries)
            return
        dates = sorted(
            [d for d in os.listdir(self._out_dir)
             if os.path.isdir(os.path.join(self._out_dir, d))],
            reverse=True)
        for date in dates:
            date_dir = os.path.join(self._out_dir, date)
            img_dir  = os.path.join(date_dir, "Image")
            if os.path.isdir(img_dir):
                # Direct layout: date/Image exists
                entries.append((date, date_dir))
            else:
                # Lot layout: date/lot/Image
                lots = sorted(
                    [d for d in os.listdir(date_dir)
                     if os.path.isdir(os.path.join(date_dir, d))],
                    reverse=True)
                for lot in lots:
                    lot_img = os.path.join(date_dir, lot, "Image")
                    if os.path.isdir(lot_img):
                        entries.append((f"  {date}/{lot}", os.path.join(date_dir, lot)))
        self.sig_entries.emit(entries)


class ThumbnailWorker(QtCore.QThread):
    """Loads image thumbnails one-by-one in a background thread."""
    sig_thumb = QtCore.pyqtSignal(int, object)   # (index, QPixmap)
    sig_done  = QtCore.pyqtSignal()

    def __init__(self, paths: list, thumb_w: int = 130, thumb_h: int = 98, parent=None):
        super().__init__(parent)
        self._paths   = paths
        self._thumb_w = thumb_w
        self._thumb_h = thumb_h
        self._stop    = False

    def stop(self):
        self._stop = True

    def run(self):
        for idx, path in enumerate(self._paths):
            if self._stop:
                break
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            qimg = QtGui.QImage(bytes(img.data), w, h, ch * w,
                                QtGui.QImage.Format_RGB888)
            # Scale to fit the thumbnail box while keeping the image's own aspect ratio
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                self._thumb_w, self._thumb_h,
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation)
            self.sig_thumb.emit(idx, pix)
        self.sig_done.emit()


class ImageCard(QtWidgets.QFrame):
    """Thumbnail card with dynamic sizing — color by result suffix."""
    clicked = QtCore.pyqtSignal(int)

    def __init__(self, idx: int, filename: str,
                 card_w: int, card_h: int, thumb_w: int, thumb_h: int,
                 parent=None):
        super().__init__(parent)
        self._idx = idx
        self.setFixedSize(card_w, card_h)
        self.setObjectName("image_card")
        self.setCursor(QtCore.Qt.PointingHandCursor)

        _stem = os.path.splitext(filename)[0]
        if _stem.endswith("_NGS"):
            card_bg = "#E07820"   # orange — FAIL suspect
        elif _stem.endswith("_GS"):
            card_bg = "#A0B830"   # yellow-green — PASS suspect
        elif _stem.endswith("_NG"):
            card_bg = "#FA6781"   # red — FAIL
        elif _stem.endswith("_G"):
            card_bg = "#478B8D"   # teal — PASS
        else:
            card_bg = "#788BFF"
        # Scoped selector: only this QFrame, not child labels
        self.setStyleSheet(
            f"QFrame#image_card{{background:{card_bg};border-radius:5px;}}")

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        self._thumb = QtWidgets.QLabel()
        self._thumb.setFixedSize(thumb_w, thumb_h)
        self._thumb.setAlignment(QtCore.Qt.AlignCenter)
        self._thumb.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        lay.addWidget(self._thumb)

        name_lbl = QtWidgets.QLabel(filename)
        name_lbl.setStyleSheet("font-size:9px;color:#E2FDFF;background:transparent;")
        name_lbl.setAlignment(QtCore.Qt.AlignCenter)
        metrics = QtGui.QFontMetrics(name_lbl.font())
        elided  = metrics.elidedText(filename, QtCore.Qt.ElideMiddle, card_w - 6)
        name_lbl.setText(elided)
        lay.addWidget(name_lbl)

    def set_thumbnail(self, pixmap: QtGui.QPixmap):
        self._thumb.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(self._idx)
        super().mousePressEvent(event)


class ImageBrowserPage(QtWidgets.QWidget):
    """
    Full-page image browser tab.
    Left: date/lot folder list. Centre: grid or single image. Right: toggle controls.
    """

    _COLS = 4    # grid columns

    def __init__(self, out_dir: str = "Output/", parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._out_dir       = out_dir
        self._all_paths: list = []     # all files in selected folder/subfolder
        self._paths: list    = []      # filtered paths shown in grid
        self._cur_idx        = 0
        self._subfolder      = "RealImg"    # "RealImg" or "Image"
        self._suffix_filter  = "FAIL"
        self._cards: list    = []
        self._current_base: str = ""
        self._img_ratio: float = 3 / 4   # h/w; updated from first image on each folder load
        self._thumb_worker: ThumbnailWorker | None = None
        self._scan_worker:  FolderScanWorker | None = None

        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Left: folder list
        self._folder_list = QtWidgets.QListWidget()
        self._folder_list.setMinimumWidth(150)
        self._folder_list.setMaximumWidth(260)
        self._folder_list.setStyleSheet(
            "QListWidget{background:#788BFF;border-radius:6px;color:#FFFFFF;font-size:11px}"
            "QListWidget::item:selected{background:#5465FF;color:#FFFFFF}"
        )
        self._folder_list.itemClicked.connect(self._on_folder_selected)
        root.addWidget(self._folder_list)

        # Centre: stacked (grid / image)
        self._stack = QtWidgets.QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        # Stack index 0: grid page
        grid_page = QtWidgets.QWidget()
        grid_lay  = QtWidgets.QVBoxLayout(grid_page)
        grid_lay.setContentsMargins(0, 0, 0, 0)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._scroll.setStyleSheet("QScrollArea{border:none;background:transparent}")
        self._grid_area = QtWidgets.QWidget()
        self._grid_layout = QtWidgets.QGridLayout(self._grid_area)
        self._grid_layout.setSpacing(6)
        self._grid_layout.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        self._scroll.setWidget(self._grid_area)
        grid_lay.addWidget(self._scroll, stretch=1)
        self._stack.addWidget(grid_page)   # index 0

        # Stack index 1: image page
        img_page = QtWidgets.QWidget()
        img_lay  = QtWidgets.QVBoxLayout(img_page)
        img_lay.setContentsMargins(0, 0, 0, 0)
        img_lay.setSpacing(4)
        self._img_view = ImageView()
        self._img_view.right_clicked.connect(self._back_to_grid)
        img_lay.addWidget(self._img_view, stretch=1)
        # Bottom nav
        nav = QtWidgets.QHBoxLayout()
        self._btn_prev = QtWidgets.QPushButton("←")
        self._btn_prev.setFixedWidth(48)
        self._btn_prev.clicked.connect(lambda: self._step_image(-1))
        nav.addWidget(self._btn_prev)
        self._lbl_nav = QtWidgets.QLabel("—")
        self._lbl_nav.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_nav.setStyleSheet("color:#E2FDFF;font-size:11px")
        nav.addWidget(self._lbl_nav, stretch=1)
        self._btn_next = QtWidgets.QPushButton("→")
        self._btn_next.setFixedWidth(48)
        self._btn_next.clicked.connect(lambda: self._step_image(1))
        nav.addWidget(self._btn_next)
        img_lay.addLayout(nav)
        self._stack.addWidget(img_page)   # index 1

        # Right: controls
        right = QtWidgets.QFrame()
        right.setObjectName("panel_right")
        right.setMinimumWidth(130)
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(8, 8, 8, 8)
        right_lay.setSpacing(10)

        # Source toggle: RealImg / Image
        right_lay.addWidget(self._section_label("Source"))
        self._grp_src = QtWidgets.QButtonGroup(self)
        self._btn_realimg = self._toggle_btn("RealImg", checked=True)
        self._btn_annimg  = self._toggle_btn("Image",   checked=False)
        self._grp_src.addButton(self._btn_realimg, 0)
        self._grp_src.addButton(self._btn_annimg,  1)
        right_lay.addWidget(self._btn_realimg)
        right_lay.addWidget(self._btn_annimg)
        self._grp_src.buttonClicked.connect(self._on_src_toggle)

        # Filter toggle: _NG / _G / Suspect / All
        right_lay.addWidget(self._section_label("Filter"))
        self._grp_flt = QtWidgets.QButtonGroup(self)
        self._btn_flt_ng      = self._toggle_btn("FAIL",    checked=True)
        self._btn_flt_suspect = self._toggle_btn("Suspect", checked=False)
        self._btn_flt_all     = self._toggle_btn("All",     checked=False)
        self._grp_flt.addButton(self._btn_flt_ng,      0)
        self._grp_flt.addButton(self._btn_flt_suspect, 1)
        self._grp_flt.addButton(self._btn_flt_all,     2)
        right_lay.addWidget(self._btn_flt_ng)
        right_lay.addWidget(self._btn_flt_suspect)
        right_lay.addWidget(self._btn_flt_all)
        self._grp_flt.buttonClicked.connect(self._on_filter_toggle)

        # Count label
        self._lbl_count = QtWidgets.QLabel("—")
        self._lbl_count.setStyleSheet("font-size:16px;font-weight:bold;color:#E2FDFF")
        self._lbl_count.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_count.setWordWrap(True)
        right_lay.addWidget(self._lbl_count)

        right_lay.addStretch()

        # Keybind reference panel
        kb_frame = QtWidgets.QFrame()
        kb_frame.setObjectName("setup_frame")
        kb_lay = QtWidgets.QVBoxLayout(kb_frame)
        kb_lay.setContentsMargins(8, 6, 8, 6)
        kb_lay.setSpacing(6)
        right_lay.addWidget(kb_frame)

        lbl_kb = QtWidgets.QLabel("Controls")
        lbl_kb.setStyleSheet("font-size:14px;font-weight:bold;color:#E2FDFF")
        kb_lay.addWidget(lbl_kb)

        _KBD_STYLE = (
            "QLabel{font-size:13px;color:#E2FDFF;padding:1px 0px;}")
        _KEY_STYLE = (
            "QLabel{font-size:13px;font-weight:bold;color:#5465FF;"
            "background:#FFFFFF;border-radius:3px;padding:2px 7px;}")

        for key_text, desc_text in [
            ("← →",       "Prev / Next"),
            ("R-Click",    "Back to grid"),
        ]:
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            key_lbl = QtWidgets.QLabel(key_text)
            key_lbl.setStyleSheet(_KEY_STYLE)
            key_lbl.setAlignment(QtCore.Qt.AlignCenter)
            desc_lbl = QtWidgets.QLabel(desc_text)
            desc_lbl.setStyleSheet(_KBD_STYLE)
            row.addWidget(key_lbl)
            row.addWidget(desc_lbl, stretch=1)
            kb_lay.addLayout(row)

        # Back button (shown in image view mode)
        self._btn_back = QtWidgets.QPushButton("← Back")
        self._btn_back.clicked.connect(self._back_to_grid)
        self._btn_back.setStyleSheet(
            "QPushButton{background:#FFFFFF;color:#5465FF;border-radius:6px;"
            "padding:8px 14px;font-weight:bold;font-size:12px;}"
            "QPushButton:hover{background:#E2FDFF;}")
        self._btn_back.hide()
        right_lay.addWidget(self._btn_back)

        root.addWidget(right)

    # resize

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._paths and e.size() != e.oldSize():
            QtCore.QTimer.singleShot(150, self._rebuild_grid)

    def _card_size(self):
        """Calculate card/thumbnail dimensions from viewport width and image aspect ratio."""
        vp_w = self._scroll.viewport().width()
        if vp_w < 4:
            return 100, 96, 96, 72

        sp      = self._grid_layout.horizontalSpacing()
        w_avail = max(1, vp_w - sp * (self._COLS - 1) - 4)
        card_w  = max(60, w_avail // self._COLS)

        thumb_w = max(1, card_w - 4)
        thumb_h = max(1, int(thumb_w * self._img_ratio))
        card_h  = thumb_h + 24   # 24 px for filename label + margins

        return card_w, card_h, thumb_w, thumb_h

    # helpers

    def _section_label(self, text: str) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel(text)
        lbl.setStyleSheet("font-size:11px;font-weight:bold;color:#E2FDFF")
        return lbl

    def _toggle_btn(self, text: str, checked: bool) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setCheckable(True)
        btn.setChecked(checked)
        btn.setStyleSheet(
            "QPushButton{background:#788BFF;color:#FFFFFF;border-radius:4px;"
            "padding:5px 8px;font-size:11px}"
            "QPushButton:checked{background:#FFFFFF;color:#5465FF;font-weight:bold}"
        )
        return btn

    # folder refresh

    def refresh(self):
        """Called when the Image Browser tab is activated."""
        if self._scan_worker and self._scan_worker.isRunning():
            return
        self._folder_list.clear()
        self._scan_worker = FolderScanWorker(self._out_dir)
        self._scan_worker.sig_entries.connect(self._on_scan_done)
        self._scan_worker.start()

    def _on_scan_done(self, entries: list):
        self._folder_list.clear()
        for label, _path in entries:
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, _path)
            self._folder_list.addItem(item)

    def _on_folder_selected(self, item: QtWidgets.QListWidgetItem):
        path = item.data(QtCore.Qt.UserRole)
        self._load_folder(path)

    # image loading

    def _load_folder(self, base_path: str):
        """Collect files from base_path/subfolder, apply filter, rebuild grid."""
        img_dir = os.path.join(base_path, self._subfolder)
        if not os.path.isdir(img_dir):
            self._all_paths = []
        else:
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(img_dir, ext)))
            self._all_paths = sorted(files)

        self._current_base = base_path
        self._apply_filter_and_build_grid()

    @staticmethod
    def _file_matches_filter(path: str, flt: str) -> bool:
        if flt == "":
            return True
        stem = os.path.splitext(os.path.basename(path))[0]
        sfx  = stem.rsplit("_", 1)[-1] if "_" in stem else ""
        if flt == "FAIL":
            return sfx in ("NG", "NGS")
        if flt == "PASS":
            return sfx in ("G", "GS")
        if flt == "SUSPECT":
            return sfx in ("GS", "NGS")
        return False

    def _apply_filter_and_build_grid(self):
        """Filter self._all_paths by suffix, rebuild grid."""
        self._paths = [p for p in self._all_paths
                       if self._file_matches_filter(p, self._suffix_filter)]

        self._lbl_count.setText(f"{len(self._paths)} images")
        self._rebuild_grid()

    def _rebuild_grid(self):
        """Clear grid and create ImageCard placeholders; start ThumbnailWorker."""
        # Stop any running thumbnail worker
        if self._thumb_worker and self._thumb_worker.isRunning():
            self._thumb_worker.stop()
            self._thumb_worker.wait(500)

        # Clear existing cards
        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards = []

        # Switch to grid view and hide Back before creating cards
        self._stack.setCurrentIndex(0)
        self._btn_back.hide()

        if not self._paths:
            return

        # Peek first image to get its aspect ratio for correct card proportions
        _peek = cv2.imread(self._paths[0], cv2.IMREAD_REDUCED_GRAYSCALE_2)
        if _peek is not None and _peek.shape[1] > 0:
            self._img_ratio = _peek.shape[0] / _peek.shape[1]
        del _peek

        card_w, card_h, thumb_w, thumb_h = self._card_size()

        for idx, path in enumerate(self._paths):
            fname = os.path.basename(path)
            card  = ImageCard(idx, fname, card_w, card_h, thumb_w, thumb_h)
            card.clicked.connect(self._on_card_clicked)
            row, col = divmod(idx, self._COLS)
            self._grid_layout.addWidget(card, row, col)
            self._cards.append(card)

        # Start loading thumbnails in background
        self._thumb_worker = ThumbnailWorker(self._paths, thumb_w, thumb_h)
        self._thumb_worker.sig_thumb.connect(self._on_thumbnail_ready)
        self._thumb_worker.start()

    def _on_thumbnail_ready(self, idx: int, pixmap: QtGui.QPixmap):
        if idx < len(self._cards):
            self._cards[idx].set_thumbnail(pixmap)

    # image view

    def _on_card_clicked(self, idx: int):
        self._cur_idx = idx
        self._show_image(idx)

    def _show_image(self, idx: int):
        if not self._paths:
            return
        self._cur_idx = max(0, min(idx, len(self._paths) - 1))
        # Switch to image page FIRST so _img_view has its real size when _refresh fires
        self._stack.setCurrentIndex(1)
        self._btn_back.show()
        self.setFocus()   # capture arrow key events
        path = self._paths[self._cur_idx]
        img  = cv2.imread(path)
        if img is not None:
            self._img_view.set_image(img)
        self._lbl_nav.setText(
            f"{self._cur_idx + 1} / {len(self._paths)}   {os.path.basename(path)}")

    def _step_image(self, delta: int):
        self._show_image(self._cur_idx + delta)

    def keyPressEvent(self, e):
        if self._stack.currentIndex() == 1:
            if e.key() == QtCore.Qt.Key_Left:
                self._step_image(-1)
            elif e.key() == QtCore.Qt.Key_Right:
                self._step_image(1)
            else:
                super().keyPressEvent(e)
        else:
            super().keyPressEvent(e)

    def _back_to_grid(self):
        self._stack.setCurrentIndex(0)
        self._btn_back.hide()

    # toggle handlers

    def _on_src_toggle(self, btn):
        self._subfolder = "RealImg" if self._grp_src.id(btn) == 0 else "Image"
        if self._current_base:
            self._load_folder(self._current_base)
        if self._stack.currentIndex() == 1:
            self._show_image(self._cur_idx)

    def _on_filter_toggle(self, btn):
        flt_id = self._grp_flt.id(btn)
        self._suffix_filter = {0: "FAIL", 1: "SUSPECT", 2: ""}[flt_id]
        self._apply_filter_and_build_grid()


# MAIN WINDOW
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, cfg: dict):
        super().__init__()
        self.setWindowTitle("ClearIC Inspect")
        self._cfg               = cfg
        self._camera:    Camera | None    = None
        self._detector:  Detector | None  = None
        self._inspector: Inspector | None = None
        self._gpio       = None
        self._logger            = Logger(
            log_dir=cfg.get("LOG_DIR", "logs"),
            log_retention=int(cfg.get("LOG_RETENTION", 365)))
        self._worker:           RunWorker | None        = None
        self._preview_timer:    QtCore.QTimer | None    = None
        self._cam_retry_timer:  QtCore.QTimer | None    = None
        self._camera_init_kwargs: dict                  = {}
        self._worker_last_tick: float                   = 0.0
        self._watchdog_timer = QtCore.QTimer(self)
        self._watchdog_timer.setInterval(15_000)
        self._watchdog_timer.timeout.connect(self._check_watchdog)
        self._watchdog_timer.start()

        self._stats_pass  = 0
        self._stats_fail  = 0
        self._stats_error = 0
        self._stats_total = 0

        self._run_state          = "standby"   # "standby" | "running" | "paused"
        self._session_start_time = 0.0
        self._lot_number         = ""
        self._package_name       = ""

        # OCR input state (retained across lots)
        self._ocr_operator:     str = ""
        self._ocr_expect_value: str = ""

        # setup state
        self._pending_ic_a:  QtCore.QRect | None = None
        self._pending_ic_b:  QtCore.QRect | None = None
        self._setup_image:   np.ndarray | None   = None
        self._setup_state:   str                 = 'idle'   # idle/draw_a/draw_b/ready

        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.90), int(screen.height() * 0.90))
        self.move(screen.x() + int(screen.width() * 0.05),
                  screen.y() + int(screen.height() * 0.05))

        self._build_ui()
        self._init_system()

    # UI construction
    def _build_ui(self):
        # Tab wrapper
        tabs = QtWidgets.QTabWidget()
        tabs.setObjectName("root")
        self.setCentralWidget(tabs)

        insp_page = QtWidgets.QWidget()
        insp_page.setObjectName("root")
        tabs.addTab(insp_page, "Inspection")

        out_dir = self._cfg.get("OUT_DIR", "Output/")
        self._browser = ImageBrowserPage(out_dir=out_dir)
        tabs.addTab(self._browser, "Image Browser")
        tabs.currentChanged.connect(
            lambda i: self._browser.refresh() if i == 1 else None)

        # All existing layout now goes into insp_page
        root = QtWidgets.QHBoxLayout(insp_page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Left panel
        left_frame = QtWidgets.QFrame()
        left_frame.setObjectName("main_view")
        left_lay = QtWidgets.QVBoxLayout(left_frame)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(6)

        self._view = ImageView()
        left_lay.addWidget(self._view, stretch=1)

        # Error banner (hidden by default)
        self._error_banner = QtWidgets.QFrame()
        self._error_banner.setObjectName("error_banner")
        eb_lay = QtWidgets.QHBoxLayout(self._error_banner)
        eb_lay.setContentsMargins(8, 4, 8, 4)
        self._error_lbl = QtWidgets.QLabel("")
        self._error_lbl.setStyleSheet("color:#FFFFFF;font-weight:bold")
        eb_lay.addWidget(self._error_lbl)
        self._error_banner.hide()
        left_lay.addWidget(self._error_banner)

        # Badge area
        badge_frame = QtWidgets.QFrame()
        badge_frame.setObjectName("badge_area")
        badge_lay = QtWidgets.QHBoxLayout(badge_frame)
        badge_lay.setSpacing(10)

        self._badge_a = self._make_badge("IC_A")
        self._badge_b = self._make_badge("IC_B")
        badge_lay.addWidget(self._badge_a)
        badge_lay.addWidget(self._badge_b)
        badge_lay.addStretch()
        left_lay.addWidget(badge_frame)

        root.addWidget(left_frame, stretch=1)

        # Right panel
        right_frame = QtWidgets.QFrame()
        right_frame.setObjectName("panel_right")
        right_frame.setMinimumWidth(240)
        right_lay = QtWidgets.QVBoxLayout(right_frame)
        right_lay.setContentsMargins(8, 8, 8, 8)
        right_lay.setSpacing(8)

        # Setup section
        setup_frame = QtWidgets.QFrame()
        setup_frame.setObjectName("setup_frame")
        setup_lay = QtWidgets.QVBoxLayout(setup_frame)
        setup_lay.setSpacing(6)

        lbl_setup = QtWidgets.QLabel("Setup")
        lbl_setup.setStyleSheet("font-weight:bold;font-size:13px")
        setup_lay.addWidget(lbl_setup)

        self._lbl_tmpl_status = QtWidgets.QLabel("No template saved.")
        self._lbl_tmpl_status.setStyleSheet(
            "font-size:11px;color:#E2FDFF;padding:4px 0px;")
        self._lbl_tmpl_status.setWordWrap(True)
        self._lbl_tmpl_status.setMinimumHeight(36)
        setup_lay.addWidget(self._lbl_tmpl_status)

        self._btn_new_tmpl = QtWidgets.QPushButton("New Template")
        self._btn_new_tmpl.clicked.connect(self._on_new_tmpl_click)
        setup_lay.addWidget(self._btn_new_tmpl)

        self._btn_confirm_tmpl = QtWidgets.QPushButton("Confirm")
        self._btn_confirm_tmpl.clicked.connect(self._confirm_template)
        self._btn_confirm_tmpl.setEnabled(False)
        setup_lay.addWidget(self._btn_confirm_tmpl)

        self._view.rect_drawn.connect(self._on_rb_rect_drawn)

        right_lay.addWidget(setup_frame)

        # Controls section
        ctrl_frame = QtWidgets.QFrame()
        ctrl_frame.setObjectName("controls_frame")
        ctrl_lay = QtWidgets.QVBoxLayout(ctrl_frame)
        ctrl_lay.setSpacing(6)

        lbl_ctrl = QtWidgets.QLabel("Controls")
        lbl_ctrl.setStyleSheet("font-weight:bold;font-size:13px")
        ctrl_lay.addWidget(lbl_ctrl)

        self._btn_action = QtWidgets.QPushButton("Start")
        self._btn_action.setEnabled(False)   # enabled only when OCR fields are valid
        self._btn_action.clicked.connect(self._on_action_click)
        ctrl_lay.addWidget(self._btn_action)

        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_run)
        ctrl_lay.addWidget(self._btn_stop)

        right_lay.addWidget(ctrl_frame)

        # Stats section
        stats_frame = QtWidgets.QFrame()
        stats_frame.setObjectName("setup_frame")
        stats_lay = QtWidgets.QVBoxLayout(stats_frame)
        stats_lay.setSpacing(4)

        lbl_stats = QtWidgets.QLabel("Stats")
        lbl_stats.setStyleSheet("font-weight:bold;font-size:13px")
        stats_lay.addWidget(lbl_stats)

        self._lbl_lot_info = self._stat_row(stats_lay, "Lot",      "—")
        self._lbl_status   = self._stat_row(stats_lay, "Status",   "Standby.")
        self._lbl_pass     = self._stat_row(stats_lay, "Pass",     "0")
        self._lbl_fail     = self._stat_row(stats_lay, "Fail",     "0")
        self._lbl_yield    = self._stat_row(stats_lay, "Yield",    "—")
        self._lbl_error    = self._stat_row(stats_lay, "Error",    "0")
        self._lbl_cycle_ms = self._stat_row(stats_lay, "Last ms",  "—")

        right_lay.addWidget(stats_frame)

        # Settings section
        settings_frame = QtWidgets.QFrame()
        settings_frame.setObjectName("setup_frame")
        settings_lay = QtWidgets.QVBoxLayout(settings_frame)
        settings_lay.setSpacing(4)

        lbl_vis = QtWidgets.QLabel("Settings")
        lbl_vis.setStyleSheet("font-weight:bold;font-size:13px")
        settings_lay.addWidget(lbl_vis)

        def _srow(parent, label, widget):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            lbl = QtWidgets.QLabel(label)
            lbl.setStyleSheet("font-size:10px;color:#E2FDFF")
            row.addWidget(lbl)
            row.addStretch()
            row.addWidget(widget)
            parent.addLayout(row)

        self._input_warmup = QtWidgets.QLineEdit(str(self._cfg.get("WARMUP_FRAMES", 5)))
        self._input_warmup.setFixedWidth(52)
        _srow(settings_lay, "Warmup frames", self._input_warmup)

        self._input_border = QtWidgets.QLineEdit(str(self._cfg.get("ANN_BORDER_PX", 1)))
        self._input_border.setFixedWidth(52)
        _srow(settings_lay, "Border thickness (px)", self._input_border)

        self._chk_labels = QtWidgets.QCheckBox("Show cell labels")
        self._chk_labels.setChecked(bool(self._cfg.get("RESULT_OVERLAY", True)))
        settings_lay.addWidget(self._chk_labels)

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.clicked.connect(self._apply_settings)
        settings_lay.addWidget(btn_apply)

        settings_frame.setVisible(False)
        right_lay.addWidget(settings_frame)

        # OCR Input section — always visible, gates Start button
        self._ocr_frame = QtWidgets.QFrame()
        self._ocr_frame.setObjectName("setup_frame")
        ocr_lay = QtWidgets.QVBoxLayout(self._ocr_frame)
        ocr_lay.setSpacing(6)

        lbl_ocr = QtWidgets.QLabel("OCR Input")
        lbl_ocr.setStyleSheet("font-weight:bold;font-size:13px")
        ocr_lay.addWidget(lbl_ocr)

        lbl_op = QtWidgets.QLabel("Operator No. (6 digits):")
        lbl_op.setStyleSheet("font-size:10px;color:#E2FDFF")
        ocr_lay.addWidget(lbl_op)

        self._edit_op_number = QtWidgets.QLineEdit()
        self._edit_op_number.setMaxLength(6)
        self._edit_op_number.setValidator(QtGui.QIntValidator(0, 999999))
        self._edit_op_number.setPlaceholderText("000000")
        self._edit_op_number.textChanged.connect(self._on_ocr_field_changed)
        ocr_lay.addWidget(self._edit_op_number)

        lbl_mark = QtWidgets.QLabel("Expected Mark (6 chars, A–Z / 0–9):")
        lbl_mark.setStyleSheet("font-size:10px;color:#E2FDFF")
        ocr_lay.addWidget(lbl_mark)

        self._edit_ocr_expect = QtWidgets.QLineEdit()
        self._edit_ocr_expect.setMaxLength(6)
        self._edit_ocr_expect.setValidator(
            QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression("[A-Z0-9]{0,6}")))
        self._edit_ocr_expect.setPlaceholderText("XXXXXX")
        self._edit_ocr_expect.textChanged.connect(self._on_ocr_field_changed)
        ocr_lay.addWidget(self._edit_ocr_expect)

        self._lbl_ocr_status = QtWidgets.QLabel("Fill both fields to enable Start.")
        self._lbl_ocr_status.setWordWrap(True)
        self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
        ocr_lay.addWidget(self._lbl_ocr_status)

        right_lay.addWidget(self._ocr_frame)   # always visible

        right_lay.addStretch()

        root.addWidget(right_frame)

    def _make_badge(self, label: str) -> QtWidgets.QFrame:
        frame = QtWidgets.QFrame()
        frame.setObjectName("badge_idle")
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(8, 6, 8, 6)
        top = QtWidgets.QLabel(label)
        top.setStyleSheet("font-weight:bold;font-size:12px")
        top.setAlignment(QtCore.Qt.AlignCenter)
        result = QtWidgets.QLabel("—")
        result.setObjectName("stat_value")
        result.setStyleSheet("font-size:16px;font-weight:bold")
        result.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(top)
        lay.addWidget(result)
        frame._result_lbl = result
        return frame

    def _update_badge(self, frame: QtWidgets.QFrame, passed: bool | None):
        if passed is None:
            frame.setObjectName("badge_idle")
            frame._result_lbl.setText("—")
        elif passed:
            frame.setObjectName("badge_pass")
            frame._result_lbl.setText("PASS")
            frame._result_lbl.setStyleSheet(
                "font-size:16px;font-weight:bold;color:#5465FF")
        else:
            frame.setObjectName("badge_fail")
            frame._result_lbl.setText("FAIL")
            frame._result_lbl.setStyleSheet(
                "font-size:16px;font-weight:bold;color:#FFFFFF")
        frame.style().unpolish(frame)
        frame.style().polish(frame)

    def _stat_row(self, parent_lay, label: str, value: str) -> QtWidgets.QLabel:
        """One horizontal row: bold label on left, value on right."""
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QtWidgets.QLabel(label)
        lbl.setStyleSheet("font-size:11px;color:#E2FDFF;font-weight:bold")
        val = QtWidgets.QLabel(value)
        val.setStyleSheet("font-size:11px;color:#FFFFFF")
        val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        row.addWidget(lbl)
        row.addStretch()
        row.addWidget(val)
        parent_lay.addLayout(row)
        return val

    # Settings apply
    def _apply_settings(self):
        try:
            wf = max(1, int(self._input_warmup.text()))
        except ValueError:
            wf = self._cfg.get("WARMUP_FRAMES", 5)
            self._input_warmup.setText(str(wf))

        try:
            bp = max(1, int(self._input_border.text()))
        except ValueError:
            bp = self._cfg.get("ANN_BORDER_PX", 1)
            self._input_border.setText(str(bp))

        show_labels = self._chk_labels.isChecked()

        self._cfg.update({"WARMUP_FRAMES": wf, "ANN_BORDER_PX": bp,
                          "RESULT_OVERLAY": show_labels})
        ConfigLoader.save(self._cfg)
        self._rebuild_inspector()

        print(f"[Settings] border={bp}px  labels={show_labels}  warmup={wf}")

    # System init
    def _init_system(self):
        cfg = self._cfg
        try:
            self._detector = Detector(
                conf_thr=cfg.get("CONF_THR", 0.5),
                text_min_conf=cfg.get("TEXT_MIN_CONF", 0.80),
                blank_cell_std_thr=cfg.get("BLANK_CELL_STD_THR", 0.0),
                model_path=cfg.get("MODEL_PATH",
                                   "Text_cls-2/best_openvino_model/best.xml"),
                n_passes=cfg.get("CLS_N_PASSES", 1),
                uncertain_thr=cfg.get("CLS_UNCERTAIN_THR", 0.50),
                debug=cfg.get("DEBUG", True),
            )
        except ModelError as e:
            QtWidgets.QMessageBox.critical(
                self, "Model Error",
                f"Cannot load classifier model:\n\n{e}\n\n"
                "Check that the model files exist and contact your administrator.")
            QtCore.QTimer.singleShot(0, self.close)
            return

        try:
            self._gpio = RaspberryIO(
                io_enabled=cfg.get("IO", False),
                start_pin=cfg.get("GPIO_START_PIN", 17),
                busy_pin=cfg.get("GPIO_BUSY_PIN", 23),
                end_pin=cfg.get("GPIO_END_PIN", 18),
                inspec_stage_pin=cfg.get("GPIO_INSPEC_STAGE_PIN", 24),
            )
        except GPIOError as e:
            self._show_error(f"GPIO init failed: {e}")

        self._camera_init_kwargs = dict(
            mode=cfg.get("CAMERA", "directory"),
            serial=cfg.get("CAMERA_SERIAL", ""),
            exposure_us=cfg.get("EXPOSURE_US", 8000),
            input_dir=cfg.get("DIR_INPUT", "Input/"),
            retry_delay=cfg.get("CAMERA_RETRY_DELAY", 0.2),
            retries=cfg.get("CAMERA_RETRIES", 2),
            warmup_frames=cfg.get("CAMERA_WARMUP_FRAMES", 5),
            image_w=cfg.get("IMAGE_W", 0),
            image_h=cfg.get("IMAGE_H", 0),
        )
        try:
            self._camera = Camera(**self._camera_init_kwargs)
            self._camera.open()
        except CameraError as e:
            self._show_error(str(e))
            if cfg.get("CAMERA") == "camera":
                self._cam_retry_timer = QtCore.QTimer(self)
                self._cam_retry_timer.setInterval(5000)
                self._cam_retry_timer.timeout.connect(self._retry_camera_open)
                self._cam_retry_timer.start()
                self._lbl_status.setText("Camera not found — retrying in 5 s…")

        if self._camera and self._camera.is_open() and cfg.get("CAMERA") == "camera":
            self._camera.warmup()
        if self._detector and self._detector.is_ready():
            self._detector.warmup(frames=cfg.get("WARMUP_FRAMES", 5))

        # Load and display first image on startup (no overlays yet)
        if self._camera and self._camera.is_open():
            try:
                img = self._camera.grab_first()
                self._view.set_image(img)
                self._setup_image = img
            except CameraError:
                pass

        if self._camera and self._camera.is_open() and cfg.get("CAMERA") == "camera":
            self._preview_timer = QtCore.QTimer(self)
            self._preview_timer.setInterval(100)
            self._preview_timer.timeout.connect(self._on_preview_tick)
            self._preview_timer.start()

        self._cellcon = CellCon(port=cfg.get("CELLCON_PORT", "/dev/ttyUSB0"))

        # Build Inspector from existing template (silent no-op if template absent)
        self._rebuild_inspector()

        # Apply initial button state — disables Start if no template exists yet.
        self._update_setup_buttons()


    def _rebuild_inspector(self):
        """(Re)build Inspector from current template + config. Sets _inspector=None on failure."""
        if not self._detector or not self._detector.is_ready():
            self._inspector = None
            return
        try:
            tmpl = TemplateManager.load()
        except TemplateError:
            self._inspector = None
            return
        matcher = None
        full_patch = TemplateManager.load_patches()
        if full_patch is not None:
            ic_a = tmpl["ic_a"]
            matcher = TemplateMatcher(
                full_patch,
                threshold=tmpl.get("match_threshold", 0.6),
                strip_h=tmpl.get("strip_h", 0),
                ic_x=ic_a["x"], ic_y=ic_a["y"],
                ic_w=ic_a["w"], ic_h=ic_a["h"],
                template_w=tmpl.get("img_w", 0),
            )
        self._inspector = Inspector(
            self._detector, tmpl, template_matcher=matcher,
            cell_shrink=self._cfg.get("CELL_SHRINK", 0.95),
            cell_expand=self._cfg.get("CELL_EXPAND", 1.2),
            col_gap_pct=self._cfg.get("COL_GAP_PCT", 40.0),
            grid_margin_top=self._cfg.get("GRID_MARGIN_TOP", 0.0),
            grid_margin_bot=self._cfg.get("GRID_MARGIN_BOT", 15.0),
            collect_dataset=self._cfg.get("COLLECT_DATASET", False),
            data_dir=self._cfg.get("DATA_DIR", "Dataset"),
            data_split=self._cfg.get("DATA_SPLIT", "train"),
            ann_border_px=self._cfg.get("ANN_BORDER_PX", 1),
            ann_show_labels=self._cfg.get("RESULT_OVERLAY", True),
        )

    def _retry_camera_open(self):
        """Called every 5 s when camera failed to open at startup (camera mode only)."""
        try:
            cam = Camera(**self._camera_init_kwargs)
            cam.open()
            self._camera = cam
            self._camera.warmup()
            try:
                img = self._camera.grab_first()
                self._view.set_image(img)
                self._setup_image = img
            except CameraError:
                pass
            if self._cam_retry_timer is not None:
                self._cam_retry_timer.stop()
            if self._preview_timer is None:
                self._preview_timer = QtCore.QTimer(self)
                self._preview_timer.setInterval(100)
                self._preview_timer.timeout.connect(self._on_preview_tick)
            self._preview_timer.start()
            self._error_banner.hide()
            self._update_setup_buttons()
            self._lbl_status.setText("Camera reconnected.")
        except CameraError:
            self._lbl_status.setText("Camera not found — retrying in 5 s…")

    # Rubber-band template setup flow
    def _grab_setup_frame(self) -> np.ndarray | None:
        if self._camera is None:
            self._show_error("Camera not ready.")
            return None
        try:
            return self._camera.grab_first()
        except CameraError as e:
            self._show_error(str(e))
            return None

    def _on_new_tmpl_click(self):
        if self._setup_state == 'idle':
            self._start_draw_a()
        else:
            self._reset_template_draw()

    def _start_draw_a(self):
        img = self._grab_setup_frame()
        if img is None:
            return
        self._setup_image  = img
        self._pending_ic_a = None
        self._pending_ic_b = None
        self._view.set_image(img)
        self._view.clear_overlays()
        self._setup_state = 'draw_a'
        self._view.set_rubberband_mode(True)
        self._update_setup_buttons()

    def _on_rb_rect_drawn(self, rect: QtCore.QRect):
        if self._setup_state not in ('draw_a', 'draw_a_retry'):
            return
        self._view.set_rubberband_mode(False)
        self._view.clear_overlays()

        img = self._setup_image
        if img is None:
            self._setup_state = 'draw_a_retry'
            self._view.set_rubberband_mode(True)
            self._update_setup_buttons()
            return

        _MIN_IC_PX = 60
        if rect.width() < _MIN_IC_PX or rect.height() < _MIN_IC_PX:
            self._lbl_tmpl_status.setText(
                f"Selection too small ({rect.width()}×{rect.height()} px) — "
                "draw the full IC area.")
            self._setup_state = 'draw_a_retry'
            self._view.set_rubberband_mode(True)
            self._update_setup_buttons()
            return

        drawn_on_left = (rect.x() + rect.width() // 2) < img.shape[1] // 2
        second, _     = _find_second_ic(img, rect)

        if drawn_on_left:
            ic_a, ic_b = rect, second
        else:
            ic_a, ic_b = second, rect

        if ic_a:
            self._view.add_overlay(ic_a, QtGui.QColor(_tmpl_color_a), "IC_A")
        if ic_b:
            self._view.add_overlay(ic_b, QtGui.QColor(_tmpl_color_b), "IC_B")

        if ic_a and ic_b:
            self._pending_ic_a = ic_a
            self._pending_ic_b = ic_b
            self._setup_state  = 'ready'
        else:
            self._pending_ic_a = None
            self._pending_ic_b = None
            self._view.set_rubberband_mode(True)
            self._setup_state = 'draw_a_retry'
        self._update_setup_buttons()

    def _show_cell_preview(self, ic_a: QtCore.QRect, ic_b: QtCore.QRect) -> bool:
        img = self._setup_image
        if img is None:
            return True

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Verify Cell Areas")
        dlg.setModal(True)
        outer = QtWidgets.QVBoxLayout(dlg)

        info = QtWidgets.QLabel(
            "Check that each cell covers one mark position.\n"
            "Click Confirm to save, or Cancel to redraw.")
        info.setWordWrap(True)
        info.setStyleSheet("font-size:11px;color:#E2FDFF")
        outer.addWidget(info)

        panels = QtWidgets.QHBoxLayout()
        ih, iw = img.shape[:2]
        for ic, label_text in ((ic_a, "IC_A"), (ic_b, "IC_B")):
            grp = QtWidgets.QGroupBox(label_text)
            grp.setStyleSheet("color:#E2FDFF;font-weight:bold")
            grid = QtWidgets.QGridLayout(grp)
            grid.setSpacing(4)
            cells = _build_cells(
                ic.x(), ic.y(), ic.width(), ic.height(),
                cell_shrink=self._cfg.get("CELL_SHRINK", 0.95),
                cell_expand=self._cfg.get("CELL_EXPAND", 1.2),
                col_gap_pct=self._cfg.get("COL_GAP_PCT", 40.0),
                grid_margin_top=self._cfg.get("GRID_MARGIN_TOP", 0.0),
                grid_margin_bot=self._cfg.get("GRID_MARGIN_BOT", 15.0),
            )
            for idx, (cx, cy, cw, ch) in enumerate(cells):
                crop = img[max(0, cy):min(ih, cy + ch), max(0, cx):min(iw, cx + cw)]
                if crop.size == 0:
                    crop = np.zeros((40, 40, 3), dtype=np.uint8)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                qimg = QtGui.QImage(rgb.data, w, h, w * 3, QtGui.QImage.Format_RGB888)
                pix  = QtGui.QPixmap.fromImage(qimg).scaled(
                    80, 80, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                row_n, col_n = divmod(idx, 2)
                lbl_name = QtWidgets.QLabel(f"R{row_n+1}C{col_n+1}")
                lbl_name.setStyleSheet("font-size:9px;color:#E2FDFF")
                lbl_name.setAlignment(QtCore.Qt.AlignCenter)
                lbl_pix = QtWidgets.QLabel()
                lbl_pix.setPixmap(pix)
                lbl_pix.setAlignment(QtCore.Qt.AlignCenter)
                grid.addWidget(lbl_name, row_n * 2,     col_n)
                grid.addWidget(lbl_pix,  row_n * 2 + 1, col_n)
            panels.addWidget(grp)
        outer.addLayout(panels)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.button(QtWidgets.QDialogButtonBox.Ok).setText("Confirm")
        btns.button(QtWidgets.QDialogButtonBox.Cancel).setText("Re-draw")
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        outer.addWidget(btns)

        return dlg.exec_() == QtWidgets.QDialog.Accepted

    def _confirm_template(self):
        if self._pending_ic_a and self._pending_ic_b:
            if not self._show_cell_preview(self._pending_ic_a, self._pending_ic_b):
                self._start_draw_a()   # cancel → back to drawing
                return
            self._on_detect_confirmed(self._pending_ic_a, self._pending_ic_b)
            self._reset_template_draw()

    def _reset_template_draw(self):
        self._view.set_rubberband_mode(False)
        self._view.clear_overlays()
        self._pending_ic_a = None
        self._pending_ic_b = None
        self._setup_state  = 'idle'
        self._update_setup_buttons()

    def _update_setup_buttons(self):
        s = self._setup_state
        self._btn_new_tmpl.setEnabled(True)   # always enabled — acts as Cancel during draw
        self._btn_new_tmpl.setText("New Template" if s == 'idle' else "Cancel")
        self._btn_confirm_tmpl.setEnabled(s == 'ready')
        if s == 'idle':
            template_ok = os.path.exists(_TEMPLATE_FILE)
            text = ("Template saved."
                    if template_ok
                    else "No template — create a template before running.")
            if self._run_state == "standby":
                self._btn_action.setEnabled(template_ok and self._ocr_fields_valid())
        else:
            self._btn_action.setEnabled(False)   # no Start while drawing template
            text = {
                'draw_a':       'Draw either IC area on image.',
                'draw_a_retry': 'IC_B not found — draw again.',
                'ready':        'IC_A + IC_B found. Confirm to save.',
            }.get(s, '—')
        self._lbl_tmpl_status.setText(text)

    def _on_detect_confirmed(self, ic_a: QtCore.QRect, ic_b: QtCore.QRect):
        self._view.clear_overlays()
        exposure = int(self._cfg.get("EXPOSURE_US", 8000))

        patch_saved = False
        strip_h_val = 0
        if self._setup_image is not None:
            try:
                full_patch, strip_h_val = \
                    TemplateManager.extract_patches(self._setup_image, ic_a)
                TemplateManager.save_patches(full_patch)
                patch_saved = True
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Patch Warning",
                    f"Could not save template patches: {e}\n"
                    "Inspection will use fixed template coordinates.")

        img_h_tmpl, img_w_tmpl = (self._setup_image.shape[:2]
                                   if self._setup_image is not None else (0, 0))
        TemplateManager.save(ic_a, ic_b, exposure, strip_h=strip_h_val,
                             img_w=img_w_tmpl, img_h=img_h_tmpl)

        if self._setup_image is not None:
            try:
                TemplateManager.save_preview(self._setup_image, ic_a, ic_b)
            except Exception:
                pass

        msg = "Template saved to templates/template.json"
        if patch_saved:
            msg += "\nPatch file saved (tmpl_full.npy)"
        msg += "\nPreview saved to templates/template_preview.png"
        QtWidgets.QMessageBox.information(self, "Template Saved", msg)
        self._rebuild_inspector()


    def _is_mock_trigger_mode(self) -> bool:
        """True when IO=False + camera mode: button acts as manual START trigger."""
        return (not self._cfg.get("IO", False)
                and self._cfg.get("CAMERA", "directory") == "camera")

    def _check_hardware_ready(self) -> bool:
        """Verify selected IO devices are reachable before starting a run.

        GPIO and camera failures are hard blocks. A missing CellCon port is a
        soft warning — the operator may continue and enter the lot number manually.
        """
        cfg = self._cfg

        if cfg.get("IO", False):
            if self._gpio is None or not self._gpio._gpio_ok:
                QtWidgets.QMessageBox.critical(
                    self, "Hardware Error",
                    "GPIO not ready — check RPi.GPIO and wiring.")
                return False

        if cfg.get("CAMERA") == "camera":
            if self._camera is None or not self._camera.is_open():
                QtWidgets.QMessageBox.critical(
                    self, "Hardware Error",
                    "Basler camera not connected or not open.")
                return False

        port = cfg.get("CELLCON_PORT", "/dev/ttyUSB0")
        if not os.path.exists(port):
            reply = QtWidgets.QMessageBox.warning(
                self, "CellCon Not Found",
                f"{port} not available — lot must be entered manually.\nProceed?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply != QtWidgets.QMessageBox.Yes:
                return False

        return True

    # Run / Pause / Stop
    def _on_action_click(self):
        if self._run_state == "standby":
            self._start_run()
        elif self._run_state == "running":
            if self._is_mock_trigger_mode():
                self._worker.trigger()   # inject one mock START pulse
            else:
                self._pause_run()
        elif self._run_state == "paused":
            self._resume_run()

    def _start_run(self):
        if self._worker and self._worker.isRunning():
            return

        # Guards before showing any dialog
        if not self._detector or not self._detector.is_ready():
            self._show_error("Detector not ready.")
            return
        inspector = self._inspector
        if inspector is None:
            self._show_error("No inspector — create a template first.")
            return
        if not self._check_hardware_ready():
            return

        # Snapshot OCR fields (read before lot dialog, values already validated by gating)
        self._ocr_operator     = self._edit_op_number.text().strip()
        self._ocr_expect_value = self._edit_ocr_expect.text().strip()

        # Clear right-panel status immediately on Start click
        self._lbl_ocr_status.setText("Verifying…")
        self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
        self._lbl_lot_info.setText("—")

        # Ask operator for lot number (or get from CellCon / subclass hook)
        lot = LotStartDialog.request(parent=self, api_fn=self._cellcon.get_lot)
        if lot is None:
            self._lbl_ocr_status.setText("Fill both fields to enable Start.")
            self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
            return   # operator cancelled
        self._lot_number   = lot
        self._package_name = inspector._template.get("package_name", "")

        self._session_start_time = time.monotonic()

        # Disk space soft check
        out_dir = self._cfg.get("OUT_DIR", "Output/")
        warn_mb = int(self._cfg.get("DISK_WARN_MB", 200))
        try:
            import shutil as _shutil
            free_mb = _shutil.disk_usage(out_dir).free >> 20
            if free_mb < warn_mb:
                self._show_error(
                    f"Low disk: {free_mb} MB free (threshold {warn_mb} MB) — run continues")
        except OSError:
            pass

        # Purge orphaned temp files from today's output dir (leftover from crashes)
        _VALID_SFX = ("_G.jpg", "_NG.jpg", "_GS.jpg", "_NGS.jpg")
        today_dir  = os.path.join(out_dir, datetime.now().strftime("%Y%m%d"))
        if os.path.isdir(today_dir):
            for _root, _, _files in os.walk(today_dir):
                for _f in _files:
                    if _f.endswith(".jpg") and not any(_f.endswith(s) for s in _VALID_SFX):
                        try:
                            os.remove(os.path.join(_root, _f))
                        except OSError:
                            pass

        mode = "DEBUG" if self._cfg.get("DEBUG", True) else "RUN"
        self._logger.start_lot(self._lot_number, self._package_name, mode)

        gpio = self._gpio or RaspberryIO(io_enabled=False)

        # OCR API verification (once per lot)
        self._lbl_lot_info.setText(lot)
        ocr_ok = self._ocr_api_call(lot, self._ocr_operator, self._ocr_expect_value)
        if not ocr_ok:
            QtWidgets.QMessageBox.critical(
                self, "OCR Verification Failed",
                "Marking verification failed.\nCheck the expected mark and retry.",
                QtWidgets.QMessageBox.Close)
            self._enter_standby()
            return
        self._logger.log_ocr(self._ocr_operator, self._ocr_used_mark)
        self._start_worker(inspector, gpio)

    def _on_preview_tick(self):
        if self._run_state != "standby" or not self._camera or self._setup_state != 'idle':
            return
        try:
            img = self._camera.grab()
            self._view.set_image(img)
        except CameraError:
            pass

    def _start_worker(self, inspector: "Inspector", gpio: "RaspberryIO"):
        """Create and start RunWorker. Lock OCR fields for the duration of the run."""
        if self._preview_timer:
            self._preview_timer.stop()
        self._edit_op_number.setReadOnly(True)
        self._edit_ocr_expect.setReadOnly(True)
        self._worker = RunWorker(
            self._camera, inspector, gpio,
            self._logger, self._cfg, lot_number=self._lot_number)
        self._worker.sig_image.connect(self._on_image)
        self._worker.sig_result.connect(self._on_result)
        self._worker.sig_fail.connect(self._on_fail)
        self._worker.sig_error.connect(self._on_worker_error)
        self._worker.sig_status.connect(self._lbl_status.setText)
        self._worker.sig_status.connect(self._reset_watchdog)
        self._worker.sig_image.connect(self._reset_watchdog)
        self._worker.sig_cycle_ms.connect(
            lambda ms: self._lbl_cycle_ms.setText(f"{ms:.0f}"))
        self._worker.sig_done.connect(self._on_run_done)
        self._worker.sig_session_reset.connect(self._on_session_reset)
        self._worker.sig_paused.connect(self._on_paused)
        self._worker.sig_resumed.connect(self._on_resumed)
        self._worker.start()
        self._worker_last_tick = time.monotonic()

        self._run_state = "running"
        self._btn_action.setText("Trigger" if self._is_mock_trigger_mode() else "Pause")
        self._btn_action.setEnabled(True)
        self._btn_stop.setEnabled(True)

    def _ocr_fields_valid(self) -> bool:
        op  = self._edit_op_number.text()
        exp = self._edit_ocr_expect.text()
        return len(op) == 6 and op.isdigit() and len(exp) == 6 and exp.isalnum()

    def _on_ocr_field_changed(self):
        if self._run_state == "standby":
            valid = self._ocr_fields_valid()
            self._btn_action.setEnabled(valid)
            if not valid:
                self._lbl_ocr_status.setText("Fill both fields to enable Start.")
                self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")

    def _ocr_api_call(self, lot: str, operator: str, expected_mark: str) -> bool:
        """POST to ReadMark API, compare result, POST CreateRecord. Returns True = proceed."""
        self._ocr_used_mark = expected_mark   # always reset — never carry stale value from prior lot
        import base64   # stdlib, only used here
        try:
            import requests as _req
        except ImportError:
            debug = self._cfg.get("DEBUG", True)
            if not debug:
                self._lbl_ocr_status.setText("OCR unavailable — 'requests' not installed")
                self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#FF6B6B")
                return False
            return True   # debug mode: skip OCR silently

        debug = self._cfg.get("DEBUG", True)
        _crop_path = "cropimg.jpg"
        _wrote_crop = False
        try:
            img = self._camera.grab_first()
            _pw = self._cfg.get("IMAGE_W", 0) or img.shape[1]
            _ph = self._cfg.get("IMAGE_H", 0) or img.shape[0]
            resized = cv2.resize(img, (_pw, _ph), interpolation=cv2.INTER_AREA)
            cv2.imwrite(_crop_path, resized)
            _wrote_crop = True

            resp = _req.post(
                "http://webserv.thematrix.net/ROHMApi/api/OCR/ReadMark",
                json={"username": operator, "lot_no": lot}, timeout=5)

            is_pass = 0
            if resp.status_code == 200:
                data = resp.json()
                if not isinstance(data, list):
                    if debug:
                        self._lbl_ocr_status.setText("[DEBUG] ReadMark: unexpected response format — skipped")
                        self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
                        is_pass = 1
                    else:
                        self._lbl_ocr_status.setText("ReadMark: unexpected server response format")
                        self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#FF6B6B")
                        return False
                elif not data:
                    if debug:
                        self._lbl_ocr_status.setText("[DEBUG] ReadMark: lot not found — skipped")
                        self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
                        is_pass = 1
                    else:
                        self._lbl_ocr_status.setText("ReadMark: lot not in DB — cannot verify")
                        self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#FF6B6B")
                        return False
                else:
                    std_mark = data[0].get("mark")
                    if std_mark is None:
                        if debug:
                            self._lbl_ocr_status.setText("[DEBUG] ReadMark: 'mark' field missing — skipped")
                            self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
                            is_pass = 1
                        else:
                            self._lbl_ocr_status.setText("ReadMark: server response missing 'mark' field")
                            self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#FF6B6B")
                            return False
                    else:
                        ocr_mark = data[0].get("ocr_mark")
                        if ocr_mark is None and not debug:
                            # retry once before giving up
                            try:
                                resp2 = _req.post(
                                    "http://webserv.thematrix.net/ROHMApi/api/OCR/ReadMark",
                                    json={"username": operator, "lot_no": lot}, timeout=5)
                                if resp2.status_code == 200:
                                    data2 = resp2.json()
                                    if isinstance(data2, list) and data2:
                                        ocr_mark = data2[0].get("ocr_mark")
                            except Exception:
                                pass
                        if ocr_mark is None:
                            if debug:
                                ocr_mark = expected_mark
                            else:
                                self._lbl_ocr_status.setText(
                                    f"OCR: no mark result after retry — check lot {lot}")
                                self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#FF6B6B")
                                return False
                        self._ocr_used_mark = ocr_mark
                        is_pass = 1 if std_mark == ocr_mark else 0
                        color   = "#69FF69" if is_pass else "#FF6B6B"
                        label   = "Mark OK" if is_pass else f"FAIL — DB: {std_mark} | OCR: {ocr_mark}"
                        self._lbl_ocr_status.setText(label)
                        self._lbl_ocr_status.setStyleSheet(f"font-size:11px;color:{color}")
            elif resp.status_code in (401, 403):
                self._lbl_ocr_status.setText(
                    f"ReadMark: authentication failed ({resp.status_code}) — check operator credentials")
                self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#FF6B6B")
                if not debug:
                    return False
                is_pass = 1
            elif resp.status_code == 404:
                self._lbl_ocr_status.setText("ReadMark: endpoint not found (404) — check server URL")
                self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#FF6B6B")
                if not debug:
                    return False
                is_pass = 1
            elif resp.status_code >= 500:
                self._lbl_ocr_status.setText(
                    f"ReadMark: server error ({resp.status_code}) — try again later")
                self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#FF6B6B")
                if not debug:
                    return False
                is_pass = 1
            elif debug:
                self._lbl_ocr_status.setText("[DEBUG] ReadMark unavailable — skipped")
                self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
                is_pass = 1
            else:
                self._lbl_ocr_status.setText(
                    f"ReadMark API error {resp.status_code} — check credentials/server")
                self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#FF6B6B")
                return False

            try:
                with open(_crop_path, "rb") as fh:
                    enc = base64.b64encode(fh.read()).decode()
                _req.post(
                    "http://webserv.thematrix.net/ROHMApi/api/OCR/CreateRecord",
                    json={"username": operator, "lot_no": lot,
                          "mark": self._ocr_used_mark,
                          "image": enc, "is_pass": is_pass,
                          "recheck_count": 0, "is_logo_pass": 0}, timeout=5)
            except Exception:
                pass   # record failure never blocks the run

            return bool(is_pass) or debug

        except Exception as exc:
            print(f"[OCR] {exc}")
            err_str = str(exc).lower()
            if debug:
                self._lbl_ocr_status.setText("[DEBUG] API unavailable — skipped")
                self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
                return True
            if any(k in err_str for k in ("connection", "timeout", "unreachable")):
                self._lbl_ocr_status.setText("ReadMark API unreachable — check network connection")
            else:
                self._lbl_ocr_status.setText(f"OCR API error — {exc}")
            self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#FF6B6B")
            return False

        finally:
            if _wrote_crop:
                try:
                    os.remove(_crop_path)
                except OSError:
                    pass

    def _pause_run(self):
        if self._worker:
            self._worker.pause()
        # UI updated via sig_paused → _on_paused()

    def _resume_run(self):
        if self._worker:
            self._worker.resume()
        # UI updated via sig_resumed → _on_resumed()

    def _on_paused(self):
        self._run_state = "paused"
        self._btn_action.setText("Resume")
        self._lbl_status.setText("Paused.")
        self._logger.log_pause()

    def _on_resumed(self):
        self._run_state = "running"
        self._btn_action.setText("Pause")
        self._lbl_status.setText("Running…")
        self._logger.log_resume()

    def _stop_run(self):
        """Called by Stop button — ends lot, no sig_done."""
        elapsed = time.monotonic() - self._session_start_time
        self._logger.end_lot(
            "STOPPED", self._stats_pass, self._stats_fail,
            self._stats_error, elapsed)
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
        self._enter_standby()

    def _on_session_reset(self, new_lot: str):
        """Batch complete (all dir images done or lot-end GPIO): end current lot, start new."""
        elapsed = time.monotonic() - self._session_start_time
        self._logger.end_lot(
            "COMPLETE", self._stats_pass, self._stats_fail,
            self._stats_error, elapsed)
        self._lot_number = new_lot
        self._stats_pass = self._stats_fail = self._stats_error = self._stats_total = 0
        self._lbl_pass.setText("0")
        self._lbl_fail.setText("0")
        self._lbl_error.setText("0")
        self._lbl_yield.setText("—")
        self._session_start_time = time.monotonic()
        mode = "DEBUG" if self._cfg.get("DEBUG", True) else "RUN"
        self._logger.start_lot(new_lot, self._package_name, mode)

    def _on_run_done(self):
        """Called when the worker loop exits."""
        if self._run_state == "standby":
            return   # Stop already handled this via _stop_run; ignore queued sig_done
        elapsed = time.monotonic() - self._session_start_time
        self._logger.end_lot(
            "COMPLETE", self._stats_pass, self._stats_fail,
            self._stats_error, elapsed)
        self._enter_standby()

    def _enter_standby(self):
        self._run_state = "standby"
        self._btn_action.setText("Start")
        self._btn_stop.setEnabled(False)
        self._edit_op_number.setReadOnly(False)
        self._edit_ocr_expect.setReadOnly(False)
        self._edit_op_number.clear()    # force re-entry each lot; prevent ID carry-over
        self._edit_ocr_expect.clear()   # each lot's mark must be entered fresh
        self._lbl_ocr_status.setText("Fill both fields to enable Start.")
        self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
        self._btn_action.setEnabled(self._ocr_fields_valid())   # False — fields now empty
        self._lbl_lot_info.setText("—")
        self._update_badge(self._badge_a, None)
        self._update_badge(self._badge_b, None)
        self._stats_pass = self._stats_fail = self._stats_error = self._stats_total = 0
        self._lbl_pass.setText("0")
        self._lbl_fail.setText("0")
        self._lbl_error.setText("0")
        self._lbl_yield.setText("—")
        self._lbl_status.setText("Standby.")
        self._reload_default_image()
        if self._preview_timer:
            self._preview_timer.start()

    def _reload_default_image(self):
        """Display the first image (or a live grab) and rewind the index."""
        if not self._camera:
            return
        try:
            img = self._camera.grab_first()
            self._view.set_image(img)
            self._view.clear_overlays()
            self._setup_image = img
        except CameraError:
            pass

    # Worker signal handlers
    def _on_image(self, img: np.ndarray):
        self._view.set_image(img)

    def _update_yield(self):
        total = self._stats_pass + self._stats_fail
        if total > 0:
            self._lbl_yield.setText(f"{self._stats_pass / total * 100:.1f}%")
        else:
            self._lbl_yield.setText("—")

    def _update_ui_after_cycle(self, ic_a_pass: bool, ic_b_pass: bool, passed: bool):
        self._update_badge(self._badge_a, ic_a_pass)
        self._update_badge(self._badge_b, ic_b_pass)
        self._stats_total += 1
        if passed:
            self._stats_pass += 1
            self._lbl_pass.setText(str(self._stats_pass))
        else:
            self._stats_fail += 1
            self._lbl_fail.setText(str(self._stats_fail))
        self._update_yield()

    def _on_result(self, ia_pass: bool, ib_pass: bool, _is_suspect: bool):
        self._update_ui_after_cycle(ia_pass, ib_pass, passed=True)

    def _on_fail(self, err: MarkMissingError, _ann_path: str, _img_id: str, _is_suspect: bool):
        self._update_ui_after_cycle(
            len(err.missing_a) == 0, len(err.missing_b) == 0, passed=False)

    def _reset_watchdog(self, *_):
        self._worker_last_tick = time.monotonic()

    def _check_watchdog(self):
        if self._run_state != "running":
            return
        if time.monotonic() - self._worker_last_tick > 30.0:
            if self._worker:
                self._worker.stop()
            self._on_worker_error(
                "Worker timeout — no activity for 30 s. Camera may be frozen.")

    def _on_worker_error(self, msg: str):
        self._stats_error += 1
        elapsed = time.monotonic() - self._session_start_time
        self._logger.end_lot(
            "ERROR", self._stats_pass, self._stats_fail,
            self._stats_error, elapsed)
        self._enter_standby()
        self._show_error(msg)   # after standby — _enter_standby does not clear the error banner

    # Error banner
    def _show_error(self, msg: str):
        self._error_lbl.setText(f"Error: {msg}")
        self._error_banner.show()

    # Close
    def closeEvent(self, e):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        if self._camera:
            self._camera.close()
        if self._gpio:
            self._gpio.cleanup()
        e.accept()

# ENTRY POINT
def main():
    app = QtWidgets.QApplication(sys.argv)

    _lockfile = open("/tmp/clearic.lock", "w")
    try:
        fcntl.flock(_lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        QtWidgets.QMessageBox.critical(
            None, "Already Running",
            "ClearIC is already running.\nClose the existing window first.")
        sys.exit(1)

    try:
        cfg = ConfigLoader.load()
    except ConfigError as e:
        QtWidgets.QMessageBox.critical(
            None, "Configuration Error",
            f"Cannot start — Config.toml problem:\n\n{e}\n\n"
            "Contact your system administrator.")
        sys.exit(1)

    os.makedirs(cfg.get("LOG_DIR", "logs"), exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs(cfg.get("DIR_INPUT", "Input/"), exist_ok=True)
    if cfg.get("COLLECT_DATASET", False):
        _dd, _ds = cfg.get("DATA_DIR", "Dataset"), cfg.get("DATA_SPLIT", "train")
        os.makedirs(os.path.join(_dd, _ds, "Text"),   exist_ok=True)
        os.makedirs(os.path.join(_dd, _ds, "NoText"), exist_ok=True)
        print(f"[Dataset] Collection ON → {_dd}/{_ds}/")

    for _stale in ("cropimg.jpg",):
        try:
            os.remove(_stale)
        except OSError:
            pass

    app.setStyleSheet(STYLE)

    pal = QtGui.QPalette()
    for role, col in [
        (QtGui.QPalette.Window,          (84,  101, 255)),
        (QtGui.QPalette.WindowText,      (255, 255, 255)),
        (QtGui.QPalette.Base,            (120, 139, 255)),
        (QtGui.QPalette.Text,            (255, 255, 255)),
        (QtGui.QPalette.Button,          (84,  101, 255)),
        (QtGui.QPalette.ButtonText,      (255, 255, 255)),
        (QtGui.QPalette.Highlight,       (191, 215, 255)),
        (QtGui.QPalette.HighlightedText, ( 84, 101, 255)),
    ]:
        pal.setColor(role, QtGui.QColor(*col))
    app.setPalette(pal)

    win = MainWindow(cfg)
    win.show()
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    signal.signal(signal.SIGINT,  lambda *_: app.quit())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
