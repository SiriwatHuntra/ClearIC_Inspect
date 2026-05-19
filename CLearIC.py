"""
ClearIC Inspect
===============
Clear-package IC laser-mark inspection via ROI crop + OpenVINO classifier.
Each ROI cell is cropped and classified as Text (present) or NoText (absent).

Sections (in order)
-------------------
  ConfigLoader        Config.json loader with defaults
  Stage / ErrorFlag   State enums
  Exceptions          InspectionError hierarchy
  Image               Image dataclass + ID generator
  Camera              Basler camera or directory source
  RaspberryIO         BCM GPIO handler (mockable)
  Detector            OpenVINO 2-class classifier (Text / NoText)
  TemplateManager     Load/save IC bounding-box template
  Inspector           12-cell ROI crop-then-classify logic
  Logger              Daily-rotating JSON-lines log
  STYLE               Qt stylesheet
  FailDialog          Modal FAIL popup
  ImageView           Zoomable image widget with overlays
  SetupPanel          Floating auto-detect confirm panel
  RunWorker           QThread inspection loop
  MainWindow          Single-page PyQt5 UI
  main / __main__     Entry point
"""

import sys
import os
import csv
import json
import glob
import math
import time
import threading
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

import gc
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore


# =========================================================
# CONFIG LOADER
# =========================================================
class ConfigLoader:
    CONFIG_FILE = "Config.json"
    DEFAULT_CONFIG = {
        # Camera / detection
        "CAMERA":               "directory",
        "CONF_THR":             0.5,
        "TEXT_MIN_CONF":        0.80,
        "BLANK_CELL_STD_THR":   0.0,
        "NMS_IOU_THR":          0.45,
        "CAMERA_SERIAL":        "",
        "EXPOSURE_US":          8000,
        # Dev flags
        "DEBUG":                True,
        "IO":                   False,
        "MODE":                 "DEBUG",
        "COLLECT_DATASET":      False,
        "DIR_INPUT":            "Input/",
        "OUT_DIR":              "Output/",
        "MODEL_PATH":           "Text_cls-2/best_openvino_model/best.xml",
        "TEMPLATE_MODEL_PATH":  "IC_Search_openvino_model/IC_Search.xml",
        # Camera tuning
        "CAMERA_WARMUP_FRAMES": 5,
        "CAMERA_RETRY_DELAY":   0.2,
        "CAMERA_RETRIES":       2,
        "RETRY_DELAY_MS":       250,
        # GPIO pins
        "GPIO_START_PIN":       17,
        "GPIO_DONE_PIN":        27,
        "GPIO_BUSY_PIN":        23,
        "GPIO_LOT_END_PIN":     18,
        "GPIO_FAIL_A_PIN":      24,
        "GPIO_FAIL_B_PIN":      25,
        "MOCK_START_DELAY_MS":  200,
        "MOCK_DONE_DELAY_MS":   100,
        "TRIGGER_SETTLE_MS":    50,
        # Cell grid geometry
        "CELL_SHRINK":          0.95,
        "CELL_EXPAND":          1.2,
        "COL_GAP_PCT":          40.0,
        "GRID_MARGIN_TOP":      0.0,
        "GRID_MARGIN_BOT":      15.0,
        # Dataset collection
        "DATA_DIR":             "Dataset",
        "DATA_SPLIT":           "train",
        # Logging
        "LOG_DIR":              "logs",
        "LOG_RETENTION":        365,
        # Annotation settings (non-color)
        "ANN_BORDER_PX":        1,
        "ANN_SHOW_LABELS":      True,
        "WARMUP_FRAMES":        5,
    }
    _VALID_CAMERA = {"camera", "directory"}

    @classmethod
    def load(cls) -> dict:
        if not os.path.exists(cls.CONFIG_FILE):
            raise ConfigError("Config.json not found — create it before running.")
        try:
            with open(cls.CONFIG_FILE, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise ConfigError(f"Config.json unreadable: {e}")
        cfg = dict(cls.DEFAULT_CONFIG)
        for k in cls.DEFAULT_CONFIG:
            if k in data:
                cfg[k] = data[k]
        if cfg["CAMERA"] not in cls._VALID_CAMERA:
            raise ConfigError(f"CAMERA must be one of {cls._VALID_CAMERA}")
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
        if not isinstance(cfg["TRIGGER_SETTLE_MS"], (int, float)) or cfg["TRIGGER_SETTLE_MS"] < 0:
            raise ConfigError("TRIGGER_SETTLE_MS must be a non-negative number")
        for pin_key in ("GPIO_START_PIN", "GPIO_DONE_PIN", "GPIO_BUSY_PIN",
                        "GPIO_LOT_END_PIN", "GPIO_FAIL_A_PIN", "GPIO_FAIL_B_PIN"):
            if not isinstance(cfg[pin_key], int) or not (1 <= cfg[pin_key] <= 27):
                raise ConfigError(f"{pin_key} must be a BCM pin number (1–27)")
        return cfg

    @classmethod
    def save(cls, data: dict):
        with open(cls.CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def update(cls, updates: dict):
        """Merge partial updates into saved config. Only known keys are accepted."""
        cfg = cls.load()
        for k, v in updates.items():
            if k in cls.DEFAULT_CONFIG:
                cfg[k] = v
        cls.save(cfg)
        return cfg

# =========================================================
# STAGE & ERROR FLAGS
# =========================================================
class Stage(Enum):
    STANDBY  = "STANDBY"
    BUSY     = "BUSY"
    ERROR    = "ERROR"
    SHUTDOWN = "SHUTDOWN"

class ErrorFlag(Enum):
    NONE     = "NONE"
    CAMERA   = "CAMERA"
    MODEL    = "MODEL"
    GPIO     = "GPIO"
    TEMPLATE = "TEMPLATE"

# =========================================================
# EXCEPTIONS
# =========================================================
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

# =========================================================
# IMAGE DATACLASS + ID GENERATOR
# =========================================================
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

# =========================================================
# CAMERA
# =========================================================
class Camera:
    """
    Unified camera source.
    CAMERA='camera'    : Basler pypylon InstantCamera
    CAMERA='directory' : reads files from Input/ in sorted order, loops
    """

    def __init__(self, mode: str, serial: str = "",
                 exposure_us: int = 8000, input_dir: str = "Input",
                 retry_delay: float = 0.2, retries: int = 2,
                 warmup_frames: int = 5):
        self._mode        = mode
        self._serial      = serial
        self._exposure_us = exposure_us
        self._input_dir   = input_dir
        self._retry_delay = retry_delay
        self._retries     = retries
        self._warmup_frames = warmup_frames

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

    # ---- open ----
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
            self._camera.ExposureTimeAbs.SetValue(float(self._exposure_us))
            self._camera.PixelFormat.SetValue("Mono8")
            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            print(f"[Camera] Opened. Exposure={self._exposure_us} µs")
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

    # ---- grab ----
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
        return img

    def peek_filename(self) -> str:
        """Return the filename that the next grab() will read (directory mode only)."""
        if self._mode != "directory" or not self._files:
            return ""
        return os.path.basename(self._files[self._idx % len(self._files)])

    # ---- misc ----
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

    def is_open(self) -> bool:
        if self._mode == "camera":
            return self._camera is not None and self._camera.IsOpen()
        return bool(self._files)

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

# =========================================================
# RASPBERRY IO
# =========================================================
class RaspberryIO:
    """
    BCM-mode GPIO handler with serial lighting controller.
    Falls back to mock logging when IO=False or RPi.GPIO unavailable.
    Serial port (/dev/ttyUSB0) is opened only when io_enabled=True.
    """

    def __init__(self, io_enabled: bool = True,
                 start_pin: int = 17, done_pin: int = 27,
                 busy_pin: int = 23, lot_end_pin: int = 18,
                 fail_a_pin: int = 24, fail_b_pin: int = 25,
                 mock_start_delay_ms: int = 200, mock_done_delay_ms: int = 100):
        self._enabled             = io_enabled
        self._gpio_ok             = False
        self._GPIO                = None
        self._ser                 = None
        self._start_pin           = start_pin
        self._done_pin            = done_pin
        self._busy_pin            = busy_pin
        self._lot_end_pin         = lot_end_pin
        self._fail_a_pin          = fail_a_pin
        self._fail_b_pin          = fail_b_pin
        self._mock_start_delay_ms = mock_start_delay_ms
        self._mock_done_delay_ms  = mock_done_delay_ms

        if not io_enabled:
            print("[IO] IO=False — mock mode.")
            return

        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self._start_pin,   GPIO.IN,  pull_up_down=GPIO.PUD_UP)   # active LOW
            GPIO.setup(self._done_pin,    GPIO.IN,  pull_up_down=GPIO.PUD_UP)   # active LOW
            GPIO.setup(self._lot_end_pin, GPIO.IN,  pull_up_down=GPIO.PUD_UP)   # active LOW
            GPIO.setup(self._busy_pin,    GPIO.OUT, initial=GPIO.LOW)           # idle = LOW
            GPIO.setup(self._fail_a_pin,  GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self._fail_b_pin,  GPIO.OUT, initial=GPIO.LOW)
            self._gpio_ok = True
            print("[IO] GPIO initialised (BCM mode).")
        except Exception as e:
            raise GPIOError(f"GPIO init failed: {e}")

        try:
            import serial as _serial
            self._ser = _serial.Serial(
                port='/dev/ttyUSB0', baudrate=38400,
                parity=_serial.PARITY_NONE,
                stopbits=_serial.STOPBITS_ONE,
                bytesize=_serial.EIGHTBITS, timeout=1)
            print("[IO] Serial port OK")
        except Exception:
            self._ser = None
            print("[IO] Serial device not found")

    def _out(self, pin: int, high: bool, pin_name: str = ""):
        if self._gpio_ok:
            self._GPIO.output(pin, self._GPIO.HIGH if high else self._GPIO.LOW)
        else:
            state = "HIGH" if high else "LOW"
            print(f"[IO MOCK] {pin_name or pin} → {state}")

    def set_fail_a(self, v: bool):
        self._out(self._fail_a_pin, v, "FAIL_A_PIN")

    def set_fail_b(self, v: bool):
        self._out(self._fail_b_pin, v, "FAIL_B_PIN")

    def set_busy(self, v: bool):
        self._out(self._busy_pin, v, "BUSY_PIN")

    def is_lot_end_signaled(self) -> bool:
        """Non-blocking: True if LOT_END_PIN is currently LOW. Always False in mock mode."""
        if not self._gpio_ok:
            return False
        return self._GPIO.input(self._lot_end_pin) == self._GPIO.LOW

    def clear_outputs(self):
        self._out(self._busy_pin,   False, "BUSY_PIN")
        self._out(self._fail_a_pin, False, "FAIL_A_PIN")
        self._out(self._fail_b_pin, False, "FAIL_B_PIN")

    def wait_for_start(self, stop_flag_fn) -> bool:
        """Block until START_PIN goes LOW (active LOW) or stop_flag_fn() returns True."""
        if not self._gpio_ok:
            deadline = time.monotonic() + self._mock_start_delay_ms / 1000
            while time.monotonic() < deadline:
                if stop_flag_fn():
                    return False
                time.sleep(0.02)
            if stop_flag_fn():
                return False
            print("[IO MOCK] START_PIN pulse")
            return True
        GPIO = self._GPIO
        while not stop_flag_fn():
            if GPIO.input(self._start_pin) == GPIO.LOW:
                time.sleep(0.05)                        # 50ms debounce
                if GPIO.input(self._start_pin) == GPIO.LOW:
                    return True
            time.sleep(0.005)
        return False

    def wait_for_done(self, stop_flag_fn) -> bool:
        """Block until DONE_PIN goes LOW (active LOW) or stop_flag_fn() returns True."""
        if not self._gpio_ok:
            deadline = time.monotonic() + self._mock_done_delay_ms / 1000
            while time.monotonic() < deadline:
                if stop_flag_fn():
                    return False
                time.sleep(0.02)
            if stop_flag_fn():
                return False
            print("[IO MOCK] DONE_PIN pulse")
            return True
        GPIO = self._GPIO
        while not stop_flag_fn():
            if GPIO.input(self._done_pin) == GPIO.LOW:
                time.sleep(0.05)                        # 50ms debounce
                if GPIO.input(self._done_pin) == GPIO.LOW:
                    return True
            time.sleep(0.005)
        return False

    def is_done_signaled(self) -> bool:
        """Non-blocking: True if DONE_PIN is currently LOW (active LOW). Always False in mock mode."""
        if not self._gpio_ok:
            return False
        return self._GPIO.input(self._done_pin) == self._GPIO.LOW

    def drain_start_pin(self, timeout_ms: int = 500):
        """Wait until START_PIN returns HIGH (idle) to discard stale LOW after resume."""
        if not self._gpio_ok:
            print("[IO MOCK] drain_start_pin")
            return
        GPIO = self._GPIO
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            if GPIO.input(self._start_pin) == GPIO.HIGH:
                return
            time.sleep(0.01)

    def cleanup(self):
        if self._gpio_ok:
            try:
                self._GPIO.cleanup()
            except Exception:
                pass

# =========================================================
# DETECTOR  (OpenVINO Classifier — 2-class)
# =========================================================
_CLS_INPUT_SIZE = 224   # YOLO-cls default input size
SHRINK_SCALE   = 1    # fraction to shrink detected IC boxes before slicing cells

class Detector:
    """
    OpenVINO image classifier for ClearIC mark inspection.
    Each ROI cell crop is classified as Text (mark present) or NoText (absent).
    Output shape: [1, 2]  — index 0 = NoText, index 1 = Text
    """

    def __init__(self, conf_thr: float = 0.5, text_min_conf: float = 0.80,
                 blank_cell_std_thr: float = 0.0,
                 model_path: str = "Text_cls-2/best_openvino_model/best.xml",
                 **_):
        self._conf_thr           = conf_thr
        self._text_min_conf      = text_min_conf
        self._blank_cell_std_thr = blank_cell_std_thr
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
            result  = self._compiled(blob)
            probs     = result[0][0]                         # [2] — softmax already applied by YOLO-cls
            text_prob = float(probs[1])                     # P(Text)
            # Require Text probability to clear TEXT_MIN_CONF; anything below → NoText.
            # Asymmetric on purpose: guards unmarked products without penalising NoText.
            if text_prob >= self._text_min_conf:
                return 1, text_prob
            return 0, float(probs[0])
        except Exception as e:
            print(f"[Detector] Classify error: {e}")
            return 0, 0.0

    def detect_all(self, _: np.ndarray) -> list:
        """Stub — classifier cannot detect IC positions. Draw IC areas in Setup to create template."""
        return []

# Dataset collection run counter
_data_run_counter = 0
_dataset_lock     = threading.Lock()

# CLAHE preprocessor applied to each cell crop before classification
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# =========================================================
# VISUAL CONSTANTS  (fixed — not configurable at runtime)
# =========================================================
_ann_color_ok = "#00C800"   # hex — Text  / PASS cell border
_ann_color_ng = "#DD0000"   # hex — NoText / FAIL cell border
_tmpl_color_a = "#FFD700"   # hex — IC_A overlay in setup view
_tmpl_color_b = "#00E5FF"   # hex — IC_B overlay in setup view

def _hex_to_bgr(h: str) -> tuple:
    """Convert '#RRGGBB' hex string to OpenCV BGR 3-tuple."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

def _valid_hex(s: str) -> bool:
    s = s.strip().lstrip("#")
    return len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s)


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

# =========================================================
# TEMPLATE MANAGER
# =========================================================
_TEMPLATE_FILE    = "templates/template.json"
_TEMPLATE_FULL    = "templates/tmpl_full.npy"   # combined patch (top strip + IC + bot strip)
_TEMPLATE_BOT     = "templates/tmpl_bot.npy"    # deprecated — kept for backward-compat load
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
             match_threshold: float = 0.6, strip_h: int = 0):
        os.makedirs("templates", exist_ok=True)
        data = {
            "ic_a": {"x": ic_a.x(), "y": ic_a.y(),
                     "w": ic_a.width(), "h": ic_a.height()},
            "ic_b": {"x": ic_b.x(), "y": ic_b.y(),
                     "w": ic_b.width(), "h": ic_b.height()},
            "exposure_us":     exposure_us,
            "match_threshold": match_threshold,
            "strip_h":         strip_h,
        }
        with open(_TEMPLATE_FILE, "w") as f:
            json.dump(data, f, indent=2)

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
        np.save(_TEMPLATE_FULL, full_patch)

    @staticmethod
    def load_patches():
        """
        Load combined template patch (tmpl_full.npy).
        Falls back to deprecated tmpl_bot.npy if full patch not found.
        Returns ndarray or None if absent/corrupt.
        """
        if os.path.exists(_TEMPLATE_FULL):
            try:
                return np.load(_TEMPLATE_FULL)
            except Exception as e:
                print(f"[TemplateManager] Patch load failed: {e}")
                return None
        # Backward compat: old split files — use bot strip only for matching
        if os.path.exists(_TEMPLATE_BOT):
            try:
                print("[TemplateManager] tmpl_full.npy not found, using deprecated "
                      "tmpl_bot.npy — re-save template to upgrade.")
                return np.load(_TEMPLATE_BOT)
            except Exception as e:
                print(f"[TemplateManager] Deprecated patch load failed: {e}")
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

        # ── IC boxes, cell grids, centre crosses ────────────────────────────
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

        # ── Template patch region (IC_A only) ───────────────────────────────
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

        # ── Contour edge overlay (teal) inside pin patch only ───────────────
        contour_full = _contour_template(image_bgr)
        patch_edges  = contour_full[patch_y1:patch_y2, ax:patch_x2]
        edge_mask    = patch_edges > 0
        roi          = preview[patch_y1:patch_y2, ax:patch_x2].astype(np.float32)
        teal         = np.array([180, 200, 0], dtype=np.float32)  # BGR teal
        roi[edge_mask] = roi[edge_mask] * 0.3 + teal * 0.7
        preview[patch_y1:patch_y2, ax:patch_x2] = roi.clip(0, 255).astype(np.uint8)

        cv2.imwrite(_TEMPLATE_PREVIEW, preview)

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

# =========================================================
# TEMPLATE MATCHER
# =========================================================
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
                 search_margin: int = 60):
        self._patch     = full_patch
        self._threshold = threshold
        self._strip_h   = strip_h   # px from patch top to IC top edge
        self._patch_w   = full_patch.shape[1]
        self._ic_x      = ic_x   # expected IC_A left edge from template
        self._ic_y      = ic_y   # expected IC_A top from template
        self._ic_w      = ic_w
        self._ic_h      = ic_h
        self._margin    = search_margin   # px around expected pos to search

    def locate_ic(self, image_bgr: np.ndarray) -> tuple:
        """
        Returns (QRect, score).
        Matches the pin-area patch (below IC body) against the frame.
        Preprocessing is applied only to the ±search_margin ROI for speed.
        strip_h is negative (patch starts below IC top), so:
          exp_patch_y = ic_y - strip_h = ic_y + IC_h
          ic_y        = matched_patch_y + strip_h = matched_patch_y - IC_h
        """
        img_h, img_w = image_bgr.shape[:2]
        ph, pw = self._patch.shape[:2]
        m     = self._margin
        exp_y = self._ic_y - self._strip_h

        rx1 = max(0, self._ic_x - m)
        ry1 = max(0, exp_y - m)
        rx2 = min(img_w, self._ic_x + pw + m)
        ry2 = min(img_h, exp_y + ph + m)

        roi_bgr  = image_bgr[ry1:ry2, rx1:rx2]
        filtered = _contour_template(roi_bgr)

        if filtered.shape[0] < ph or filtered.shape[1] < pw:
            # ROI too small — fall back to full-frame search
            full = _contour_template(image_bgr)
            res = cv2.matchTemplate(full, self._patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            ic_y = loc[1] + self._strip_h
            return QtCore.QRect(loc[0], ic_y, self._ic_w, self._ic_h), float(score)

        res = cv2.matchTemplate(filtered, self._patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        abs_x = loc[0] + rx1
        abs_y = loc[1] + ry1
        ic_y  = abs_y + self._strip_h

        if score < self._threshold:
            print(f"[TemplateMatcher] Low match score {score:.3f} < {self._threshold:.3f} — "
                  "using best-match position anyway")

        return QtCore.QRect(abs_x, ic_y, self._ic_w, self._ic_h), float(score)

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

# =========================================================
# INSPECTOR
# =========================================================
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
        self._ic_b_dx = template["ic_b"]["x"] - template["ic_a"]["x"]
        self._ic_b_dy = template["ic_b"]["y"] - template["ic_a"]["y"]
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
        _gcfg = {
            "CELL_SHRINK": self._cell_shrink, "CELL_EXPAND": self._cell_expand,
            "COL_GAP_PCT": self._col_gap_pct,
            "GRID_MARGIN_TOP": self._grid_margin_top,
            "GRID_MARGIN_BOT": self._grid_margin_bot,
        }
        tmpl_a_cells, tmpl_b_cells = TemplateManager.compute_rois(self._template, _gcfg)
        annotated = image_bgr  # draw in-place; caller saves raw before calling inspect()

        # Phase 1: locate ICs
        if self._template_matcher is not None:
            rt_a, score = self._template_matcher.locate_ic(image_bgr)
            rt_b = QtCore.QRect(
                rt_a.x() + self._ic_b_dx, rt_a.y() + self._ic_b_dy,
                self._template["ic_b"]["w"], self._template["ic_b"]["h"],
            )
            ic_a_cells = self._rect_to_cells(rt_a)
            ic_b_cells = self._rect_to_cells(rt_b)
            if debug:
                print(f"[Inspector] TemplateMatcher score={score:.3f}")
                print(f"[Inspector] IC_A matched: "
                      f"x={rt_a.x()} y={rt_a.y()} w={rt_a.width()} h={rt_a.height()}")
                print(f"[Inspector] IC_B by offset: "
                      f"x={rt_b.x()} y={rt_b.y()} w={rt_b.width()} h={rt_b.height()}")
        else:
            # Fixed template coordinates — no runtime IC localization
            ic_a_cells = tmpl_a_cells
            ic_b_cells = tmpl_b_cells
            if debug:
                print("[Inspector] No TemplateMatcher — using fixed template coordinates")

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
            crop    = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                crop = cv2.cvtColor(_CLAHE.apply(gray), cv2.COLOR_GRAY2BGR)
            cls_idx, conf = self._detector.classify_crop(crop)
            present   = (cls_idx == 1)   # 1 = Text (mark present)
            text_conf = conf if cls_idx == 1 else (1.0 - conf)  # Text-class probability
            hits_flags.append(present)
            text_confs.append(text_conf)
            if debug:
                lbl = "Text" if present else "NoText"
                _dbg_g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                print(f"[Cell R{row}C{col}] "
                      f"{'PRESENT' if present else 'ABSENT '} "
                      f"cls={lbl} conf={conf:.3f} text_conf={text_conf:.3f}  "
                      f"raw_std={_dbg_g.std():.1f}")
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

# =========================================================
# LOGGER
# =========================================================
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
                    "ic_b_result", "cycle_ms", "is_retry"]

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

    # ── internal helpers ────────────────────────────────────────

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
                w.writerow(["YIELD_PCT",   yield_])
                w.writerow(["END_TIME",    datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                w.writerow(["DURATION_S",  round(elapsed_s, 1)])
        except Exception as e:
            print(f"[Logger] result footer write failed: {e}", file=sys.stderr)

    # ── public interface ────────────────────────────────────────

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
                       cycle_ms: float, is_retry: bool):
        passed = (ic_a_result == "PASS" and ic_b_result == "PASS")
        event  = "PASS" if passed else "FAIL"
        # Build detail: image filename + missing cells if any
        detail_parts = [image_id]
        if ic_a_missing:
            detail_parts.append(f"miss_a={ic_a_missing}")
        if ic_b_missing:
            detail_parts.append(f"miss_b={ic_b_missing}")
        detail_parts.append(f"is_retry={1 if is_retry else 0}")
        self._op_append(event, " ".join(detail_parts), cycle_ms)
        # Result log row
        self._res_write([
            datetime.now().isoformat(),
            image_id,
            ic_a_result,
            ic_b_result,
            round(cycle_ms, 1),
            1 if is_retry else 0,
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

    def log_io_mock(self, pin_name: str, state: str):
        print(f"[IO MOCK] {pin_name} → {state}")

# =========================================================
# STYLESHEET
# =========================================================
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

# =========================================================
# FAIL DIALOG
# =========================================================
def _fmt_missing(cells: list) -> str:
    return ", ".join(f"[R{r}C{c}]" for r, c in cells)

class FailDialog(QtWidgets.QDialog):

    def __init__(self, missing_a: list, missing_b: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inspection Failed")
        self.setModal(True)
        self.setStyleSheet(
            "QDialog { background: #5465FF; border-radius: 10px; }"
            "QLabel  { color: #FFFFFF; }"
            "QPushButton { background:#FFFFFF; color:#5465FF; font-weight:bold;"
            "  border-radius:6px; padding:6px 24px; }"
        )
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10)
        lay.setContentsMargins(20, 20, 20, 20)

        title = QtWidgets.QLabel("Inspection Failed")
        title.setStyleSheet("font-size:16px;font-weight:bold;color:#FFFFFF")
        title.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(title)

        if missing_a:
            lbl = QtWidgets.QLabel(f"IC_A — missing: {_fmt_missing(missing_a)}")
            lbl.setStyleSheet("color:#EF5350;font-weight:bold")
            lbl.setWordWrap(True)
            lay.addWidget(lbl)

        if missing_b:
            lbl = QtWidgets.QLabel(f"IC_B — missing: {_fmt_missing(missing_b)}")
            lbl.setStyleSheet("color:#EF5350;font-weight:bold")
            lbl.setWordWrap(True)
            lay.addWidget(lbl)

        btn = QtWidgets.QPushButton("Acknowledge")
        btn.clicked.connect(self.accept)
        lay.addWidget(btn, alignment=QtCore.Qt.AlignCenter)

        self.adjustSize()

# =========================================================
# IMAGE VIEW
# =========================================================
class ImageView(QtWidgets.QLabel):
    """
    Zoomable image display with overlay support, stamp mode, and rubber-band drawing.
    """
    anchor_clicked = QtCore.pyqtSignal(QtCore.QPoint)   # unused but kept for future
    rect_drawn     = QtCore.pyqtSignal(QtCore.QRect)    # emitted on rubber-band release (image coords)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setObjectName("image_area")
        self._orig        = None
        self._scale       = 1.0
        self._offset      = QtCore.QPoint(0, 0)
        self._overlays    = []    # (QRect, QColor, label)
        self._stamp_mode  = False
        self._stamp_w     = 100
        self._stamp_h     = 60
        self._rb_mode     = False
        self._rb_start    = None  # QPoint in image coords
        self._rb_cur      = None  # QPoint in image coords (current drag position)
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

    # ---- image ----
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

    # ---- coordinate helper ----
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

    # ---- overlays ----
    def add_overlay(self, rect: QtCore.QRect, color: QtGui.QColor, label: str = ""):
        self._overlays.append((rect, color, label))
        self.update()

    def clear_overlays(self):
        self._overlays.clear()
        self.update()

    # ---- stamp mode ----
    def set_stamp_mode(self, on: bool, w: int = 100, h: int = 60):
        self._stamp_mode = on
        self._stamp_w    = w
        self._stamp_h    = h
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
        self.update()

    # ---- rubber-band mode ----
    def set_rubberband_mode(self, on: bool):
        self._rb_mode  = on
        self._rb_start = None
        self._rb_cur   = None
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
        self.update()

    # ---- paint ----
    def paintEvent(self, e):
        super().paintEvent(e)
        has_rb = self._rb_mode and self._rb_start and self._rb_cur
        if not self._overlays and not self._stamp_mode and not has_rb:
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
        if self._stamp_mode:
            self.update()
        elif self._rb_mode and self._rb_start:
            self._rb_cur = self._to_img(e.pos())
            self.update()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton and self._stamp_mode:
            img_pt = self._to_img(e.pos())
            w2, h2 = self._stamp_w // 2, self._stamp_h // 2
            rect   = QtCore.QRect(img_pt.x() - w2, img_pt.y() - h2,
                                  self._stamp_w, self._stamp_h)
            self.anchor_clicked.emit(img_pt)
            self.add_overlay(rect, QtGui.QColor("#FFD700"), "")
        elif e.button() == QtCore.Qt.LeftButton and self._rb_mode:
            self._rb_start = self._to_img(e.pos())
            self._rb_cur   = self._rb_start

    def mouseReleaseEvent(self, e):
        if self._rb_mode and self._rb_start and e.button() == QtCore.Qt.LeftButton:
            end  = self._to_img(e.pos())
            rect = QtCore.QRect(self._rb_start, end).normalized()
            self._rb_start = None
            self._rb_cur   = None
            self.update()
            if rect.width() > 5 and rect.height() > 5:
                self.rect_drawn.emit(rect)

# =========================================================
# RUN WORKER
# =========================================================
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

    MANUAL=True  + CAMERA='directory': waits for trigger() call per cycle
    MANUAL=False + CAMERA='directory': auto-loops with short delay between cycles
    CAMERA='camera': waits for GPIO START_PIN (or trigger() if MANUAL=True)
    """
    sig_image    = QtCore.pyqtSignal(object)          # annotated BGR ndarray
    sig_result   = QtCore.pyqtSignal(bool, bool)      # ic_a_pass, ic_b_pass
    sig_fail     = QtCore.pyqtSignal(object, str, str) # (MarkMissingError, ann_path, img_id)
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

    def _handle_lot_end(self):
        """Auto-advance lot on GPIO LOT_END signal and emit new lot number."""
        _reset_image_counter()
        self._lot_number = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sig_session_reset.emit(self._lot_number)

    def run(self):
        cam_mode = self._cfg.get("CAMERA", "directory")
        io_mock  = not self._cfg.get("IO", False)
        debug    = self._cfg.get("DEBUG", True)

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

        while not self._stop:

            # ── Wait for next cycle trigger ──────────────────────────
            if cam_mode == "camera":
                # Check lot-end pin before waiting for start
                if self._gpio.is_lot_end_signaled():
                    time.sleep(0.05)
                    if self._gpio.is_lot_end_signaled():
                        self.sig_status.emit("Lot end — saving NG images…")
                        self._handle_lot_end()

                # Production: wait for GPIO START_PIN active LOW
                self.sig_status.emit("Waiting for START signal…")
                if not self._gpio.wait_for_start(lambda: self._stop):
                    break
                if self._stop:
                    break

                # Tray settled — assert BUSY before grab
                self._gpio.set_busy(True)

                settle_ms = self._cfg.get("TRIGGER_SETTLE_MS", 0)
                if settle_ms > 0:
                    time.sleep(settle_ms / 1000.0)
            else:
                # Auto directory: brief yield, then check DONE_PIN / Stop
                time.sleep(0.05)
                if self._stop:
                    break
                if self._gpio.is_done_signaled():
                    self._camera.reset()
                    _reset_image_counter()
                    new_lot = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._lot_number = new_lot
                    self.sig_session_reset.emit(new_lot)
                    continue

            # ── Capture guard ────────────────────────────────────────
            if self._stop:
                break
            t0 = time.perf_counter()
            try:
                img_bgr = self._camera.grab()
            except CameraError as e:
                self._logger.log_error("CAMERA_ERROR", str(e),
                                       (time.perf_counter() - t0) * 1000)
                self.sig_error.emit(f"Camera error: {e}")
                self.sig_status.emit("ERROR — machine blocked, restart required.")
                break

            img_id = _next_image_id()

            # Save raw with temp name before result is known
            out_dir  = self._cfg.get("OUT_DIR", "Output/")
            real_dir, ann_dir = _output_dirs(out_dir, self._lot_number)
            tmp_real = os.path.join(real_dir, f"{img_id}.jpg")
            cv2.imwrite(tmp_real, img_bgr)

            self.sig_status.emit("Inspecting…")

            # ── Inspect (with one retry on fail) ────────────────────
            is_retry    = False
            miss_a      = []
            miss_b      = []
            ann         = img_bgr
            tmpl_error  = False

            try:
                self._inspector.inspect(img_bgr, debug=debug)
                # pass — img_bgr annotated in-place; miss_a/miss_b stay []

            except TemplateError as te:
                tmpl_error = True
                miss_a = miss_b = [[r, c] for r in range(1, 4) for c in range(1, 3)]
                if debug:
                    print(f"[RunWorker] Alignment rejected: {te}")

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
                break

            # ── Finalize paths and save ──────────────────────────────
            pass_a   = not miss_a
            pass_b   = not miss_b
            passed   = pass_a and pass_b and not tmpl_error
            suffix   = "_G" if passed else "_NG"
            cycle_ms = (time.perf_counter() - t0) * 1000

            final_real = os.path.join(real_dir, f"{img_id}{suffix}.jpg")
            ann_path   = os.path.join(ann_dir,  f"{img_id}{suffix}.jpg")
            try:
                os.rename(tmp_real, final_real)
            except OSError:
                final_real = tmp_real   # rename failed, keep original name

            cv2.imwrite(ann_path, ann)

            # ── Emit signals and log ─────────────────────────────────
            self.sig_image.emit(img_bgr)
            self.sig_cycle_ms.emit(cycle_ms)

            if passed:
                self.sig_result.emit(True, True)
                self._logger.log_inspection(
                    img_id, "PASS", [], "PASS", [], cycle_ms, is_retry)
                self._gpio.set_fail_a(False)
                self._gpio.set_fail_b(False)
            else:
                err = MarkMissingError(miss_a, miss_b, ann)
                self.sig_fail.emit(err, ann_path, img_id)
                self._logger.log_inspection(
                    img_id,
                    "FAIL" if miss_a else "PASS", miss_a,
                    "FAIL" if miss_b else "PASS", miss_b,
                    cycle_ms, is_retry)
                self._gpio.set_fail_a(bool(miss_a) or tmpl_error)
                self._gpio.set_fail_b(bool(miss_b) or tmpl_error)

            self._gpio.set_busy(False)

            try:
                del img_bgr
            except NameError:
                pass

            _cycle += 1
            if _cycle % 100 == 0:
                gc.collect()

            # ── End-of-cycle handshake ───────────────────────────────
            if cam_mode == "camera":
                # Production: hold outputs until machine signals DONE
                self.sig_status.emit("Holding — waiting for DONE signal…")
                self._gpio.wait_for_done(lambda: self._stop)
                self._gpio.clear_outputs()
            else:
                if not self._camera.has_more():
                    self._camera.reset()
                    break                           # directory done → standby
                if self._gpio.is_done_signaled():
                    self._camera.reset()
                    _reset_image_counter()
                    new_lot = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._lot_number = new_lot
                    self.sig_session_reset.emit(new_lot)   # IO signal → new batch

            # ── Pause checkpoint ─────────────────────────────────────
            # Sits after DONE handshake + clear_outputs so the machine
            # always receives ACK before the loop suspends.
            if not self._running.is_set():
                self.sig_paused.emit()
                self._running.wait()          # blocks until resume() or stop()
                if self._stop:
                    break
                if self._drain_needed.is_set():
                    self._gpio.drain_start_pin()
                    self._drain_needed.clear()
                self.sig_resumed.emit()

        self._gpio.clear_outputs()
        if cam_mode != "camera":
            self.sig_done.emit()
        self.sig_status.emit("Standby.")

# =========================================================
# LOT START DIALOG
# =========================================================
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
    def request(cls, parent=None) -> str | None:
        """Returns lot number string, or None if operator cancelled."""
        api_lot = cls.get_lot_number_from_api()
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


# =========================================================
# IMAGE BROWSER — worker threads + widgets
# =========================================================

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

        if "_NG" in filename:
            card_bg = "#FA6781"
        elif "_G" in filename:
            card_bg = "#478B8D"
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
        self._out_dir       = out_dir
        self._all_paths: list = []     # all files in selected folder/subfolder
        self._paths: list    = []      # filtered paths shown in grid
        self._cur_idx        = 0
        self._subfolder      = "RealImg"    # "RealImg" or "Image"
        self._suffix_filter  = "_NG"        # "_NG", "_G", or "" (all)
        self._cards: list    = []
        self._current_base: str = ""
        self._thumb_worker: ThumbnailWorker | None = None
        self._scan_worker:  FolderScanWorker | None = None

        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Left: folder list ────────────────────────────────
        self._folder_list = QtWidgets.QListWidget()
        self._folder_list.setFixedWidth(200)
        self._folder_list.setStyleSheet(
            "QListWidget{background:#788BFF;border-radius:6px;color:#FFFFFF;font-size:11px}"
            "QListWidget::item:selected{background:#5465FF;color:#FFFFFF}"
        )
        self._folder_list.itemClicked.connect(self._on_folder_selected)
        root.addWidget(self._folder_list)

        # ── Centre: stacked (grid / image) ──────────────────
        self._stack = QtWidgets.QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        # Stack index 0: grid page
        grid_page = QtWidgets.QWidget()
        grid_lay  = QtWidgets.QVBoxLayout(grid_page)
        grid_lay.setContentsMargins(0, 0, 0, 0)
        self._scroll = QtWidgets.QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
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

        # ── Right: controls ──────────────────────────────────
        right = QtWidgets.QFrame()
        right.setObjectName("panel_right")
        right.setFixedWidth(160)
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

        # Filter toggle: _NG / _G / All
        right_lay.addWidget(self._section_label("Filter"))
        self._grp_flt = QtWidgets.QButtonGroup(self)
        self._btn_flt_ng  = self._toggle_btn("_NG", checked=True)
        self._btn_flt_g   = self._toggle_btn("_G",  checked=False)
        self._btn_flt_all = self._toggle_btn("All", checked=False)
        self._grp_flt.addButton(self._btn_flt_ng,  0)
        self._grp_flt.addButton(self._btn_flt_g,   1)
        self._grp_flt.addButton(self._btn_flt_all, 2)
        right_lay.addWidget(self._btn_flt_ng)
        right_lay.addWidget(self._btn_flt_g)
        right_lay.addWidget(self._btn_flt_all)
        self._grp_flt.buttonClicked.connect(self._on_filter_toggle)

        # Count label
        self._lbl_count = QtWidgets.QLabel("—")
        self._lbl_count.setStyleSheet("font-size:16px;font-weight:bold;color:#E2FDFF")
        self._lbl_count.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_count.setWordWrap(True)
        right_lay.addWidget(self._lbl_count)

        right_lay.addStretch()

        # Back button (shown in image view mode)
        self._btn_back = QtWidgets.QPushButton("← Back")
        self._btn_back.clicked.connect(self._back_to_grid)
        self._btn_back.hide()
        right_lay.addWidget(self._btn_back)

        root.addWidget(right)

    # ── resize ───────────────────────────────────────────────

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._paths and e.size() != e.oldSize():
            QtCore.QTimer.singleShot(150, self._rebuild_grid)

    def _card_size(self, n_images: int):
        """Calculate card/thumbnail dimensions so all rows fit in the viewport."""
        vp_w = self._scroll.viewport().width()
        vp_h = self._scroll.viewport().height()
        if vp_w < 4 or vp_h < 4:    # not yet laid out — return safe defaults
            return 100, 96, 96, 72

        n_rows = max(1, math.ceil(n_images / self._COLS))
        sp     = self._grid_layout.horizontalSpacing()
        sp_v   = self._grid_layout.verticalSpacing()

        # Width: 4 cards fill viewport width
        w_avail = max(1, vp_w - sp * (self._COLS - 1) - 4)
        card_w  = max(60, w_avail // self._COLS)

        # Height: all rows fit in viewport height
        h_avail = max(1, vp_h - sp_v * (n_rows - 1) - 4)
        card_h  = max(50, h_avail // n_rows)

        # Thumbnail area fills card minus 2px margin and 22px filename label
        thumb_w = card_w - 4
        thumb_h = max(1, card_h - 22)
        # card_h already set above — no ratio cap; Qt scales the image within the box

        return card_w, card_h, max(1, thumb_w), max(1, thumb_h)

    # ── helpers ─────────────────────────────────────────────

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

    # ── folder refresh ───────────────────────────────────────

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

    # ── image loading ────────────────────────────────────────

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

    def _apply_filter_and_build_grid(self):
        """Filter self._all_paths by suffix, rebuild grid."""
        if self._suffix_filter:
            filtered = [p for p in self._all_paths
                        if self._suffix_filter in os.path.basename(p)]
            # Fall back to all if no matching suffix found (old files without suffix)
            self._paths = filtered if filtered else self._all_paths
        else:
            self._paths = list(self._all_paths)

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

        card_w, card_h, thumb_w, thumb_h = self._card_size(len(self._paths))

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

    # ── image view ───────────────────────────────────────────

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
        path = self._paths[self._cur_idx]
        img  = cv2.imread(path)
        if img is not None:
            self._img_view.set_image(img)
        self._lbl_nav.setText(
            f"{self._cur_idx + 1} / {len(self._paths)}   {os.path.basename(path)}")

    def _step_image(self, delta: int):
        self._show_image(self._cur_idx + delta)

    def _back_to_grid(self):
        self._stack.setCurrentIndex(0)
        self._btn_back.hide()

    # ── toggle handlers ──────────────────────────────────────

    def _on_src_toggle(self, btn):
        self._subfolder = "RealImg" if self._grp_src.id(btn) == 0 else "Image"
        if self._current_base:
            self._load_folder(self._current_base)
        if self._stack.currentIndex() == 1:
            self._show_image(self._cur_idx)

    def _on_filter_toggle(self, btn):
        flt_id = self._grp_flt.id(btn)
        self._suffix_filter = {0: "_NG", 1: "_G", 2: ""}[flt_id]
        self._apply_filter_and_build_grid()


# =========================================================
# MAIN WINDOW
# =========================================================
class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, cfg: dict):
        super().__init__()
        self.setWindowTitle("ClearIC Inspect")
        self._cfg               = cfg
        self._camera:    Camera | None  = None
        self._detector:  Detector | None = None
        self._gpio       = None
        self._logger            = Logger(
            log_dir=cfg.get("LOG_DIR", "logs"),
            log_retention=int(cfg.get("LOG_RETENTION", 365)))
        self._worker:           RunWorker | None        = None

        self._stats_pass  = 0
        self._stats_fail  = 0
        self._stats_error = 0
        self._stats_total = 0

        self._run_state          = "standby"   # "standby" | "running" | "paused"
        self._session_start_time = 0.0
        self._lot_number         = ""
        self._package_name       = ""

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

    # ----------------------------------------------------------
    # UI construction
    # ----------------------------------------------------------
    def _build_ui(self):
        # ── Tab wrapper ──────────────────────────────────────
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

        # ── Left panel ───────────────────────────────────────
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

        # ── Right panel ──────────────────────────────────────
        right_frame = QtWidgets.QFrame()
        right_frame.setObjectName("panel_right")
        right_frame.setFixedWidth(280)
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
        self._btn_new_tmpl.clicked.connect(self._start_draw_a)
        setup_lay.addWidget(self._btn_new_tmpl)

        self._btn_confirm_tmpl = QtWidgets.QPushButton("Confirm")
        self._btn_confirm_tmpl.clicked.connect(self._confirm_template)
        self._btn_confirm_tmpl.setEnabled(False)
        self._btn_confirm_tmpl.setStyleSheet(
            "background:#FFFFFF;color:#5465FF;border-radius:6px;"
            "padding:6px 14px;font-weight:bold;")
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
        self._chk_labels.setChecked(bool(self._cfg.get("ANN_SHOW_LABELS", True)))
        settings_lay.addWidget(self._chk_labels)

        btn_apply = QtWidgets.QPushButton("Apply")
        btn_apply.clicked.connect(self._apply_settings)
        settings_lay.addWidget(btn_apply)

        settings_frame.setVisible(False)
        right_lay.addWidget(settings_frame)
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

    # ----------------------------------------------------------
    # Settings apply
    # ----------------------------------------------------------
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
                          "ANN_SHOW_LABELS": show_labels})
        ConfigLoader.save(self._cfg)

        print(f"[Settings] border={bp}px  labels={show_labels}  warmup={wf}")

    # ----------------------------------------------------------
    # System init
    # ----------------------------------------------------------
    def _init_system(self):
        cfg = self._cfg
        try:
            self._detector = Detector(
                conf_thr=cfg.get("CONF_THR", 0.5),
                text_min_conf=cfg.get("TEXT_MIN_CONF", 0.80),
                blank_cell_std_thr=cfg.get("BLANK_CELL_STD_THR", 0.0),
                model_path=cfg.get("MODEL_PATH",
                                   "Text_cls-2/best_openvino_model/best.xml"),
            )
        except ModelError as e:
            self._show_error(f"Classifier model load failed: {e}")

        try:
            self._gpio = RaspberryIO(
                io_enabled=cfg.get("IO", False),
                start_pin=cfg.get("GPIO_START_PIN", 17),
                done_pin=cfg.get("GPIO_DONE_PIN", 27),
                busy_pin=cfg.get("GPIO_BUSY_PIN", 23),
                lot_end_pin=cfg.get("GPIO_LOT_END_PIN", 18),
                fail_a_pin=cfg.get("GPIO_FAIL_A_PIN", 24),
                fail_b_pin=cfg.get("GPIO_FAIL_B_PIN", 25),
                mock_start_delay_ms=cfg.get("MOCK_START_DELAY_MS", 200),
                mock_done_delay_ms=cfg.get("MOCK_DONE_DELAY_MS", 100),
            )
        except GPIOError as e:
            self._show_error(f"GPIO init failed: {e}")

        try:
            self._camera = Camera(
                mode=cfg.get("CAMERA", "directory"),
                serial=cfg.get("CAMERA_SERIAL", ""),
                exposure_us=cfg.get("EXPOSURE_US", 8000),
                input_dir=cfg.get("DIR_INPUT", "Input/"),
                retry_delay=cfg.get("CAMERA_RETRY_DELAY", 0.2),
                retries=cfg.get("CAMERA_RETRIES", 2),
                warmup_frames=cfg.get("CAMERA_WARMUP_FRAMES", 5),
            )
            self._camera.open()
        except CameraError as e:
            self._show_error(f"Camera init failed: {e}")

        if self._camera and cfg.get("CAMERA") == "camera":
            self._camera.warmup()
        if self._detector and self._detector.is_ready():
            self._detector.warmup(frames=cfg.get("WARMUP_FRAMES", 5))

        # Load and display first image on startup (no overlays yet)
        if self._camera:
            try:
                img = self._camera.grab_first()
                self._view.set_image(img)
                self._setup_image = img
            except CameraError:
                pass

        # Apply initial button state — disables Start if no template exists yet.
        self._update_setup_buttons()


    # ----------------------------------------------------------
    # Rubber-band template setup flow
    # ----------------------------------------------------------
    def _grab_setup_frame(self) -> np.ndarray | None:
        if self._camera is None:
            self._show_error("Camera not ready.")
            return None
        try:
            return self._camera.grab_first()
        except CameraError as e:
            self._show_error(str(e))
            return None

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

    def _confirm_template(self):
        if self._pending_ic_a and self._pending_ic_b:
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
        self._btn_new_tmpl.setEnabled(s in ('idle', 'ready'))
        self._btn_confirm_tmpl.setEnabled(s == 'ready')
        if s == 'idle':
            template_ok = os.path.exists(_TEMPLATE_FILE)
            text = ("Template saved."
                    if template_ok
                    else "No template — create a template before running.")
            # Gate the Start button: only allow run when a template exists.
            if self._run_state == "standby":
                self._btn_action.setEnabled(template_ok)
        else:
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

        TemplateManager.save(ic_a, ic_b, exposure, strip_h=strip_h_val)

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


    # ----------------------------------------------------------
    # Run / Pause / Stop
    # ----------------------------------------------------------
    def _on_action_click(self):
        if self._run_state == "standby":
            self._start_run()
        elif self._run_state == "running":
            self._pause_run()
        elif self._run_state == "paused":
            self._resume_run()

    def _start_run(self):
        if self._worker and self._worker.isRunning():
            return
        try:
            tmpl = TemplateManager.load()
        except TemplateError as e:
            self._show_error(str(e))
            return
        if not self._detector or not self._detector.is_ready():
            self._show_error("Detector not ready.")
            return

        # Ask operator for lot number (or get from API hook)
        lot = LotStartDialog.request(parent=self)
        if lot is None:
            return   # operator cancelled
        self._lot_number   = lot
        self._package_name = tmpl.get("package_name", "")

        self._session_start_time = time.monotonic()

        mode = "DEBUG" if self._cfg.get("DEBUG", True) else "RUN"
        self._logger.start_lot(self._lot_number, self._package_name, mode)

        # IC localization via TemplateMatcher when patch exists; else fixed template coords.
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
            )

        inspector = Inspector(
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
            ann_show_labels=self._cfg.get("ANN_SHOW_LABELS", True),
        )
        gpio      = self._gpio or RaspberryIO(io_enabled=False)

        self._worker = RunWorker(
            self._camera, inspector, gpio, self._logger, self._cfg,
            lot_number=self._lot_number)
        self._worker.sig_image.connect(self._on_image)
        self._worker.sig_result.connect(self._on_result)
        self._worker.sig_fail.connect(self._on_fail)
        self._worker.sig_error.connect(self._on_worker_error)
        self._worker.sig_status.connect(self._lbl_status.setText)
        self._worker.sig_cycle_ms.connect(
            lambda ms: self._lbl_cycle_ms.setText(f"{ms:.0f}"))
        self._worker.sig_done.connect(self._on_run_done)
        self._worker.sig_session_reset.connect(self._on_session_reset)
        self._worker.sig_paused.connect(self._on_paused)
        self._worker.sig_resumed.connect(self._on_resumed)
        self._worker.start()

        self._run_state = "running"
        self._btn_action.setText("Pause")
        self._btn_stop.setEnabled(True)

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
        """Called when the worker loop exits (directory mode done)."""
        elapsed = time.monotonic() - self._session_start_time
        self._logger.end_lot(
            "COMPLETE", self._stats_pass, self._stats_fail,
            self._stats_error, elapsed)
        self._enter_standby()

    def _enter_standby(self):
        self._run_state = "standby"
        self._btn_action.setText("Start")
        self._btn_action.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._update_badge(self._badge_a, None)
        self._update_badge(self._badge_b, None)
        self._stats_pass = self._stats_fail = self._stats_error = self._stats_total = 0
        self._lbl_pass.setText("0")
        self._lbl_fail.setText("0")
        self._lbl_error.setText("0")
        self._lbl_yield.setText("—")
        self._lbl_status.setText("Standby.")
        self._reload_default_image()

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

    # ----------------------------------------------------------
    # Worker signal handlers
    # ----------------------------------------------------------
    def _on_image(self, img: np.ndarray):
        self._view.set_image(img)

    def _update_yield(self):
        total = self._stats_pass + self._stats_fail
        if total > 0:
            self._lbl_yield.setText(f"{self._stats_pass / total * 100:.1f}%")
        else:
            self._lbl_yield.setText("—")

    def _on_result(self, ia_pass: bool, ib_pass: bool):
        self._update_badge(self._badge_a, ia_pass)
        self._update_badge(self._badge_b, ib_pass)
        self._stats_pass  += 1
        self._stats_total += 1
        self._lbl_pass.setText(str(self._stats_pass))
        self._update_yield()

    def _on_fail(self, err: MarkMissingError, ann_path: str, img_id: str):
        ic_a_pass = len(err.missing_a) == 0
        ic_b_pass = len(err.missing_b) == 0
        self._update_badge(self._badge_a, ic_a_pass)
        self._update_badge(self._badge_b, ic_b_pass)
        self._stats_fail  += 1
        self._stats_total += 1
        self._lbl_fail.setText(str(self._stats_fail))
        self._update_yield()

    def _on_worker_error(self, msg: str):
        self._stats_error += 1
        self._lbl_error.setText(str(self._stats_error))
        self._show_error(msg)
        elapsed = time.monotonic() - self._session_start_time
        self._logger.end_lot(
            "ERROR", self._stats_pass, self._stats_fail,
            self._stats_error, elapsed)
        self._run_state = "standby"
        self._btn_action.setText("Start")
        self._btn_action.setEnabled(True)
        self._btn_stop.setEnabled(False)

    # ----------------------------------------------------------
    # Error banner
    # ----------------------------------------------------------
    def _show_error(self, msg: str):
        self._error_lbl.setText(f"Error: {msg}")
        self._error_banner.show()

    def _clear_error(self):
        self._error_banner.hide()

    # ----------------------------------------------------------
    # Close
    # ----------------------------------------------------------
    def closeEvent(self, e):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        if self._camera:
            self._camera.close()
        if self._gpio:
            self._gpio.cleanup()
        e.accept()

# =========================================================
# ENTRY POINT
# =========================================================
def main():
    try:
        cfg = ConfigLoader.load()
    except ConfigError as e:
        print(f"[Config] {e}")
        sys.exit(1)

    os.makedirs(cfg.get("LOG_DIR", "logs"), exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs(cfg.get("DIR_INPUT", "Input/"), exist_ok=True)
    if cfg.get("COLLECT_DATASET", False):
        _dd, _ds = cfg.get("DATA_DIR", "Dataset"), cfg.get("DATA_SPLIT", "train")
        os.makedirs(os.path.join(_dd, _ds, "Text"),   exist_ok=True)
        os.makedirs(os.path.join(_dd, _ds, "NoText"), exist_ok=True)
        print(f"[Dataset] Collection ON → {_dd}/{_ds}/")

    app = QtWidgets.QApplication(sys.argv)
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
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
