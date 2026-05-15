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
import json
import glob
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
# HARDCODED DEV FLAGS  (edit here before running — not in Config.json)
# =========================================================
DEBUG     = True      # verbose logs, annotated output
IO        = False     # True = drive GPIO / False = mock (log only)
MODE      = "DEBUG"   # "DEBUG" or "RUN"
DIR_INPUT = "Input/"   # input image folder for CAMERA="directory" mode
OUT_DIR   = "Output/" # Output image foler for annotated results (created on first run)
MODEL_PATH          = "Text_cls-2/best_openvino_model/best.xml"       # YOLO-cls classifier (cell inspection)
TEMPLATE_MODEL_PATH = "IC_Search_openvino_model/IC_Search.xml"               # (unused — kept for reference only)
COLLECT_DATASET = False  # True = save cropped cell images to dataset/ for retraining

# =========================================================
# CONFIG LOADER
# =========================================================
class ConfigLoader:
    CONFIG_FILE = "Config.json"
    DEFAULT_CONFIG = {
        "CAMERA":              "directory",
        "CONF_THR":            0.5,
        "TEXT_MIN_CONF":       0.80,
        "BLANK_CELL_STD_THR":  0.0,
        "CAMERA_SERIAL":       "",
        "EXPOSURE_US":         8000,
    }
    _VALID_CAMERA = {"camera", "directory"}

    @classmethod
    def load(cls) -> dict:
        if not os.path.exists(cls.CONFIG_FILE):
            cls.save(cls.DEFAULT_CONFIG)
            return dict(cls.DEFAULT_CONFIG)
        try:
            with open(cls.CONFIG_FILE, "r") as f:
                data = json.load(f)
        except Exception as e:
            raise ConfigError(f"Config.json unreadable: {e}")
        cfg = dict(cls.DEFAULT_CONFIG)
        # Only apply keys that belong in Config.json (ignore dev flags if present)
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
        return cfg

    @classmethod
    def save(cls, data: dict):
        with open(cls.CONFIG_FILE, "w") as f:
            json.dump(data, f, indent=2)

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
                 annotated: "np.ndarray | None" = None):
        self.missing_a = missing_a
        self.missing_b = missing_b
        self.annotated = annotated
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

@dataclass
class Image:
    id:        str
    raw:       np.ndarray
    annotated: np.ndarray = field(default=None)

# =========================================================
# CAMERA
# =========================================================
_CAMERA_WARMUP_FRAMES = 5
_CAMERA_RETRY_DELAY   = 0.2
_CAMERA_RETRIES       = 2

class Camera:
    """
    Unified camera source.
    CAMERA='camera'    : Basler pypylon InstantCamera
    CAMERA='directory' : reads files from Input/ in sorted order, loops
    """

    def __init__(self, mode: str, serial: str = "",
                 exposure_us: int = 8000, input_dir: str = "Input"):
        self._mode        = mode
        self._serial      = serial
        self._exposure_us = exposure_us
        self._input_dir   = input_dir

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
        for attempt in range(_CAMERA_RETRIES + 1):
            try:
                img = self._grab_once()
                if img is not None:
                    return img
            except CameraError:
                raise
            except Exception as e:
                if attempt < _CAMERA_RETRIES:
                    time.sleep(_CAMERA_RETRY_DELAY)
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
            for _ in range(_CAMERA_WARMUP_FRAMES):
                try:
                    self._grab_basler()
                except Exception:
                    pass
            print(f"[Camera] Warmup done ({_CAMERA_WARMUP_FRAMES} frames).")

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
START_PIN  = 17
DONE_PIN   = 27
ACK_PIN    = 22
FAIL_A_PIN = 24
FAIL_B_PIN = 25

# Mock-mode self-pulse delays (IO=False). Replace with real IO when hardware is ready.
_MOCK_START_DELAY_MS = 200   # simulated delay before START_PIN fires each cycle
_MOCK_DONE_DELAY_MS  = 100   # simulated delay after ACK before DONE_PIN fires

class RaspberryIO:
    """
    BCM-mode GPIO handler.
    Falls back to mock logging when IO=False or RPi.GPIO unavailable.
    """

    def __init__(self, io_enabled: bool = True):
        self._enabled = io_enabled
        self._gpio_ok = False
        self._GPIO    = None

        if not io_enabled:
            print("[IO] IO=False — mock mode.")
            return

        try:
            import RPi.GPIO as GPIO
            self._GPIO = GPIO
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(START_PIN,  GPIO.IN,  pull_up_down=GPIO.PUD_DOWN)
            GPIO.setup(DONE_PIN,   GPIO.IN,  pull_up_down=GPIO.PUD_DOWN)
            GPIO.setup(ACK_PIN,    GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(FAIL_A_PIN, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(FAIL_B_PIN, GPIO.OUT, initial=GPIO.LOW)
            self._gpio_ok = True
            print("[IO] GPIO initialised (BCM mode).")
        except Exception as e:
            raise GPIOError(f"GPIO init failed: {e}")

    def _out(self, pin: int, high: bool, pin_name: str = ""):
        if self._gpio_ok:
            self._GPIO.output(pin, self._GPIO.HIGH if high else self._GPIO.LOW)
        else:
            state = "HIGH" if high else "LOW"
            print(f"[IO MOCK] {pin_name or pin} → {state}")

    def set_fail_a(self, v: bool):
        self._out(FAIL_A_PIN, v, "FAIL_A_PIN")

    def set_fail_b(self, v: bool):
        self._out(FAIL_B_PIN, v, "FAIL_B_PIN")

    def pulse_ack(self, ms: int = 50):
        self._out(ACK_PIN, True,  "ACK_PIN")
        time.sleep(ms / 1000.0)
        self._out(ACK_PIN, False, "ACK_PIN")

    def clear_outputs(self):
        self._out(ACK_PIN,    False, "ACK_PIN")
        self._out(FAIL_A_PIN, False, "FAIL_A_PIN")
        self._out(FAIL_B_PIN, False, "FAIL_B_PIN")

    def wait_for_start(self, stop_flag_fn) -> bool:
        """Block until START_PIN rising edge or stop_flag_fn() returns True."""
        if not self._gpio_ok:
            # Mock: self-pulse after _MOCK_START_DELAY_MS, honouring stop requests.
            deadline = time.monotonic() + _MOCK_START_DELAY_MS / 1000
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
            if GPIO.input(START_PIN) == GPIO.HIGH:
                time.sleep(0.005)
                if GPIO.input(START_PIN) == GPIO.HIGH:
                    return True
            time.sleep(0.005)
        return False

    def wait_for_done(self, stop_flag_fn) -> bool:
        """Block until DONE_PIN rising edge or stop_flag_fn() returns True."""
        if not self._gpio_ok:
            # Mock: self-pulse after _MOCK_DONE_DELAY_MS, honouring stop requests.
            deadline = time.monotonic() + _MOCK_DONE_DELAY_MS / 1000
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
            if GPIO.input(DONE_PIN) == GPIO.HIGH:
                time.sleep(0.005)
                if GPIO.input(DONE_PIN) == GPIO.HIGH:
                    return True
            time.sleep(0.005)
        return False

    def is_done_signaled(self) -> bool:
        """Non-blocking: True if DONE_PIN is currently HIGH. Always False in mock mode."""
        if not self._gpio_ok:
            return False
        return self._GPIO.input(DONE_PIN) == self._GPIO.HIGH

    def drain_start_pin(self, timeout_ms: int = 500):
        """Wait until START_PIN is LOW (or timeout) to discard stale HIGH after resume."""
        if not self._gpio_ok:
            print("[IO MOCK] drain_start_pin")
            return
        GPIO = self._GPIO
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            if not GPIO.input(START_PIN):
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

    MODEL_XML = MODEL_PATH

    def __init__(self, conf_thr: float = 0.5, text_min_conf: float = 0.80,
                 blank_cell_std_thr: float = 0.0, **_):
        self._conf_thr           = conf_thr
        self._text_min_conf      = text_min_conf
        self._blank_cell_std_thr = blank_cell_std_thr
        self._compiled = None
        self._ready    = False
        try:
            import openvino as ov
            if not os.path.exists(self.MODEL_XML):
                raise ModelError(f"Model not found: {self.MODEL_XML}")
            core  = ov.Core()
            model = core.read_model(self.MODEL_XML)
            self._compiled = core.compile_model(model, "CPU", {
                "INFERENCE_PRECISION_HINT": "f32",
                "PERFORMANCE_HINT":         "LATENCY",
            })
            self._ready = True
            print(f"[Detector] OpenVINO classifier loaded: {self.MODEL_XML}")
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

# =========================================================
# CELL GRID CONSTANTS
# =========================================================
_CELL_SHRINK    = 0.90   # IC rect shrunk before slicing (keeps grid off raw edges)
_CELL_EXPAND    = 1.05   # each cell expanded after slicing (overlapping neighbours)
_COL_GAP_PCT    = 40.0   # column gap as % of (shrunk) IC width
_GRID_MARGIN_TOP = 10.0  # % of shrunk IC height — dead band before row 1
_GRID_MARGIN_BOT = 10.0  # % of shrunk IC height — dead band after row 3

# Dataset collection (used only when COLLECT_DATASET = True)
_DATA_DIR   = "Dataset"
_DATA_SPLIT = "train"    # "train" | "val" | "test"
_data_run_counter = 0
_dataset_lock     = threading.Lock()

# =========================================================
# VISUAL SETTINGS  (live-editable from the Settings panel)
# =========================================================
_ann_border_px   = 1           # ROI cell border thickness (px)
_ann_show_labels = True        # draw R{row}C{col} inside each cell
_ann_color_ok    = "#00C800"   # hex — Text  / PASS cell border
_ann_color_ng    = "#DD0000"   # hex — NoText / FAIL cell border
_tmpl_color_a    = "#FFD700"   # hex — IC_A overlay in setup view
_tmpl_color_b    = "#00E5FF"   # hex — IC_B overlay in setup view
_warmup_frames   = 5           # classifier warmup passes on startup

def _hex_to_bgr(h: str) -> tuple:
    """Convert '#RRGGBB' hex string to OpenCV BGR 3-tuple."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

def _valid_hex(s: str) -> bool:
    s = s.strip().lstrip("#")
    return len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s)


def _build_cells(x: int, y: int, w: int, h: int) -> list:
    """
    Build the 3-row × 2-col cell list for one IC bounding rect.

    Steps:
      1. Apply horizontal shrink (_CELL_SHRINK, L/R) and independent
         vertical margins (_GRID_MARGIN_TOP / _GRID_MARGIN_BOT, top/bot).
      2. Slice the resulting rect into a 3×2 grid with _COL_GAP_PCT applied.
      3. Expand every cell by _CELL_EXPAND (centred), so adjacent cells
         overlap — text marks near a boundary are covered by both cells.
    """
    # Step 1 — shrink (centred)
    sw = max(1, int(w * _CELL_SHRINK))
    sh = max(1, int(h * _CELL_SHRINK))
    sx = x + (w - sw) // 2
    sy = y + (h - sh) // 2

    # Step 2 — vertical margins then 3×2 grid on usable area
    usable_y0 = sy + int(sh * _GRID_MARGIN_TOP / 100.0)
    usable_y1 = sy + sh - int(sh * _GRID_MARGIN_BOT / 100.0)
    usable_h  = max(1, usable_y1 - usable_y0)
    col_gap   = int(sw * _COL_GAP_PCT / 100.0)
    cw        = max(1, (sw - col_gap) // 2)
    ch        = max(1, usable_h // 3)
    col_starts = [sx, sx + cw + col_gap]

    # Step 3 — expand each cell (centred)
    exp_w = max(1, int(cw * _CELL_EXPAND))
    exp_h = max(1, int(ch * _CELL_EXPAND))
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
                     cell_hits: list, ic_label: str, run_num: int):
    """
    Save each ROI cell crop to Dataset/<split>/Text/ or .../NoText/.
    Called only when COLLECT_DATASET = True.
    Filename: {run_num:06d}_IC{label}_{idx:02d}.png
    """
    ih, iw = image_bgr.shape[:2]
    for idx, (cx, cy, cw, ch) in enumerate(cells):
        class_name = "Text" if cell_hits[idx] else "NoText"
        folder = os.path.join(_DATA_DIR, _DATA_SPLIT, class_name)
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

def _bilateral_binary(image_bgr: np.ndarray) -> np.ndarray:
    """BGR → Otsu binary via bilateral-filtered grayscale (params tuned for Basler optics)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.bilateralFilter(gray, 9, 75, 75)
    _, binary = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

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
        Extract a single combined patch: [top_strip | IC body | bot_strip] and apply
        bilateral-binary preprocessing.

        Strip height = IC_H * 0.5 below the IC (bot strip only; top strip disabled).
        Returns (full_patch, strip_h) where strip_h is the pixel offset from the patch
        top to the IC top edge (0 because patch starts at IC top).
        """
        x, y = ic_rect.x(), ic_rect.y()
        w, h = ic_rect.width(), ic_rect.height()
        h1 = max(1, int(h * 0.5))  # strip height = 50% of IC height

        img_h, img_w = image_bgr.shape[:2]
        # y_start = max(0, y - h1)  # top strip disabled
        y_start = y
        y_end   = min(img_h, y + h + h1)
        x_end   = min(x + w, img_w)

        full_bin = _bilateral_binary(image_bgr[y_start:y_end, x:x_end])
        strip_h  = 0  # y - y_start  # top strip disabled; patch starts at IC top

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
        Save an annotated preview image showing what the program detected:
        - IC_A box (yellow) and IC_B box (cyan) with labels
        - 3×2 cell grid inside each IC box
        - Top/bottom strip ROIs used for template matching (magenta), at their
          actual computed positions per the strip formula
        Saved to templates/template_preview.png for visual verification.
        """
        os.makedirs("templates", exist_ok=True)
        img_h = image_bgr.shape[0]
        preview = image_bgr.copy()

        for rect, color, label in [
            (ic_a, (0, 255, 255), "IC_A"),   # yellow in BGR
            (ic_b, (255, 215, 0), "IC_B"),   # cyan in BGR
        ]:
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
            # Outer IC box
            cv2.rectangle(preview, (x, y), (x + w, y + h), color, 2)
            cv2.putText(preview, label, (x + 4, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # 3×2 cell grid
            cw, ch = w // 2, h // 3
            for row in range(3):
                for col in range(2):
                    cx, cy = x + col * cw, y + row * ch
                    cv2.rectangle(preview, (cx, cy), (cx + cw, cy + ch), color, 1)
                    cv2.putText(preview, f"R{row+1}C{col+1}",
                                (cx + 2, cy + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            # Center cross at IC centroid
            cx, cy = x + w // 2, y + h // 2
            arm = max(12, min(w, h) // 6)
            cv2.line(preview, (cx - arm, cy), (cx + arm, cy), (255, 255, 255), 2)
            cv2.line(preview, (cx, cy - arm), (cx, cy + arm), (255, 255, 255), 2)
            cv2.circle(preview, (cx, cy), 3, (255, 255, 255), -1)
            # Strip ROI — same geometry as extract_patches
            h1 = max(1, int(h * 0.5))
            y1 = max(0, y - h1)
            y2 = max(0, min(y + h, img_h - h1))
            cv2.rectangle(preview, (x, y1), (x + w, y1 + h1), (255, 0, 255), 2)
            cv2.rectangle(preview, (x, y2), (x + w, y2 + h1), (255, 0, 255), 2)
            cv2.putText(preview, "TOP leads", (x + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
            cv2.putText(preview, "BOT leads", (x + 2, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

        cv2.imwrite(_TEMPLATE_PREVIEW, preview)

    @staticmethod
    def compute_rois(template: dict) -> tuple:
        """Returns (ic_a_cells, ic_b_cells) — list of 6 (x,y,w,h) per IC."""
        def _cells(box: dict) -> list:
            return _build_cells(box["x"], box["y"], box["w"], box["h"])
        return _cells(template["ic_a"]), _cells(template["ic_b"])

# =========================================================
# TEMPLATE MATCHER
# =========================================================
class TemplateMatcher:
    """
    Locates IC_A in a new image using a single bilateral-binary combined patch
    (top strip + IC body + bottom strip) matched with cv2.TM_CCOEFF_NORMED.

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

    def _match_in_roi(self, filtered: np.ndarray, patch: np.ndarray,
                      exp_x: int, exp_y: int) -> tuple:
        """
        Search for patch within ±margin around (exp_x, exp_y).
        Returns (abs_x, abs_y, score). Falls back to full image if ROI is too small.
        """
        img_h, img_w = filtered.shape[:2]
        ph, pw = patch.shape[:2]
        m = self._margin

        rx1 = max(0, exp_x - m)
        ry1 = max(0, exp_y - m)
        rx2 = min(img_w, exp_x + pw + m)
        ry2 = min(img_h, exp_y + ph + m)
        roi  = filtered[ry1:ry2, rx1:rx2]

        if roi.shape[0] < ph or roi.shape[1] < pw:
            # ROI too small — fall back to full-image search
            res = cv2.matchTemplate(filtered, patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            return loc[0], loc[1], float(score)

        res = cv2.matchTemplate(roi, patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        return loc[0] + rx1, loc[1] + ry1, float(score)

    def locate_ic(self, image_bgr: np.ndarray) -> tuple:
        """
        Returns (QRect, score). Raises TemplateError when score < threshold.
        Matches the combined (top strip + IC body + bot strip) patch against the frame.
        Searches only within ±search_margin of the expected position.
        """
        # Apply same preprocessing as extract_patches: grayscale → bilateral → Otsu
        filtered = _bilateral_binary(image_bgr)

        # Expected top of combined patch in image (IC_y minus the top-strip offset)
        exp_y = self._ic_y - self._strip_h

        mx, my, score = self._match_in_roi(filtered, self._patch, self._ic_x, exp_y)

        if score < self._threshold:
            raise TemplateError(
                f"Match score {score:.3f} < {self._threshold:.3f} — "
                "IC rotation or misalignment detected")

        ic_y = my + self._strip_h
        ic_x = mx
        return QtCore.QRect(ic_x, ic_y, self._patch_w, self._ic_h), score

# =========================================================
# INSPECTOR
# =========================================================
class Inspector:
    """
    Crops each ROI cell from the image and classifies it as Text / NoText.
    Raises MarkMissingError if either IC has any cell classified as NoText.
    """

    def __init__(self, detector: Detector, template: dict,
                 template_matcher: "TemplateMatcher | None" = None):
        self._detector         = detector
        self._template         = template
        self._template_matcher = template_matcher
        self._ic_b_dx = template["ic_b"]["x"] - template["ic_a"]["x"]
        self._ic_b_dy = template["ic_b"]["y"] - template["ic_a"]["y"]

    def inspect(self, image_bgr: np.ndarray,
                debug: bool = False) -> tuple:
        """
        Returns (ic_a_pass, ic_b_pass, missing_a, missing_b, annotated_bgr).
        Raises MarkMissingError if either IC fails.
        Raises TemplateError if template matching rejects the frame.

        Phase 1 — locate IC_A via TemplateMatcher (preferred) or fixed template coords.
        Phase 2 — crop each ROI cell and classify as Text / NoText.
        """
        tmpl_a_cells, tmpl_b_cells = TemplateManager.compute_rois(self._template)
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
        missing_a, hits_a = self._check_ic(image_bgr, ic_a_cells, annotated, debug)
        missing_b, hits_b = self._check_ic(image_bgr, ic_b_cells, annotated, debug)

        if COLLECT_DATASET:
            global _data_run_counter
            with _dataset_lock:
                _data_run_counter += 1
                run_num = _data_run_counter
            _save_cell_crops(image_bgr, ic_a_cells, hits_a, "A", run_num)
            _save_cell_crops(image_bgr, ic_b_cells, hits_b, "B", run_num)

        if missing_a or missing_b:
            raise MarkMissingError(missing_a, missing_b, annotated)

        return True, True, [], [], annotated

    @staticmethod
    def _rect_to_cells(rect: QtCore.QRect) -> list:
        """Divide a QRect into a shrunk+expanded 3-row × 2-col cell grid."""
        return _build_cells(rect.x(), rect.y(), rect.width(), rect.height())

    def _check_ic(self, image_bgr: np.ndarray, cells: list,
                  annotated: np.ndarray, debug: bool) -> tuple:
        """
        Crop each ROI cell from image_bgr and classify as Text / NoText.
        Returns (missing, hits_flags).
        """
        ih, iw = image_bgr.shape[:2]
        color_ok   = _hex_to_bgr(_ann_color_ok)
        color_ng   = _hex_to_bgr(_ann_color_ng)
        missing    = []
        hits_flags = []
        for idx, (cx, cy, cw, ch) in enumerate(cells):
            row = idx // 2 + 1
            col = idx %  2 + 1
            x1, y1 = max(0, cx),       max(0, cy)
            x2, y2 = min(iw, cx + cw), min(ih, cy + ch)
            crop    = image_bgr[y1:y2, x1:x2]
            cls_idx, conf = self._detector.classify_crop(crop)
            present = (cls_idx == 1)   # 1 = Text (mark present)
            hits_flags.append(present)
            if debug:
                lbl = "Text" if present else "NoText"
                _dbg_g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
                print(f"[Cell R{row}C{col}] "
                      f"{'PRESENT' if present else 'ABSENT '} "
                      f"cls={lbl} conf={conf:.3f}  raw_std={_dbg_g.std():.1f}")
            color = color_ok if present else color_ng
            cv2.rectangle(annotated,
                          (max(0, cx), max(0, cy)),
                          (min(iw, cx + cw), min(ih, cy + ch)),
                          color, _ann_border_px)
            if _ann_show_labels:
                label = f"R{row}C{col}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                tx = max(0, cx) + (cw - tw) // 2
                ty = max(0, cy) + (ch + th) // 2
                cv2.putText(annotated, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if not present:
                missing.append([row, col])
        return missing, hits_flags

# =========================================================
# LOGGER
# =========================================================
_LOG_DIR = "logs"
_LOG_RETENTION = 365

class Logger:

    def __init__(self, log_dir: str = _LOG_DIR):
        self._dir         = log_dir
        self._session_log: str | None = None
        os.makedirs(log_dir, exist_ok=True)
        self._rotate()

    def _log_path(self) -> str:
        return os.path.join(self._dir,
                            f"inspect_{datetime.now():%Y%m%d}.log")

    def _rotate(self):
        logs = sorted(glob.glob(os.path.join(self._dir, "inspect_*.log")))
        while len(logs) > _LOG_RETENTION:
            try:
                os.remove(logs.pop(0))
            except OSError:
                pass

    def _append(self, record: dict):
        path = self._session_log if self._session_log else self._log_path()
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[Logger] Write failed: {e}", file=sys.stderr)

    def start_session(self, mode: str):
        self._rotate()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_log = os.path.join(self._dir, f"inspect_{ts}.log")
        self._append({"event": "SESSION_START",
                       "timestamp": datetime.now().isoformat(),
                       "mode": mode})

    def log_session_end(self, reason: str,
                        pass_ct: int, fail_ct: int, err_ct: int,
                        elapsed_s: float):
        self._append({"event": "SESSION_END",
                       "timestamp": datetime.now().isoformat(),
                       "reason": reason,
                       "total_pass": pass_ct,
                       "total_fail": fail_ct,
                       "total_error": err_ct,
                       "duration_s": round(elapsed_s, 1)})
        self._session_log = None

    def log_pause(self):
        self._append({"event": "PAUSE",
                       "timestamp": datetime.now().isoformat()})

    def log_resume(self):
        self._append({"event": "RESUME",
                       "timestamp": datetime.now().isoformat()})

    def log_inspection(self, image_id: str,
                       ic_a_result: str, ic_a_missing: list,
                       ic_b_result: str, ic_b_missing: list,
                       cycle_ms: float, mode: str, io_mock: bool):
        self._append({
            "timestamp":   datetime.now().isoformat(),
            "image_id":    image_id,
            "ic_a_result": ic_a_result,
            "ic_a_missing": ic_a_missing,
            "ic_b_result": ic_b_result,
            "ic_b_missing": ic_b_missing,
            "cycle_time_ms": round(cycle_ms, 1),
            "mode":        mode,
            "io_mock":     io_mock,
        })

    def log_error(self, error_type: str, message: str, cycle_ms: float = 0):
        self._append({
            "timestamp":     datetime.now().isoformat(),
            "event":         "ERROR",
            "error_type":    error_type,
            "error_message": message,
            "cycle_time_ms": round(cycle_ms, 1),
        })

    def log_io_mock(self, pin_name: str, state: str):
        print(f"[IO MOCK] {pin_name} → {state}")
        self._append({
            "timestamp": datetime.now().isoformat(),
            "event":     "IO_MOCK",
            "pin":       pin_name,
            "state":     state,
        })

# =========================================================
# STYLESHEET
# =========================================================
STYLE = """
QMainWindow, QWidget#root {
    background: #5465FF;
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

    def _refresh(self):
        if self._orig is None:
            return
        h, w = self._orig.shape[:2]
        rgb  = cv2.cvtColor(self._orig, cv2.COLOR_BGR2RGB)
        qi   = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix  = QtGui.QPixmap.fromImage(qi)
        lw, lh = self.width(), self.height()
        if lw > 0 and lh > 0:
            pix = pix.scaled(lw, lh, QtCore.Qt.KeepAspectRatio,
                             QtCore.Qt.SmoothTransformation)
        if self._orig.shape[1] > 0:
            self._scale = pix.width() / self._orig.shape[1]
        self._offset = QtCore.QPoint((lw - pix.width())  // 2,
                                     (lh - pix.height()) // 2)
        self.setPixmap(pix)
        if self._overlays:
            self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
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
class RunWorker(QtCore.QThread):
    """
    Background inspection loop.

    MANUAL=True  + CAMERA='directory': waits for trigger() call per cycle
    MANUAL=False + CAMERA='directory': auto-loops with short delay between cycles
    CAMERA='camera': waits for GPIO START_PIN (or trigger() if MANUAL=True)
    """
    sig_image    = QtCore.pyqtSignal(object)     # annotated BGR ndarray
    sig_result   = QtCore.pyqtSignal(bool, bool) # ic_a_pass, ic_b_pass
    sig_fail     = QtCore.pyqtSignal(object)     # MarkMissingError
    sig_error    = QtCore.pyqtSignal(str)
    sig_status   = QtCore.pyqtSignal(str)
    sig_cycle_ms = QtCore.pyqtSignal(float)
    sig_done     = QtCore.pyqtSignal()           # all directory images processed → standby
    sig_paused   = QtCore.pyqtSignal()
    sig_resumed  = QtCore.pyqtSignal()

    def __init__(self, camera: Camera, inspector: Inspector,
                 gpio: RaspberryIO, logger: Logger,
                 cfg: dict, parent=None):
        super().__init__(parent)
        self._camera    = camera
        self._inspector = inspector
        self._gpio      = gpio
        self._logger    = logger
        self._cfg       = cfg
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

    def run(self):
        cam_mode = self._cfg.get("CAMERA", "directory")
        io_mock  = not IO
        debug    = DEBUG

        # Camera preflight — verify camera is reachable before entering the loop.
        if cam_mode == "camera":
            try:
                self._camera.grab_first()
            except CameraError as e:
                self.sig_error.emit(f"Camera not found: {e}")
                self.sig_status.emit("ERROR — camera not found, cannot run.")
                return

        self.sig_status.emit("Running…")
        _cycle = 0

        while not self._stop:

            # ── Wait for next cycle trigger ──────────────────────────
            if cam_mode == "camera":
                # Production: wait for GPIO START_PIN rising edge
                self.sig_status.emit("Waiting for START signal…")
                if not self._gpio.wait_for_start(lambda: self._stop):
                    break
                if self._stop:
                    break
            else:
                # Auto directory: brief yield, then check DONE_PIN / Stop
                time.sleep(0.05)
                if self._stop or self._gpio.is_done_signaled():
                    break

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
                # No ACK pulsed — machine stays blocked waiting; operator must reset.
                break

            img_id = _next_image_id()

            # Save raw before any processing
            real_path, ann_path = _output_paths(img_id)
            cv2.imwrite(real_path, img_bgr)

            self.sig_status.emit("Inspecting…")

            # ── Inspect ─────────────────────────────────────────────
            try:
                self._inspector.inspect(img_bgr, debug=debug)
                # img_bgr is now annotated in-place

                cycle_ms = (time.perf_counter() - t0) * 1000
                self.sig_image.emit(img_bgr)
                self.sig_result.emit(True, True)
                self.sig_cycle_ms.emit(cycle_ms)
                self._logger.log_inspection(
                    img_id, "PASS", [], "PASS", [],
                    cycle_ms, MODE, io_mock)

                cv2.imwrite(ann_path, img_bgr)

                # GPIO: both FAIL pins LOW → pulse ACK
                self._gpio.set_fail_a(False)
                self._gpio.set_fail_b(False)
                self._gpio.pulse_ack()

            except MarkMissingError as e:
                cycle_ms = (time.perf_counter() - t0) * 1000
                # img_bgr IS e.annotated (drawn in-place); raw already at real_path

                self.sig_image.emit(img_bgr)
                self.sig_fail.emit(e)
                self.sig_cycle_ms.emit(cycle_ms)
                self._logger.log_inspection(
                    img_id,
                    "FAIL" if e.missing_a else "PASS", e.missing_a,
                    "FAIL" if e.missing_b else "PASS", e.missing_b,
                    cycle_ms, MODE, io_mock)

                cv2.imwrite(ann_path, img_bgr)

                # GPIO: set FAIL pins then pulse ACK
                self._gpio.set_fail_a(bool(e.missing_a))
                self._gpio.set_fail_b(bool(e.missing_b))
                self._gpio.pulse_ack()

            except TemplateError as e:
                # Rotation/misalignment rejection — signal machine FAIL for both ICs,
                # then continue the loop (next frame may align correctly).
                cycle_ms = (time.perf_counter() - t0) * 1000
                all_cells = [[r, c] for r in range(1, 4) for c in range(1, 3)]
                err = MarkMissingError(all_cells, all_cells, None)
                # ann_path intentionally NOT written — no annotation drawn for alignment errors;
                # only raw (real_path) is on disk for this cycle.

                self.sig_image.emit(img_bgr)
                self.sig_fail.emit(err)
                self.sig_cycle_ms.emit(cycle_ms)
                self._logger.log_inspection(
                    img_id, "FAIL", all_cells, "FAIL", all_cells,
                    cycle_ms, MODE, io_mock)

                self._gpio.set_fail_a(True)
                self._gpio.set_fail_b(True)
                self._gpio.pulse_ack()

                if debug:
                    print(f"[RunWorker] Alignment rejected: {e}")

            except Exception as e:
                cycle_ms = (time.perf_counter() - t0) * 1000
                self._logger.log_error("RUNTIME_ERROR", str(e), cycle_ms)
                self.sig_error.emit(f"Unexpected error: {e}")
                self.sig_status.emit("ERROR — machine blocked, restart required.")
                # No ACK pulsed — machine stays blocked waiting; operator must reset.
                break

            finally:
                del img_bgr

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
                # Directory mode: stop on last image or DONE_PIN signal
                if not self._camera.has_more():
                    self._camera.reset()   # rewind for next run
                    break                  # natural end → sig_done
                if self._gpio.is_done_signaled():
                    break                  # machine stop signal → sig_done

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
# SETUP HELPERS
# =========================================================
def _find_second_ic(image_bgr: np.ndarray,
                    ref_rect: QtCore.QRect,
                    conf_thr: float = 0.4) -> tuple:
    """
    Search the opposite image half for a second IC using the ref_rect crop as a
    template.  Preprocessing matches extract_patches (bilateral → Otsu binary).

    Returns (QRect, score).  QRect is None if score < conf_thr.
    """
    x, y, w, h = ref_rect.x(), ref_rect.y(), ref_rect.width(), ref_rect.height()
    img_h, img_w = image_bgr.shape[:2]

    binary = _bilateral_binary(image_bgr)

    # Crop template from ref IC position (clamped to image bounds)
    ty1, ty2 = max(0, y), min(img_h, y + h)
    tx1, tx2 = max(0, x), min(img_w, x + w)
    template = binary[ty1:ty2, tx1:tx2]
    if template.size == 0:
        return None, 0.0

    # Search in opposite half
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
# MAIN WINDOW
# =========================================================
def _output_paths(img_id: str) -> tuple:
    """
    Returns (real_path, annotated_path) for today's date-based output folders.
      date/RealImg/img_id.jpg   — raw image
      date/Image/img_id.jpg     — annotated image
    Creates directories on first call.
    """
    prefix = OUT_DIR
    date     = datetime.now().strftime("%Y%m%d")
    real_dir = os.path.join(prefix, date, "RealImg")
    ann_dir  = os.path.join(prefix, date, "Image")
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(ann_dir,  exist_ok=True)
    return (os.path.join(real_dir, f"{img_id}.jpg"),
            os.path.join(ann_dir,  f"{img_id}.jpg"))

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, cfg: dict):
        super().__init__()
        self.setWindowTitle("ClearIC Inspect")
        self._cfg               = cfg
        self._camera:    Camera | None  = None
        self._detector:  Detector | None = None
        self._gpio       = None
        self._logger            = Logger()
        self._worker:           RunWorker | None        = None

        self._stats_pass  = 0
        self._stats_fail  = 0
        self._stats_error = 0

        self._run_state          = "standby"   # "standby" | "running" | "paused"
        self._session_start_time = 0.0

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
        central = QtWidgets.QWidget()
        central.setObjectName("root")
        self.setCentralWidget(central)

        root = QtWidgets.QHBoxLayout(central)
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

        lbl_exp = QtWidgets.QLabel("Exposure (µs)")
        lbl_exp.setStyleSheet("font-weight:bold")
        setup_lay.addWidget(lbl_exp)
        self._input_exposure = QtWidgets.QLineEdit(
            str(self._cfg.get("EXPOSURE_US", 8000)))
        setup_lay.addWidget(self._input_exposure)

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

        self._input_warmup = QtWidgets.QLineEdit(str(_warmup_frames))
        self._input_warmup.setFixedWidth(52)
        _srow(settings_lay, "Warmup frames", self._input_warmup)

        self._input_border = QtWidgets.QLineEdit(str(_ann_border_px))
        self._input_border.setFixedWidth(52)
        _srow(settings_lay, "Border thickness (px)", self._input_border)

        self._chk_labels = QtWidgets.QCheckBox("Show cell labels")
        self._chk_labels.setChecked(_ann_show_labels)
        settings_lay.addWidget(self._chk_labels)

        self._input_color_ok = QtWidgets.QLineEdit(_ann_color_ok)
        self._input_color_ok.setFixedWidth(80)
        _srow(settings_lay, "PASS cell color", self._input_color_ok)

        self._input_color_ng = QtWidgets.QLineEdit(_ann_color_ng)
        self._input_color_ng.setFixedWidth(80)
        _srow(settings_lay, "FAIL cell color", self._input_color_ng)

        self._input_color_a = QtWidgets.QLineEdit(_tmpl_color_a)
        self._input_color_a.setFixedWidth(80)
        _srow(settings_lay, "IC_A overlay color", self._input_color_a)

        self._input_color_b = QtWidgets.QLineEdit(_tmpl_color_b)
        self._input_color_b.setFixedWidth(80)
        _srow(settings_lay, "IC_B overlay color", self._input_color_b)

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
        global _ann_border_px, _ann_show_labels
        global _ann_color_ok, _ann_color_ng
        global _tmpl_color_a, _tmpl_color_b, _warmup_frames

        try:
            _warmup_frames = max(1, int(self._input_warmup.text()))
        except ValueError:
            self._input_warmup.setText(str(_warmup_frames))

        try:
            _ann_border_px = max(1, int(self._input_border.text()))
        except ValueError:
            self._input_border.setText(str(_ann_border_px))

        _ann_show_labels = self._chk_labels.isChecked()

        for attr, field in [
            ("_ann_color_ok", self._input_color_ok),
            ("_ann_color_ng", self._input_color_ng),
            ("_tmpl_color_a", self._input_color_a),
            ("_tmpl_color_b", self._input_color_b),
        ]:
            val = field.text().strip()
            if not val.startswith("#"):
                val = "#" + val
            if _valid_hex(val):
                globals()[attr] = val
            else:
                field.setText(globals()[attr])

        print(f"[Settings] border={_ann_border_px}px  labels={_ann_show_labels}  "
              f"ok={_ann_color_ok}  ng={_ann_color_ng}  "
              f"ic_a={_tmpl_color_a}  ic_b={_tmpl_color_b}  warmup={_warmup_frames}")

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
            )
        except ModelError as e:
            self._show_error(f"Classifier model load failed: {e}")

        try:
            self._gpio = RaspberryIO(io_enabled=IO)
        except GPIOError as e:
            self._show_error(f"GPIO init failed: {e}")

        try:
            self._camera = Camera(
                mode=cfg.get("CAMERA", "directory"),
                serial=cfg.get("CAMERA_SERIAL", ""),
                exposure_us=cfg.get("EXPOSURE_US", 8000),
                input_dir=DIR_INPUT,
            )
            self._camera.open()
        except CameraError as e:
            self._show_error(f"Camera init failed: {e}")

        if self._camera and cfg.get("CAMERA") == "camera":
            self._camera.warmup()
        if self._detector and self._detector.is_ready():
            self._detector.warmup(frames=_warmup_frames)

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

        img     = self._setup_image
        img_w   = img.shape[1] if img is not None else 0
        cx      = rect.x() + rect.width() // 2
        on_left = (img_w == 0) or (cx < img_w // 2)

        if on_left:
            ic_a = rect
            ic_b, _ = _find_second_ic(img, rect) if img is not None else (None, 0.0)
        else:
            ic_b = rect
            ic_a, _ = _find_second_ic(img, rect) if img is not None else (None, 0.0)

        if ic_a:
            self._view.add_overlay(ic_a, QtGui.QColor(_tmpl_color_a), "IC_A")
        if ic_b:
            self._view.add_overlay(ic_b, QtGui.QColor(_tmpl_color_b), "IC_B")

        if ic_a and ic_b:
            self._pending_ic_a = ic_a
            self._pending_ic_b = ic_b
            self._setup_state  = 'ready'
        else:
            # Auto-search failed — re-enable rubber-band for immediate retry
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
        try:
            exposure = int(self._input_exposure.text())
        except ValueError:
            exposure = 8000

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

        self._setup_state = 'idle'
        self._update_setup_buttons()


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

        # Reset counters for new session
        self._stats_pass = self._stats_fail = self._stats_error = 0
        self._lbl_pass.setText("0")
        self._lbl_fail.setText("0")
        self._lbl_error.setText("0")
        self._session_start_time = time.monotonic()

        mode = "DEBUG" if DEBUG else "RUN"
        self._logger.start_session(mode)

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

        inspector = Inspector(self._detector, tmpl, template_matcher=matcher)
        gpio      = self._gpio or RaspberryIO(io_enabled=False)

        self._worker = RunWorker(
            self._camera, inspector, gpio, self._logger, self._cfg)
        self._worker.sig_image.connect(self._on_image)
        self._worker.sig_result.connect(self._on_result)
        self._worker.sig_fail.connect(self._on_fail)
        self._worker.sig_error.connect(self._on_worker_error)
        self._worker.sig_status.connect(self._lbl_status.setText)
        self._worker.sig_cycle_ms.connect(
            lambda ms: self._lbl_cycle_ms.setText(f"{ms:.0f}"))
        self._worker.sig_done.connect(self._on_run_done)
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
        """Called by Stop button — ends session, no sig_done."""
        elapsed = time.monotonic() - self._session_start_time
        self._logger.log_session_end(
            "STOPPED", self._stats_pass, self._stats_fail,
            self._stats_error, elapsed)
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
        self._enter_standby()

    def _on_run_done(self):
        """Called when all images processed naturally OR DONE_PIN received."""
        elapsed = time.monotonic() - self._session_start_time
        self._logger.log_session_end(
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

    def _on_result(self, ia_pass: bool, ib_pass: bool):
        self._update_badge(self._badge_a, ia_pass)
        self._update_badge(self._badge_b, ib_pass)
        self._stats_pass += (1 if ia_pass else 0) + (1 if ib_pass else 0)
        self._lbl_pass.setText(str(self._stats_pass))
        self._lbl_fail.setText(str(self._stats_fail))

    def _on_fail(self, err: MarkMissingError):
        ic_a_pass = len(err.missing_a) == 0
        ic_b_pass = len(err.missing_b) == 0
        self._update_badge(self._badge_a, ic_a_pass)
        self._update_badge(self._badge_b, ic_b_pass)
        self._stats_pass += (1 if ic_a_pass else 0) + (1 if ic_b_pass else 0)
        self._stats_fail += (0 if ic_a_pass else 1) + (0 if ic_b_pass else 1)
        self._lbl_pass.setText(str(self._stats_pass))
        self._lbl_fail.setText(str(self._stats_fail))

    def _on_worker_error(self, msg: str):
        self._stats_error += 1
        self._lbl_error.setText(str(self._stats_error))
        self._show_error(msg)
        elapsed = time.monotonic() - self._session_start_time
        self._logger.log_session_end(
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

    os.makedirs(_LOG_DIR,   exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs(DIR_INPUT,   exist_ok=True)
    if COLLECT_DATASET:
        os.makedirs(os.path.join(_DATA_DIR, _DATA_SPLIT, "Text"),   exist_ok=True)
        os.makedirs(os.path.join(_DATA_DIR, _DATA_SPLIT, "NoText"), exist_ok=True)
        print(f"[Dataset] Collection ON → {_DATA_DIR}/{_DATA_SPLIT}/")

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
