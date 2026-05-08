"""
ClearIC Inspect
===============
Clear-package IC laser-mark inspection via YOLO/OpenVINO ROI presence detection.

Sections (in order)
-------------------
  ConfigLoader        Config.json loader with defaults
  Stage / ErrorFlag   State enums
  Exceptions          InspectionError hierarchy
  Image               Image dataclass + ID generator
  Camera              Basler camera or directory source
  RaspberryIO         BCM GPIO handler (mockable)
  Detector            OpenVINO YOLO inference (2-class)
  TemplateManager     Load/save IC bounding-box template
  Inspector           12-cell ROI inspection logic
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
import math
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

# =========================================================
# HARDCODED DEV FLAGS  (edit here before running — not in Config.json)
# =========================================================
DEBUG  = True      # verbose logs, annotated output
IO     = False     # True = drive GPIO / False = mock (log only)
MODE   = "DEBUG"   # "DEBUG" or "RUN"

# =========================================================
# CONFIG LOADER
# =========================================================
class ConfigLoader:
    CONFIG_FILE = "Config.json"
    DEFAULT_CONFIG = {
        "CAMERA":        "directory",
        "CONF_THR":      0.5,
        "NMS_IOU_THR":   0.45,
        "CAMERA_SERIAL": "",
        "EXPOSURE_US":   8000,
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
_img_counter = 0

def _next_image_id() -> str:
    global _img_counter
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

# =========================================================
# RASPBERRY IO
# =========================================================
START_PIN  = 17
DONE_PIN   = 27
ACK_PIN    = 22
FAIL_A_PIN = 24
FAIL_B_PIN = 25

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
            while not stop_flag_fn():
                time.sleep(0.02)
                return True
            return False
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

    def cleanup(self):
        if self._gpio_ok:
            try:
                self._GPIO.cleanup()
            except Exception:
                pass

# =========================================================
# DETECTOR  (OpenVINO YOLO — 2-class)
# =========================================================
_INPUT_SIZE = 640

class Detector:
    """
    OpenVINO YOLO8 wrapper for ClearIC_Insp model.
    Output: [1, 6, 8400]  (cx, cy, w, h, conf_IC_Presence, conf_Text)
    No built-in NMS — applied manually.
    """

    MODEL_XML = "ClearIC_Insp_openvino_model/ClearIC_Insp.xml"

    def __init__(self, conf_thr: float = 0.9, nms_thr: float = 0.45):
        self._conf_thr  = conf_thr
        self._nms_thr   = nms_thr
        self._compiled  = None
        self._ready     = False
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
            print(f"[Detector] OpenVINO model loaded: {self.MODEL_XML}")
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f"Model load failed: {e}")

    def is_ready(self) -> bool:
        return self._ready

    def warmup(self, frames: int = 5):
        blank = np.zeros((_INPUT_SIZE, _INPUT_SIZE, 3), dtype=np.uint8)
        for _ in range(frames):
            self._run_inference(blank)
        print(f"[Detector] Warmup done ({frames} frames).")

    def _run_inference(self, image_bgr: np.ndarray) -> list:
        """
        Letterbox → blob → infer → NMS.
        Returns list of (x1, y1, w, h, class_idx) 5-tuples.
          class_idx 0 = IC_Presence (large mold outline)
          class_idx 1 = Text        (individual mark/character)
        """
        if not self._ready or self._compiled is None:
            return []
        try:
            if image_bgr.ndim == 2:
                image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
            ih, iw = image_bgr.shape[:2]
            sz = _INPUT_SIZE

            scale   = min(sz / iw, sz / ih)
            nw, nh  = int(iw * scale), int(ih * scale)
            resized = cv2.resize(image_bgr, (nw, nh))
            pad_buf = np.full((sz, sz, 3), 114, dtype=np.uint8)
            pad_x   = (sz - nw) // 2
            pad_y   = (sz - nh) // 2
            pad_buf[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

            blob   = pad_buf[:, :, ::-1].astype(np.float32) / 255.0
            blob   = blob.transpose(2, 0, 1)[np.newaxis]
            result = self._compiled(blob)
            preds  = result[0][0].T                         # [8400, 6]

            # confidence = max of class scores; class = argmax of scores
            conf      = preds[:, 4:6].max(axis=1)
            class_idx = preds[:, 4:6].argmax(axis=1)
            mask      = conf >= self._conf_thr
            preds     = preds[mask]
            conf      = conf[mask]
            class_idx = class_idx[mask]
            if len(preds) == 0:
                return []

            raw_boxes, scores, classes = [], [], []
            for i, row in enumerate(preds):
                cx, cy, bw, bh = row[:4]
                x1 = max(0,  int((cx - bw / 2 - pad_x) / scale))
                y1 = max(0,  int((cy - bh / 2 - pad_y) / scale))
                x2 = min(iw, int((cx + bw / 2 - pad_x) / scale))
                y2 = min(ih, int((cy + bh / 2 - pad_y) / scale))
                if x2 > x1 and y2 > y1:
                    raw_boxes.append([x1, y1, x2 - x1, y2 - y1])
                    scores.append(float(conf[i]))
                    classes.append(int(class_idx[i]))

            if not raw_boxes:
                return []

            indices = cv2.dnn.NMSBoxes(
                raw_boxes, scores, self._conf_thr, self._nms_thr)
            if len(indices) == 0:
                return []
            kept = indices.flatten() if hasattr(indices, "flatten") else list(indices)
            # Returns (x, y, w, h, class_idx, conf) 6-tuples
            return [(raw_boxes[i][0], raw_boxes[i][1],
                     raw_boxes[i][2], raw_boxes[i][3],
                     classes[i], scores[i])
                    for i in kept]
        except Exception as e:
            print(f"[Detector] Inference error: {e}")
            return []

    def detect_full_image(self, image_bgr: np.ndarray) -> list:
        """
        Run YOLO on the full image.
        Returns only Text class (1) boxes as (x1, y1, w, h, conf) 5-tuples.
        IC_Presence (class 0) boxes are excluded — they cover the entire IC area
        and would make every cell appear present even when fonts are absent.
        Confidence is included so Inspector can annotate each detection.
        """
        return [(x, y, w, h, cf) for x, y, w, h, cls, cf in self._run_inference(image_bgr)
                if cls == 1]

    def locate_ics(self, image_bgr: np.ndarray) -> tuple:
        """
        Find the two IC_Presence bounding boxes (class 0) in the image.
        Returns (ic_a_rect, ic_b_rect) sorted left→right by X position.
        Either element is None if fewer than 2 IC_Presence boxes are found.

        This is the key to correct A/B assignment: position in the current
        image determines the label, not the saved template coordinates.
        """
        boxes = self._run_inference(image_bgr)
        # Keep only IC_Presence class (0), sort by area descending to get full IC boxes
        presence = [(x, y, w, h) for x, y, w, h, cls, _cf in boxes if cls == 0]
        if not presence:
            return None, None
        # Among IC_Presence boxes, take the two largest (mold outlines)
        presence.sort(key=lambda b: -(b[2] * b[3]))
        top2 = presence[:2]
        # Sort left→right by X position
        top2.sort(key=lambda b: b[0])

        if len(top2) == 2:
            return QtCore.QRect(*top2[0]), QtCore.QRect(*top2[1])

        # Only one IC found — return it in first slot; Inspector will assign correct A/B
        return QtCore.QRect(*top2[0]), None

    def detect_all(self, image_bgr: np.ndarray) -> list:
        """
        Return up to 2 IC_Presence QRects for template setup, sorted left→right by X.
        Picks the two largest boxes first (to get the IC mold outlines),
        then orders them by X so index 0 = IC_A (left), index 1 = IC_B (right).
        """
        boxes = self._run_inference(image_bgr)
        if not boxes:
            return []
        presence = [(x, y, w, h) for x, y, w, h, cls, _cf in boxes if cls == 0]
        if not presence:
            return []
        presence.sort(key=lambda b: -(b[2] * b[3]))   # largest first
        top2 = presence[:2]
        top2.sort(key=lambda b: b[0])                  # left→right by X
        return [QtCore.QRect(x, y, w, h) for x, y, w, h in top2]

# =========================================================
# TEMPLATE MANAGER
# =========================================================
_TEMPLATE_FILE    = "templates/template.json"
_TEMPLATE_TOP     = "templates/tmpl_top.npy"
_TEMPLATE_BOT     = "templates/tmpl_bot.npy"
_TEMPLATE_PREVIEW = "templates/template_preview.png"

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
             match_threshold: float = 0.6,
             strip_top_y_offset: int = 0, strip_bot_y_offset: int = 0,
             strip_h: int = 0):
        os.makedirs("templates", exist_ok=True)
        data = {
            "ic_a": {"x": ic_a.x(), "y": ic_a.y(),
                     "w": ic_a.width(), "h": ic_a.height()},
            "ic_b": {"x": ic_b.x(), "y": ic_b.y(),
                     "w": ic_b.width(), "h": ic_b.height()},
            "exposure_us":        exposure_us,
            "match_threshold":    match_threshold,
            "strip_top_y_offset": strip_top_y_offset,
            "strip_bot_y_offset": strip_bot_y_offset,
            "strip_h":            strip_h,
        }
        with open(_TEMPLATE_FILE, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def extract_patches(image_bgr: np.ndarray, ic_rect: QtCore.QRect) -> tuple:
        """
        Crop top and bottom strips using the defined formula and apply bilateral filter.

        Given IC at (X, Y) with size (W, H):
          H1 = H2 = H * 0.5
          W1 = W2 = W
          Y1 = Y - H * 0.75   (top strip origin, above IC)
          Y2 = Y + H * 0.75   (bottom strip origin, straddles IC bottom edge)

        Returns (top_filtered, bot_filtered, top_y_offset, bot_y_offset, strip_h)
        where *_y_offset = strip_y_clamped - IC_y  (used to reconstruct IC pos from match).
        """
        x, y = ic_rect.x(), ic_rect.y()
        w, h = ic_rect.width(), ic_rect.height()
        h1 = max(1, int(h * 0.5))
        y_center = y + h // 2

        y1 = int(y_center - h * 0.75)
        y2 = int(y_center + h * 0.75)

        img_h, img_w = image_bgr.shape[:2]
        y1c = max(0, y1)
        y2c = max(0, min(y2, img_h - h1))
        x_end = min(x + w, img_w)

        top_raw = image_bgr[y1c:y1c + h1, x:x_end]
        bot_raw = image_bgr[y2c:y2c + h1, x:x_end]
        top_filt = cv2.bilateralFilter(top_raw, 9, 75, 75)
        bot_filt = cv2.bilateralFilter(bot_raw, 9, 75, 75)

        # Offset = strip_top_y - IC_y (negative means strip is above IC origin)
        top_y_offset = y1c - y
        bot_y_offset = y2c - y
        return top_filt, bot_filt, top_y_offset, bot_y_offset, h1

    @staticmethod
    def save_patches(top_patch: np.ndarray, bot_patch: np.ndarray):
        """Save bilateral-filtered strip patches as .npy files."""
        os.makedirs("templates", exist_ok=True)
        np.save(_TEMPLATE_TOP, top_patch)
        np.save(_TEMPLATE_BOT, bot_patch)

    @staticmethod
    def load_patches() -> tuple:
        """
        Load template patches. Returns (top, bot) ndarrays, or (None, None)
        if files are absent or corrupt (backward-compatible with old templates).
        """
        if not os.path.exists(_TEMPLATE_TOP) or not os.path.exists(_TEMPLATE_BOT):
            return None, None
        try:
            top = np.load(_TEMPLATE_TOP)
            bot = np.load(_TEMPLATE_BOT)
            return top, bot
        except Exception as e:
            print(f"[TemplateManager] Patch load failed: {e}")
            return None, None

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
            # Strip ROI positions — same formula as extract_patches
            h1 = max(1, int(h * 0.5))
            y1 = int(cy - h * 0.8)
            y2 = int(y + h * 0.8)
            cv2.rectangle(preview, (x, y1), (x + w, y1 + h1), (255, 0, 255), 2)
            cv2.rectangle(preview, (x, y2), (x + w, y2 + h1), (255, 0, 255), 2)
            cv2.putText(preview, f"TOP y={y1}", (x + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)
            cv2.putText(preview, f"BOT y={y2}", (x + 2, y2 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

        cv2.imwrite(_TEMPLATE_PREVIEW, preview)

    @staticmethod
    def compute_rois(template: dict) -> tuple:
        """
        Returns (ic_a_cells, ic_b_cells).
        Each is a list of 6 (x, y, w, h) tuples — 3 rows × 2 cols.
        Left col = left half, right col = right half; rows = thirds.
        """
        def _cells(box: dict) -> list:
            x, y = box["x"], box["y"]
            w, h = box["w"], box["h"]
            cw   = w // 2
            ch   = h // 3
            cells = []
            for row in range(3):
                for col in range(2):
                    cells.append((x + col * cw, y + row * ch, cw, ch))
            return cells
        return _cells(template["ic_a"]), _cells(template["ic_b"])

# =========================================================
# TEMPLATE MATCHER
# =========================================================
class TemplateMatcher:
    """
    Locates IC_A in a new image using bilateral-filtered top/bottom strip
    template matching (cv2.TM_CCOEFF_NORMED). Replaces Detector.locate_ics()
    for Phase 1 of inspection.

    If the averaged match score falls below threshold a TemplateError is raised —
    this acts as a rotation/misalignment rejection gate.
    """

    def __init__(self, top_patch: np.ndarray, bot_patch: np.ndarray,
                 threshold: float = 0.6,
                 strip_top_y_offset: int = 0, strip_bot_y_offset: int = 0,
                 ic_h: int = 0):
        self._top            = top_patch
        self._bot            = bot_patch
        self._threshold      = threshold
        self._patch_w        = top_patch.shape[1]
        self._top_y_offset   = strip_top_y_offset  # strip_top_y - IC_y at creation
        self._bot_y_offset   = strip_bot_y_offset  # strip_bot_y - IC_y at creation
        self._ic_h           = ic_h

    def locate_ic(self, image_bgr: np.ndarray) -> tuple:
        """
        Returns (QRect, score). Raises TemplateError when score < threshold
        (rotation or misalignment — caller should treat the frame as FAIL).

        IC position is reconstructed using stored strip offsets:
          ic_y = matched_strip_y - strip_y_offset
        """
        filtered = cv2.bilateralFilter(image_bgr, 9, 75, 75)

        res_top = cv2.matchTemplate(filtered, self._top, cv2.TM_CCOEFF_NORMED)
        _, score_top, _, loc_top = cv2.minMaxLoc(res_top)

        res_bot = cv2.matchTemplate(filtered, self._bot, cv2.TM_CCOEFF_NORMED)
        _, score_bot, _, loc_bot = cv2.minMaxLoc(res_bot)

        score = (score_top + score_bot) / 2.0
        if score < self._threshold:
            raise TemplateError(
                f"Match score {score:.3f} < {self._threshold:.3f} — "
                "IC rotation or misalignment detected")

        # Reconstruct IC position: ic_y = matched_strip_top_y - offset
        ic_y_from_top = loc_top[1] - self._top_y_offset
        ic_y_from_bot = loc_bot[1] - self._bot_y_offset
        ic_y = (ic_y_from_top + ic_y_from_bot) // 2
        ic_x = (loc_top[0] + loc_bot[0]) // 2
        return QtCore.QRect(ic_x, ic_y, self._patch_w, self._ic_h), score

# =========================================================
# INSPECTOR
# =========================================================
class Inspector:
    """
    Runs YOLO once on the full image then checks each ROI cell for box intersections.
    Raises MarkMissingError if either IC has any cell without an overlapping detection.
    """

    def __init__(self, detector: Detector, template: dict,
                 template_matcher: "TemplateMatcher | None" = None):
        self._detector        = detector
        self._template        = template
        self._template_matcher = template_matcher

    def inspect(self, image_bgr: np.ndarray,
                debug: bool = False) -> tuple:
        """
        Returns (ic_a_pass, ic_b_pass, missing_a, missing_b, annotated_bgr).
        Raises MarkMissingError if either IC fails.
        Raises TemplateError if template matching rejects the frame (rotation/misalignment).

        Phase 1 — locate IC_A via TemplateMatcher (preferred) or YOLO fallback.
          TemplateMatcher: bilateral-filtered strip matching; IC_B derived from offset.
          YOLO fallback:   Detector.locate_ics() with proximity swap logic.
        Phase 2 — YOLO text detection checked against each IC's cell grid.
        """
        tmpl_a_cells, tmpl_b_cells = TemplateManager.compute_rois(self._template)
        annotated = image_bgr.copy()

        # Phase 1: locate ICs
        if self._template_matcher is not None:
            # Template matching path — raises TemplateError if score below threshold
            rt_a, score = self._template_matcher.locate_ic(image_bgr)
            dx = self._template["ic_b"]["x"] - self._template["ic_a"]["x"]
            dy = self._template["ic_b"]["y"] - self._template["ic_a"]["y"]
            rt_b = QtCore.QRect(
                rt_a.x() + dx, rt_a.y() + dy,
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
            # YOLO fallback path
            rt_a, rt_b = self._detector.locate_ics(image_bgr)

            # Single IC found: assign to correct A/B slot using template center proximity.
            # Prevents: IC_A absent, IC_B present → YOLO finds 1 IC → incorrectly assigned to A.
            if rt_a is not None and rt_b is None:
                tmpl_a_cx = (self._template["ic_a"]["x"] +
                             self._template["ic_a"]["w"] // 2)
                tmpl_b_cx = (self._template["ic_b"]["x"] +
                             self._template["ic_b"]["w"] // 2)
                found_cx  = rt_a.x() + rt_a.width() // 2
                if abs(found_cx - tmpl_b_cx) < abs(found_cx - tmpl_a_cx):
                    rt_a, rt_b = None, rt_a

            ic_a_cells = self._rect_to_cells(rt_a) if rt_a else tmpl_a_cells
            ic_b_cells = self._rect_to_cells(rt_b) if rt_b else tmpl_b_cells

            if debug:
                if rt_a:
                    print(f"[Inspector] IC_A located (YOLO): "
                          f"x={rt_a.x()} y={rt_a.y()} w={rt_a.width()} h={rt_a.height()}")
                else:
                    print("[Inspector] IC_A not found → template fallback → FAIL")
                if rt_b:
                    print(f"[Inspector] IC_B located (YOLO): "
                          f"x={rt_b.x()} y={rt_b.y()} w={rt_b.width()} h={rt_b.height()}")
                else:
                    print("[Inspector] IC_B not found → template fallback → FAIL")

        # Phase 2: check all Text detection boxes against each IC's cells
        all_boxes = self._detector.detect_full_image(image_bgr)  # (x,y,w,h,conf)

        # Draw every Text detection with cyan box + confidence label
        for bx, by, bw, bh, bc in all_boxes:
            cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh), (255, 200, 0), 1)
            cv2.putText(annotated, f"{bc:.2f}",
                        (bx, by - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (255, 200, 0), 1)

        missing_a = self._check_ic(all_boxes, ic_a_cells, annotated, debug)
        missing_b = self._check_ic(all_boxes, ic_b_cells, annotated, debug)

        if missing_a or missing_b:
            raise MarkMissingError(missing_a, missing_b, annotated)

        return True, True, [], [], annotated

    @staticmethod
    def _rect_to_cells(rect: QtCore.QRect) -> list:
        """Divide a QRect into a 3-row × 2-col grid of (x, y, w, h) cells."""
        x, y  = rect.x(),     rect.y()
        w, h  = rect.width(), rect.height()
        cw, ch = w // 2,       h // 3
        return [(x + col * cw, y + row * ch, cw, ch)
                for row in range(3) for col in range(2)]

    @staticmethod
    def _boxes_intersect(bx1, by1, bw, bh, cx, cy, cw, ch) -> bool:
        """True if box (bx1,by1,bw,bh) overlaps cell (cx,cy,cw,ch)."""
        return (bx1 < cx + cw and bx1 + bw > cx and
                by1 < cy + ch and by1 + bh > cy)

    def _check_ic(self, boxes: list, cells: list,
                  annotated: np.ndarray, debug: bool) -> list:
        missing = []
        for idx, (cx, cy, cw, ch) in enumerate(cells):
            row = idx // 2 + 1
            col = idx %  2 + 1
            # Collect every box that intersects this cell (for debug count)
            hits = [
                (bx, by, bw, bh, cf)
                for bx, by, bw, bh, cf in boxes
                if self._boxes_intersect(bx, by, bw, bh, cx, cy, cw, ch)
            ]
            present = len(hits) > 0
            if debug:
                hit_confs = [f"{cf:.3f}" for *_, cf in hits]
                print(f"[Cell R{row}C{col}] "
                      f"{'PRESENT' if present else 'ABSENT '} "
                      f"hits={len(hits)} confs=[{', '.join(hit_confs)}]")
            color = (0, 200, 0) if present else (0, 0, 220)
            x2 = min(annotated.shape[1], cx + cw)
            y2 = min(annotated.shape[0], cy + ch)
            cv2.rectangle(annotated, (max(0, cx), max(0, cy)), (x2, y2), color, 2)
            if debug:
                cv2.putText(annotated, f"R{row}C{col}",
                            (max(0, cx) + 2, max(0, cy) + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                            (0, 200, 0) if present else (0, 0, 220), 1)
            if not present:
                missing.append([row, col])
        return missing

# =========================================================
# LOGGER
# =========================================================
_LOG_DIR = "logs"
_LOG_RETENTION = 365

class Logger:

    def __init__(self, log_dir: str = _LOG_DIR):
        self._dir = log_dir
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
        try:
            with open(self._log_path(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[Logger] Write failed: {e}", file=sys.stderr)

    def log_inspection(self, image_id: str,
                       ic_a_result: str, ic_a_missing: list,
                       ic_b_result: str, ic_b_missing: list,
                       cycle_ms: float, mode: str, io_mock: bool):
        self._rotate()
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
            cells = ", ".join(f"[R{r}C{c}]" for r, c in missing_a)
            lbl = QtWidgets.QLabel(f"IC_A — missing: {cells}")
            lbl.setStyleSheet("color:#EF5350;font-weight:bold")
            lbl.setWordWrap(True)
            lay.addWidget(lbl)

        if missing_b:
            cells = ", ".join(f"[R{r}C{c}]" for r, c in missing_b)
            lbl = QtWidgets.QLabel(f"IC_B — missing: {cells}")
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
    Zoomable image display with overlay support and stamp mode.
    Ported from Ref_sample.py ImageView (L3597–3833).
    """
    anchor_clicked = QtCore.pyqtSignal(QtCore.QPoint)   # unused but kept for future

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

    # ---- paint ----
    def paintEvent(self, e):
        super().paintEvent(e)
        if not self._overlays and not self._stamp_mode:
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

        painter.end()

    def mouseMoveEvent(self, _e):
        if self._stamp_mode:
            self.update()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton and self._stamp_mode:
            img_pt = self._to_img(e.pos())
            w2, h2 = self._stamp_w // 2, self._stamp_h // 2
            rect   = QtCore.QRect(img_pt.x() - w2, img_pt.y() - h2,
                                  self._stamp_w, self._stamp_h)
            self.anchor_clicked.emit(img_pt)
            self.add_overlay(rect, QtGui.QColor("#FFD700"), "")

# =========================================================
# SETUP DIALOG  (Auto Detect flow: Auto-Detect → Popup → Retry / Confirm)
# =========================================================
class SetupDialog(QtWidgets.QDialog):
    """
    Non-modal popup for template creation.
    Opens after "Auto Detect" runs; shows detection status and
    lets the user Retry (re-runs detection) or Confirm (saves template).
    Overlays on the main ImageView update live while the dialog is open.
    """
    confirmed = QtCore.pyqtSignal(QtCore.QRect, QtCore.QRect)   # ic_a, ic_b

    def __init__(self, retry_fn, cancel_fn, parent=None):
        super().__init__(parent, QtCore.Qt.WindowTitleHint |
                         QtCore.Qt.WindowCloseButtonHint)
        self.setWindowTitle("Auto Detect — IC Template")
        self.setModal(False)
        self.setMinimumWidth(340)
        self.setStyleSheet(
            "QDialog  { background:#5465FF; }"
            "QLabel   { color:#FFFFFF; }"
            "QPushButton { background:#788BFF; color:#FFFFFF; border-radius:6px;"
            "  padding:6px 14px; font-weight:bold; border:none; }"
            "QPushButton:disabled { background:#9BB1FF; color:#BFD7FF; }"
        )

        self._retry_fn  = retry_fn
        self._cancel_fn = cancel_fn
        self._ic_a: QtCore.QRect | None = None
        self._ic_b: QtCore.QRect | None = None

        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(12)
        lay.setContentsMargins(20, 20, 20, 20)

        title = QtWidgets.QLabel("Auto Detect IC Positions")
        title.setStyleSheet("font-size:14px;font-weight:bold;color:#E2FDFF")
        title.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(title)

        self._lbl_status = QtWidgets.QLabel("Detecting…")
        self._lbl_status.setStyleSheet("color:#BFD7FF;font-size:11px")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_status.setMinimumHeight(36)
        lay.addWidget(self._lbl_status)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.setSpacing(8)

        self._btn_retry   = QtWidgets.QPushButton("Retry")
        self._btn_confirm = QtWidgets.QPushButton("Confirm")
        self._btn_cancel  = QtWidgets.QPushButton("Cancel")

        self._btn_confirm.setStyleSheet(
            "background:#FFFFFF;color:#5465FF;border-radius:6px;"
            "padding:6px 14px;font-weight:bold;")
        self._btn_confirm.setEnabled(False)
        self._btn_retry.setEnabled(False)

        self._btn_retry.clicked.connect(self._on_retry)
        self._btn_confirm.clicked.connect(self._on_confirm)
        self._btn_cancel.clicked.connect(self._on_cancel_clicked)

        btn_row.addWidget(self._btn_retry)
        btn_row.addStretch()
        btn_row.addWidget(self._btn_confirm)
        btn_row.addWidget(self._btn_cancel)
        lay.addLayout(btn_row)

        self.adjustSize()

    def show_result(self, ic_a: QtCore.QRect | None, ic_b: QtCore.QRect | None,
                    n_detected: int, candidate_idx: int = 0):
        self._ic_a = ic_a
        self._ic_b = ic_b
        both = ic_a is not None and ic_b is not None
        if both:
            self._lbl_status.setText(
                f"Candidate {candidate_idx + 1} of {n_detected} shown as IC_A.\n"
                "Retry to cycle. Confirm when correct.")
            self._lbl_status.setStyleSheet("color:#BFD7FF;font-size:11px")
        elif n_detected == 1:
            self._lbl_status.setText(
                "Only 1 IC detected — cannot assign IC_B.\n"
                "Adjust image or click Retry.")
            self._lbl_status.setStyleSheet("color:#EF5350;font-size:11px")
        else:
            self._lbl_status.setText(
                "No ICs detected.\nAdjust image or click Retry.")
            self._lbl_status.setStyleSheet("color:#EF5350;font-size:11px")
        self._btn_confirm.setEnabled(both)
        self._btn_retry.setEnabled(True)

    def _on_retry(self):
        self._btn_retry.setEnabled(False)
        self._btn_confirm.setEnabled(False)
        self._lbl_status.setText("Detecting…")
        self._lbl_status.setStyleSheet("color:#BFD7FF;font-size:11px")
        if self._retry_fn:
            self._retry_fn()

    def _on_confirm(self):
        if self._ic_a and self._ic_b:
            self.confirmed.emit(self._ic_a, self._ic_b)
            self.accept()

    def _on_cancel_clicked(self):
        if self._cancel_fn:
            self._cancel_fn()
        self.reject()

    def closeEvent(self, e):
        if self._cancel_fn:
            self._cancel_fn()
        e.accept()

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

    def __init__(self, camera: Camera, inspector: Inspector,
                 gpio: RaspberryIO, logger: Logger,
                 cfg: dict, parent=None):
        super().__init__(parent)
        self._camera    = camera
        self._inspector = inspector
        self._gpio      = gpio
        self._logger    = logger
        self._cfg       = cfg
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        cam_mode = self._cfg.get("CAMERA", "directory")
        mode     = MODE
        io_mock  = not IO
        debug    = DEBUG
        os.makedirs("output", exist_ok=True)
        os.makedirs(os.path.join("Input", "results"), exist_ok=True)
        input_results = os.path.join("Input", "results")

        self.sig_status.emit("Running…")

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
                break

            img_id = _next_image_id()
            self.sig_status.emit("Inspecting…")

            # ── Inspect ─────────────────────────────────────────────
            try:
                _ia, _ib, _ma, _mb, annotated = \
                    self._inspector.inspect(img_bgr, debug=debug)

                cycle_ms = (time.perf_counter() - t0) * 1000
                self.sig_image.emit(annotated)
                self.sig_result.emit(True, True)
                self.sig_cycle_ms.emit(cycle_ms)
                self._logger.log_inspection(
                    img_id, "PASS", [], "PASS", [],
                    cycle_ms, mode, io_mock)

                # Save annotated PASS image to output/ and Input/results/
                cv2.imwrite(os.path.join("output",       f"{img_id}.png"), annotated)
                cv2.imwrite(os.path.join(input_results,  f"{img_id}.png"), annotated)

                # GPIO: both FAIL pins LOW → pulse ACK
                self._gpio.set_fail_a(False)
                self._gpio.set_fail_b(False)
                self._gpio.pulse_ack()

            except MarkMissingError as e:
                cycle_ms = (time.perf_counter() - t0) * 1000
                annotated = e.annotated if e.annotated is not None \
                            else img_bgr.copy()

                self.sig_image.emit(annotated)
                self.sig_fail.emit(e)
                self.sig_cycle_ms.emit(cycle_ms)
                self._logger.log_inspection(
                    img_id,
                    "FAIL" if e.missing_a else "PASS", e.missing_a,
                    "FAIL" if e.missing_b else "PASS", e.missing_b,
                    cycle_ms, mode, io_mock)

                # FAIL: save raw + annotated to output/ and annotated to Input/results/
                cv2.imwrite(os.path.join("output",      f"{img_id}_R.png"), img_bgr)
                cv2.imwrite(os.path.join("output",      f"{img_id}.png"),   annotated)
                cv2.imwrite(os.path.join(input_results, f"{img_id}.png"),   annotated)

                # GPIO: set FAIL pins then pulse ACK
                self._gpio.set_fail_a(bool(e.missing_a))
                self._gpio.set_fail_b(bool(e.missing_b))
                self._gpio.pulse_ack()

            except TemplateError as e:
                # Rotation/misalignment rejection — signal machine FAIL for both ICs,
                # then continue the loop (next frame may align correctly).
                cycle_ms = (time.perf_counter() - t0) * 1000
                all_cells = [[r, c] for r in range(1, 4) for c in range(1, 3)]
                err = MarkMissingError(all_cells, all_cells, img_bgr.copy())

                self.sig_image.emit(img_bgr)
                self.sig_fail.emit(err)
                self.sig_cycle_ms.emit(cycle_ms)
                self._logger.log_inspection(
                    img_id, "FAIL", all_cells, "FAIL", all_cells,
                    cycle_ms, mode, io_mock)
                cv2.imwrite(os.path.join("output",      f"{img_id}_R.png"), img_bgr)
                cv2.imwrite(os.path.join("output",      f"{img_id}.png"),   img_bgr)
                cv2.imwrite(os.path.join(input_results, f"{img_id}.png"),   img_bgr)

                self._gpio.set_fail_a(True)
                self._gpio.set_fail_b(True)
                self._gpio.pulse_ack()

                print(f"[RunWorker] Alignment rejected: {e}")

            except Exception as e:
                cycle_ms = (time.perf_counter() - t0) * 1000
                self._logger.log_error("RUNTIME_ERROR", str(e), cycle_ms)
                self.sig_error.emit(f"Unexpected error: {e}")
                break

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

        self._gpio.clear_outputs()
        if not self._stop and cam_mode != "camera":
            self.sig_done.emit()
        self.sig_status.emit("Standby.")

# =========================================================
# MAIN WINDOW
# =========================================================
_OUTPUT_DIR = "output"

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, cfg: dict):
        super().__init__()
        self.setWindowTitle("ClearIC Inspect")
        self._cfg      = cfg
        self._camera:  Camera | None = None
        self._detector: Detector | None = None
        self._gpio     = None
        self._logger   = Logger()
        self._worker:  RunWorker | None = None

        self._stats_pass  = 0
        self._stats_fail  = 0
        self._stats_error = 0

        # setup state
        self._setup_dlg:           SetupDialog | None = None
        self._pending_ic_a:        QtCore.QRect | None = None
        self._pending_ic_b:        QtCore.QRect | None = None
        self._detect_candidates:   list = []
        self._detect_candidate_idx: int = 0
        self._current_detect_image: np.ndarray | None = None

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

        self._btn_detect = QtWidgets.QPushButton("Auto Detect")
        self._btn_detect.clicked.connect(self._start_auto_detect)
        setup_lay.addWidget(self._btn_detect)

        right_lay.addWidget(setup_frame)

        # Controls section
        ctrl_frame = QtWidgets.QFrame()
        ctrl_frame.setObjectName("controls_frame")
        ctrl_lay = QtWidgets.QVBoxLayout(ctrl_frame)
        ctrl_lay.setSpacing(6)

        lbl_ctrl = QtWidgets.QLabel("Controls")
        lbl_ctrl.setStyleSheet("font-weight:bold;font-size:13px")
        ctrl_lay.addWidget(lbl_ctrl)

        self._btn_run = QtWidgets.QPushButton("Start")
        self._btn_run.clicked.connect(self._start_run)
        ctrl_lay.addWidget(self._btn_run)

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
    # System init
    # ----------------------------------------------------------
    def _init_system(self):
        cfg = self._cfg
        try:
            self._detector = Detector(
                conf_thr=cfg.get("CONF_THR", 0.9),
                nms_thr=cfg.get("NMS_IOU_THR", 0.45),
            )
        except ModelError as e:
            self._show_error(f"Model load failed: {e}")

        try:
            self._gpio = RaspberryIO(io_enabled=IO)
        except GPIOError as e:
            self._show_error(f"GPIO init failed: {e}")

        try:
            self._camera = Camera(
                mode=cfg.get("CAMERA", "directory"),
                serial=cfg.get("CAMERA_SERIAL", ""),
                exposure_us=cfg.get("EXPOSURE_US", 8000),
            )
            self._camera.open()
        except CameraError as e:
            self._show_error(f"Camera init failed: {e}")

        if self._camera and cfg.get("CAMERA") == "camera":
            self._camera.warmup()
        if self._detector and self._detector.is_ready():
            self._detector.warmup(frames=5)

        # Load and display first image on startup (no overlays yet)
        if self._camera:
            try:
                img = self._camera.grab()
                self._view.set_image(img)
                self._current_detect_image = img
                if cfg.get("CAMERA") == "directory":
                    self._camera.reset()   # rewind so run starts from image 1
            except CameraError:
                pass


    # ----------------------------------------------------------
    # Auto Detect flow
    # ----------------------------------------------------------
    def _start_auto_detect(self):
        if not self._detector or not self._detector.is_ready():
            self._show_error("Detector not ready — check model file.")
            return
        if self._camera is None:
            self._show_error("Camera not ready.")
            return

        try:
            img = self._camera.grab()
            if self._cfg.get("CAMERA") == "directory":
                self._camera.reset()
        except CameraError as e:
            self._show_error(str(e))
            return

        self._current_detect_image = img
        self._view.set_image(img)
        self._view.clear_overlays()

        # Open the popup dialog before running detection so it's visible immediately
        if self._setup_dlg is None or not self._setup_dlg.isVisible():
            self._setup_dlg = SetupDialog(
                retry_fn=self._retry_detection,
                cancel_fn=self._cancel_detect,
                parent=self,
            )
            self._setup_dlg.confirmed.connect(self._on_detect_confirmed)
            # Centre popup over main window
            self._setup_dlg.adjustSize()
            geo   = self.geometry()
            dlg_w = self._setup_dlg.width()
            dlg_h = self._setup_dlg.height()
            self._setup_dlg.move(
                geo.x() + (geo.width()  - dlg_w) // 2,
                geo.y() + (geo.height() - dlg_h) // 2,
            )
            self._setup_dlg.show()

        self._run_detection(img)

    def _retry_detection(self):
        """Cycle to the next sorted YOLO candidate (no new inference needed)."""
        if self._detect_candidates:
            self._detect_candidate_idx += 1
            self._show_candidate()
        elif self._current_detect_image is not None:
            self._run_detection(self._current_detect_image)

    def _run_detection(self, img: np.ndarray):
        if not self._detector:
            return
        rects = self._detector.detect_all(img)
        # Sort by (x + y) ascending → leftmost-topmost candidate first
        self._detect_candidates    = sorted(rects, key=lambda r: r.x() + r.y())
        self._detect_candidate_idx = 0
        self._show_candidate()

    def _show_candidate(self):
        """Highlight the current candidate as IC_A and the next one as IC_B."""
        cands = self._detect_candidates
        if not cands:
            if self._setup_dlg and self._setup_dlg.isVisible():
                self._setup_dlg.show_result(None, None, 0, 0)
            return
        idx  = self._detect_candidate_idx % len(cands)
        ic_a = cands[idx]
        ic_b = cands[(idx + 1) % len(cands)] if len(cands) >= 2 else None
        self._view.clear_overlays()
        self._view.add_overlay(ic_a, QtGui.QColor("#FFD700"), "IC_A")
        if ic_b:
            self._view.add_overlay(ic_b, QtGui.QColor("#00E5FF"), "IC_B")
        self._pending_ic_a = ic_a
        self._pending_ic_b = ic_b
        if self._setup_dlg and self._setup_dlg.isVisible():
            self._setup_dlg.show_result(ic_a, ic_b, len(cands), idx)

    def _on_detect_confirmed(self, ic_a: QtCore.QRect, ic_b: QtCore.QRect):
        self._pending_ic_a = ic_a
        self._pending_ic_b = ic_b
        self._view.clear_overlays()
        try:
            exposure = int(self._input_exposure.text())
        except ValueError:
            exposure = 8000

        # Extract bilateral-filtered strip patches from the confirmed IC_A rect
        patch_saved = False
        top_y_off = bot_y_off = strip_h_val = 0
        if self._current_detect_image is not None:
            try:
                top_patch, bot_patch, top_y_off, bot_y_off, strip_h_val = \
                    TemplateManager.extract_patches(self._current_detect_image, ic_a)
                TemplateManager.save_patches(top_patch, bot_patch)
                patch_saved = True
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self, "Patch Warning",
                    f"Could not save template patches: {e}\n"
                    "Inspection will fall back to YOLO IC localization.")

        TemplateManager.save(ic_a, ic_b, exposure,
                             strip_top_y_offset=top_y_off,
                             strip_bot_y_offset=bot_y_off,
                             strip_h=strip_h_val)

        if self._current_detect_image is not None:
            try:
                TemplateManager.save_preview(self._current_detect_image, ic_a, ic_b)
            except Exception:
                pass  # preview is non-critical

        msg = "Template saved to templates/template.json"
        if patch_saved:
            msg += "\nPatch files saved (tmpl_top.npy / tmpl_bot.npy)"
        msg += "\nPreview saved to templates/template_preview.png"
        QtWidgets.QMessageBox.information(self, "Template Saved", msg)

    def _cancel_detect(self):
        self._view.clear_overlays()
        self._pending_ic_a = None
        self._pending_ic_b = None


    # ----------------------------------------------------------
    # Run / Stop
    # ----------------------------------------------------------
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

        top_patch, bot_patch = TemplateManager.load_patches()
        if top_patch is not None and bot_patch is not None:
            matcher = TemplateMatcher(
                top_patch, bot_patch,
                threshold=tmpl.get("match_threshold", 0.6),
                strip_top_y_offset=tmpl.get("strip_top_y_offset", 0),
                strip_bot_y_offset=tmpl.get("strip_bot_y_offset", 0),
                ic_h=tmpl["ic_a"]["h"],
            )
            print("[MainWindow] TemplateMatcher loaded — using template matching for IC localization.")
        else:
            matcher = None
            print("[MainWindow] No patch files found — falling back to YOLO locate_ics().")

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
        self._worker.start()

        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)

    def _stop_run(self):
        """Called by Stop button — early stop, no sig_done."""
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
        self._enter_standby()

    def _on_run_done(self):
        """Called when all images processed naturally OR DONE_PIN received."""
        self._enter_standby()

    def _enter_standby(self):
        self._btn_run.setEnabled(True)
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
            if self._cfg.get("CAMERA") == "directory":
                self._camera.reset()          # ensure index at 0
            img = self._camera.grab()
            self._view.set_image(img)
            self._view.clear_overlays()
            self._current_detect_image = img
            if self._cfg.get("CAMERA") == "directory":
                self._camera.reset()          # rewind again so next run starts at 0
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
        self._btn_run.setEnabled(True)
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

    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    os.makedirs(_LOG_DIR, exist_ok=True)
    os.makedirs("templates", exist_ok=True)

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
