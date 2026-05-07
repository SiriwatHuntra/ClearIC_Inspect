import time
import threading
import os, sys
from datetime import datetime
from enum import Enum

import numpy as np
import cv2 as cv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel,
    QPushButton, QLineEdit, QHBoxLayout, QVBoxLayout,
    QDialog, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont, QIntValidator, QDoubleValidator

try:
    import RPi.GPIO as GPIO
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False

try:
    from pypylon import pylon
    _PYLON_AVAILABLE = True
except ImportError:
    _PYLON_AVAILABLE = False


# ─── Config Flags ─────────────────────────────────────────────────────────────

DEBUG  = True
CAMERA = "directory"   # "camera" | "directory"
IO     = False          # True = real GPIO pins, False = mock / log
MODE   = "DEBUG"        # "RUN" | "DEBUG"

_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR     = os.path.join(_BASE_DIR, "Input")
OUTPUT_NG_DIR = os.path.join(_BASE_DIR, "output", "Ng")
OUTPUT_OK_DIR = os.path.join(_BASE_DIR, "output", "GooD")

for _d in (IMAGE_DIR, OUTPUT_NG_DIR, OUTPUT_OK_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── GPIO Pin Assignment (BCM) ────────────────────────────────────────────────

START_PIN  = 17   # IN  — rising edge = start inspection
DONE_PIN   = 27   # IN  — rising edge from machine = stop / return to standby
ACK_PIN    = 22   # OUT — pulse HIGH when result is ready
FAIL_A_PIN = 24   # OUT — HIGH = IC_A failed
FAIL_B_PIN = 25   # OUT — HIGH = IC_B failed

_ACK_PULSE_S     = 0.1
_CAPTURE_RETRIES = 2
_RETRY_DELAY_S   = 0.2
_BASLER_TIMEOUT  = 5000   # ms

# ─── Camera / Processing Constants ───────────────────────────────────────────

CAMERA_SERIAL      = ""       # Basler serial; empty = first available
CAMERA_EXPOSURE_US = 10000    # µs — synced with UI Exposure field
CAMERA_WARMUP      = 3        # warmup frames after open

PROC_W, PROC_H = 640, 640     # resize target for OpenVINO YOLO

# ─── Model Classes ────────────────────────────────────────────────────────────
# PASS cell = any detection present (class irrelevant), FAIL = no detection
MODEL_CLASSES = ("IC_Presence", "Text")


# ─── Stage & Error Flags ──────────────────────────────────────────────────────

class Stage(Enum):
    STANDBY = "STANDBY"
    BUSY    = "BUSY"
    ERROR   = "ERROR"

class ErrorFlag(Enum):
    NONE           = None
    CAMERA_ERROR   = "CAMERA_ERROR"
    MODEL_ERROR    = "MODEL_ERROR"
    TEMPLATE_ERROR = "TEMPLATE_ERROR"
    GPIO_ERROR     = "GPIO_ERROR"
    CONFIG_ERROR   = "CONFIG_ERROR"


# ─── Exceptions ───────────────────────────────────────────────────────────────

class InspectionError(Exception):
    pass

class MarkMissingError(InspectionError):
    def __init__(self, ic_position: str, missing_cells: list):
        self.ic_position   = ic_position
        self.missing_cells = missing_cells
        super().__init__(f"IC_{ic_position}: mark missing at {missing_cells}")

class SystemError(InspectionError):
    pass

class CameraError(SystemError):
    pass

class ModelError(SystemError):
    pass

class TemplateError(SystemError):
    pass

class GPIOError(SystemError):
    pass

class ConfigError(InspectionError):
    pass


# ─── Image ────────────────────────────────────────────────────────────────────

_img_counter      = 0
_img_counter_lock = threading.Lock()

def _next_image_id() -> str:
    global _img_counter
    with _img_counter_lock:
        _img_counter += 1
        return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{_img_counter:03d}"


class Image:
    """Immutable container for one captured frame.

    frame    — BGR numpy array (H × W × 3)
    image_id — unique ID: YYYYMMDD_HHMMSS_NNN
    """

    def __init__(self, frame: np.ndarray, image_id: str):
        self.frame    = frame
        self.image_id = image_id

    @property
    def shape(self) -> tuple:
        return self.frame.shape

    def __repr__(self) -> str:
        h, w = self.frame.shape[:2]
        return f"<Image id={self.image_id} size={w}×{h}>"


# ─── ImageIO ─────────────────────────────────────────────────────────────────

class ImageIO:
    """Load / save / list images — mirrors Ref_sample ImageIO pattern.

    load()        → BGR ndarray normalised to PROC_H × PROC_W
    save()        → writes BGR or grayscale as-is
    list_images() → sorted absolute paths for a folder
    """

    _EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

    def load(self, path: str) -> np.ndarray:
        img = cv.imread(path, cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load: {path}")
        if img.ndim == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        if img.shape[:2] != (PROC_H, PROC_W):
            img = cv.resize(img, (PROC_W, PROC_H), interpolation=cv.INTER_AREA)
        return img

    def save(self, path: str, img: np.ndarray):
        cv.imwrite(path, img)

    def list_images(self, folder: str) -> list:
        if not os.path.isdir(folder):
            return []
        return sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in self._EXTS
        )


# ─── Camera ───────────────────────────────────────────────────────────────────

class Camera:
    """Acquires frames from a Basler camera or image files in Input/.

    CAMERA="camera"    — live Basler feed via Pylon SDK (Mono8 → BGR).
    CAMERA="directory" — loads sorted image files from IMAGE_DIR, cycling.

    capture() retries up to 2× (200 ms apart) before raising CameraError.
    """

    def __init__(self):
        self._cam      = None
        self._files: list = []
        self._index    = 0
        self._image_io = ImageIO()

        if CAMERA == "camera":
            self._open_basler()
        elif CAMERA == "directory":
            self._open_directory()
        else:
            raise CameraError(f"Invalid CAMERA flag: '{CAMERA}'")

    # ── Basler ────────────────────────────────────────────────────────────────

    def _open_basler(self):
        if not _PYLON_AVAILABLE:
            raise CameraError("pypylon not installed")
        try:
            tl     = pylon.TlFactory.GetInstance()
            devs   = tl.EnumerateDevices()
            device = None
            if CAMERA_SERIAL:
                for d in devs:
                    if d.GetSerialNumber() == CAMERA_SERIAL:
                        device = tl.CreateDevice(d)
                        break
            if device is None:
                device = tl.CreateFirstDevice()

            self._cam = pylon.InstantCamera(device)
            self._cam.Open()
            self._cam.ExposureAuto.SetValue("Off")
            self._cam.ExposureTimeAbs.SetValue(float(CAMERA_EXPOSURE_US))
            self._cam.PixelFormat.SetValue("Mono8")
            self._cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            for _ in range(CAMERA_WARMUP):
                r = self._cam.RetrieveResult(
                    _BASLER_TIMEOUT, pylon.TimeoutHandling_ThrowException)
                r.Release()
            print(f"[Camera] serial={CAMERA_SERIAL or 'first'} "
                  f"exposure={CAMERA_EXPOSURE_US}µs warmup={CAMERA_WARMUP}")
        except Exception as exc:
            raise CameraError(f"Basler open failed: {exc}") from exc

    def set_exposure(self, us: int):
        if CAMERA == "camera" and self._cam is not None:
            try:
                self._cam.ExposureTimeAbs.SetValue(float(us))
            except Exception as e:
                print(f"[Camera] Exposure error: {e}")

    # ── Directory ─────────────────────────────────────────────────────────────

    def _open_directory(self):
        os.makedirs(IMAGE_DIR, exist_ok=True)
        self._scan_directory()

    def _scan_directory(self):
        self._files = self._image_io.list_images(IMAGE_DIR)

    # ── Capture ───────────────────────────────────────────────────────────────

    def capture(self) -> Image:
        last_exc = None
        for attempt in range(1 + _CAPTURE_RETRIES):
            try:
                return Image(self._grab(), _next_image_id())
            except CameraError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < _CAPTURE_RETRIES:
                    time.sleep(_RETRY_DELAY_S)
        if CAMERA == "directory":
            self._index += 1
        raise CameraError(
            f"Capture failed after {_CAPTURE_RETRIES} retries: {last_exc}"
        ) from last_exc

    def _grab(self) -> np.ndarray:
        return self._grab_basler() if CAMERA == "camera" else self._grab_directory()

    def _grab_basler(self) -> np.ndarray:
        result = self._cam.RetrieveResult(
            _BASLER_TIMEOUT, pylon.TimeoutHandling_ThrowException)
        if not result.GrabSucceeded():
            desc = result.GetErrorDescription()
            result.Release()
            raise RuntimeError(desc)
        arr = result.GetArray().copy()
        result.Release()
        bgr = cv.cvtColor(arr, cv.COLOR_GRAY2BGR)
        return cv.resize(bgr, (PROC_W, PROC_H), interpolation=cv.INTER_AREA)

    def _grab_directory(self) -> np.ndarray:
        if self._index % max(len(self._files), 1) == 0:
            self._scan_directory()
        if not self._files:
            raise CameraError("No images in Input/ — add image files to continue")
        path = self._files[self._index % len(self._files)]
        try:
            frame = self._image_io.load(path)
        except FileNotFoundError as exc:
            raise RuntimeError(str(exc)) from exc
        self._index += 1
        return frame

    def release(self):
        if self._cam is not None:
            try:
                self._cam.StopGrabbing()
                self._cam.Close()
            except Exception:
                pass
            self._cam = None


# ─── RaspberryIO ──────────────────────────────────────────────────────────────

class RaspberryIO:
    """Manages all GPIO I/O.

    IO=True  — drives physical BCM pins via RPi.GPIO.
    IO=False — mocks every state change as a printed log.
    """

    def __init__(self):
        self._start_cb: callable = None
        self._done_cb:  callable = None
        self._lock = threading.Lock()

        if IO:
            if not _GPIO_AVAILABLE:
                raise GPIOError("RPi.GPIO not available — cannot use IO=True")
            self._setup_gpio()

    def _setup_gpio(self):
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(START_PIN, GPIO.IN,  pull_up_down=GPIO.PUD_DOWN)
            GPIO.setup(DONE_PIN,  GPIO.IN,  pull_up_down=GPIO.PUD_DOWN)
            for pin in (ACK_PIN, FAIL_A_PIN, FAIL_B_PIN):
                GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            GPIO.add_event_detect(START_PIN, GPIO.RISING,
                                  callback=self._on_start_edge, bouncetime=50)
            GPIO.add_event_detect(DONE_PIN,  GPIO.RISING,
                                  callback=self._on_done_edge,  bouncetime=50)
        except Exception as exc:
            raise GPIOError(f"GPIO init failed: {exc}") from exc

    def _on_start_edge(self, _):
        if self._start_cb:
            self._start_cb()

    def _on_done_edge(self, _):
        if self._done_cb:
            self._done_cb()

    def register_start_callback(self, fn: callable):
        self._start_cb = fn

    def register_done_callback(self, fn: callable):
        self._done_cb = fn

    def set_result(self, fail_a: bool, fail_b: bool):
        if IO and _GPIO_AVAILABLE:
            GPIO.output(FAIL_A_PIN, GPIO.HIGH if fail_a else GPIO.LOW)
            GPIO.output(FAIL_B_PIN, GPIO.HIGH if fail_b else GPIO.LOW)
        else:
            a = "HIGH" if fail_a else "LOW"
            b = "HIGH" if fail_b else "LOW"
            print(f"[IO MOCK] FAIL_A → {a}  FAIL_B → {b}")

    def pulse_ack(self):
        if IO and _GPIO_AVAILABLE:
            GPIO.output(ACK_PIN, GPIO.HIGH)
            time.sleep(_ACK_PULSE_S)
            GPIO.output(ACK_PIN, GPIO.LOW)
        else:
            print("[IO MOCK] ACK → HIGH (pulse)")

    def clear_outputs(self):
        if IO and _GPIO_AVAILABLE:
            for pin in (FAIL_A_PIN, FAIL_B_PIN, ACK_PIN):
                GPIO.output(pin, GPIO.LOW)
        else:
            for name in ("FAIL_A", "FAIL_B", "ACK"):
                print(f"[IO MOCK] {name} → LOW")

    def release(self):
        if IO and _GPIO_AVAILABLE:
            try:
                GPIO.cleanup()
            except Exception:
                pass


# ─── ImageView ────────────────────────────────────────────────────────────────

class ImageView(QLabel):
    """Camera frame display with ROI overlay — mirrors Ref_sample ImageView.

    set_image(bgr)      — store frame and defer scaled refresh.
    set_rois(rois)      — overlay ROI boxes; rois = [(x,y,w,h,bgr_color,label), ...]
    clear_rois()        — remove overlays.
    resizeEvent()       — auto-rescale stored frame on window resize.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._orig: np.ndarray | None = None
        self._rois: list = []
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "background-color: #E2FDFF; border-radius: 8px; color: #788BFF;"
        )
        self.setText("No image")

    def set_image(self, frame: np.ndarray):
        """Accept BGR (H×W×3) — store and defer refresh."""
        self._orig = frame.copy()
        QTimer.singleShot(0, self._refresh)

    def set_rois(self, rois: list):
        """rois = [(x, y, w, h, bgr_color, label), ...]"""
        self._rois = rois
        self._refresh()

    def clear_rois(self):
        self._rois = []
        self._refresh()

    def _refresh(self):
        if self._orig is None:
            return
        frame = self._orig.copy()
        for x, y, w, h, color, label in self._rois:
            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if label:
                cv.putText(frame, label, (x + 2, y + 14),
                           cv.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv.LINE_AA)

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        fh, fw = rgb.shape[:2]
        qimg = QImage(rgb.data, fw, fh, fw * 3, QImage.Format_RGB888).copy()
        pix  = QPixmap.fromImage(qimg)
        lw, lh = self.width(), self.height()
        if lw > 0 and lh > 0:
            pix = pix.scaled(lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()


# ─── Stylesheet ───────────────────────────────────────────────────────────────
#
#  Palette
#  #5465FF  deepest  — window bg, button fill
#  #788BFF  dark     — panel / frame bg
#  #9BB1FF  base     — card surface bg  (Setup, Controls, Stats, Badges)
#  #BFD7FF  light    — badge / accent surfaces, PASS indicator
#  #E2FDFF  lightest — image display area bg
#  #FFFFFF  white    — input field bg
#  #EF5350  red      — FAIL badge, error banner

STYLE = """
    QMainWindow, QWidget {
        background-color: #5465FF;
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
    }
    QFrame {
        border-radius: 8px;
    }
    QPushButton {
        background-color: #5465FF;
        color: #FFFFFF;
        border-radius: 8px;
        padding: 6px 14px;
        border: 1px solid #788BFF;
        min-height: 28px;
    }
    QPushButton:disabled {
        background-color: #9BB1FF;
        color: #BFD7FF;
        border: 1px solid #9BB1FF;
    }
    QLineEdit {
        background-color: #FFFFFF;
        color: #5465FF;
        border-radius: 6px;
        border: 2px solid #5465FF;
        padding: 4px 8px;
        min-height: 26px;
    }
    QLineEdit:focus {
        border: 2px solid #9BB1FF;
    }
    QLabel {
        color: #FFFFFF;
        background-color: transparent;
    }
"""


# ─── Fail Dialog ──────────────────────────────────────────────────────────────

class FailDialog(QDialog):
    """Modal popup shown when any IC fails inspection."""

    def __init__(self, ic_a_missing: list, ic_b_missing: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inspection Failed")
        self.setModal(True)
        self.setStyleSheet(
            "QDialog { background-color: #5465FF; }"
            "QLabel  { color: #FFFFFF; background-color: transparent; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(10)

        title = QLabel("Inspection Failed")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        layout.addWidget(title)

        if ic_a_missing:
            layout.addWidget(QLabel(f"IC_A  FAIL — missing cells: {ic_a_missing}"))
        if ic_b_missing:
            layout.addWidget(QLabel(f"IC_B  FAIL — missing cells: {ic_b_missing}"))

        ack_btn = QPushButton("Acknowledge")
        ack_btn.clicked.connect(self.accept)
        layout.addWidget(ack_btn, alignment=Qt.AlignCenter)


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    _sig_start = pyqtSignal()
    _sig_done  = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClearIC Inspect")
        self.setMinimumSize(1280, 720)
        self.resize(1400, 800)
        self.setStyleSheet(STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_main_view(), stretch=3)
        root.addWidget(self._build_right_panel(), stretch=1)

        # ── System state ──────────────────────────────────────────────────────
        self._stage      = Stage.STANDBY
        self._error_flag = ErrorFlag.NONE
        self._stat_pass  = 0
        self._stat_fail  = 0
        self._stat_error = 0

        # ── Qt signal bridge (GPIO thread → main thread) ──────────────────────
        self._sig_start.connect(self._on_start)
        self._sig_done.connect(self._on_done)

        # ── Init GPIO ─────────────────────────────────────────────────────────
        try:
            self._rio = RaspberryIO()
            self._rio.register_start_callback(self._sig_start.emit)
            self._rio.register_done_callback(self._sig_done.emit)
        except GPIOError as exc:
            self._rio = None
            self._set_error(ErrorFlag.GPIO_ERROR, str(exc))

        # ── Init Camera ───────────────────────────────────────────────────────
        try:
            self._cam = Camera()
        except CameraError as exc:
            self._cam = None
            self._set_error(ErrorFlag.CAMERA_ERROR, str(exc))

        self._refresh_status()

    # ── Main view (left) ──────────────────────────────────────────────────────

    def _build_main_view(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #788BFF; border-radius: 8px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._view = ImageView()
        layout.addWidget(self._view, stretch=1)

        self.error_banner = QLabel()
        self.error_banner.setAlignment(Qt.AlignCenter)
        self.error_banner.setStyleSheet(
            "background-color: #EF5350; color: #FFFFFF;"
            "border-radius: 8px; padding: 8px;"
        )
        self.error_banner.hide()
        layout.addWidget(self.error_banner)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)
        bottom_row.addWidget(self._build_badges())
        bottom_row.addWidget(self._build_stats_section())
        bottom_row.addStretch()
        layout.addLayout(bottom_row)

        return frame

    def _build_badges(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #9BB1FF; border-radius: 8px;")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.badge_a = self._make_badge("IC_A")
        self.badge_b = self._make_badge("IC_B")
        layout.addWidget(self.badge_a)
        layout.addWidget(self.badge_b)
        return frame

    def _make_badge(self, label: str) -> QLabel:
        badge = QLabel(f"{label}\n—")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(110, 56)
        badge.setFont(QFont("Segoe UI", 11, QFont.Bold))
        badge.setStyleSheet(
            "background-color: #788BFF; color: #FFFFFF;"
            "border-radius: 8px; padding: 4px; border: 1px solid #BFD7FF;"
        )
        return badge

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #5465FF; border-radius: 8px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)
        layout.addWidget(self._build_setup_section())
        layout.addWidget(self._build_controls_section())
        layout.addStretch()
        return frame

    def _build_setup_section(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #788BFF; border-radius: 8px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._section_title("Setup"))

        self.inp_exposure   = self._labeled_input(layout, "Exposure (µs)",
                                                  str(CAMERA_EXPOSURE_US),
                                                  QIntValidator(1, 1_000_000))
        self.inp_exposure.editingFinished.connect(self._on_exposure_changed)

        self.inp_scale      = self._labeled_input(layout, "Scale / ratio",
                                                  "1.0", QDoubleValidator(0.1, 10.0, 3))
        self.inp_col_offset = self._labeled_input(layout, "Column offset (px)",
                                                  "0",   QIntValidator(0, 2000))
        self.inp_ic_b_x     = self._labeled_input(layout, "IC_B offset X (px)",
                                                  "0",   QIntValidator(-4000, 4000))
        self.inp_ic_b_y     = self._labeled_input(layout, "IC_B offset Y (px)",
                                                  "0",   QIntValidator(-4000, 4000))

        self.btn_set_anchor  = QPushButton("Set Anchor")
        self.btn_preview_roi = QPushButton("Preview ROIs")
        self.btn_save_tmpl   = QPushButton("Save Template")

        for btn in (self.btn_set_anchor, self.btn_preview_roi, self.btn_save_tmpl):
            layout.addWidget(btn)

        self.btn_set_anchor.clicked.connect(self.on_set_anchor)
        self.btn_preview_roi.clicked.connect(self.on_preview_roi)
        self.btn_save_tmpl.clicked.connect(self.on_save_template)

        return frame

    def _build_controls_section(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #788BFF; border-radius: 8px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._section_title("Controls"))

        self.btn_trigger = QPushButton("Manual Trigger")
        self.btn_trigger.clicked.connect(self.on_manual_trigger)
        layout.addWidget(self.btn_trigger)

        return frame

    def _build_stats_section(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #9BB1FF; border-radius: 8px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.lbl_status = QLabel("Status: Standby")
        self.lbl_status.setFont(QFont("Segoe UI", 10, QFont.Bold))
        layout.addWidget(self.lbl_status)

        self.lbl_pass       = QLabel("Pass:  0")
        self.lbl_fail       = QLabel("Fail:  0")
        self.lbl_error_cnt  = QLabel("Error: 0")
        self.lbl_cycle_time = QLabel("Last cycle: — ms")

        for lbl in (self.lbl_pass, self.lbl_fail, self.lbl_error_cnt, self.lbl_cycle_time):
            layout.addWidget(lbl)

        return frame

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _section_title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        return lbl

    def _labeled_input(self, layout: QVBoxLayout, label: str,
                       default: str = "0", validator=None) -> QLineEdit:
        lbl = QLabel(label)
        lbl.setFont(QFont("Segoe UI", 9, QFont.Bold))
        layout.addWidget(lbl)
        field = QLineEdit(default)
        field.setStyleSheet(
            "QLineEdit {"
            "  background-color: #FFFFFF;"
            "  color: #5465FF;"
            "  border-radius: 6px;"
            "  border: 2px solid #5465FF;"
            "  padding: 4px 8px;"
            "  min-height: 26px;"
            "}"
            "QLineEdit:focus { border: 2px solid #9BB1FF; }"
        )
        if validator is not None:
            field.setValidator(validator)
        layout.addWidget(field)
        return field

    def _int_val(self, field: QLineEdit, default: int = 0) -> int:
        try:
            return int(field.text())
        except ValueError:
            return default

    def _float_val(self, field: QLineEdit, default: float = 0.0) -> float:
        try:
            return float(field.text())
        except ValueError:
            return default

    def _on_exposure_changed(self):
        us = self._int_val(self.inp_exposure, CAMERA_EXPOSURE_US)
        if self._cam is not None:
            self._cam.set_exposure(us)

    # ── Inspection cycle ──────────────────────────────────────────────────────

    def _on_start(self):
        """Runs on Qt main thread via _sig_start signal."""
        if self._stage == Stage.BUSY:
            print("[IO] START ignored — busy")
            return

        self._stage = Stage.BUSY
        self.error_banner.hide()
        self._refresh_status()
        self.btn_trigger.setEnabled(False)

        t0 = time.time()
        try:
            if self._cam is None:
                raise CameraError("Camera not initialized")

            img = self._cam.capture()
            self._view.set_image(img.frame)

            # ── Detection placeholder ──────────────────────────────────────────
            # TODO: replace with real YOLO/OpenVINO inference
            # Classes: "IC_Presence", "Text"
            # Per cell: PASS = any detection present, FAIL = no detection
            # Per IC:   PASS = all 6 cells pass
            ic_a_passed  = True
            ic_b_passed  = True
            ic_a_missing: list = []
            ic_b_missing: list = []
            # ──────────────────────────────────────────────────────────────────

            self._update_badge("A", ic_a_passed)
            self._update_badge("B", ic_b_passed)

            if self._rio:
                self._rio.set_result(
                    fail_a=not ic_a_passed,
                    fail_b=not ic_b_passed,
                )
                self._rio.pulse_ack()

            cycle_ms = (time.time() - t0) * 1000
            overall  = ic_a_passed and ic_b_passed

            if overall:
                self._stat_pass += 1
            else:
                self._stat_fail += 1
                FailDialog(ic_a_missing, ic_b_missing, self).exec_()

            self._refresh_stats(cycle_ms)

        except CameraError as exc:
            self._stat_error += 1
            self._set_error(ErrorFlag.CAMERA_ERROR, f"Camera: {exc}")
            if self._rio:
                self._rio.set_result(fail_a=False, fail_b=False)
                self._rio.pulse_ack()

        except Exception as exc:
            self._stat_error += 1
            self._set_error(ErrorFlag.CAMERA_ERROR, str(exc))

        finally:
            if self._error_flag == ErrorFlag.NONE:
                self._stage = Stage.STANDBY
            self.btn_trigger.setEnabled(True)
            self._refresh_status()
            if (DEBUG and CAMERA == "directory"
                    and self._cam is not None
                    and self._error_flag == ErrorFlag.NONE):
                QTimer.singleShot(0, self._sig_done.emit)

    def _on_done(self):
        """DONE_PIN handler — clear outputs, return to standby.

        DEBUG + CAMERA="directory": auto-advance to the next image.
        """
        if self._rio:
            self._rio.clear_outputs()
        self._stage      = Stage.STANDBY
        self._error_flag = ErrorFlag.NONE
        self.error_banner.hide()
        self._refresh_status()

        if DEBUG and CAMERA == "directory" and self._cam is not None:
            QTimer.singleShot(0, self._sig_start.emit)

    # ── Internal UI updaters ──────────────────────────────────────────────────

    def _update_badge(self, ic: str, passed: bool):
        badge = self.badge_a if ic == "A" else self.badge_b
        color = "#BFD7FF" if passed else "#EF5350"
        badge.setText(f"IC_{ic}\n{'PASS' if passed else 'FAIL'}")
        badge.setStyleSheet(
            f"background-color: #788BFF; color: {color};"
            f"border-radius: 8px; padding: 4px; border: 2px solid {color};"
        )

    def _refresh_status(self):
        mapping = {
            Stage.STANDBY: ("Standby", "#FFFFFF"),
            Stage.BUSY:    ("Running", "#BFD7FF"),
            Stage.ERROR:   ("Error",   "#EF5350"),
        }
        text, color = mapping[self._stage]
        self.lbl_status.setText(f"Status: {text}")
        self.lbl_status.setStyleSheet(f"color: {color}; background-color: transparent;")

    def _refresh_stats(self, cycle_ms: float = None):
        self.lbl_pass.setText(f"Pass:  {self._stat_pass}")
        self.lbl_fail.setText(f"Fail:  {self._stat_fail}")
        self.lbl_error_cnt.setText(f"Error: {self._stat_error}")
        if cycle_ms is not None:
            self.lbl_cycle_time.setText(f"Last cycle: {cycle_ms:.0f} ms")

    def _set_error(self, flag: ErrorFlag, message: str):
        self._error_flag = flag
        self._stage      = Stage.ERROR
        self.error_banner.setText(message)
        self.error_banner.show()
        self._refresh_status()

    # ── Slots ─────────────────────────────────────────────────────────────────

    def on_set_anchor(self):
        pass

    def on_preview_roi(self):
        pass

    def on_save_template(self):
        pass

    def on_manual_trigger(self):
        self._sig_start.emit()

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._rio:
            self._rio.clear_outputs()
            self._rio.release()
        if self._cam:
            self._cam.release()
        event.accept()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
