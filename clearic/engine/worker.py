import os
import gc
import time
import threading
from datetime import datetime

import cv2
from PyQt5 import QtCore

from ..utils.exceptions import CameraError, TemplateError, MarkMissingError
from ..utils.models import _next_image_id, _reset_image_counter
from ..utils.logger import Logger
from ..io.camera import Camera
from ..io.gpio import RaspberryIO
from .inspector import Inspector
from .detector import _TOTAL_CELLS


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
    sig_image    = QtCore.pyqtSignal(object)
    sig_result   = QtCore.pyqtSignal(bool, bool, bool)
    sig_fail     = QtCore.pyqtSignal(object, str, str, bool)
    sig_error    = QtCore.pyqtSignal(str)
    sig_status   = QtCore.pyqtSignal(str)
    sig_cycle_ms = QtCore.pyqtSignal(float)
    sig_done          = QtCore.pyqtSignal()
    sig_session_reset = QtCore.pyqtSignal(str)
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
        self._running.set()

    def pause(self):
        self._running.clear()

    def resume(self):
        self._drain_needed.set()
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
            self._gpio.clear_outputs()

        while not self._stop:

            if cam_mode == "camera":
                self.sig_status.emit("Waiting for START signal…")
                if not self._gpio.wait_for_start(lambda: self._stop):
                    break
                if self._stop:
                    break

                self._gpio.set_busy(True)
            else:
                time.sleep(0.05)
                if self._stop:
                    break

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
                reconnected = False
                for attempt in range(reconnect_attempts):
                    self.sig_status.emit(
                        f"Camera lost — reconnecting {attempt + 1}/{reconnect_attempts}…")
                    if self._camera.reconnect(1, reconnect_delay):
                        reconnected = True
                        break
                if reconnected:
                    continue
                self.sig_error.emit(f"Camera error: {e}")
                self.sig_status.emit("ERROR — camera lost, restart required.")
                self._gpio.clear_outputs()
                break

            img_id = _next_image_id()

            out_dir  = self._cfg.get("OUT_DIR", "Output/")
            real_dir, ann_dir = _output_dirs(out_dir, self._lot_number)
            tmp_real = os.path.join(real_dir, f"{img_id}.jpg")
            cv2.imwrite(tmp_real, img_bgr)

            self.sig_status.emit("Inspecting…")

            is_retry    = False
            miss_a      = []
            miss_b      = []
            ann         = img_bgr

            try:
                self._inspector.inspect(img_bgr, debug=debug)

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

            cycle_ms = (time.perf_counter() - t0) * 1000

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

            save_image = suffix != "_G"

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
                self._gpio.set_inspec_stage(not is_overall_pass)
                time.sleep(0.010)
                self._gpio.pulse_end_pin()
                self._gpio.set_inspec_stage(True)

            try:
                del img_bgr
            except NameError:
                pass

            _cycle += 1
            if _cycle % 100 == 0:
                gc.collect()

            if cam_mode != "camera":
                if not self._camera.has_more():
                    self._camera.reset()
                    break

            if not self._running.is_set():
                self.sig_paused.emit()
                self._running.wait()
                if self._stop:
                    break
                if self._drain_needed.is_set():
                    self._gpio.drain_start_pin()
                    self._drain_needed.clear()
                self.sig_resumed.emit()

        if cam_mode == "camera":
            self._gpio.clear_outputs()
        self.sig_done.emit()
        self.sig_status.emit("Standby.")
