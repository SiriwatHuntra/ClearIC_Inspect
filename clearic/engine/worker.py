import os
import gc
import time
import threading
from datetime import datetime

import cv2
from PyQt5 import QtCore

from ..utils.exceptions import CameraError, TemplateError, MarkMissingError, LowMatchError
from ..utils.models import _next_image_id, _reset_image_counter
from ..utils.logger import Logger
from ..io import Camera
from ..io.hardware import RaspberryIO
from .inspector import Inspector
from .detector import _TOTAL_CELLS


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


def _resolve_ic(missing_first: list,
                confs_first: list,
                confs_second: list,
                w2: float = 0.7,
                w1: float = 0.3,
                pass_thr: float = 0.90,) -> list:
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
        if w2 * c2 + w1 * c1 < pass_thr:
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
    sig_warn          = QtCore.pyqtSignal(str) # soft warning — run continues

    def __init__(self, camera: Camera, inspector: Inspector,
                 gpio: RaspberryIO, logger: Logger,
                 cfg: dict, lot_number: str = "",
                 lighting=None, parent=None):
        super().__init__(parent)
        self._camera     = camera
        self._inspector  = inspector
        self._gpio       = gpio
        self._logger     = logger
        self._cfg        = cfg
        self._lot_number = lot_number
        self._lighting   = lighting
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

        self.sig_status.emit("Running…")
        _reset_image_counter()
        _cycle = 0

        if cam_mode == "camera":
            self._gpio.clear_outputs()  # ensure known-idle state before first cycle
            try:
                self._camera.open()
                self._camera.warmup()
            except CameraError as e:
                self.sig_error.emit(f"Camera error: {e}")
                self.sig_status.emit("ERROR — camera failed to open, restart required.")
                self.sig_done.emit()
                self.sig_status.emit("Standby.")
                return

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
                if self._lighting:
                    self._lighting.on()
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
                # Camera mode: attempt reconnect then continue
                self.sig_status.emit("Camera grab failed — reconnecting…")
                if self._camera.is_open():
                    self._camera.close()
                reconnected = False
                for attempt in range(reconnect_attempts):
                    time.sleep(reconnect_delay)
                    self.sig_status.emit(
                        f"Reconnecting {attempt + 1}/{reconnect_attempts}…")
                    try:
                        self._camera.open()
                        self._camera.warmup()
                        reconnected = True
                        break
                    except CameraError:
                        pass
                if not reconnected:
                    self.sig_error.emit("Camera lost — restart required.")
                    if self._lighting:
                        self._lighting.off()
                    break
                self._gpio.set_busy(False)
                continue

            img_id = _next_image_id()

            out_dir  = self._cfg.get("OUT_DIR", "Output/")
            real_dir, ann_dir = _output_dirs(out_dir, self._lot_number)

            self.sig_status.emit("Inspecting…")

            # Inspect (with one retry on fail)
            is_retry    = False
            miss_a      = []
            miss_b      = []
            ann         = img_bgr
            raw_bgr     = img_bgr.copy()   # preserve unannotated frame; inspect() annotates in-place
            self.sig_image.emit(raw_bgr)   # show raw frame immediately after capture

            try:
                self._inspector.inspect(img_bgr, debug=debug)
                # pass — img_bgr annotated in-place; miss_a/miss_b stay []

            except LowMatchError as lme:
                # Transient bad frame — skip this cycle, do not break the loop
                self.sig_status.emit(f"Low match — skipped. ({lme})")
                if cam_mode == "camera":
                    self._gpio.set_busy(False)
                    if self._lighting:
                        self._lighting.off()
                continue

            except TemplateError as te:
                cycle_ms = (time.perf_counter() - t0) * 1000
                self._logger.log_error("TEMPLATE_ERROR", str(te), cycle_ms)
                if cam_mode == "directory":
                    self.sig_status.emit(f"Skipping {img_id}: {te}")
                    continue
                self.sig_error.emit(f"Template error: {te}")
                self.sig_status.emit("ERROR — template invalid, restart required.")
                self._gpio.clear_outputs()
                if self._camera.is_open():
                    self._camera.close()
                if self._lighting:
                    self._lighting.off()
                break

            except MarkMissingError as e1:
                if cam_mode == "camera":
                    is_retry = True
                    retry_delay = self._cfg.get("RETRY_DELAY_MS", 250) / 1000
                    # Lighting cycle: off → probe → on before re-grab
                    if self._lighting:
                        self._lighting.off()
                        if not self._lighting.probe():
                            self.sig_status.emit("Retry: lighting no response ⚠")
                        self._lighting.on()
                    time.sleep(retry_delay)   # camera exposure settles under fresh lighting
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
                            w2      = self._cfg.get("RETRY_W2", 0.7)
                            w1      = self._cfg.get("RETRY_W1", 0.3)
                            thr     = self._cfg.get("RETRY_PASS_THR", 0.90)
                            miss_a  = (_resolve_ic(e1.missing_a, e1.confs_a, e2.confs_a, w2, w1, thr)
                                       if e1.missing_a else [])
                            miss_b  = (_resolve_ic(e1.missing_b, e1.confs_b, e2.confs_b, w2, w1, thr)
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
                if cam_mode == "camera":
                    self._gpio.clear_outputs()
                    if self._camera.is_open():
                        self._camera.close()
                    if self._lighting:
                        self._lighting.off()
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
                cv2.imwrite(final_real, raw_bgr)
                cv2.imwrite(ann_path, ann)
            else:
                ann_path = ""

            # Emit signals and log
            self.sig_image.emit(ann)
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
                is_overall_pass = not (miss_a or miss_b)
                self._gpio.set_inspec_stage(not is_overall_pass)  # LOW=pass, HIGH=NG
                time.sleep(self._gpio._GPIO_PRE_END_SEC)                      # hold stage for machine to read
                self._gpio.pulse_end_pin()                        # LOW 40 ms → machine reads INSPEC_STAGE
                self._gpio.set_busy(False)                        # BUSY LOW after END pulse done
                self._gpio.set_inspec_stage(True)                 # restore idle HIGH
                if self._lighting:
                    self._lighting.off()
                # camera stays open — closed at lot end by exit guard

            try:
                del img_bgr
            except NameError:
                pass

            _cycle += 1
            if _cycle % 100 == 0:
                gc.collect()

            if _cycle % 500 == 0:
                import shutil as _shutil
                try:
                    free_mb = _shutil.disk_usage(out_dir).free >> 20
                    if free_mb < int(self._cfg.get("DISK_WARN_MB", 200)):
                        self.sig_warn.emit(
                            f"Low disk: {free_mb} MB free — free space or images may not save")
                except OSError:
                    pass

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
            if self._camera.is_open():
                self._camera.close()
            if self._lighting:
                self._lighting.off()
        self.sig_done.emit()   # always emit; _on_run_done guards against double-call
        self.sig_status.emit("Standby.")
