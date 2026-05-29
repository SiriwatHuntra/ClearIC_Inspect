import os
import time
from datetime import datetime

import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore

from ..utils.exceptions import (CameraError, ModelError, GPIOError,
                                TemplateError, MarkMissingError)
from ..utils.config import ConfigLoader
from ..utils.logger import Logger
from ..io.camera import Camera
from ..io.gpio import RaspberryIO, CellCon
from ..engine.detector import Detector
from ..engine.template import (TemplateManager, TemplateMatcher,
                               _TEMPLATE_FILE, _find_second_ic, _build_cells)
from ..engine.inspector import Inspector, _tmpl_color_a, _tmpl_color_b
from ..engine.worker import RunWorker
from .image_view import ImageView
from .browser import ImageBrowserPage
from .dialogs import LotStartDialog


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

        self._run_state          = "standby"
        self._session_start_time = 0.0
        self._lot_number         = ""
        self._package_name       = ""

        self._ocr_operator:     str = ""
        self._ocr_expect_value: str = ""

        self._pending_ic_a:  QtCore.QRect | None = None
        self._pending_ic_b:  QtCore.QRect | None = None
        self._setup_image:   np.ndarray | None   = None
        self._setup_state:   str                 = 'idle'

        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(int(screen.width() * 0.90), int(screen.height() * 0.90))
        self.move(screen.x() + int(screen.width() * 0.05),
                  screen.y() + int(screen.height() * 0.05))

        self._build_ui()
        self._init_system()

    def _build_ui(self):
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

        root = QtWidgets.QHBoxLayout(insp_page)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        left_frame = QtWidgets.QFrame()
        left_frame.setObjectName("main_view")
        left_lay = QtWidgets.QVBoxLayout(left_frame)
        left_lay.setContentsMargins(8, 8, 8, 8)
        left_lay.setSpacing(6)

        self._view = ImageView()
        left_lay.addWidget(self._view, stretch=1)

        self._error_banner = QtWidgets.QFrame()
        self._error_banner.setObjectName("error_banner")
        eb_lay = QtWidgets.QHBoxLayout(self._error_banner)
        eb_lay.setContentsMargins(8, 4, 8, 4)
        self._error_lbl = QtWidgets.QLabel("")
        self._error_lbl.setStyleSheet("color:#FFFFFF;font-weight:bold")
        eb_lay.addWidget(self._error_lbl)
        self._error_banner.hide()
        left_lay.addWidget(self._error_banner)

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

        right_frame = QtWidgets.QFrame()
        right_frame.setObjectName("panel_right")
        right_frame.setMinimumWidth(240)
        right_lay = QtWidgets.QVBoxLayout(right_frame)
        right_lay.setContentsMargins(8, 8, 8, 8)
        right_lay.setSpacing(8)

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

        ctrl_frame = QtWidgets.QFrame()
        ctrl_frame.setObjectName("controls_frame")
        ctrl_lay = QtWidgets.QVBoxLayout(ctrl_frame)
        ctrl_lay.setSpacing(6)

        lbl_ctrl = QtWidgets.QLabel("Controls")
        lbl_ctrl.setStyleSheet("font-weight:bold;font-size:13px")
        ctrl_lay.addWidget(lbl_ctrl)

        self._btn_action = QtWidgets.QPushButton("Start")
        self._btn_action.setEnabled(False)
        self._btn_action.clicked.connect(self._on_action_click)
        ctrl_lay.addWidget(self._btn_action)

        self._btn_stop = QtWidgets.QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_run)
        ctrl_lay.addWidget(self._btn_stop)

        right_lay.addWidget(ctrl_frame)

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

        right_lay.addWidget(self._ocr_frame)
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
        self._rebuild_inspector()
        print(f"[Settings] border={bp}px  labels={show_labels}  warmup={wf}")

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
        self._rebuild_inspector()
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
            ann_show_labels=self._cfg.get("ANN_SHOW_LABELS", True),
        )

    def _retry_camera_open(self):
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
                self._start_draw_a()
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
        self._btn_new_tmpl.setEnabled(True)
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
            self._btn_action.setEnabled(False)
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
        return (not self._cfg.get("IO", False)
                and self._cfg.get("CAMERA", "directory") == "camera")

    def _on_action_click(self):
        if self._run_state == "standby":
            self._start_run()
        elif self._run_state == "running":
            if self._is_mock_trigger_mode():
                self._worker.trigger()
            else:
                self._pause_run()
        elif self._run_state == "paused":
            self._resume_run()

    def _start_run(self):
        if self._worker and self._worker.isRunning():
            return

        if not self._detector or not self._detector.is_ready():
            self._show_error("Detector not ready.")
            return
        inspector = self._inspector
        if inspector is None:
            self._show_error("No inspector — create a template first.")
            return

        self._ocr_operator     = self._edit_op_number.text().strip()
        self._ocr_expect_value = self._edit_ocr_expect.text().strip()

        self._lbl_ocr_status.setText("Verifying…")
        self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
        self._lbl_lot_info.setText("—")

        lot = LotStartDialog.request(parent=self, api_fn=self._cellcon.get_lot)
        if lot is None:
            self._lbl_ocr_status.setText("Fill both fields to enable Start.")
            self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
            return
        self._lot_number   = lot
        self._package_name = inspector._template.get("package_name", "")

        self._session_start_time = time.monotonic()

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

    def _start_worker(self, inspector: Inspector, gpio: RaspberryIO):
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
        self._ocr_used_mark = expected_mark
        import base64
        try:
            import requests as _req
        except ImportError:
            debug = self._cfg.get("DEBUG", True)
            if not debug:
                self._lbl_ocr_status.setText("OCR unavailable — 'requests' not installed")
                self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#FF6B6B")
                return False
            return True

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
                pass

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

    def _resume_run(self):
        if self._worker:
            self._worker.resume()

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
        elapsed = time.monotonic() - self._session_start_time
        self._logger.end_lot(
            "STOPPED", self._stats_pass, self._stats_fail,
            self._stats_error, elapsed)
        if self._worker:
            self._worker.stop()
            self._worker.wait(3000)
        self._enter_standby()

    def _on_session_reset(self, new_lot: str):
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
        if self._run_state == "standby":
            return
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
        self._edit_op_number.clear()
        self._edit_ocr_expect.clear()
        self._lbl_ocr_status.setText("Fill both fields to enable Start.")
        self._lbl_ocr_status.setStyleSheet("font-size:11px;color:#E2FDFF")
        self._btn_action.setEnabled(self._ocr_fields_valid())
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
        if not self._camera:
            return
        try:
            img = self._camera.grab_first()
            self._view.set_image(img)
            self._view.clear_overlays()
            self._setup_image = img
        except CameraError:
            pass

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
        self._show_error(msg)

    def _show_error(self, msg: str):
        self._error_lbl.setText(f"Error: {msg}")
        self._error_banner.show()

    def closeEvent(self, e):
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        if self._camera:
            self._camera.close()
        if self._gpio:
            self._gpio.cleanup()
        e.accept()
