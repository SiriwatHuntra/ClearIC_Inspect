import os
import glob

import cv2
from PyQt5 import QtWidgets, QtGui, QtCore

from .image_view import ImageView


class FolderScanWorker(QtCore.QThread):
    """Scans Output/ directory tree; emits flat list of (label, leaf_dir_path)."""
    sig_entries = QtCore.pyqtSignal(list)

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
                entries.append((date, date_dir))
            else:
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
    sig_thumb = QtCore.pyqtSignal(int, object)
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
            card_bg = "#E07820"
        elif _stem.endswith("_GS"):
            card_bg = "#A0B830"
        elif _stem.endswith("_NG"):
            card_bg = "#FA6781"
        elif _stem.endswith("_G"):
            card_bg = "#478B8D"
        else:
            card_bg = "#788BFF"
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

    _COLS = 4

    def __init__(self, out_dir: str = "Output/", parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._out_dir       = out_dir
        self._all_paths: list = []
        self._paths: list    = []
        self._cur_idx        = 0
        self._subfolder      = "RealImg"
        self._suffix_filter  = "FAIL"
        self._cards: list    = []
        self._current_base: str = ""
        self._img_ratio: float = 3 / 4
        self._thumb_worker: ThumbnailWorker | None = None
        self._scan_worker:  FolderScanWorker | None = None

        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        self._folder_list = QtWidgets.QListWidget()
        self._folder_list.setMinimumWidth(150)
        self._folder_list.setMaximumWidth(260)
        self._folder_list.setStyleSheet(
            "QListWidget{background:#788BFF;border-radius:6px;color:#FFFFFF;font-size:11px}"
            "QListWidget::item:selected{background:#5465FF;color:#FFFFFF}"
        )
        self._folder_list.itemClicked.connect(self._on_folder_selected)
        root.addWidget(self._folder_list)

        self._stack = QtWidgets.QStackedWidget()
        root.addWidget(self._stack, stretch=1)

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
        self._stack.addWidget(grid_page)

        img_page = QtWidgets.QWidget()
        img_lay  = QtWidgets.QVBoxLayout(img_page)
        img_lay.setContentsMargins(0, 0, 0, 0)
        img_lay.setSpacing(4)
        self._img_view = ImageView()
        self._img_view.right_clicked.connect(self._back_to_grid)
        img_lay.addWidget(self._img_view, stretch=1)
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
        self._stack.addWidget(img_page)

        right = QtWidgets.QFrame()
        right.setObjectName("panel_right")
        right.setMinimumWidth(130)
        right_lay = QtWidgets.QVBoxLayout(right)
        right_lay.setContentsMargins(8, 8, 8, 8)
        right_lay.setSpacing(10)

        right_lay.addWidget(self._section_label("Source"))
        self._grp_src = QtWidgets.QButtonGroup(self)
        self._btn_realimg = self._toggle_btn("RealImg", checked=True)
        self._btn_annimg  = self._toggle_btn("Image",   checked=False)
        self._grp_src.addButton(self._btn_realimg, 0)
        self._grp_src.addButton(self._btn_annimg,  1)
        right_lay.addWidget(self._btn_realimg)
        right_lay.addWidget(self._btn_annimg)
        self._grp_src.buttonClicked.connect(self._on_src_toggle)

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

        self._lbl_count = QtWidgets.QLabel("—")
        self._lbl_count.setStyleSheet("font-size:16px;font-weight:bold;color:#E2FDFF")
        self._lbl_count.setAlignment(QtCore.Qt.AlignCenter)
        self._lbl_count.setWordWrap(True)
        right_lay.addWidget(self._lbl_count)

        right_lay.addStretch()

        kb_frame = QtWidgets.QFrame()
        kb_frame.setObjectName("setup_frame")
        kb_lay = QtWidgets.QVBoxLayout(kb_frame)
        kb_lay.setContentsMargins(8, 6, 8, 6)
        kb_lay.setSpacing(6)
        right_lay.addWidget(kb_frame)

        lbl_kb = QtWidgets.QLabel("Controls")
        lbl_kb.setStyleSheet("font-size:14px;font-weight:bold;color:#E2FDFF")
        kb_lay.addWidget(lbl_kb)

        _KBD_STYLE = "QLabel{font-size:13px;color:#E2FDFF;padding:1px 0px;}"
        _KEY_STYLE = ("QLabel{font-size:13px;font-weight:bold;color:#5465FF;"
                      "background:#FFFFFF;border-radius:3px;padding:2px 7px;}")

        for key_text, desc_text in [
            ("← →",    "Prev / Next"),
            ("R-Click", "Back to grid"),
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

        self._btn_back = QtWidgets.QPushButton("← Back")
        self._btn_back.clicked.connect(self._back_to_grid)
        self._btn_back.setStyleSheet(
            "QPushButton{background:#FFFFFF;color:#5465FF;border-radius:6px;"
            "padding:8px 14px;font-weight:bold;font-size:12px;}"
            "QPushButton:hover{background:#E2FDFF;}")
        self._btn_back.hide()
        right_lay.addWidget(self._btn_back)

        root.addWidget(right)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._paths and e.size() != e.oldSize():
            QtCore.QTimer.singleShot(150, self._rebuild_grid)

    def _card_size(self):
        vp_w = self._scroll.viewport().width()
        if vp_w < 4:
            return 100, 96, 96, 72
        sp      = self._grid_layout.horizontalSpacing()
        w_avail = max(1, vp_w - sp * (self._COLS - 1) - 4)
        card_w  = max(60, w_avail // self._COLS)
        thumb_w = max(1, card_w - 4)
        thumb_h = max(1, int(thumb_w * self._img_ratio))
        card_h  = thumb_h + 24
        return card_w, card_h, thumb_w, thumb_h

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

    def _load_folder(self, base_path: str):
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
        self._paths = [p for p in self._all_paths
                       if self._file_matches_filter(p, self._suffix_filter)]
        self._lbl_count.setText(f"{len(self._paths)} images")
        self._rebuild_grid()

    def _rebuild_grid(self):
        if self._thumb_worker and self._thumb_worker.isRunning():
            self._thumb_worker.stop()
            self._thumb_worker.wait(500)

        while self._grid_layout.count():
            item = self._grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._cards = []

        self._stack.setCurrentIndex(0)
        self._btn_back.hide()

        if not self._paths:
            return

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

        self._thumb_worker = ThumbnailWorker(self._paths, thumb_w, thumb_h)
        self._thumb_worker.sig_thumb.connect(self._on_thumbnail_ready)
        self._thumb_worker.start()

    def _on_thumbnail_ready(self, idx: int, pixmap: QtGui.QPixmap):
        if idx < len(self._cards):
            self._cards[idx].set_thumbnail(pixmap)

    def _on_card_clicked(self, idx: int):
        self._cur_idx = idx
        self._show_image(idx)

    def _show_image(self, idx: int):
        if not self._paths:
            return
        self._cur_idx = max(0, min(idx, len(self._paths) - 1))
        self._stack.setCurrentIndex(1)
        self._btn_back.show()
        self.setFocus()
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
