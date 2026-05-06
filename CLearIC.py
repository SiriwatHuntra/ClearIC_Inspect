import numpy as np
import os, sys
import cv2 as cv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFrame, QLabel,
    QPushButton, QDoubleSpinBox, QSpinBox, QHBoxLayout, QVBoxLayout,
    QDialog, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QFont


# ─── Exceptions ───────────────────────────────────────────────────────────────

class InspectionError(Exception):
    pass


class MarkMissingError(InspectionError):
    def __init__(self, ic_position: str, missing_cells: list):
        self.ic_position = ic_position      # "A" or "B"
        self.missing_cells = missing_cells  # [[row, col], ...]
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


# ─── Stylesheet ───────────────────────────────────────────────────────────────

STYLE = """
    QMainWindow, QWidget {
        background-color: #263238;
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
    }
    QFrame {
        border-radius: 8px;
    }
    QPushButton {
        background-color: #FFFFFF;
        color: #263238;
        border-radius: 8px;
        padding: 6px 12px;
        border: 1px solid #00BCD4;
    }
    QPushButton:hover {
        background-color: #00BCD4;
        color: #FFFFFF;
        border: 1px solid #00BCD4;
    }
    QDoubleSpinBox, QSpinBox {
        background-color: #37474F;
        color: #FFFFFF;
        border: 1px solid #455A64;
        border-radius: 4px;
        padding: 2px 4px;
    }
    QDoubleSpinBox:focus, QSpinBox:focus {
        border: 1px solid #00BCD4;
    }
    QLabel {
        color: #ECEFF1;
    }
"""


# ─── Fail Dialog ──────────────────────────────────────────────────────────────

class FailDialog(QDialog):
    """Modal popup shown when any IC fails inspection."""

    def __init__(self, ic_a_missing: list, ic_b_missing: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inspection Failed")
        self.setModal(True)
        self.setStyleSheet("background-color: #37474F; color: #FFFFFF; border-radius: 8px; border: 1px solid #00BCD4;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        title = QLabel("Inspection Failed")
        title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        layout.addWidget(title)

        if ic_a_missing:
            layout.addWidget(QLabel(f"IC_A FAIL — missing cells: {ic_a_missing}"))
        if ic_b_missing:
            layout.addWidget(QLabel(f"IC_B FAIL — missing cells: {ic_b_missing}"))

        ack_btn = QPushButton("Acknowledge")
        ack_btn.setStyleSheet(
            "background-color: #37474F; color: #00BCD4; border-radius: 8px; padding: 6px 24px; border: 1px solid #00BCD4;"
        )
        ack_btn.clicked.connect(self.accept)
        layout.addWidget(ack_btn, alignment=Qt.AlignCenter)

    def show_fail(self, ic_a_missing: list, ic_b_missing: list):
        """Re-populate and re-open the dialog for a new failure."""
        self.__init__(ic_a_missing, ic_b_missing, self.parent())
        self.exec_()


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ClearIC Inspect")
        self.setMinimumSize(1100, 680)
        self.setStyleSheet(STYLE)

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        root.addWidget(self._build_main_view(), stretch=3)
        root.addWidget(self._build_right_panel(), stretch=1)

    # ── Main view (left) ──────────────────────────────────────────────────────

    def _build_main_view(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #37474F; border-radius: 8px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Camera / image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet("background-color: #263238; border-radius: 8px; border: 1px solid #455A64;")
        self.image_label.setText("No image")
        layout.addWidget(self.image_label, stretch=1)

        # Error banner (hidden by default)
        self.error_banner = QLabel()
        self.error_banner.setAlignment(Qt.AlignCenter)
        self.error_banner.setStyleSheet(
            "background-color: #EF5350; color: #FFFFFF; border-radius: 8px; padding: 8px;"
        )
        self.error_banner.hide()
        layout.addWidget(self.error_banner)

        # Bottom row — IC badges (left) + stats (middle)
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(8)

        badges_frame = QFrame()
        badges_frame.setStyleSheet("background-color: #455A64; border-radius: 8px;")
        badges_layout = QHBoxLayout(badges_frame)
        badges_layout.setContentsMargins(8, 8, 8, 8)
        badges_layout.setSpacing(8)
        self.badge_a = self._make_badge("IC_A")
        self.badge_b = self._make_badge("IC_B")
        badges_layout.addWidget(self.badge_a)
        badges_layout.addWidget(self.badge_b)

        bottom_row.addWidget(badges_frame)
        bottom_row.addWidget(self._build_stats_section())
        bottom_row.addStretch()
        layout.addLayout(bottom_row)

        return frame

    def _make_badge(self, label: str) -> QLabel:
        badge = QLabel(f"{label}\n—")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(110, 56)
        badge.setFont(QFont("Segoe UI", 11, QFont.Bold))
        badge.setStyleSheet("background-color: #37474F; border-radius: 8px; padding: 4px; border: 1px solid #455A64;")
        return badge

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #37474F; border-radius: 8px; border: 1px solid #00BCD4;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        layout.addWidget(self._build_setup_section())
        layout.addWidget(self._build_controls_section())
        layout.addStretch()

        return frame

    def _build_setup_section(self) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background-color: #455A64; border-radius: 8px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._section_title("Setup"))

        self.exposure_spin = self._labeled_spinbox(layout, "Exposure (µs)", 1, 1_000_000, 10000)
        self.scale_spin    = self._labeled_double_spinbox(layout, "Scale / ratio", 0.1, 10.0, 1.0)
        self.col_offset_spin = self._labeled_spinbox(layout, "Column offset (px)", 0, 2000, 0)
        self.ic_b_offset_x  = self._labeled_spinbox(layout, "IC_B offset X (px)", -4000, 4000, 0)
        self.ic_b_offset_y  = self._labeled_spinbox(layout, "IC_B offset Y (px)", -4000, 4000, 0)

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
        frame.setStyleSheet("background-color: #455A64; border-radius: 8px;")
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
        frame.setStyleSheet("background-color: #455A64; border-radius: 8px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self.lbl_status     = QLabel("Status: Idle")
        self.lbl_status.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.lbl_status.setStyleSheet("color: #00BCD4;")
        layout.addWidget(self.lbl_status)

        self.lbl_pass       = QLabel("Pass:       0")
        self.lbl_fail       = QLabel("Fail:       0")
        self.lbl_start_time = QLabel("Start at:  —")
        self.lbl_cycle_time = QLabel("Last cycle: — ms")

        for lbl in (self.lbl_pass, self.lbl_fail, self.lbl_start_time, self.lbl_cycle_time):
            lbl.setStyleSheet("color: #ECEFF1;")
            layout.addWidget(lbl)

        return frame

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _section_title(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl.setStyleSheet("color: #00BCD4;")
        return lbl

    def _labeled_spinbox(self, layout: QVBoxLayout, label: str,
                         min_val: int, max_val: int, default: int) -> QSpinBox:
        layout.addWidget(QLabel(label))
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        layout.addWidget(spin)
        return spin

    def _labeled_double_spinbox(self, layout: QVBoxLayout, label: str,
                                min_val: float, max_val: float, default: float) -> QDoubleSpinBox:
        layout.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setSingleStep(0.1)
        layout.addWidget(spin)
        return spin

    # ── Public update API (called by inspection logic) ────────────────────────

    def update_image(self, frame: np.ndarray):
        """Display a BGR numpy frame in the image label."""
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.image_label.setPixmap(
            QPixmap.fromImage(img).scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

    def update_result(self, ic: str, passed: bool):
        """Update IC_A or IC_B badge. ic = 'A' or 'B'."""
        badge = self.badge_a if ic == "A" else self.badge_b
        text  = f"IC_{ic}\n{'PASS' if passed else 'FAIL'}"
        border = "#00BCD4" if passed else "#EF5350"
        color  = "#00BCD4" if passed else "#EF5350"
        badge.setText(text)
        badge.setStyleSheet(
            f"background-color: #37474F; color: {color}; border: 1px solid {border};"
            "border-radius: 8px; padding: 4px;"
        )

    def show_error(self, message: str):
        """Show red error banner with message."""
        self.error_banner.setText(message)
        self.error_banner.show()

    def clear_error(self):
        self.error_banner.hide()

    def update_status(self, state: str):
        """state: 'Idle', 'Run', or 'Error'"""
        color = {"Idle": "#00BCD4", "Run": "#FFFFFF", "Error": "#EF5350"}.get(state, "#00BCD4")
        self.lbl_status.setText(f"Status: {state}")
        self.lbl_status.setStyleSheet(f"color: {color};")

    def update_stats(self, passed: int, failed: int, start_time: str, cycle_ms: float):
        self.lbl_pass.setText(f"Pass:       {passed}")
        self.lbl_fail.setText(f"Fail:       {failed}")
        self.lbl_start_time.setText(f"Start at:  {start_time}")
        self.lbl_cycle_time.setText(f"Last cycle: {cycle_ms:.0f} ms")

    def show_fail_dialog(self, ic_a_missing: list, ic_b_missing: list):
        FailDialog(ic_a_missing, ic_b_missing, self).exec_()

    # ── Slots (wired to buttons) ──────────────────────────────────────────────

    def on_set_anchor(self):
        pass

    def on_preview_roi(self):
        pass

    def on_save_template(self):
        pass

    def on_manual_trigger(self):
        pass


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

