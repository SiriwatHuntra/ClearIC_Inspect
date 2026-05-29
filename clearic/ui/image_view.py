import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore


class ImageView(QtWidgets.QLabel):
    """
    Zoomable image display with overlay support, stamp mode, and rubber-band drawing.
    """
    rect_drawn    = QtCore.pyqtSignal(QtCore.QRect)
    right_clicked = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setObjectName("image_area")
        self._orig        = None
        self._scale       = 1.0
        self._offset      = QtCore.QPoint(0, 0)
        self._overlays    = []
        self._rb_mode     = False
        self._rb_start    = None
        self._rb_cur      = None
        self.setMouseTracking(True)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)

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
        if lw < 2 or lh < 2:
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

    def add_overlay(self, rect: QtCore.QRect, color: QtGui.QColor, label: str = ""):
        self._overlays.append((rect, color, label))
        self.update()

    def clear_overlays(self):
        self._overlays.clear()
        self.update()

    def set_rubberband_mode(self, on: bool):
        self._rb_mode  = on
        self._rb_start = None
        self._rb_cur   = None
        self.setCursor(QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)
        self.update()

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
