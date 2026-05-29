import os
import json

import cv2
import numpy as np
from PyQt5 import QtCore

from ..utils.exceptions import TemplateError


_TEMPLATE_FILE    = "templates/template.json"
_TEMPLATE_FULL    = "templates/tmpl_full.npy"
_TEMPLATE_PREVIEW = "templates/template_preview.png"


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
    sw = max(1, int(w * cell_shrink))
    sh = max(1, int(h * cell_shrink))
    sx = x + (w - sw) // 2
    sy = y + (h - sh) // 2

    usable_y0 = sy + int(sh * grid_margin_top / 100.0)
    usable_y1 = sy + sh - int(sh * grid_margin_bot / 100.0)
    usable_h  = max(1, usable_y1 - usable_y0)
    col_gap   = int(sw * col_gap_pct / 100.0)
    cw        = max(1, (sw - col_gap) // 2)
    ch        = max(1, usable_h // 3)
    col_starts = [sx, sx + cw + col_gap]

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


def _adaptive_binary(image_bgr: np.ndarray) -> np.ndarray:
    """BGR → dense adaptive-threshold binary. Used for setup-time IC auto-detection."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 21, 5)


def _contour_template(image_bgr: np.ndarray) -> np.ndarray:
    """BGR → binary edge map for template matching.
    Pipeline: Gaussian blur → Otsu-driven Canny → dilate.
    """
    gray    = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    otsu_thr, _ = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges  = cv2.Canny(blurred, otsu_thr * 0.5, otsu_thr)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(edges, kernel, iterations=1)


def _find_second_ic(image_bgr: np.ndarray,
                    ref_rect: QtCore.QRect,
                    conf_thr: float = 0.4) -> tuple:
    """
    Search the opposite image half for a second IC using the ref_rect crop as a
    template. Uses dense adaptive binary for reliable setup-time matching.

    Returns (QRect, score). QRect is None if score < conf_thr.
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
    if (x + w // 2) < mid:
        search   = binary[:, mid:]
        x_offset = mid
    else:
        search   = binary[:, :mid]
        x_offset = 0

    if search.shape[1] < template.shape[1] or search.shape[0] < template.shape[0]:
        return None, 0.0

    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(result)

    if score >= conf_thr:
        return QtCore.QRect(loc[0] + x_offset, loc[1], w, h), float(score)
    return None, float(score)


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
             match_threshold: float = 0.6, strip_h: int = 0,
             img_w: int = 0, img_h: int = 0):
        os.makedirs("templates", exist_ok=True)
        data = {
            "ic_a": {"x": ic_a.x(), "y": ic_a.y(),
                     "w": ic_a.width(), "h": ic_a.height()},
            "ic_b": {"x": ic_b.x(), "y": ic_b.y(),
                     "w": ic_b.width(), "h": ic_b.height()},
            "exposure_us":     exposure_us,
            "match_threshold": match_threshold,
            "strip_h":         strip_h,
            "img_w":           img_w,
            "img_h":           img_h,
        }
        _tmp = _TEMPLATE_FILE + ".tmp"
        with open(_tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(_tmp, _TEMPLATE_FILE)

    @staticmethod
    def extract_patches(image_bgr: np.ndarray, ic_rect: QtCore.QRect) -> tuple:
        """
        Extract the pin-area patch ONLY (below the IC body):
          patch spans [X1, Y2] → [X2, Y3]
          where Y2 = ic bottom, Y3 = Y2 + pin_height (50% of IC height)

        Returns (patch, strip_h) where strip_h = y - y_start = -(IC height).
        """
        x, y = ic_rect.x(), ic_rect.y()
        w, h = ic_rect.width(), ic_rect.height()
        h1 = max(1, int(h * 0.5))

        img_h, img_w = image_bgr.shape[:2]
        y_start = y + h
        y_end   = min(img_h, y + h + h1)
        x_end   = min(x + w, img_w)

        full_bin = _contour_template(image_bgr)[y_start:y_end, x:x_end]
        strip_h  = y - y_start

        return full_bin, strip_h

    @staticmethod
    def save_patches(full_patch: np.ndarray):
        """Save combined patch as tmpl_full.npy."""
        os.makedirs("templates", exist_ok=True)
        _tmp = _TEMPLATE_FULL.replace(".npy", "_tmp.npy")
        np.save(_tmp, full_patch)
        os.replace(_tmp, _TEMPLATE_FULL)

    @staticmethod
    def load_patches():
        """Load template patch (tmpl_full.npy). Returns ndarray or None if absent/corrupt."""
        if not os.path.exists(_TEMPLATE_FULL):
            return None
        try:
            return np.load(_TEMPLATE_FULL)
        except Exception as e:
            print(f"[TemplateManager] Patch load failed: {e}")
            return None

    @staticmethod
    def save_preview(image_bgr: np.ndarray,
                     ic_a: QtCore.QRect, ic_b: QtCore.QRect):
        """
        Save an annotated preview image:
        - IC_A (yellow) and IC_B (cyan) boxes with 3×2 cell grids and labels
        - Magenta outline of the actual template patch region saved for IC_A
        - Teal overlay of the _contour_template edges within that patch region
        Saved to templates/template_preview.png for visual verification.
        """
        os.makedirs("templates", exist_ok=True)
        img_h, img_w = image_bgr.shape[:2]
        preview = image_bgr.copy()

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

        ax, ay = ic_a.x(), ic_a.y()
        aw, ah = ic_a.width(), ic_a.height()
        h1       = max(1, int(ah * 0.5))
        patch_y1 = ay + ah
        patch_y2 = min(img_h, ay + ah + h1)
        patch_x2 = min(ax + aw, img_w)

        cv2.rectangle(preview,
                      (ax, patch_y1), (patch_x2, patch_y2),
                      (255, 0, 255), 2)
        cv2.putText(preview, "Pin patch",
                    (ax + 2, patch_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 255), 1)

        contour_full = _contour_template(image_bgr)
        patch_edges  = contour_full[patch_y1:patch_y2, ax:patch_x2]
        edge_mask    = patch_edges > 0
        roi          = preview[patch_y1:patch_y2, ax:patch_x2].astype(np.float32)
        teal         = np.array([180, 200, 0], dtype=np.float32)
        roi[edge_mask] = roi[edge_mask] * 0.3 + teal * 0.7
        preview[patch_y1:patch_y2, ax:patch_x2] = roi.clip(0, 255).astype(np.uint8)

        _tmp = _TEMPLATE_PREVIEW + ".tmp.png"
        cv2.imwrite(_tmp, preview)
        os.replace(_tmp, _TEMPLATE_PREVIEW)

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


class TemplateMatcher:
    """
    Locates IC_A in a new image using a single adaptive-binary combined patch
    (IC body + bottom strip) matched with cv2.TM_CCOEFF_NORMED.
    """

    def __init__(self, full_patch: np.ndarray,
                 threshold: float = 0.6,
                 strip_h: int = 0,
                 ic_x: int = 0, ic_y: int = 0,
                 ic_w: int = 0, ic_h: int = 0,
                 search_margin: int = 60,
                 template_w: int = 0):
        self._patch       = full_patch
        self._threshold   = threshold
        self._strip_h     = strip_h
        self._patch_w     = full_patch.shape[1]
        self._ic_x        = ic_x
        self._ic_y        = ic_y
        self._ic_w        = ic_w
        self._ic_h        = ic_h
        self._margin      = search_margin
        self._template_w  = template_w

    def locate_ic(self, image_bgr: np.ndarray) -> tuple:
        """
        Returns (QRect, score).
        Matches the pin-area patch against the frame using contour preprocessing.
        """
        img_h, img_w = image_bgr.shape[:2]

        if self._template_w > 0 and abs(img_w / self._template_w - 1.0) > 0.01:
            scale = img_w / self._template_w
            ph0, pw0 = self._patch.shape[:2]
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            patch = cv2.resize(self._patch,
                               (max(1, int(pw0 * scale)), max(1, int(ph0 * scale))),
                               interpolation=interp)
            ic_x = int(self._ic_x * scale)
            ic_y_tmpl = int(self._ic_y * scale)
            ic_w = max(1, int(self._ic_w * scale))
            ic_h = max(1, int(self._ic_h * scale))
            strip_h = int(self._strip_h * scale)
            m = int(self._margin * scale)
        else:
            patch = self._patch
            ic_x, ic_y_tmpl = self._ic_x, self._ic_y
            ic_w, ic_h = self._ic_w, self._ic_h
            strip_h = self._strip_h
            m = self._margin

        ph, pw = patch.shape[:2]
        exp_y = ic_y_tmpl - strip_h

        rx1 = max(0, ic_x - m)
        ry1 = max(0, exp_y - m)
        rx2 = min(img_w, ic_x + pw + m)
        ry2 = min(img_h, exp_y + ph + m)

        roi_bgr = image_bgr[ry1:ry2, rx1:rx2]

        if roi_bgr.size == 0 or roi_bgr.shape[0] < ph or roi_bgr.shape[1] < pw:
            full = _contour_template(image_bgr)
            res = cv2.matchTemplate(full, patch, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            found_ic_y = loc[1] + strip_h
            return QtCore.QRect(loc[0], found_ic_y, ic_w, ic_h), float(score)

        filtered = _contour_template(roi_bgr)
        res = cv2.matchTemplate(filtered, patch, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        abs_x = loc[0] + rx1
        abs_y = loc[1] + ry1
        found_ic_y = abs_y + strip_h

        if score < self._threshold:
            print(f"[TemplateMatcher] Low match score {score:.3f} < {self._threshold:.3f} — "
                  "using best-match position anyway")

        return QtCore.QRect(abs_x, found_ic_y, ic_w, ic_h), float(score)
