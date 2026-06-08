import os
import threading

import cv2
import numpy as np
from PyQt5 import QtCore

from ..utils.exceptions import MarkMissingError, TemplateError, LowMatchError
from .detector import Detector
from .template import TemplateMatcher, _build_cells, _safe_crop


# Dataset collection run counter
_data_run_counter = 0
_dataset_lock     = threading.Lock()

# VISUAL CONSTANTS  (fixed — not configurable at runtime)
_ann_color_ok = "#00C800"   # hex — Text  / PASS cell border
_ann_color_ng = "#DD0000"   # hex — NoText / FAIL cell border
_tmpl_color_a = "#FFD700"   # hex — IC_A overlay in setup view
_tmpl_color_b = "#00E5FF"   # hex — IC_B overlay in setup view


def _hex_to_bgr(h: str) -> tuple:
    """Convert '#RRGGBB' hex string to OpenCV BGR 3-tuple."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def _save_cell_crops(image_bgr: np.ndarray, cells: list,
                     cell_hits: list, ic_label: str, run_num: int,
                     data_dir: str = "Dataset", data_split: str = "train"):
    """
    Save each ROI cell crop to Dataset/<split>/Text/ or .../NoText/.
    Called only when COLLECT_DATASET = True.
    Filename: {run_num:06d}_IC{label}_{idx:02d}.png
    """
    for idx, (cx, cy, cw, ch) in enumerate(cells):
        class_name = "Text" if cell_hits[idx] else "NoText"
        folder = os.path.join(data_dir, data_split, class_name)
        os.makedirs(folder, exist_ok=True)
        crop = _safe_crop(image_bgr, cx, cy, cw, ch)
        if crop.size > 0:
            fname = f"{run_num:06d}_IC{ic_label}_{idx:02d}.png"
            cv2.imwrite(os.path.join(folder, fname), crop)


# INSPECTOR
class Inspector:
    """
    Crops each ROI cell from the image and classifies it as Text / NoText.
    Raises MarkMissingError if either IC has any cell classified as NoText.
    """

    def __init__(self, detector: Detector, template: dict,
                 template_matcher: "TemplateMatcher | None" = None,
                 cell_shrink: float = 0.95, cell_expand: float = 1.2,
                 col_gap_pct: float = 40.0,
                 grid_margin_top: float = 0.0, grid_margin_bot: float = 15.0,
                 collect_dataset: bool = False,
                 data_dir: str = "Dataset", data_split: str = "train",
                 ann_border_px: int = 1, ann_show_labels: bool = True):
        self._detector         = detector
        self._template         = template
        self._template_matcher = template_matcher
        self._ic_b_dx_tmpl = template["ic_b"]["x"] - template["ic_a"]["x"]
        self._ic_b_dy_tmpl = template["ic_b"]["y"] - template["ic_a"]["y"]
        self._template_w   = int(template.get("img_w", 0))
        self._template_h   = int(template.get("img_h", 0))
        self._cell_shrink     = cell_shrink
        self._cell_expand     = cell_expand
        self._col_gap_pct     = col_gap_pct
        self._grid_margin_top = grid_margin_top
        self._grid_margin_bot = grid_margin_bot
        self._collect_dataset = collect_dataset
        self._data_dir        = data_dir
        self._data_split      = data_split
        self._ann_border_px   = ann_border_px
        self._ann_show_labels = ann_show_labels

    def inspect(self, image_bgr: np.ndarray,
                debug: bool = False) -> tuple:
        """
        Returns (ic_a_pass, ic_b_pass, missing_a, missing_b, annotated_bgr).
        Raises MarkMissingError if either IC fails.
        Raises TemplateError if template matching rejects the frame.

        Phase 1 — locate IC_A via TemplateMatcher (preferred) or fixed template coords.
        Phase 2 — crop each ROI cell and classify as Text / NoText.
        """
        img_h, img_w = image_bgr.shape[:2]

        # Scale factor: template coords → current image space.
        # Legacy templates (no img_w/img_h) use 1:1 — no change in behaviour.
        if self._template_w > 0 and self._template_h > 0:
            sx = img_w / self._template_w
            sy = img_h / self._template_h
        else:
            sx, sy = 1.0, 1.0

        ic_b_dx = int(self._ic_b_dx_tmpl * sx)
        ic_b_dy = int(self._ic_b_dy_tmpl * sy)

        def _scale_r(r):
            return {"x": int(r["x"] * sx), "y": int(r["y"] * sy),
                    "w": max(1, int(r["w"] * sx)), "h": max(1, int(r["h"] * sy))}

        ic_a_s = _scale_r(self._template["ic_a"])
        ic_b_s = _scale_r(self._template["ic_b"])

        # Guard: verify image is large enough to cover both scaled IC regions
        min_w = max(ic_a_s["x"] + ic_a_s["w"], ic_b_s["x"] + ic_b_s["w"])
        min_h = max(ic_a_s["y"] + ic_a_s["h"], ic_b_s["y"] + ic_b_s["h"])
        if img_w < min_w or img_h < min_h:
            raise TemplateError(
                f"Image {img_w}×{img_h} too small — template requires at least {min_w}×{min_h}")

        annotated = image_bgr  # draw in-place; caller saves raw before calling inspect()

        # Phase 1: locate ICs
        if self._template_matcher is not None:
            rt_a, score = self._template_matcher.locate_ic(image_bgr)
            if score < self._template_matcher._threshold:
                raise LowMatchError(
                    f"Template match {score:.3f} < {self._template_matcher._threshold:.3f} — frame skipped")
            rt_b = QtCore.QRect(
                rt_a.x() + ic_b_dx, rt_a.y() + ic_b_dy,
                ic_b_s["w"], ic_b_s["h"],
            )
            ic_a_cells = self._rect_to_cells(rt_a)
            ic_b_cells = self._rect_to_cells(rt_b)
            if debug:
                print(f"[Inspector] scale=({sx:.3f},{sy:.3f}) "
                      f"TemplateMatcher score={score:.3f}")
                print(f"[Inspector] IC_A matched: "
                      f"x={rt_a.x()} y={rt_a.y()} w={rt_a.width()} h={rt_a.height()}")
                print(f"[Inspector] IC_B by offset: "
                      f"x={rt_b.x()} y={rt_b.y()} w={rt_b.width()} h={rt_b.height()}")
        else:
            # Fixed scaled coords — no runtime IC localization
            ic_a_cells = _build_cells(
                ic_a_s["x"], ic_a_s["y"], ic_a_s["w"], ic_a_s["h"],
                self._cell_shrink, self._cell_expand,
                self._col_gap_pct, self._grid_margin_top, self._grid_margin_bot)
            ic_b_cells = _build_cells(
                ic_b_s["x"], ic_b_s["y"], ic_b_s["w"], ic_b_s["h"],
                self._cell_shrink, self._cell_expand,
                self._col_gap_pct, self._grid_margin_top, self._grid_margin_bot)
            if debug:
                print(f"[Inspector] scale=({sx:.3f},{sy:.3f}) "
                      "No TemplateMatcher — using scaled fixed template coordinates")

        # Phase 2: crop each cell and classify as Text / NoText
        missing_a, hits_a, confs_a = self._check_ic(image_bgr, ic_a_cells, annotated, debug)
        missing_b, hits_b, confs_b = self._check_ic(image_bgr, ic_b_cells, annotated, debug)

        if self._collect_dataset:
            global _data_run_counter
            with _dataset_lock:
                _data_run_counter += 1
                run_num = _data_run_counter
            _save_cell_crops(image_bgr, ic_a_cells, hits_a, "A", run_num,
                             self._data_dir, self._data_split)
            _save_cell_crops(image_bgr, ic_b_cells, hits_b, "B", run_num,
                             self._data_dir, self._data_split)

        if missing_a or missing_b:
            raise MarkMissingError(missing_a, missing_b, annotated, confs_a, confs_b)

        return True, True, [], [], annotated

    def _rect_to_cells(self, rect: QtCore.QRect) -> list:
        """Divide a QRect into a shrunk+expanded 3-row × 2-col cell grid."""
        return _build_cells(rect.x(), rect.y(), rect.width(), rect.height(),
                            self._cell_shrink, self._cell_expand,
                            self._col_gap_pct,
                            self._grid_margin_top, self._grid_margin_bot)

    def _check_ic(self, image_bgr: np.ndarray, cells: list,
                  annotated: np.ndarray, debug: bool) -> tuple:
        """
        Crop each ROI cell from image_bgr and classify as Text / NoText.
        Returns (missing, hits_flags, text_confs).
        text_confs: per-cell Text-class confidence (6 floats, 0–1).
        """
        ih, iw = image_bgr.shape[:2]
        color_ok    = _hex_to_bgr(_ann_color_ok)
        color_ng    = _hex_to_bgr(_ann_color_ng)
        missing     = []
        hits_flags  = []
        text_confs  = []
        for idx, (cx, cy, cw, ch) in enumerate(cells):
            row = idx // 2 + 1
            col = idx %  2 + 1
            crop = _safe_crop(image_bgr, cx, cy, cw, ch)
            cls_idx, conf = self._detector.classify_crop(crop) if crop.size > 0 else (0, 0.0)
            present   = (cls_idx == 1)   # 1 = Text (mark present)
            text_conf = conf if cls_idx == 1 else (1.0 - conf)  # Text-class probability
            hits_flags.append(present)
            text_confs.append(text_conf)
            if debug:
                lbl = "Text" if present else "NoText"
                std_str = f"{crop.std():.1f}" if crop.size > 0 else "n/a"
                print(f"[Cell R{row}C{col}] "
                      f"{'PRESENT' if present else 'ABSENT '} "
                      f"cls={lbl} conf={conf:.3f} text_conf={text_conf:.3f}  "
                      f"raw_std={std_str}")
            color = color_ok if present else color_ng
            cv2.rectangle(annotated,
                          (max(0, cx), max(0, cy)),
                          (min(iw, cx + cw), min(ih, cy + ch)),
                          color, self._ann_border_px)
            if self._ann_show_labels:
                label = f"R{row}C{col}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                tx = max(0, cx) + (cw - tw) // 2
                ty = max(0, cy) + (ch + th) // 2
                cv2.putText(annotated, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            if not present:
                missing.append([row, col])
        return missing, hits_flags, text_confs
