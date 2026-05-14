# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Single-file: `CLearIC.py`. All code lives there; no modules.

---

## Run

```bash
source .venv/bin/activate
python CLearIC.py
```

Requires a display (physical screen or `DISPLAY`). PyQt5 is system-package only — do not pip-install it.

**Dependencies split:**
- `apt` only: `python3-pyqt5`, `python3-rpi.gpio`, `python3-numpy`, `python3-opencv`
- `.venv` pip: `openvino`, `pypylon` (pypylon also needs Basler Pylon SDK at system level)

---

## Config: Two Separate Systems

### Hardcoded dev flags *(top of `CLearIC.py`)*

| Flag | Values | Effect |
|---|---|---|
| `DEBUG` | `True/False` | Verbose logs + save annotated every cycle |
| `IO` | `True/False` | Drive GPIO / mock (log only) |
| `MODE` | `"RUN"/"DEBUG"` | Written into log records |
| `MODEL_PATH` | path | OpenVINO `.xml` for cell classifier |
| `COLLECT_DATASET` | `True/False` | Save cell crops to `Dataset/` for retraining |

### `Config.json` *(runtime, no source edit needed)*

| Key | Effect |
|---|---|
| `CAMERA` | `"camera"` = live Basler · `"directory"` = load from `Input/` |
| `CONF_THR` | Classifier confidence threshold |
| `CAMERA_SERIAL` | Basler serial filter (`""` = first found) |
| `EXPOSURE_US` | Camera exposure µs |

---

## Stage & Error Flags *(runtime state)*

| Enum | Values |
|---|---|
| `Stage` | `STANDBY · BUSY · ERROR · SHUTDOWN` |
| `ErrorFlag` | `NONE · CAMERA · MODEL · GPIO · TEMPLATE` |

---

## Inspection Flow

```
START_PIN (or Manual Trigger)
  → Camera.grab()                          → image_bgr (ndarray)
  → save raw to Output/.../RealImg/        (before processing)
  → TemplateMatcher.locate_ic(image_bgr)   → (QRect rt_a, score)
  → rt_b = rt_a offset by template (ic_b_offset_x/y)
  → Inspector._rect_to_cells(rt_a/b)       → cells_a / cells_b  [(cx,cy,cw,ch)×6]
  → Inspector._check_ic(image_bgr, cells)  → annotates in-place → (missing[], hits[])
  → save annotated to Output/.../Image/
  → set FAIL_A_PIN / FAIL_B_PIN  →  pulse ACK_PIN
  → wait DONE_PIN  →  clear pins  →  STANDBY
```

Template-match fallback: if score < `match_threshold`, use fixed template coords.

---

## Key Functions

### `_build_cells(x, y, w, h) → list[(cx,cy,cw,ch)]`
Converts one IC rect into 6 ROI cells (3 rows × 2 cols). Applies `_CELL_SHRINK`, margins, `_COL_GAP_PCT`, `_CELL_EXPAND`. Row-major order R1C1→R3C2.

### `Inspector._check_ic(image_bgr, cells, annotated, debug) → (missing, hits_flags)`
Crop → CLAHE (`_CLAHE` module-level object) → resize 224×224 → `Detector.classify_crop()`. Draws borders + labels onto `annotated` in place. `missing`: `[[row,col],…]` for NoText cells.

### `Inspector.inspect(image_bgr, debug) → (pass_a, pass_b, missing_a, missing_b, annotated_bgr)`
Locate → cells → `_check_ic` × 2. `annotated_bgr` IS `image_bgr` (in-place, no copy).
Raises `MarkMissingError` on missing cells, `TemplateError` on alignment rejection.

### `TemplateMatcher.locate_ic(image_bgr) → (QRect, score)`
Bilateral-strip match on a horizontal band. Returns IC_A rect + score.

### `RunWorker.run()`
QThread loop: grab → save raw → inspect → save annotated → GPIO → wait DONE.
Signals: `sig_image`, `sig_result`, `sig_fail`, `sig_error`, `sig_status`, `sig_cycle_ms`, `sig_done`, `sig_paused`, `sig_resumed`.

---

## Cell Grid Constants *(top of file)*

| Constant | Default | Effect |
|---|---|---|
| `_CELL_SHRINK` | `0.90` | Shrink IC rect before slicing |
| `_GRID_MARGIN_TOP` | `10.0` | Top dead-band before row 1, % of IC height |
| `_GRID_MARGIN_BOT` | `10.0` | Bottom dead-band after row 3, % of IC height |
| `_COL_GAP_PCT` | `40.0` | Gap between L and R column, % of IC width |
| `_CELL_EXPAND` | `1.05` | Per-cell expansion after slicing |

---

## Setup & IO

### Setup Flow *(one-time per product)*
1. Capture/load reference image
2. Draw IC_A rect → confirm or auto-detect
3. Set IC_B offset (x, y from IC_A anchor)
4. Preview 12 ROI boxes → adjust → Save Template

### GPIO Pins *(BCM, `IO=True` only)*

| Signal | Pin | Dir |
|---|---|---|
| `START_PIN` | 17 | IN↓ — rising edge starts cycle |
| `DONE_PIN` | 27 | IN↓ — rising edge returns to STANDBY |
| `ACK_PIN` | 22 | OUT — pulse when result ready |
| `FAIL_A_PIN` | 24 | OUT — HIGH = IC_A fail |
| `FAIL_B_PIN` | 25 | OUT — HIGH = IC_B fail |

---

## Files & Output

```
ClearIC_Inspect/
├── CLearIC.py
├── Config.json
├── Text_cls-2/best_openvino_model/   # ACTIVE cell classifier (best.xml + best.bin)
├── IC_Search_openvino_model/         # unused — kept for reference
├── templates/                        # template.json + tmpl_top.npy + tmpl_bot.npy
├── template_auto/                    # auto-detect template output
├── Output/YYYYMMDD/RealImg/          # raw captures (every cycle)
├── Output/YYYYMMDD/Image/            # annotated captures (every cycle)
├── logs/                             # inspect_YYYYMMDD.log (daily rotation, 365-day retention)
├── Input/                            # source images for CAMERA="directory"
├── Dataset/                          # cell crops (COLLECT_DATASET=True)
└── Test/                             # trainModel.py, Converter.py, ImagePlayGround.py
```

`IMAGE_ID` format: `YYYYMMDD_HHMMSS_NNN`. Log: JSON-lines, one record per inspection.

### Model Retraining
Train with `Test/trainModel.py` (Ultralytics YOLO). Export to OpenVINO with `Test/Converter.py`, then update `MODEL_PATH` in `CLearIC.py`.
