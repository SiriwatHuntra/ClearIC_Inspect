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

## Config: `Config.toml` *(all runtime settings)*

All configuration lives in `Config.toml` — no hardcoded dev flags. `ConfigLoader` merges it against `DEFAULT_CONFIG` on startup.

**Note:** `USE_CAMERA` is the Config.toml key. `ConfigLoader` converts it internally to `CAMERA = "camera"/"directory"` at load time — do not set `CAMERA` directly in Config.toml.

### Camera & capture

| Key | Default | Effect |
|---|---|---|
| `USE_CAMERA` | `false` | `true` = live Basler camera · `false` = load from `Input/` |
| `CAMERA_SERIAL` | `""` | Basler serial filter (`""` = first found) |
| `EXPOSURE_US` | `8000` | Camera exposure µs |
| `IMAGE_W` | `0` | Camera resolution override width (0 = camera native) |
| `IMAGE_H` | `0` | Camera resolution override height (0 = camera native) |
| `CAMERA_WARMUP_FRAMES` | `5` | Frames to grab and discard on camera open |
| `CAMERA_RETRY_DELAY` | `0.2` | Seconds between grab retries on fail |
| `CAMERA_RETRIES` | `2` | Number of grab retries before raising CameraError |
| `RECONNECT_ATTEMPTS` | `3` | Camera reconnect attempts on disconnect |
| `RECONNECT_DELAY_S` | `5.0` | Seconds between reconnect attempts |

### Classifier & inspection

| Key | Default | Effect |
|---|---|---|
| `MODEL_PATH` | `"Text_cls-2/best_openvino_model/best.xml"` | OpenVINO classifier |
| `CONF_THR` | `0.5` | Classifier confidence threshold (legacy, superseded by TEXT_MIN_CONF) |
| `TEXT_MIN_CONF` | `0.80` | Minimum Text-class probability to call a cell PASS |
| `TEXT_NG_THRESHOLD` | `2` | Missing-cell count at or above this → NG (below → suspect-pass `_GS`) |
| `BLANK_CELL_STD_THR` | `0.0` | Pixel-std below this → force NoText without running model (0 = disabled) |
| `CLS_N_PASSES` | `1` | Inference passes per cell (averaged); deterministic model — 1 is sufficient |
| `CLS_UNCERTAIN_THR` | `0.50` | Log a warning when text_prob is in this uncertain zone (debug only) |
| `WARMUP_FRAMES` | `5` | Classifier warmup passes on startup |
| `RETRY_DELAY_MS` | `10` | Delay (ms) before retry grab on MarkMissingError |

### Grid geometry

| Key | Default | Effect |
|---|---|---|
| `CELL_SHRINK` | `0.95` | Shrink IC rect before slicing (centred) |
| `CELL_EXPAND` | `1.2` | Per-cell expansion after slicing (centred overlap) |
| `COL_GAP_PCT` | `40.0` | Gap between L and R column, % of IC width |
| `GRID_MARGIN_TOP` | `0.0` | Top dead-band before row 1, % of IC height |
| `GRID_MARGIN_BOT` | `15.0` | Bottom dead-band after row 3, % of IC height |

### I/O & GPIO

| Key | Default | Effect |
|---|---|---|
| `IO` | `false` | `true` = drive real BCM GPIO; `false` = mock/log only (manual trigger per shot) |
| `CELLCON_PORT` | `"/dev/ttyUSB0"` | Serial port for Cell-con lot tracker |

GPIO pin keys: `GPIO_START_PIN` (17), `GPIO_BUSY_PIN` (23), `GPIO_END_PIN` (18), `GPIO_INSPEC_STAGE_PIN` (24).

### Output & logging

| Key | Default | Effect |
|---|---|---|
| `OUT_DIR` | `"Output/"` | Root output directory |
| `DIR_INPUT` | `"Input/"` | Source images for `USE_CAMERA=false` |
| `LOG_DIR` | `"logs"` | Log file directory |
| `LOG_RETENTION` | `365` | Max log files kept per pattern |
| `ANN_BORDER_PX` | `1` | Cell annotation border thickness |
| `RESULT_OVERLAY` | `true` | Show R1C1 labels on cell overlays |
| `COLLECT_DATASET` | `false` | Save cell crops to `Dataset/` for retraining |
| `DATA_DIR` | `"Dataset"` | Root directory for collected crops |
| `DATA_SPLIT` | `"train"` | Subfolder under DATA_DIR (`train` or `val`) |
| `DISK_WARN_MB` | `200` | Warn in UI when free disk space drops below this |

### Other

| Key | Default | Effect |
|---|---|---|
| `DEBUG` | `true` | Verbose console logs + annotated image saved every cycle |
| `MODE` | `"DEBUG"` | String written into log records |

---

## Error Flags *(runtime state)*

| Enum | Values |
|---|---|
| `ErrorFlag` | `NONE · CAMERA · MODEL · GPIO · TEMPLATE` |

---

## Inspection Flow

```
LotStartDialog → operator enters lot number (or fetched from CellCon via serial)
  → RunWorker starts
  → Camera.grab()                            → image_bgr (ndarray)
  → save raw to Output/YYYYMMDD/lot/RealImg/ (tmp name, before result known)
  → TemplateMatcher.locate_ic(image_bgr)     → (QRect rt_a, score)
    (if no patch file → fixed template coords used instead)
  → rt_b = rt_a offset by (ic_b_dx, ic_b_dy) from template
  → Inspector._check_ic(image_bgr, cells)    → (missing[], hits[], confs[])  ×2
  → on MarkMissingError: one retry grab with confidence-weighted resolution
  → classify result by n_missing (total missing cells across both ICs):
      n_missing == 0                      → _G   (PASS, not saved by default)
      0 < n_missing < TEXT_NG_THRESHOLD   → _GS  (suspect PASS — saved)
      n_missing >= TEXT_NG_THRESHOLD      → _NGS (suspect NG — saved)
      n_missing >= total cells (12)       → _NG  (full NG — saved)
  → rename raw to {img_id}_{suffix}.jpg
  → save annotated to Output/YYYYMMDD/lot/Image/ (all non-_G results)
  → GPIO: set_busy(False) → set_inspec_stage (LOW=PASS, HIGH=NG)
  → sleep 10 ms → pulse END_PIN LOW 40 ms → set_inspec_stage(HIGH, idle)
  → loop to next START
```

Low match score from TemplateMatcher: prints a warning but uses the best-match position regardless (no hard rejection). `TemplateError` is only raised by the Inspector if something else fails.

---

## Key Functions

### `_build_cells(x, y, w, h, ...) → list[(cx,cy,cw,ch)]`
Converts one IC rect into 6 ROI cells (3 rows × 2 cols). Steps: shrink (centred) → apply top/bot margins → slice 3×2 grid with col gap → expand each cell. Row-major order R1C1→R3C2.

### `Inspector._check_ic(image_bgr, cells, annotated, debug) → (missing, hits_flags, text_confs)`
Crop each raw cell from `image_bgr` → `Detector.classify_crop()`. Draws borders + labels onto `annotated` in place. Returns `missing`: `[[row,col],…]` for NoText cells; `text_confs`: per-cell Text-class probability (6 floats).

### `Inspector.inspect(image_bgr, debug) → (pass_a, pass_b, missing_a, missing_b, annotated_bgr)`
Locate via TemplateMatcher (or fixed coords) → cells → `_check_ic` × 2. `annotated_bgr` IS `image_bgr` (in-place, no copy). Raises `MarkMissingError` on missing cells.

### `TemplateMatcher.locate_ic(image_bgr) → (QRect, score)`
Applies Canny-contour preprocessing then `cv2.matchTemplate` on the saved pin-area patch (50% of IC height below the IC body). Searches within `±search_margin` of expected position. Always returns best-match position; logs a warning if score < threshold.

### `_resolve_ic(missing_first, confs_first, confs_second) → still_missing`
Confidence-weighted retry: `w = 0.7 * conf_second + 0.3 * conf_first`. Cell is PASS only if `w >= 0.90`. Applied only to cells that were missing on the first attempt.

### `RunWorker.run()`
QThread loop: wait START → grab → save raw → inspect (+ one retry) → rename/save annotated → GPIO (INSPEC_STAGE + END_PIN pulse).
Signals: `sig_image`, `sig_result`, `sig_fail`, `sig_error`, `sig_status`, `sig_cycle_ms`, `sig_done`, `sig_session_reset`, `sig_paused`, `sig_resumed`.

---

## Setup Flow *(one-time per product)*

1. Click **New Template** → grabs a fresh frame from the camera (or directory).
2. Draw a rubber-band rect around **either** IC on the image.
3. System auto-detects the second IC (`_find_second_ic` — adaptive-binary template match on the opposite image half).
4. Both IC_A (yellow) and IC_B (cyan) shown as overlays.
5. Click **Confirm** → saves `template.json`, `tmpl_full.npy` (pin-area patch), and `template_preview.png`.

If IC_B is not found automatically, the UI prompts to draw again (`draw_a_retry` state).

---

## GPIO Pins *(BCM, `IO=True` only)*

| Signal | Pin | Dir |
|---|---|---|
| `START_PIN` | 17 | IN — active HIGH 10 ms pulse (machine signals ready for one shot) |
| `BUSY_PIN` | 23 | OUT — HIGH during full inspection + retry |
| `END_PIN` | 18 | OUT — normally HIGH; pulses LOW 40 ms after inspection done |
| `INSPEC_STAGE_PIN` | 24 | OUT — normally HIGH; LOW = both ICs pass, HIGH = any fail |

---

## Logging *(dual-CSV)*

| File | Contents |
|---|---|
| `logs/op_YYYYMMDD.csv` | One row per event: `timestamp, event, lot_number, detail, cycle_ms` |
| `logs/result_{lot}_{ts}.csv` | Header block + one row per inspection + footer summary |

Events: `SESSION_START`, `SESSION_END`, `PASS`, `PASS_SUSPECT`, `FAIL`, `FAIL_SUSPECT`, `ERROR`, `PAUSE`, `RESUME`.

---

## UI Classes

| Class | Description |
|---|---|
| `MainWindow` | Two-tab window: Inspection + Image Browser |
| `ImageView` | Zoomable image widget; rubber-band draw mode for setup |
| `LotStartDialog` | Pre-run dialog for lot number; uses CellCon or `get_lot_number_from_api()` hook |
| `FailDialog` | Modal FAIL popup (currently unused in RunWorker — no longer auto-shown) |
| `ImageBrowserPage` | Browse `Output/` by date/lot; thumbnail grid with RealImg/Image toggle; NG-only filter |
| `ImageCard` | Single thumbnail card (color-coded by suffix: _G / _GS / _NGS / _NG) |
| `ThumbnailWorker` | Background QThread that loads thumbnails one-by-one |
| `FolderScanWorker` | Background QThread that scans the Output/ directory tree |
| `CellCon` | Serial interface to Cell-con lot tracker (`LA\r\n` → `LS,<lot>`) on `CELLCON_PORT` |

---

## Files & Output

```
ClearIC_Inspect/
├── CLearIC.py
├── Config.toml
├── Text_cls-2/best_openvino_model/   # ACTIVE cell classifier (best.xml + best.bin)
├── templates/
│   ├── template.json                 # ic_a/ic_b coords, match_threshold, strip_h
│   ├── tmpl_full.npy                 # pin-area patch for TemplateMatcher
│   └── template_preview.png          # annotated preview saved on confirm
├── Output/YYYYMMDD/lot_number/
│   ├── RealImg/{img_id}_{suffix}.jpg # raw captures
│   └── Image/{img_id}_{suffix}.jpg  # annotated captures (non-_G only)
├── logs/
│   ├── op_YYYYMMDD.csv               # daily operation log
│   └── result_{lot}_{ts}.csv         # per-lot result log
├── Input/                            # source images for USE_CAMERA=false
├── Dataset/                          # cell crops (COLLECT_DATASET=true)
└── Test/                             # trainModel.py, Converter.py, ImagePlayGround.py
```

`IMAGE_ID` format: `YYYYMMDD_HHMMSS_NNN` (thread-safe counter).

Output suffixes: `_G` (clean pass), `_GS` (suspect pass), `_NGS` (suspect NG), `_NG` (full NG). Threshold between `_GS` and `_NGS` is `TEXT_NG_THRESHOLD`. Clean-pass `_G` images are not saved by default.

---

## Skills (from `skills-lock.json`)

Installed from `JuliusBrussee/caveman` via the Claude Code skills system:

| Skill | Invoke | Purpose |
|---|---|---|
| `cavecrew` | `/cavecrew` | Multi-agent crew orchestration |
| `caveman` | `/caveman` | General caveman assistant |
| `caveman-commit` | `/caveman-commit` | Draft and create git commits |
| `caveman-compress` | `/caveman-compress` | Compress/summarize context |
| `caveman-help` | `/caveman-help` | Help and usage guide |
| `caveman-review` | `/caveman-review` | Code review |
| `caveman-stats` | `/caveman-stats` | Project statistics |

---

## Model Retraining

Train with `Test/trainModel.py` (Ultralytics YOLO-cls). Export to OpenVINO with `Test/Converter.py`, then update `MODEL_PATH` in `Config.json`.

Classifier output: `[1, 2]` — index 0 = NoText (absent), index 1 = Text (present). Training images are manually cropped raw images — no preprocessing. At inference, raw cell crops are fed directly to the model (no CLAHE). `TEXT_MIN_CONF` gates the Text class asymmetrically (below threshold → NoText, regardless of raw NoText probability).

---

## Git

**Never perform any git operation** (commit, merge, push, rebase, branch, reset, stash, tag, cherry-pick, etc.) automatically or proactively. The user operates git exclusively. Only run a git command if the user gives an explicit, direct instruction for that specific command in the current turn.
