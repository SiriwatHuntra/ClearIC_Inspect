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

## High-Level Data Flow

```
Camera / Input dir
      │
      ▼
  image_bgr (ndarray, BGR)
      │
      ├─── raw_bgr = image_bgr.copy()          ← preserve unannotated frame
      │
      ▼
  TemplateMatcher.locate_ic(image_bgr)
      │    _contour_template(full image) → crop search window → matchTemplate
      ▼
  rt_a (QRect)  +  rt_b = rt_a + (ic_b_dx, ic_b_dy)
      │
      ▼
  Inspector._check_ic(image_bgr, cells) × 2
      │    crop each cell → Detector.classify_crop() → Text / NoText
      ▼
  n_missing → result suffix (_G / _GS / _NGS / _NG)
      │
      ├─ _G  → zero disk writes
      └─ other → cv2.imwrite(raw_bgr) + cv2.imwrite(annotated)
      │
      ▼
  GPIO: INSPEC_STAGE + END_PIN pulse
```

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
| `BLOB_MIN_RATIO` | `0.0` | Remove blobs < ratio × largest blob from `_contour_template` output; `0.0` = disabled; `0.2` removes IC-corner reflections |
| `TEMPLATE_MATCH_THR` | `0.6` | Minimum match score for IC_A locate; below this logs a warning |
| `TEMPLATE_FIND_CONF_THR` | `0.4` | Minimum match score to accept IC_B during auto-detection at setup |

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
| `LOG_RETENTION` | `730` | Days to retain log files (date-based rotation; 2-year default) |
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
  → raw_bgr = img_bgr.copy()                (preserve unannotated frame; inspect() annotates in-place)
  → TemplateMatcher.locate_ic(image_bgr)     → (QRect rt_a, score)
    (if no patch file → fixed template coords used instead)
  → rt_b = rt_a offset by (ic_b_dx, ic_b_dy) from template
  → Inspector._check_ic(image_bgr, cells)    → (missing[], hits[], confs[])  ×2
  → on MarkMissingError: one retry grab with confidence-weighted resolution
  → classify result by n_missing (total missing cells across both ICs):
      n_missing == 0                      → _G   (PASS — zero disk writes)
      0 < n_missing < TEXT_NG_THRESHOLD   → _GS  (suspect PASS — saved)
      n_missing >= TEXT_NG_THRESHOLD      → _NGS (suspect NG — saved)
      n_missing >= total cells (12)       → _NG  (full NG — saved)
  → if non-_G: write raw_bgr → RealImg/{img_id}_{suffix}.jpg
               write annotated → Image/{img_id}_{suffix}.jpg
  → GPIO: set_busy(False) → set_inspec_stage (LOW=PASS, HIGH=NG)
  → sleep 10 ms → pulse END_PIN LOW 40 ms → set_inspec_stage(HIGH, idle)
  → every 500 cycles: check disk free; emit sig_warn banner if below DISK_WARN_MB
  → loop to next START
```

Low match score from TemplateMatcher: prints a warning but uses the best-match position regardless (no hard rejection). `TemplateError` is only raised by the Inspector if something else fails.

---

## Inspection-Time Locate Flow — `TemplateMatcher.locate_ic()`

```
image_bgr
  │
  ├─ _contour_template(image_bgr)  → full_filtered   ← full image, always
  │
  ├─ compute search window: ic_x ± search_margin, exp_pin_y ± search_margin
  │
  ├─ filtered = full_filtered[ry1:ry2, rx1:rx2]      ← crop AFTER preprocess
  │
  └─ matchTemplate(filtered, saved_patch, TM_CCOEFF_NORMED)
       → best loc → ic_a QRect
```

IC_B position: `ic_a_rect` + fixed `(ic_b_dx, ic_b_dy)` offset saved in `template.json`.

---

## Key Functions

### `_build_cells(x, y, w, h, ...) → list[(cx,cy,cw,ch)]`
Converts one IC rect into 6 ROI cells (3 rows × 2 cols). Steps: shrink (centred) → apply top/bot margins → slice 3×2 grid with col gap → expand each cell. Row-major order R1C1→R3C2.

```
IC rect
  → shrink by CELL_SHRINK (centred)
  → apply GRID_MARGIN_TOP / GRID_MARGIN_BOT dead-bands
  → split 3 rows × 2 cols with COL_GAP_PCT gap between columns
  → expand each cell by CELL_EXPAND (centred overlap)
  → row-major: R1C1, R1C2, R2C1, R2C2, R3C1, R3C2
```

### `Inspector._check_ic(image_bgr, cells, annotated, debug) → (missing, hits_flags, text_confs)`
Crop each raw cell from `image_bgr` → `Detector.classify_crop()`. Draws borders + labels onto `annotated` in place. Returns `missing`: `[[row,col],…]` for NoText cells; `text_confs`: per-cell Text-class probability (6 floats).

### `Inspector.inspect(image_bgr, debug) → (pass_a, pass_b, missing_a, missing_b, annotated_bgr)`
Locate via TemplateMatcher (or fixed coords) → cells → `_check_ic` × 2. `annotated_bgr` IS `image_bgr` (in-place, no copy). Raises `MarkMissingError` on missing cells.

### `_contour_template(image_bgr) → np.ndarray`
**Must receive the full image.** Returns a binary bright-region map (region-based, not edge-based).
Pipeline: `medianBlur(5)` → background-divide (`GaussianBlur σ=50`) → `Otsu` threshold → `morphOpen(9×9)` → `morphClose(5×5)`.
Background division normalises lot-to-lot brightness variation without amplifying tape texture (unlike CLAHE). Morph open removes tape-noise blobs; morph close fills holes inside pin blobs.
Called identically at template-save time, search time, and IC_B setup detection so all three see the same global histogram.

### `TemplateMatcher.locate_ic(image_bgr) → (QRect, score)`
Calls `_contour_template(image_bgr)` on the **full frame** first, then crops the `±search_margin` search window from the result. Matches the saved pin-area blob patch (50% of IC height below body) with `TM_CCOEFF_NORMED`. Always returns best-match position; logs a warning if score < threshold.

### `_resolve_ic(missing_first, confs_first, confs_second) → still_missing`
Confidence-weighted retry: `w = 0.7 * conf_second + 0.3 * conf_first`. Cell is PASS only if `w >= 0.90`. Applied only to cells that were missing on the first attempt.

### `RunWorker.run()`
QThread loop: wait START → grab → `raw_bgr = img_bgr.copy()` → inspect (+ one retry) → write files if non-`_G` → GPIO (INSPEC_STAGE + END_PIN pulse) → every 500 cycles emit `sig_warn` if disk low.
Signals: `sig_image`, `sig_result`, `sig_fail`, `sig_error`, `sig_warn`, `sig_status`, `sig_cycle_ms`, `sig_done`, `sig_session_reset`, `sig_paused`, `sig_resumed`.

---

## Template Preprocessing — `_contour_template()`

All three template operations share a single preprocessing function:

| Caller | When | Purpose |
|---|---|---|
| `TemplateManager.extract_patches()` | Setup — save template | Build pin-blob patch |
| `TemplateMatcher.locate_ic()` | Every inspection | Find IC_A position |
| `_find_second_ic()` | Setup — detect IC_B | Find IC_B position |

**Rule:** always pass the **full image**. Otsu and background-blur need the global pixel histogram. Passing a crop changes the threshold and breaks consistency between template-save time and search time.

**Pipeline:**
```
BGR image
  │
  ├─ cvtColor → gray
  ├─ medianBlur(5)              remove sensor grain; preserve pin edges
  ├─ GaussianBlur(σ=50) → bg   estimate illumination field (slow-varying)
  ├─ divide(gray, bg, ×255)    normalise: removes dark-lot vs bright-lot global shift
  ├─ Otsu threshold             binary: bright pins survive, dark tape falls away
  ├─ morphOpen(9×9)            remove small tape-noise blobs (< pin size)
  └─ morphClose(5×5)           fill holes inside pin blobs → solid regions
```

**Why not CLAHE:** amplifies local tape texture → tape pixels classified as "bright" → noise fills the template patch. Background-divide normalises global brightness without amplifying local texture contrast.

**Output:** binary image — white = bright pin/IC regions, black = dark tape/background.

---

## Classifier — `Detector`

- Model: OpenVINO IR (`best.xml` + `best.bin`)
- Input: raw cell crop (no preprocessing, no CLAHE)
- Output: `[1, 2]` — index 0 = NoText probability, index 1 = Text probability
- Decision: `text_prob >= TEXT_MIN_CONF` → PASS; else → NoText (FAIL)
- `BLANK_CELL_STD_THR > 0`: skip model if pixel std below threshold (blank-cell shortcut)

---

## Setup Flow *(one-time per product)*

1. Click **New Template** → grabs a fresh frame from the camera (or directory).
2. Draw a rubber-band rect around **either** IC on the image.
3. System auto-detects the second IC (`_find_second_ic` — region-based template match on the opposite image half). Template region = drawn IC rect **extended downward 50%** to include pin blobs, making a tight IC body box distinctive enough to match. Uses `_contour_template` for consistency with inspection-time matching.
4. Both IC_A (yellow) and IC_B (cyan) shown as overlays.
5. Click **Confirm** → saves `template.json`, `tmpl_full.npy` (pin-area patch), and `template_preview.png`.

If IC_B is not found automatically, the UI prompts to draw again (`draw_a_retry` state).

**Detailed flow:**
```
User clicks "New Template"
  │
  ▼
Grab one frame (camera or Input/)
  │
  ▼
User rubber-bands IC_A rect on ImageView
  │
  ▼
_find_second_ic(image_bgr, ic_a_rect)
  │  _contour_template(full image) → full_map
  │  template = full_map[ic_a_y : ic_a_y + ic_a_h + pin_h, ic_a_x : ic_a_x + ic_a_w]
  │             ↑ IC body + 50% pin area below → distinctive even with tight box
  │  search = full_map[:, right half]  (or left half)
  │  TM_CCOEFF_NORMED → best match → ic_b_rect (height = ic_a_h only)
  ▼
Both IC_A (yellow) + IC_B (cyan) shown; user clicks Confirm
  │
  ▼
TemplateManager.extract_patches(image_bgr, ic_a_rect)
  │  _contour_template(full image) → crop pin area [ic_bottom : ic_bottom + 50%h]
  │  strip_h = -(ic_h)  (patch top is IC height below IC top)
  ▼
Save: template.json, tmpl_full.npy, template_preview.png
```

**Pin patch geometry:**
```
ic_a_y ──────────────────┐
                          │  IC body  (cells R1C1–R3C2 extracted here)
ic_a_y + ic_a_h ─────────┤
                          │  pin area  (50% of ic_h)  ← saved patch for matching
ic_a_y + ic_a_h × 1.5 ───┘
```

Draw the IC_A box tightly around the IC body face. The pin area extension is added automatically — oversizing the box causes cell misalignment.

---

## GPIO Pins *(BCM, `IO=True` only)*

| Signal | Pin | Dir |
|---|---|---|
| `START_PIN` | 17 | IN — active HIGH 10 ms pulse (machine signals ready for one shot) |
| `BUSY_PIN` | 23 | OUT — HIGH during full inspection + retry |
| `END_PIN` | 18 | OUT — normally HIGH; pulses LOW 40 ms after inspection done |
| `INSPEC_STAGE_PIN` | 24 | OUT — normally HIGH; LOW = both ICs pass, HIGH = any fail |

**GPIO Timing:**
```
inspection complete
  → set BUSY LOW
  → set INSPEC_STAGE (LOW=PASS, HIGH=NG)
  → sleep PRE_END_SEC (10 ms)
  → pulse END_PIN LOW for 40 ms
  → set INSPEC_STAGE HIGH (idle)
  ← cycle_ms is snapped BEFORE this block
```

True machine cycle ≈ `cycle_ms` + ~51 ms GPIO tail.

---

## Threading Model

| Thread | Class | Role |
|---|---|---|
| Main (GUI) | `MainWindow` | Qt event loop, UI updates |
| Worker | `RunWorker(QThread)` | Camera grab → inspect → GPIO |
| Thumbnail loader | `ThumbnailWorker(QThread)` | Load thumbnails for browser |
| Folder scanner | `FolderScanWorker(QThread)` | Scan Output/ directory tree |

Cross-thread communication: PyQt5 signals only. Signal routing:
- `sig_warn` → `_show_error` (banner only, run continues)
- `sig_error` → `_on_worker_error` → `_enter_standby` (stops run)

---

## Logging *(dual-CSV, daily files)*

| File | Contents |
|---|---|
| `logs/op_YYYYMMDD.csv` | One row per event: `timestamp, event, lot_number, detail, cycle_ms` |
| `logs/result_YYYYMMDD.csv` | All lots for the day, one per day; lots separated by `# --- LOT_START ---` / `# --- LOT_END ---` blocks |

Each lot block in `result_YYYYMMDD.csv`: header rows (LOT_NUMBER, PACKAGE, START_TIME, MODE) → column header → inspection rows → summary footer → blank line.

Rotation: both `op_*.csv` and `result_*.csv` are deleted when their date is older than `LOG_RETENTION` days. Legacy `result_{lot}_{ts}.csv` filenames don't match the date pattern and are silently skipped (kept until manually deleted).

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
│   └── result_YYYYMMDD.csv           # daily result log (all lots, LOT_START/LOT_END blocks)
├── Input/                            # source images for USE_CAMERA=false
├── Dataset/                          # cell crops (COLLECT_DATASET=true)
└── Test/                             # trainModel.py, Converter.py, ImagePlayGround.py
```

`IMAGE_ID` format: `YYYYMMDD_HHMMSS_NNN` (thread-safe counter).

Output suffixes: `_G` (clean pass), `_GS` (suspect pass), `_NGS` (suspect NG), `_NG` (full NG). Threshold between `_GS` and `_NGS` is `TEXT_NG_THRESHOLD`. Clean-pass `_G` shots produce **zero disk writes** — raw frame is held in RAM and discarded.

---

## Key Tuning Parameters

| Parameter | Location | Effect |
|---|---|---|
| `TEXT_MIN_CONF` | Config.toml | Minimum text probability for PASS |
| `TEXT_NG_THRESHOLD` | Config.toml | Missing-cell count: `_GS` vs `_NGS` boundary |
| `search_margin` | `TemplateMatcher.__init__` | ±px around expected IC_A pin-patch position |
| `match_threshold` | `template.json` | Minimum match score (below → warning only) |
| `DISK_WARN_MB` | Config.toml | Free-space warning threshold |
| `LOG_RETENTION` | Config.toml | Days to retain log files (default 730 = 2 years) |
| `BLOB_MIN_RATIO` | Config.toml | Drop blobs < ratio × largest from binary map; 0.0 = off; 0.2 removes IC-corner reflections |
| `_contour_template` open kernel | hardcoded `9×9` | Increase if tape-noise blobs survive |
| `_contour_template` bg sigma | hardcoded `50` | Increase if illumination gradient is steep |

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
