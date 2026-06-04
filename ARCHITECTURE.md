# ClearIC Inspect — Architecture

Single-file application: `CLearIC.py`. All logic lives in one file; no modules.

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

## Template Preprocessing — `_contour_template()`

All three template operations share a single preprocessing function:

| Caller | When | Purpose |
|---|---|---|
| `TemplateManager.extract_patches()` | Setup — save template | Build pin-blob patch |
| `TemplateMatcher.locate_ic()` | Every inspection | Find IC_A position |
| `_find_second_ic()` | Setup — detect IC_B | Find IC_B position |

**Rule:** always pass the **full image**. Otsu and background-blur need the global pixel
histogram. Passing a crop changes the threshold and breaks consistency between
template-save time and search time.

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

**Why not CLAHE:** amplifies local tape texture → tape pixels classified as "bright" →
noise fills the template patch. Background-divide normalises global brightness without
amplifying local texture contrast.

**Output:** binary image — white = bright pin/IC regions, black = dark tape/background.

---

## Setup Flow — One-Time Per Product

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

Draw the IC_A box tightly around the IC body face. The pin area extension is added
automatically — oversizing the box causes cell misalignment.

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

## Cell Extraction — `_build_cells()`

Converts one IC rect into 6 ROI cells (3 rows × 2 cols):

```
IC rect
  → shrink by CELL_SHRINK (centred)
  → apply GRID_MARGIN_TOP / GRID_MARGIN_BOT dead-bands
  → split 3 rows × 2 cols with COL_GAP_PCT gap between columns
  → expand each cell by CELL_EXPAND (centred overlap)
  → row-major: R1C1, R1C2, R2C1, R2C2, R3C1, R3C2
```

---

## Classifier — `Detector`

- Model: OpenVINO IR (`best.xml` + `best.bin`)
- Input: raw cell crop (no preprocessing, no CLAHE)
- Output: `[1, 2]` — index 0 = NoText probability, index 1 = Text probability
- Decision: `text_prob >= TEXT_MIN_CONF` → PASS; else → NoText (FAIL)
- `BLANK_CELL_STD_THR > 0`: skip model if pixel std below threshold (blank-cell shortcut)

---

## Threading Model

| Thread | Class | Role |
|---|---|---|
| Main (GUI) | `MainWindow` | Qt event loop, UI updates |
| Worker | `RunWorker(QThread)` | Camera grab → inspect → GPIO |
| Thumbnail loader | `ThumbnailWorker(QThread)` | Load thumbnails for browser |
| Folder scanner | `FolderScanWorker(QThread)` | Scan Output/ directory tree |

Cross-thread communication: PyQt5 signals only. `RunWorker` emits:
`sig_image`, `sig_result`, `sig_fail`, `sig_error`, `sig_warn`, `sig_status`,
`sig_cycle_ms`, `sig_done`, `sig_session_reset`, `sig_paused`, `sig_resumed`.

`sig_warn` → `_show_error` (banner only, run continues).
`sig_error` → `_on_worker_error` → `_enter_standby` (stops run).

---

## Logging

| File | Rotation | Contents |
|---|---|---|
| `logs/op_YYYYMMDD.csv` | Date-based (`LOG_RETENTION` days) | One row per event |
| `logs/result_YYYYMMDD.csv` | Date-based (`LOG_RETENTION` days) | All lots for the day, `LOT_START`/`LOT_END` blocks |

Events: `SESSION_START`, `SESSION_END`, `PASS`, `PASS_SUSPECT`, `FAIL`, `FAIL_SUSPECT`,
`ERROR`, `PAUSE`, `RESUME`.

---

## Output Files

```
Output/YYYYMMDD/lot_number/
  ├─ RealImg/{img_id}_{suffix}.jpg   raw unannotated frame (non-_G only)
  └─ Image/{img_id}_{suffix}.jpg     annotated frame (non-_G only)
```

`_G` (clean pass) → zero disk writes. `raw_bgr` copy is held in RAM then discarded.

Suffixes: `_G` · `_GS` · `_NGS` · `_NG`.
Threshold between `_GS` and `_NGS`: `TEXT_NG_THRESHOLD`.

---

## GPIO Timing (IO=True)

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
