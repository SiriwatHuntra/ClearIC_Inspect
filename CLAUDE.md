# ClearIC Inspect — Reference

Single-file: `CLearIC.py`. All code lives there; no modules.

---

## Config Flags *(top of file, not UI-changeable)*

| Flag | Values | Effect |
|---|---|---|
| `DEBUG` | `True/False` | Verbose logs + annotated output saved on every cycle |
| `CAMERA` | `"camera"/"directory"` | Live Basler or load from `Input/` |
| `IO` | `True/False` | Drive GPIO or mock (`[IO MOCK] PIN→STATE`) |
| `MODE` | `RUN/DEBUG` | Production or dev |
| `MODEL_PATH` | path | OpenVINO `.xml` |

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
  → BUSY check (drop if BUSY)
  → Camera.grab()                          → image_bgr (ndarray)
  → TemplateMatcher.locate_ic(image_bgr)   → (QRect rt_a, score)
  → rt_b = rt_a offset by template (ic_b_offset_x/y)
  → Inspector._rect_to_cells(rt_a/b)       → cells_a / cells_b  [(cx,cy,cw,ch)×6]
  → Inspector._check_ic(image_bgr, cells)  → (missing[], hits[])
  → ic_pass = all(hits)
  → set FAIL_A_PIN / FAIL_B_PIN
  → pulse ACK_PIN
  → on FAIL: save OUTPUT images + log
  → wait DONE_PIN  →  clear pins  →  STANDBY
```

Template-match fallback: if score < `match_threshold`, use fixed template coords.

---

## Key Functions

### `_build_cells(x, y, w, h) → list[(cx,cy,cw,ch)]`
Converts one IC bounding rect into 6 ROI cells (3 rows × 2 cols).  
- Applies `_CELL_SHRINK` (L/R), `_GRID_MARGIN_TOP/BOT` (top/bot), `_COL_GAP_PCT` (between cols), `_CELL_EXPAND` (per-cell).  
- Output: list of 6 `(int, int, int, int)` tuples, row-major order R1C1→R3C2.

### `Inspector._rect_to_cells(rect: QRect) → list`
Wrapper: unpacks QRect → calls `_build_cells(rect.x, rect.y, rect.width, rect.height)`.

### `Inspector._check_ic(image_bgr, cells, annotated, debug) → (missing, hits_flags)`
- `cells`: 6-tuple list from `_build_cells`  
- Crops each cell → resizes 224×224 → `Detector.classify_crop()` → bool per cell  
- `missing`: `[[row, col], …]` for NoText cells  
- `hits_flags`: `[bool×6]`  
- Draws colored borders + adaptive-scale labels onto `annotated` in place.

### `Inspector.inspect(image_bgr, debug) → (pass_a, pass_b, missing_a, missing_b, annotated_bgr)`
Top-level call. Runs locate → cells → `_check_ic` × 2. Raises `MarkMissingError` on any missing cell.

### `TemplateMatcher.locate_ic(image_bgr) → (QRect, score)`
Bilateral-strip template match. Returns IC_A bounding rect and match score.

### `TemplateManager.load() → dict`
Keys: `ic_a {x,y,w,h}`, `ic_b {x,y,w,h}`, `exposure_us`, `match_threshold`,  
`strip_top_y_offset`, `strip_bot_y_offset`, `strip_h`.

### `TemplateManager.save(ic_a: QRect, ic_b: QRect, exposure_us, match_threshold, strip_top_y_offset, strip_bot_y_offset, strip_h)`
Writes template JSON + bilateral patch `.npy` files.

### `RunWorker.run()`
Background thread. Loop: grab → inspect → set pins → pulse ACK → wait DONE.  
Signals: `result_ready`, `error_occurred`, `status_changed`.

---

## Cell Grid Constants *(top of file, not UI-changeable)*

| Constant | Default | Effect |
|---|---|---|
| `_CELL_SHRINK` | `1.00` | Horizontal L/R shrink ratio (1.0 = none) |
| `_GRID_MARGIN_TOP` | `10.0` | Top margin before row 1, % of IC height |
| `_GRID_MARGIN_BOT` | `10.0` | Bottom margin after row 3, % of IC height |
| `_COL_GAP_PCT` | `40.0` | Gap between L and R column, % of IC width |
| `_CELL_EXPAND` | `1.00` | Per-cell expansion after slicing (1.0 = none) |

---

## Setup & IO

### Setup Flow *(one-time per product)*
1. Capture/load reference image
2. Click → set IC_A anchor (top-left of IC body)
3. Set scale → auto-compute 6 ROI positions
4. Set column offset (L ↔ R horizontal shift)
5. Set IC_B offset (x, y from IC_A anchor)
6. Preview 12 ROI boxes → adjust → Save Template

### GPIO Pins *(BCM, IO=True only)*

| Signal | Pin | Dir | Description |
|---|---|---|---|
| `START_PIN` | 17 | IN↓ | Rising edge → start inspection |
| `DONE_PIN` | 27 | IN↓ | Rising edge → return to STANDBY |
| `ACK_PIN` | 22 | OUT | Pulse HIGH when result ready |
| `FAIL_A_PIN` | 24 | OUT | HIGH = IC_A failed |
| `FAIL_B_PIN` | 25 | OUT | HIGH = IC_B failed |

---

## Files & Output

### Directory Structure
```
ClearIC_Inspect/
├── CLearIC.py
├── Text_cls-2/best_openvino_model/   # best.xml + best.bin
├── templates/                         # template.json + top/bot .npy patches
├── Output/                            # FAIL images only
├── logs/                              # inspect_YYYYMMDD.log
└── Input/                             # CAMERA="directory" source images
```

### Output Images *(FAIL only)*
- `IMAGE_ID_R.jpg` — raw capture  
- `IMAGE_ID.jpg` — annotated (green border = Text/PASS, red = NoText/FAIL)  
- `IMAGE_ID` format: `YYYYMMDD_HHMMSS_NNN`

### Log Format *(JSON-lines, one record/inspection)*
Fields: `timestamp`, `image_id`, `ic_a_result`, `ic_a_missing`, `ic_b_result`, `ic_b_missing`, `cycle_time_ms`, `mode`, `io_mock`.  
DEBUG adds: per-cell `class` + `confidence`.  
Rotation: daily → `logs/inspect_YYYYMMDD.log`, 365-file retention.

### Template JSON Keys
`ic_a`, `ic_b` → `{x, y, w, h}` · `exposure_us` · `match_threshold` · `strip_top_y_offset` · `strip_bot_y_offset` · `strip_h`
