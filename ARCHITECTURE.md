# ClearIC Inspect — Architecture Reference

Three sections: class/function reference with data types, operator workflow, and Qt signal flow.

---

## 1. Program Flow & Class/Function Reference

### ConfigLoader (`CLearIC.py:46`)

Static class — no constructor.

| Method | Signature | Description |
|---|---|---|
| `load()` | `() → dict` | Read Config.toml, fill missing keys from `DEFAULT_CONFIG`, type-validate all values |
| `save(updates)` | `(dict) → None` | Overwrite specific keys in Config.toml |
| `update(updates)` | `(dict) → None` | Merge updates into existing TOML, persist |

---

### Camera (`CLearIC.py:237`)

Unified image source — Basler camera or directory loop.

**Constructor:**
```
Camera(mode, serial="", exposure_us=8000, input_dir="Input",
       retry_delay=0.2, retries=2, warmup_frames=5,
       image_w=0, image_h=0)
```

| Method | Returns | Description |
|---|---|---|
| `grab()` | `np.ndarray` (BGR uint8) | Grab one frame; raises `CameraError` on failure (retries internally) |
| `grab_first()` | `np.ndarray` (BGR uint8) | Rewind → grab → rewind; used for setup and OCR crop |
| `warmup()` | `None` | Discard N initial frames (Basler mode only) |
| `reconnect(attempts=1, delay_s=0.0)` | `bool` | Close and re-open Basler camera; returns `True` on success |
| `is_open()` | `bool` | True if camera handle is open (Basler) or image list is loaded (directory) |
| `is_healthy()` | `bool` | Delegates to `is_open()` — non-blocking liveness check |
| `has_more()` | `bool` | Directory mode: unvisited frames remain in this cycle? |
| `reset()` | `None` | Rewind directory index to beginning |

**Mode differences:**
- `"camera"` — pypylon Basler, `RetrieveResult` with 5 s timeout, grayscale→BGR conversion
- `"directory"` — reads sorted files from `Input/`, loops indefinitely, resizes if `IMAGE_W`/`IMAGE_H` set

---

### CellCon (`CLearIC.py:467`)

Serial USB interface to lot-tracking system.

**Constructor:** `CellCon(port="/dev/ttyUSB0")`

| Method | Returns | Description |
|---|---|---|
| `get_lot()` | `str` | Open serial 38400 8N1 1 s timeout → send `LA\r\n` → read up to 5 retries → parse `LS,<lot_number>` → return lot string; returns `""` on any error/timeout |

**Protocol:** Request = `"LA\r\n"`, Response prefix = `"LS,"`, lot is `parts[1]` after comma split.

---

### RaspberryIO (`CLearIC.py:505`)

GPIO controller with mock fallback when `IO=False`.

In mock mode (`IO=False`), `wait_for_start()` blocks until `trigger()` is called from the UI thread (one click = one inspection cycle). All outputs are logged to console instead of driving real pins.

**Constructor:**
```
RaspberryIO(io_enabled=True, start_pin=17, busy_pin=23,
            end_pin=18, inspec_stage_pin=24)
```

**Timing constants (class-level):**

| Constant | Value | Description |
|---|---|---|
| `_END_PIN_PULSE_SEC` | `0.040` | END_PIN LOW pulse duration (40 ms) |
| `_GPIO_PRE_END_SEC` | `0.010` | Settle delay before END_PIN pulse (10 ms) |

| Method | Returns | Description |
|---|---|---|
| `is_initialised()` | `bool` | Returns `True` if GPIO was set up successfully |
| `set_busy(v)` | `None` | Drive BUSY_PIN HIGH (`True`) or LOW (`False`) |
| `set_inspec_stage(high)` | `None` | Drive INSPEC_STAGE_PIN; `False` (LOW) = both ICs pass, `True` (HIGH) = any fail / idle |
| `pulse_end_pin()` | `None` | Pulse END_PIN LOW for `_END_PIN_PULSE_SEC` (40 ms) then HIGH; blocking — call from worker thread only |
| `trigger()` | `None` | Inject a mock START pulse (mock mode only); called from UI thread |
| `wait_for_start(stop_flag_fn)` | `bool` | Block until START_PIN goes HIGH (active HIGH 10 ms pulse); mock mode blocks until `trigger()` is called; returns `False` if `stop_flag_fn()` fires |
| `drain_start_pin(timeout_ms)` | `None` | Wait until START_PIN returns LOW (idle); clears mock trigger in mock mode — called on resume to discard stale pulse |
| `clear_outputs()` | `None` | BUSY_PIN → LOW, INSPEC_STAGE → HIGH (idle), END_PIN → HIGH (idle) |

---

### Detector (`CLearIC.py:640`)

OpenVINO 2-class classifier: NoText (absent) vs Text (present).

**Constructor:**
```
Detector(conf_thr=0.5, text_min_conf=0.80, blank_cell_std_thr=0.0,
         model_path="Text_cls-2/best_openvino_model/best.xml",
         n_passes=3, uncertain_thr=0.50, debug=False)
```

| Method | Returns | Description |
|---|---|---|
| `classify_crop(crop_bgr)` | `(class_idx: int, confidence: float)` | OpenVINO inference on raw crop → averaged over `n_passes` runs → index `0`=NoText, `1`=Text |

**Asymmetric gate:** Text class probability must exceed `text_min_conf` to call a cell PASS; raw NoText probability is not the gate — Text confidence is.

**No preprocessing inside classify_crop:** The caller provides the raw crop. CLAHE is only used in `_contour_template()` for IC pin search (template matching), not for cell classification.

---

### TemplateMatcher (`CLearIC.py:1000`)

Locates IC_A in new frames using a saved pin-area patch.

**Constructor:**
```
TemplateMatcher(full_patch, threshold=0.6, strip_h=0,
                ic_x=0, ic_y=0, ic_w=0, ic_h=0,
                search_margin=60, template_w=0)
```

| Method | Returns | Description |
|---|---|---|
| `locate_ic(image_bgr)` | `(QRect, score: float)` | Canny-contour preprocess on ±margin ROI → `cv2.matchTemplate(TM_CCOEFF_NORMED)` → IC_A bounding box + match confidence; logs warning if `score < threshold` but always returns best match |

**Geometry note:** `strip_h` is negative (patch starts below IC top = `−IC_height`), so `found_ic_y = matched_patch_y + strip_h`. Patch and search params are auto-scaled if image width differs from template.

---

### Inspector (`CLearIC.py:1131`)

Crops 12 ROI cells (6 per IC × 2 ICs) and classifies each.

**Constructor:**
```
Inspector(detector, template, template_matcher=None,
          cell_shrink=0.95, cell_expand=1.2, col_gap_pct=40.0,
          grid_margin_top=0.0, grid_margin_bot=15.0,
          collect_dataset=False, data_dir="Dataset", data_split="train",
          ann_border_px=1, ann_show_labels=True)
```

| Method | Returns | Description |
|---|---|---|
| `inspect(image_bgr, debug=False)` | `(ic_a_pass: bool, ic_b_pass: bool, missing_a: list, missing_b: list, annotated_bgr: np.ndarray)` | Locate ICs → classify cells × 2 → annotate image **in-place**; raises `MarkMissingError` on missing cells; raises `TemplateError` on bad match |
| `_check_ic(image_bgr, cells, annotated, debug)` | `(missing: [[row,col],...], hits_flags: [bool×6], text_confs: [float×6])` | Crop raw cell from image → `classify_crop()` × 6; draws colored borders + labels onto annotated |

#### `_build_cells(x, y, w, h, ...)` → `list[(cx, cy, cw, ch)]`
Converts one IC rect to 6 ROI cells. Steps: shrink (centred) → apply top/bottom margins → slice 3×2 grid with column gap → expand each cell. Row-major order: R1C1 → R1C2 → R2C1 → R2C2 → R3C1 → R3C2.

#### `_resolve_ic(missing_first, confs_first, confs_second, w2=0.7, w1=0.3, pass_thr=0.90)` → `still_missing`
Confidence-weighted retry resolution: `w = w2 × conf_second + w1 × conf_first`. Cell clears only if `w >= pass_thr`. Applied only to cells that failed on the first attempt. Weights and threshold are passed by the caller from config keys `RETRY_W2`, `RETRY_W1`, `RETRY_PASS_THR`.

---

### Logger (`CLearIC.py:1307`)

Writes two CSV logs: daily operation log and per-lot result log.

| Method | Signature | Writes |
|---|---|---|
| `start_lot(lot, package, mode)` | `(str, str, str) → None` | Result CSV header (metadata block) + column header row |
| `end_lot(reason, pass_ct, fail_ct, err_ct, elapsed_s)` | `(str, int, int, int, float) → None` | Result CSV footer: total, pass, fail, errors, yield%, end time, duration |
| `log_inspection(image_id, ic_a_result, ic_a_missing, ic_b_result, ic_b_missing, cycle_ms, is_retry, is_suspect)` | `(...) → None` | One result CSV data row + op_log PASS/FAIL/PASS_SUSPECT/FAIL_SUSPECT event |
| `log_error(error_type, msg, cycle_ms)` | `(str, str, float) → None` | op_log ERROR row |
| `log_pause()` | `() → None` | op_log PAUSE row |
| `log_resume()` | `() → None` | op_log RESUME row |
| `log_ocr(operator, expect_mark)` | `(str, str) → None` | op_log OCR_VERIFY row: `op=<operator> expect=<mark>` |

**Files:**
- `logs/op_YYYYMMDD.csv` — daily: `timestamp, event, lot_number, detail, cycle_ms`
- `logs/result_{lot}_{YYYYMMDD_HHMMSS}.csv` — per lot: header + data rows + footer

---

### RunWorker (`CLearIC.py:1776`)

Main `QThread` inspection loop.

**Constructor:**
```
RunWorker(camera, inspector, gpio, logger, cfg, lot_number="", parent=None)
```

| Method | Description |
|---|---|
| `run()` | Main loop: wait START_PIN HIGH → BUSY HIGH → grab → inspect → result → INSPEC_STAGE + END_PIN pulse → pause checkpoint |
| `trigger()` | Delegate to `gpio.trigger()` — injects a mock START pulse from the UI |
| `stop()` | Set `_stop=True`, unblock `_running` event |
| `pause()` | `_running.clear()` — blocks loop at post-GPIO checkpoint |
| `resume()` | Set `_drain_needed` flag, `_running.set()` — unblocks loop; drains stale START_PIN |

---

### `_ocr_api_call` (`CLearIC.py:3486`)

**Signature:** `(lot: str, operator: str, expected_mark: str) → bool`
Returns `True` = proceed with run, `False` = abort lot.

**Steps:**

1. Reset `_ocr_used_mark = expected_mark` (never carry stale value from prior lot)
2. Grab frame → resize to `IMAGE_W × IMAGE_H` → save `cropimg.jpg`
3. `POST /OCR/ReadMark` — `{"username": operator, "lot_no": lot}` — 5 s timeout
   - **200, list with mark:** compare `std_mark` vs `ocr_mark`
     - `ocr_mark` null → retry once; still null → DEBUG: use `expected_mark` / PROD: `return False`
     - `std_mark == ocr_mark` → `is_pass=1`; mismatch → `is_pass=0`
   - **200, empty list / missing `mark` field** → DEBUG: skip / PROD: `return False`
   - **401/403** → auth error — DEBUG: skip / PROD: `return False`
   - **404** → endpoint error — DEBUG: skip / PROD: `return False`
   - **5xx** → server error — DEBUG: skip / PROD: `return False`
4. `POST /OCR/CreateRecord` — `{"username", "lot_no", "mark", "image"(b64 JPEG), "is_pass", "recheck_count": 0, "is_logo_pass": 0}` — failure silently ignored (never blocks run)
5. Returns `bool(is_pass) or debug`
6. `finally:` remove `cropimg.jpg`

---

## 2. Operator Workflow

```
STARTUP
  └─ python CLearIC.py
       ├─ Single-instance lock check
       ├─ Config.toml loaded (abort if parse error)
       ├─ Directories ensured: logs/, templates/, Input/, Dataset/
       ├─ OpenVINO model loaded + warmup (WARMUP_FRAMES inference calls)
       ├─ GPIO initialized (or mock if IO=False in config)
       ├─ Camera opened (Basler by CAMERA_SERIAL / or Input/ directory)
       └─ Inspector built from templates/template.json
            └─ If template missing → UI shows "No template" banner

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONE-TIME SETUP  (once per product / IC layout change)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  └─ Click [New Template]
       ├─ Grab frame → display in ImageView
       ├─ Operator draws rubber-band rect around either IC
       │    → IC_A bounding box set
       ├─ _find_second_ic() auto-detects IC_B on opposite image half
       │    ├─ Found → show IC_A (yellow) + IC_B (cyan) overlays
       │    └─ Not found → prompt operator to draw IC_B manually
       ├─ Cell preview dialog:
       │    3×2 grid for IC_A and IC_B shown side-by-side
       │    Operator confirms cell crops align with printed marks
       └─ [Confirm] saves:
            templates/template.json        IC coords + match params
            templates/tmpl_full.npy        pin-area binary patch
            templates/template_preview.png annotated preview

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOT START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  └─ Fill fields:
       ├─ Operator ID (6 digits)
       └─ Expected mark (6 alphanumeric)
  └─ Click [Start]
       ├─ Lot number acquisition:
       │    1. CellCon serial: send LA\r\n → parse LS,<lot> → auto-fill
       │    2. If CellCon empty/error → LotStartDialog shown
       │       Operator types lot OR leaves blank (auto timestamp)
       │       Cancel → abort, return to standby
       ├─ OCR verification (skipped if DEBUG=True):
       │    POST ReadMark(lot, operator)
       │    ├─ Mark matches → green "Mark OK" → proceed
       │    ├─ Mismatch → red "FAIL — DB:X | OCR:Y" → abort lot
       │    └─ API unreachable/error → red error label → abort lot
       ├─ Logger.start_lot() → create result CSV
       └─ RunWorker thread starts

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
INSPECTION CYCLE  (one shot per START pulse)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [Camera mode]
  └─ Wait for START_PIN HIGH (10 ms active-HIGH pulse from machine)
     IO=False: wait for UI [Trigger] button click instead
  [Directory mode]
  └─ Auto-advance to next file in Input/

  Per cycle:
  ├─ BUSY_PIN → HIGH
  ├─ Camera.grab() → raw BGR frame
  ├─ Save raw to Output/YYYYMMDD/lot/RealImg/ (temp name)
  ├─ Inspector.inspect():
  │    TemplateMatcher.locate_ic() → IC_A QRect + match score
  │    IC_B = IC_A shifted by (ic_b_dx, ic_b_dy) from template
  │    _check_ic(IC_A cells) → missing_a, confs_a
  │    _check_ic(IC_B cells) → missing_b, confs_b
  │
  ├─ [MarkMissingError — camera mode only]:
  │    Sleep RETRY_DELAY_MS → grab second frame → re-inspect
  │    ├─ Second pass clears all cells → use second result
  │    └─ Still missing → _resolve_ic():
  │         w = RETRY_W2×conf_second + RETRY_W1×conf_first
  │         cell clears only if w ≥ RETRY_PASS_THR
  │
  ├─ PASS/FAIL determination:
  │    missing = len(missing_a) + len(missing_b)
  │    ┌─────────────────────────────────────────────┐
  │    │ 0 missing           → PASS         → _G     │
  │    │ 1 missing           → PASS+SUSPECT → _GS    │
  │    │ 2+ missing          → FAIL+SUSPECT → _NGS   │
  │    │ all 12 cells miss   → FAIL         → _NG    │
  │    └─────────────────────────────────────────────┘
  ├─ Rename raw file → Output/RealImg/{img_id}{suffix}.jpg
  ├─ Save annotated → Output/Image/{img_id}{suffix}.jpg
  ├─ UI: sig_result / sig_fail → badges update + stats counters
  ├─ BUSY_PIN → LOW
  └─ [Camera mode only]:
       INSPEC_STAGE → LOW if both ICs pass, HIGH if any fail
       sleep 10 ms
       END_PIN pulse LOW 40 ms  (machine reads INSPEC_STAGE during this window)
       INSPEC_STAGE → HIGH (idle)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PAUSE / RESUME
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [Pause] button
  └─ Worker completes current cycle (GPIO outputs cleanly restored)
     Then blocks at pause checkpoint
     sig_paused → button text "Resume" / status "Paused."

  [Resume] button
  └─ Drain stale START_PIN signal (prevent spurious next-cycle trigger)
     sig_resumed → button text "Pause" / status "Running…"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LOT END
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [Stop] button → worker exits → sig_done → standby
  └─ Logger.end_lot() writes footer: total / pass / fail / yield% / duration

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
IMAGE BROWSER TAB
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Left panel   : folder tree (date → lot)
  Centre panel : thumbnail grid (4 columns)
                  _G   → teal      PASS
                  _NG  → red       FAIL
                  _GS  → yellow-green  PASS+SUSPECT
                  _NGS → orange    FAIL+SUSPECT
  Right panel  : source toggle (RealImg raw / Image annotated)
                 filter buttons (ALL / FAIL / SUSPECT)

  Click thumbnail → single-image view with ← → navigation
  Right-click or Back → return to grid
```

---

## 3. Signal Flow & Behavior

### RunWorker Signals (`CLearIC.py:1784`)

| Signal | Arg types | Emitted when | MainWindow slot | Slot action |
|---|---|---|---|---|
| `sig_image` | `object` (np.ndarray BGR) | Every completed cycle | `_on_image()` | `_view.set_image(img)` — display annotated image |
| `sig_result` | `bool, bool, bool` (ia_pass, ib_pass, is_suspect) | PASS or PASS_SUSPECT cycle | `_on_result()` | IC_A/B badges → green; pass counter +1; yield% update |
| `sig_fail` | `object, str, str, bool` (MarkMissingError, ann_path, img_id, is_suspect) | FAIL or FAIL_SUSPECT cycle | `_on_fail()` | Failing IC badges → red; fail counter +1; yield% update |
| `sig_error` | `str` (message) | Fatal error (camera / template) | `_on_worker_error()` | End lot logging; error banner; enter standby |
| `sig_status` | `str` | Status transitions | `_lbl_status.setText()` + `_reset_watchdog()` | Update status label; reset 30 s inactivity watchdog |
| `sig_cycle_ms` | `float` | Every completed cycle | lambda | Update "Last ms" label |
| `sig_done` | — | Worker loop exits (Stop pressed) | `_on_run_done()` | End lot logging; return to standby UI state |
| `sig_session_reset` | `str` (new lot_number) | Directory batch wrap | `_on_session_reset()` | Auto-advance lot; reset stats counters; log SESSION_START |
| `sig_paused` | — | Worker reaches pause checkpoint | `_on_paused()` | Button → "Resume"; status → "Paused."; log PAUSE |
| `sig_resumed` | — | Worker unblocked after resume | `_on_resumed()` | Button → "Pause"; status → "Running…"; log RESUME |

**Watchdog:** Both `sig_image` and `sig_status` call `_reset_watchdog()`. If 30 s passes with no signal, watchdog fires → stop worker → show error banner.

---

### RunWorker Internal State Machine

```
[THREAD START]
     │
     ▼
 PREFLIGHT
  camera.grab_first() reachable?
     │ fail ──► sig_error ──► thread exits
     │ ok
     ▼
 ┌─────────────────────────────────────────────────────────┐
 │  MAIN LOOP  (while not _stop)                           │
 │                                                         │
 │  WAIT_TRIGGER                                           │
 │    [camera] wait_for_start() blocks on START_PIN HIGH   │
 │             IO=False: blocks until trigger() called     │
 │    [directory] sleep 0.05 s                             │
 │         ▼                                               │
 │  GRAB_FRAME                                             │
 │    BUSY_PIN → HIGH                                      │
 │    camera.grab() → img_bgr                              │
 │    CameraError ──► [dir] skip / [cam] reconnect or exit │
 │    img_id = next_image_id()                             │
 │    save raw to RealImg/tmp_{img_id}.jpg                 │
 │         ▼                                               │
 │  INSPECT                                                │
 │    inspector.inspect(img_bgr)                           │
 │    TemplateError ──► sig_error / skip                   │
 │    MarkMissingError (1st):                              │
 │      [camera] sleep RETRY_DELAY_MS                      │
 │               grab img_bgr2                             │
 │               inspector.inspect(img_bgr2)               │
 │               ├─ success ──► use img_bgr2 result        │
 │               ├─ still missing ──► _resolve_ic()        │
 │               └─ camera/template error ──► keep 1st     │
 │      [directory] no retry                               │
 │         ▼                                               │
 │  DETERMINE_RESULT                                       │
 │    count = len(missing_a) + len(missing_b)              │
 │    0     → PASS      _G                                 │
 │    1     → PASS+SUS  _GS                                │
 │    2–11  → FAIL+SUS  _NGS                               │
 │    12    → FAIL      _NG                                │
 │         ▼                                               │
 │  SAVE & EMIT                                            │
 │    rename raw; save annotated                           │
 │    sig_image(annotated)                                 │
 │    sig_result(...) or sig_fail(...)                     │
 │    sig_cycle_ms(elapsed_ms)                             │
 │    logger.log_inspection(...)                           │
 │         ▼                                               │
 │  GPIO OUTPUT  [camera mode only]                        │
 │    BUSY_PIN → LOW                                       │
 │    INSPEC_STAGE → LOW (both pass) or HIGH (any fail)    │
 │    sleep 10 ms                                          │
 │    pulse END_PIN LOW 40 ms                              │
 │    INSPEC_STAGE → HIGH (idle)                           │
 │         ▼                                               │
 │  PAUSE_CHECKPOINT                                       │
 │    if not _running.is_set():                            │
 │      sig_paused.emit()                                  │
 │      _running.wait()      ◄─── blocks here              │
 │      if _stop: break                                    │
 │      if _drain_needed: gpio.drain_start_pin()           │
 │      sig_resumed.emit()                                 │
 │         │                                               │
 │         └──────────────────────────────► WAIT_TRIGGER  │
 └─────────────────────────────────────────────────────────┘
     │
     ▼
 SHUTDOWN
  clear_outputs()
  sig_done.emit()
  sig_status("Standby.")
```

---

### UI Update Flow Per Cycle

```
RunWorker thread                    MainWindow (Qt main thread)
─────────────────                   ───────────────────────────
sig_image(annotated_bgr)   ──────►  _on_image()
                                       _view.set_image(img)

sig_result(a, b, suspect)  ──────►  _on_result()
                                       _update_badge(badge_a, a)
OR                                     _update_badge(badge_b, b)
                                       _stats_pass += 1
sig_fail(err, path, id, s) ──────►  _on_fail()
                                       _update_badge(badge_a, not err.missing_a)
                                       _update_badge(badge_b, not err.missing_b)
                                       _stats_fail += 1

sig_cycle_ms(ms)           ──────►  lambda: _lbl_ms.setText(f"{ms:.0f} ms")

sig_status(text)           ──────►  _lbl_status.setText(text)
                                    _reset_watchdog()

sig_image                  ──────►  _reset_watchdog()
```

### Badge Color Codes

| State | Object name | Color | Text |
|---|---|---|---|
| Idle / unknown | `badge_idle` | Grey | `—` |
| Pass | `badge_pass` | Light blue | `PASS` |
| Fail | `badge_fail` | Red | `FAIL` |
