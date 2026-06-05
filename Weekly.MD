# Weekly Change Log

---

## Week: 2026-06-02 → 2026-06-05

---

### 1. Lighting Control (new feature — `lighting_ctrl` branch, merged PR #1)

**New class: `LightingController`**
- Serial RS232 over Prolific USB adapter (38400 baud, 8N1)
- Commands: `@00L1007D\r\n` (ON), `@00L0007C\r\n` (OFF), `@00F{value:03}00{checksum:02X}\r\n` (brightness)
- Fire-and-forget: `_send()` writes to serial without blocking — no background reader thread (simplified from original design)
- On init, fires the OFF command and reads ACK to confirm controller is alive → sets `controller_ok` flag
- `probe(timeout_s=0.2)` method: sends OFF and reads ACK, updates `controller_ok` — called during retry cycle
- Config keys added: `LIGHTING_ENABLE`, `LIGHTING_USB_ID`, `LIGHTING_PORT`, `LIGHTING_VALUE`

**Port auto-detection: `_detect_ports(usb_id_hint)`**
- Lighting: claimed first via `/dev/serial/by-id/` symlink matching `LIGHTING_USB_ID` — immune to USB slot swaps
- CellCon: probed on remaining `/dev/ttyUSB*` ports using `LA\r\n` → `LS,...` handshake (3 retries × 0.5 s timeout per port)
- If either is not found: toast shows "NOT FOUND ⚠" or "NO RESPONSE ⚠" — no crash

**UI additions**
- Startup status label sequence: "Detecting hardware…" → "GPIO warmup…" → "Standby." with `processEvents()` between each
- Amber toast label under Setup panel showing Light and CellCon detection result
- "Re-detect Hardware" button in Setup panel; only active in standby state
- Hardware toast shows three possible lighting states: `OK`, `NO RESPONSE ⚠`, `NOT FOUND ⚠`

**Run lifecycle**
- Light ON at `_start_worker()`, OFF at `_enter_standby()`
- During inspection retry (`MarkMissingError`): light OFF → `probe()` → light ON → re-grab
- During template grab (`_grab_setup_frame`): `try/finally` guarantees light OFF even on camera error
- On `closeEvent`: light OFF + serial port closed

**Logging**
- `HW_PORTS` row written to result CSV header each lot: `lighting=<port> cellcon=<port>`
- Port assignments also appended to `SESSION_START` row in op CSV

**Standalone test: `Test/lighting_test.py`**
- Blink-latency tester: measures ON echo time, OFF echo time, cycle total over N cycles
- Auto-detects Prolific port via `/dev/serial/by-id/`; falls back to first `/dev/ttyUSB*`
- Prints min/max/avg stats and recommendation: `< 50 ms` = per-capture safe, `> 100 ms` = hold per-lot

**`requirements.txt`** — Added `pyserial` and `tomlkit` (previously used but unlisted)

---

### 2. Template Matching Refactor (June 4)

**`_contour_template()` algorithm changed: edge-based → region-based**

| Step | Before | After |
|---|---|---|
| Denoise | Gaussian blur | `medianBlur(5)` |
| Normalise | CLAHE | Background-divide: divide gray by `GaussianBlur(σ=50)`, scale to 0–255 |
| Threshold | Canny edge detector | Otsu threshold on normalised image |
| Clean | Dilate | `morphOpen(9×9)` (removes tape-noise blobs) |
| Fill | — | `morphClose(5×5)` (fills holes inside pin blobs) |

Background division normalises lot-to-lot brightness variation without amplifying tape texture, unlike CLAHE.
Same pipeline runs at template-save time, search time, and IC_B auto-detection — all three see identical histograms.

**Blob min ratio filter added to `_contour_template()`**
- New parameter `min_blob_ratio` (also `BLOB_MIN_RATIO` in Config.toml, default `0.0` = disabled)
- After morphology: `cv2.connectedComponentsWithStats` finds all blobs; any blob with area `< ratio × largest_blob_area` is zeroed out
- Purpose: eliminates IC-corner reflections and other non-pin artefacts that survive morphology
- Setting `0.2` removes typical corner reflections; `0.0` leaves behaviour unchanged

**TOML config keys added (matching thresholds)**
- `BLOB_MIN_RATIO` (0.0–1.0, validated)
- `TEMPLATE_MATCH_THR` — minimum score for IC_A locate warning
- `TEMPLATE_FIND_CONF_THR` — minimum score to accept IC_B during setup auto-detect

---

### 3. Low Match Score Handling (June 5)

**New exception: `LowMatchError(InspectionError)`**
- Raised in `Inspector.inspect()` when `score < _threshold` (instead of just logging a warning)
- `RunWorker` catches `LowMatchError` separately: skips the cycle (`continue`), turns light OFF, sets BUSY LOW — loop is not broken
- Status label shows: `"Low match — skipped. (score X.XXX < Y.YYY — frame skipped)"`
- Previously: low score logged a warning but the frame was still processed with potentially wrong IC position

---

### 4. Camera: Snapshot-Only Open Mode (June 5)

**Previous behaviour**: Camera opened once at run start, held open for the entire lot.
**New behaviour**: Camera opened fresh for each capture cycle, closed immediately after.

- `RunWorker` loop now calls `camera.open()` + `camera.warmup()` at the top of each camera-mode cycle (after START pin fires)
- Camera closed at end of each cycle after GPIO signals complete
- On grab failure: camera closed; next START trigger attempts a fresh open (no longer uses `reconnect()`)
- `Camera.reconnect()` method removed (dead code)
- `Camera.is_healthy()` method removed (dead code)
- `Camera._grab_once()` private wrapper removed; `grab()` calls `_grab_basler()` / `_grab_directory()` directly

**Why**: Avoids stale camera state across long idle periods between cycles; cleaner error recovery.

---

### 5. Performance Fix: Pre-write/Delete Removed (June 4)

**Previous behaviour**: Raw frame written to a temp file before result was known; renamed on pass, deleted on fail.
This meant every cycle wrote a file (even `_G` clean passes), then deleted or renamed it.

**New behaviour**: Raw frame held in RAM as `raw_bgr = img_bgr.copy()` after grab.
- `inspect()` annotates `img_bgr` in-place; `raw_bgr` remains unannotated
- Files written only for non-`_G` results (one `cv2.imwrite` call each for raw + annotated)
- Clean `_G` passes: zero disk writes

**Code removed**: two `os.rename()` paths, two `os.remove()` cleanup blocks in error handlers

---

### 6. CSV Result Log: Per-Day File + Lot Markers (June 4)

**Previous**: One file per lot run — `result_{lot}_{YYYYMMDD_HHMMSS}.csv`

**New**: One file per calendar day — `result_YYYYMMDD.csv`; all lots for that day appended sequentially

Each lot block:
```
# --- LOT_START ---
LOT_NUMBER, <lot>
PACKAGE, <pkg>
START_TIME, ...
MODE, ...
HW_PORTS, lighting=<port> cellcon=<port>
<column header>
<inspection rows…>
PASS_CT, ...
FAIL_CT, ...
YIELD_PCT, ...
END_TIME, ...
DURATION_S, ...
# --- LOT_END ---
<blank line>
```

**Log rotation changed**: from count-based (delete oldest N files) to age-based (delete files older than `LOG_RETENTION` days using `datetime.strptime` against filename date). Legacy filenames that don't match the date pattern are silently skipped. Default retention raised from 365 → 730 days.

---

### 7. Disk Space Warning (June 4)

- Every 500 cycles, `RunWorker` checks free disk space via `shutil.disk_usage(out_dir).free`
- If below `DISK_WARN_MB` (default 200 MB): emits `sig_warn` → displayed via `_show_error()` banner in UI
- Warning is soft — run continues

---

### 8. Code Cleanup (June 2 + June 4–5)

- `_save_cell_crops()` and `Inspector._check_ic()` both now use shared `_safe_crop(image, cx, cy, cw, ch)` helper instead of inline clamp math
- Logger: four duplicate `print(f"[Logger] ... failed")` strings replaced with `_log_error(operation, exc)` method
- `_build_ui()`: two repeated QFrame+QVBoxLayout+QLabel setup blocks extracted to `_make_section_frame(title, spacing, obj_name)` helper
- Duplicate camera log lines removed from `Camera` class
- `Camera.reconnect()` and `Camera.is_healthy()` removed (see §4)

---
