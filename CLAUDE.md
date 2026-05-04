# Clear Package IC Inspection

## Problem

ICs with a clear (transparent) mold make laser-engraved marks very difficult to inspect visually.

**Goal:** For each IC in the image — **IS THE MARK COMPLETE? (PASS / FAIL)**

---

## Hardware

| Item | Detail |
|---|---|
| Camera | Basler (via Pylon SDK) |
| Lighting | Coaxial — no image pre-processing needed |
| Controller | Raspberry Pi 5 |
| IO | Raspberry Pi GPIO (BCM numbering) |

---

## Runtime Flags (hardcoded in config)

These are set at the top of the config file before running. Not changeable from the UI.

| Flag | Values | Effect |
|---|---|---|
| `DEBUG` | `True / False` | Enables debug mode with detailed logs |
| `CAMERA` | `"camera" / "directory"` | Live Basler feed or load images from a folder (for dev/testing) |
| `IO` | `True / False` | If `False`, log IO signals as messages instead of driving GPIO pins |

## Modes (hardcoded in config)

| Mode | Description |
|---|---|
| `RUN` | Production mode — minimal logging, full IO, max speed |
| `DEBUG` | Development mode — verbose logs, annotated output, IO flag respected |

---

## Mark Specification

- **Type:** Laser engraved
- **Content:** Numbers and alphabets
- **Layout per IC:** 3 rows × 2 columns = **6 ROI cells**
  - Left column and right column have a horizontal offset between them
  - Gap between columns is a configurable offset parameter
- **ROI cell size:** Dynamic — derived from a scale/ratio parameter set in template setup

---

## Image Layout

Each image contains **two ICs**. IC_A is the template anchor; IC_B position is derived from a fixed pixel offset relative to IC_A.

```
┌──────────────────────────────────────────────┐
│   IC_A (anchor)        │   IC_B (fixed offset)│
│                        │                      │
│  Col_L    Col_R        │  Col_L    Col_R       │
│  ┌───┐    ┌───┐        │  ┌───┐    ┌───┐      │
│  │R1 │    │R1 │        │  │R1 │    │R1 │      │
│  ├───┤    ├───┤        │  ├───┤    ├───┤      │
│  │R2 │    │R2 │        │  │R2 │    │R2 │      │
│  ├───┤    ├───┤        │  ├───┤    ├───┤      │
│  │R3 │    │R3 │        │  │R3 │    │R3 │      │
│  └───┘    └───┘        │  └───┘    └───┘      │
└──────────────────────────────────────────────┘
```

- Total ROI cells per image: **12** (2 ICs × 2 columns × 3 rows)
- IC_B anchor = IC_A anchor + `(offset_x, offset_y)` — set once in template

---

## Detection Logic

- **Model:** YOLO via OpenVINO (model ready)
- **Post-processing:** NMS (Non-Maximum Suppression) applied after OpenVINO inference to filter overlapping bounding boxes before per-cell evaluation
- **Classes:** `letter`, `number`
- **Per ROI cell result:**
  - Mark detected → `TRUE` + class label
  - No mark → `FALSE` (None)
- **Per IC result:**
  - **PASS** — all 6 cells return TRUE
  - **FAIL** — any cell returns FALSE → trigger output + save images + log

---

## GPIO Pin Assignment (BCM)

**Inputs — Machine → Pi**

| Pin | BCM | Direction | Description |
|---|---|---|---|
| `START_PIN` | GPIO 17 | IN (pull-down) | Rising edge = Start inspection / Next image |
| `DONE_PIN` | GPIO 27 | IN (pull-down) | Rising edge from machine = Stop gracefully, return to standby |

**Outputs — Pi → Machine**

| Pin | BCM | Direction | Description |
|---|---|---|---|
| `ACK_PIN` | GPIO 22 | OUT | Pulse HIGH when result is ready — machine reads result pins on this edge |
| `RESULT_PIN` | GPIO 23 | OUT | HIGH = PASS, LOW = FAIL (held until next ACK pulse) |
| `FAIL_A_PIN` | GPIO 24 | OUT | HIGH = IC_A failed (held until next ACK pulse) |
| `FAIL_B_PIN` | GPIO 25 | OUT | HIGH = IC_B failed (held until next ACK pulse) |

> `DONE_PIN` (IN) replaces the old STOP_PIN — machine controls when the system returns to standby.
> Old output `DONE_PIN` is now `ACK_PIN` (OUT) — the Pi's "result ready" signal to the machine.
> When `IO = False`: all GPIO state changes are mocked — logged as `[IO MOCK] PIN → STATE` instead of driving physical pins.

---

## IO Signal Flow

```
STANDBY STATE
  → Waiting for START_PIN

Machine sends START_PIN (rising edge)
  → If BUSY = True: drop signal, log "[IO] START ignored — busy"
  → Set BUSY = True
  → Capture image
  → Run inspection on IC_A and IC_B
  → Set RESULT_PIN  (PASS=HIGH / FAIL=LOW)
  → Set FAIL_A_PIN  (HIGH if IC_A failed, else LOW)
  → Set FAIL_B_PIN  (HIGH if IC_B failed, else LOW)
  → Pulse ACK_PIN HIGH  ← machine reads all output pins on this edge
       [IO=False: log "[IO MOCK] RESULT=X  FAIL_A=X  FAIL_B=X  ACK→HIGH"]
  → Hold output pins
  → Set BUSY = False
  → Wait for next START_PIN  ──or──  DONE_PIN

Machine sends DONE_PIN (rising edge)
  → If BUSY = True: wait for current cycle to finish first
  → Clear RESULT_PIN, FAIL_A_PIN, FAIL_B_PIN (all LOW)
  → Log "Returning to standby"
  → Return to STANDBY STATE
```

> **Multi-IC simultaneous fail:** `FAIL_A_PIN` and `FAIL_B_PIN` are both set before `ACK_PIN` pulses — machine reads both on the same rising edge. `IO=False` logs both in one message.

---

## Performance Requirement

| Metric | Target |
|---|---|
| Full inspection per image (both ICs) | **< 1000 ms** on Raspberry Pi 5 |

---

## Inspection Flow

### Setup Mode (one-time per product type)
1. Capture reference image (via camera or load from file)
2. Click to set **IC_A anchor** (top-left corner of IC_A)
3. Set **scale/ratio** → auto-compute 6 ROI cell positions for IC_A
4. Set **column offset** (horizontal shift between Col_L and Col_R)
5. Set **IC_B offset** (x, y pixel offset from IC_A anchor)
6. Preview all 12 ROI boxes overlaid on image — adjust if needed
7. Save template

### RUN / DEBUG Inspection Loop
```
Receive START_PIN (or manual trigger in DEBUG)
  → If BUSY: drop signal
  → Set BUSY = True
  → Capture image  [or load next image from directory if CAMERA="directory"]
  → Map 12 ROI cells using saved template
  → Run YOLO/OpenVINO inference on all 12 cells → apply NMS
  → Per cell: TRUE+class or FALSE (raise MarkMissingError if any FALSE)
  → IC_A: PASS if all 6 TRUE, else FAIL
  → IC_B: PASS if all 6 TRUE, else FAIL
  → Set RESULT_PIN, FAIL_A_PIN, FAIL_B_PIN → Pulse ACK_PIN
       [IO=False: log "[IO MOCK] RESULT=X FAIL_A=X FAIL_B=X ACK→HIGH"]
  → If any FAIL:
      - Save IMAGE_ID_R.png  (original image)
      - Save IMAGE_ID.png    (annotated: ROI boxes, failed cells in red, class labels)
      - Log entry (see Logging section)
  → Update UI (result banner, stats)
  → Set BUSY = False
  → If STOP_PIN received: end loop
  → Else: wait for next START_PIN
```

---

## File Output (FAIL only)

| File | Content |
|---|---|
| `IMAGE_ID_R.png` | Raw original image, no annotation |
| `IMAGE_ID.png` | Annotated: all ROI boxes drawn, failed cells highlighted red, detected class labels shown |

`IMAGE_ID` format: `YYYYMMDD_HHMMSS_NNN` (timestamp + sequential counter)

---

## Logging

Every inspection appends one record to the log file.

**Log fields:**
| Field | Detail |
|---|---|
| `timestamp` | ISO 8601 |
| `image_id` | Links to saved files |
| `ic_a_result` | PASS / FAIL |
| `ic_a_missing` | List of `[row, col]` cells where mark was absent (empty if PASS) |
| `ic_b_result` | PASS / FAIL |
| `ic_b_missing` | List of `[row, col]` cells where mark was absent |
| `cycle_time_ms` | Full inspection duration |
| `mode` | RUN / DEBUG |
| `io_mock` | `True` if IO=False (signals were mocked, not sent) |

In **DEBUG mode**: also log per-cell confidence scores, raw YOLO detections, and NMS input/output counts.

When `IO = False`: each signal change appended as a separate log line:
```
[IO MOCK] RESULT_PIN → HIGH
[IO MOCK] FAIL_A_PIN → LOW
[IO MOCK] FAIL_B_PIN → HIGH
[IO MOCK] ACK_PIN → HIGH (pulse)
```

### Log Rotation

| Setting | Value |
|---|---|
| Rotation | Daily (new file each day at midnight) |
| Retention | 1 year (365 log files max, oldest deleted automatically) |
| Filename | `inspect_YYYYMMDD.log` |
| Location | `logs/` |

---

## Frontend (Single Page Application — PyQt)

### Framework
- **PyQt5** (or PyQt6)
- Single `QMainWindow`, no navigation between pages
- All panels in one layout — no separate windows except modal popups

### Design System

**Color Palette**
| Role | Color | Hex |
|---|---|---|
| Accent / highlight | Cyan | `#00BCD4` |
| Panel background | Grey Blue | `#546E7A` |
| Primary surface | Steel Blue | `#4472C4` |
| Text / base | White | `#FFFFFF` |
| PASS indicator | Cyan | `#00BCD4` |
| FAIL indicator | Red (standard) | `#EF5350` |
| Error banner | Red (standard) | `#EF5350` |

**Component Style**
- All containers: `QFrame` with `border-radius: 8px` (rounded edges)
- Buttons: rounded rectangles, Steel Blue fill, White text, Cyan hover
- Input fields: Grey Blue background, White text, Cyan focus border
- No decorative icons, no gradients — flat and clean
- Consistent 8px padding inside all panels

### Layout

```
┌────────────────────────────────────┬─────────────────────┐
│  MAIN VIEW  (Grey Blue bg)         │  RIGHT PANEL        │
│                                    │  (Steel Blue bg)    │
│  Live camera / last image          │                     │
│  Overlay: ROI boxes (all 12)       │  [Setup]            │
│  Overlay: detection labels (cyan)  │  Exposure time      │
│  Overlay: failed cells (red)       │  Scale / ratio      │
│                                    │  Column offset      │
│  ┌──────────┐  ┌──────────┐        │  IC_B offset        │
│  │ IC_A     │  │ IC_B     │        │  Set anchor btn     │
│  │ PASS/FAIL│  │ PASS/FAIL│        │  Preview ROIs btn   │
│  └──────────┘  └──────────┘        │  Save template btn  │
│  (rounded badge, cyan or red)      │                     │
│                                    │  [Controls]         │
│  [ERROR BANNER — red, rounded]     │  Manual trigger btn │
│                                    │                     │
│                                    │  [Stats]            │
│                                    │  Pass / Fail count  │
│                                    │  Error count        │
│                                    │  Last cycle (ms)    │
└────────────────────────────────────┴─────────────────────┘
```

### Modal Popup on FAIL
- Rounded `QDialog`, Steel Blue background
- Title: "Inspection Failed"
- List which IC (A / B) failed
- List missing cells as `[row, col]` per IC
- Single "Acknowledge" button (Cyan, rounded)

---

## Exception Design — Throw-back on False

Detection failures are raised as exceptions, not returned as values. This makes the FAIL path explicit and prevents silent misses.

### Custom Exception Hierarchy

```python
InspectionError              # base for all inspection exceptions
├── MarkMissingError         # raised when any ROI cell returns FALSE
│     .ic_position           # "A" or "B"
│     .missing_cells         # list of [row, col] that failed
├── SystemError              # base for hardware/config failures
│   ├── CameraError          # capture failed (timeout, disconnect)
│   ├── ModelError           # OpenVINO load or inference failed
│   ├── TemplateError        # template file missing or corrupt
│   └── GPIOError            # pin init or write failed (IO=True only)
└── ConfigError              # invalid flag values on startup
```

### How throw-back works

```
detector.py     →  raises MarkMissingError  if any cell is False
roi.py          →  raises TemplateError     if template cannot be loaded
camera.py       →  raises CameraError       after N retries
main.py         →  catches all and routes:
                     MarkMissingError  → FAIL path (signal + save + log)
                     CameraError       → ERROR path (stop loop, alert UI)
                     ModelError        → ERROR path (stop loop, alert UI)
                     TemplateError     → ERROR path (block inspection start)
                     GPIOError         → ERROR path (stop loop, alert UI)
                     ConfigError       → abort startup
```

---

## Error Handling

### Startup Errors (fatal — abort before entering loop)

| Error | Cause | Action |
|---|---|---|
| `ConfigError` | Invalid flag value in config.py | Print error, exit |
| `TemplateError` | No template saved for current product | Block RUN mode, force Setup first |
| `ModelError` | OpenVINO model file missing or incompatible | Print error, exit |
| `CameraError` | Basler not found at startup (CAMERA="camera") | Print error, exit |
| `GPIOError` | GPIO init failed (IO=True) | Print error, exit |

### Runtime Errors (per-cycle — recoverable where possible)

| Error | Cause | Action |
|---|---|---|
| `CameraError` | Capture timeout or disconnect mid-run | Retry 2× with 200ms delay → if still fails: set RESULT_PIN LOW, pulse DONE_PIN, log `CAMERA_ERROR`, alert UI, pause loop |
| `MarkMissingError` | Any cell returns FALSE (expected result) | FAIL path: set output pins, save images, log missing cells, update UI |
| `ModelError` | Inference crashed mid-run | Log `MODEL_ERROR`, set RESULT_PIN LOW, pulse DONE_PIN, alert UI, stop loop |
| `GPIOError` | Pin write failed mid-run | Log `GPIO_ERROR`, continue inspection but flag IO as unreliable |
| Log write failure | Disk full or permission error | Print to stderr, do not crash inspection loop |
| Cycle time > 1000ms | Inference slower than target | Log warning with actual time; do not stop — machine still gets DONE signal |

### Error State in UI

- Display a red **ERROR banner** with short description in the main view
- Sidebar shows last error message + timestamp
- Stats counter tracks error count separately from FAIL count

### Error Logging Fields (appended to main log)

```
timestamp, event=ERROR, error_type, error_message, cycle_time_ms
```

---

## Startup Sequence

```
1. Load and validate config.py → raise ConfigError if invalid
2. Verify template file exists  → raise TemplateError if missing
3. Load OpenVINO model          → raise ModelError if load fails
4. Init GPIO pins (if IO=True)  → raise GPIOError if init fails
5. Open camera / directory      → raise CameraError if not accessible
6. Warm up inference (1 dummy pass to load model into cache)
7. Log "System ready" + mode + flags
8. Pulse ACK_PIN HIGH (system ready indicator to machine)
9. Enter inspection loop
```

---

## Graceful Shutdown

Triggered by: `DONE_PIN` rising edge (from machine), `Ctrl+C`, or UI stop button.

```
1. Finish current inspection cycle (do not abort mid-inference)
2. Set RESULT_PIN LOW, FAIL_A_PIN LOW, FAIL_B_PIN LOW, ACK_PIN LOW
3. Flush log buffer to disk
4. Release camera
5. Release GPIO
6. Log "System stopped cleanly" + total pass/fail/error counts
7. Return to STANDBY STATE  (or exit if Ctrl+C / crash)
```

On unhandled crash (unexpected exception):
```
1. Log full traceback
2. Attempt GPIO release
3. Exit with non-zero code
```

---

## Project Structure (suggested)

```
ClearIC_Inspect/
├── config.py              # FLAGS: DEBUG, CAMERA, IO, MODE + GPIO pin constants
├── main.py                # Entry point + startup sequence + graceful shutdown
├── exceptions.py          # Custom exception hierarchy (InspectionError subclasses)
├── inspection/
│   ├── detector.py        # YOLO/OpenVINO inference — raises MarkMissingError on FALSE
│   ├── roi.py             # Template load, ROI mapping, cell extraction
│   └── result.py          # PASS/FAIL logic, result dataclass
├── io_handler/
│   ├── gpio_handler.py    # GPIO read/write; respects IO flag; raises GPIOError
│   └── camera.py          # Basler capture or directory loader; retry 2× on fail; raises CameraError
├── ui/
│   └── app.py             # Frontend (single page)
├── logger/
│   └── log.py             # Structured log writer (inspection + error records)
├── output/                # Saved FAIL images (auto-created)
├── logs/                  # Log files (auto-created)
├── templates/             # Saved product templates
└── models/                # OpenVINO model files
```

---

## Still Open

- [ ] Startup ACK_PIN pulse — confirm machine expects this as "ready" indicator on boot
