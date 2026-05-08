# ClearIC Inspect — Weekly Report
**Week 19 · 2026-05-08**

---

## Project Overview

Automated pass/fail inspection of laser-engraved marks on **clear-mold ICs** using a Basler camera + YOLO (OpenVINO) on Raspberry Pi 5.  
Each image contains **two ICs (IC_A and IC_B)**. Each IC has a **3-row × 2-column = 6-cell ROI grid**.  
System checks all 12 cells per image and signals results via GPIO.

---

## What It Can Do

| Capability | Detail |
|---|---|
| IC Detection (setup) | YOLO auto-detects IC bounding boxes from a reference image |
| Template Creation | One-click "Auto Detect" saves IC_A + IC_B bounding box positions (used to define the 3×2 cell grids); strip patches also saved but not yet active at runtime |
| Mark Inspection | YOLO (OpenVINO, CPU) checks all 12 ROI cells per image for laser-mark presence |
| Real-time UI | Live annotated image, IC_A/IC_B PASS/FAIL badges, cycle time, pass/fail/error counters |
| FAIL Popup | Modal dialog listing which IC and which cells (row/col) failed — requires acknowledgement |
| GPIO Signaling | Sets FAIL_A_PIN / FAIL_B_PIN HIGH and pulses ACK_PIN when result is ready |
| Image Logging | Saves raw + annotated images to dated output folders on any FAIL |
| JSON Log | Appends one record per inspection to a daily rotating log file (365-day retention) |
| Directory Mode | Runs from a folder of images instead of live camera (dev/testing) |
| Mock IO | When IO=False, all GPIO signals are printed instead of driving pins |
| Graceful Shutdown | Finishes current cycle, clears GPIO, releases camera on Stop / DONE_PIN / window close |

---

## Data Flow & Processing

```
[Camera / Image folder]
        │
        ▼
   Camera.grab()        ← BGR ndarray
        │
        ▼
   Inspector.inspect()
    │
    ├─ Phase 1 — Locate ICs  [current: YOLO only]
    │     Detector.locate_ics()
    │       class 0 = IC_Presence, two largest boxes → sorted left→right
    │     Template JSON → compute cell grids (fallback if YOLO misses one IC)
    │     ──────────────────────────────────────────────────────────────────
    │     [built, disabled] TemplateMatcher
    │       bilateral filter → Otsu threshold → strip template match
    │       IC_B derived from saved dx/dy offset — matcher=None in _start_run()
    │
    ├─ Phase 2 — Build cell grids
    │     _build_cells(): shrink IC rect (×0.95) → slice 3×2 grid
    │                     expand each cell (×1.20) for overlap
    │
    ├─ Phase 3 — YOLO text detection (full image, once)
    │     Detector.detect_full_image()
    │       letterbox → blob → OpenVINO infer → NMS → class 1 (Text) boxes
    │
    └─ Phase 4 — Cell evaluation
          dominant-overlap assignment + 5% guard band
          cell has hit → PRESENT; no hit → ABSENT
          any ABSENT → MarkMissingError raised
        │
        ▼
   RunWorker (QThread)
    ├─ PASS  → GPIO: FAIL_A=LOW, FAIL_B=LOW, pulse ACK
    ├─ FAIL  → GPIO: set FAIL pins, pulse ACK; save raw+annotated images; open FailDialog
    └─ ERROR → alert UI banner, stop loop
        │
        ▼
   MainWindow (PyQt5)
        badges / stats / image view updated via Qt signals
```

---

## User Setup (Non-Developer Steps)

### One-Time Template Setup (per product type)

1. Place a **reference image** in the input folder — or connect the Basler camera.
2. Open the app. The first image loads automatically.
3. Click **"Auto Detect"** in the Setup panel.
   - A popup shows the detected IC_A (yellow box) and IC_B (cyan box).
   - If wrong, click **Retry** to cycle through candidates.
4. Click **Confirm** when both boxes are correct.
   - Template is saved to `templates/template.json`
   - Strip patches saved to `templates/tmpl_top.npy` / `tmpl_bot.npy`
   - Preview image saved to `templates/template_preview.png`
5. Optionally adjust **Exposure (µs)** before confirming (written into the template).

### Running Inspection

1. Click **Start** — the system begins the inspection loop.
2. In **camera mode**: system waits for GPIO `START_PIN` per cycle.
3. In **directory mode**: system auto-loops through all images in the input folder.
4. Badges show **PASS / FAIL** per IC after each cycle.
5. On FAIL: a popup appears listing failed cells → click **Acknowledge**.
6. Click **Stop** (or machine sends `DONE_PIN`) to return to standby.

### What Users Do NOT Need to Touch

- Dev flags (`DEBUG`, `IO`, `MODE`, `DIR_INPUT`) — set by developer in code.
- Model files — already in `ClearIC_Insp_openvino_model/`.
- Log files — auto-created in `logs/`, auto-rotated daily.
- Output images — auto-saved in `Output/YYYYMMDD/`.

---

## Inspection Classes

| Class | Role |
|---|---|
| `Detector` | OpenVINO YOLO wrapper — runs inference, applies NMS, returns Text / IC_Presence boxes |
| `TemplateMatcher` | Locates IC_A per-frame via bilateral-filtered strip template matching; IC_B derived from offset |
| `TemplateManager` | Load/save template JSON + strip patches + preview image; computes cell grids from saved rects |
| `Inspector` | Orchestrates Phase 1 (locate) → Phase 2 (build cells) → Phase 3 (detect) → Phase 4 (evaluate); raises `MarkMissingError` on fail |
| `Camera` | Basler Pylon or directory image source — uniform `grab()` interface |
| `RaspberryIO` | BCM GPIO handler; mock-logs when `IO=False` |
| `RunWorker` | QThread loop — waits for trigger, calls Inspector, emits Qt signals, handles GPIO, saves files |
| `Logger` | Appends JSON-lines records to daily rotating log files |
| `MainWindow` | Single-page PyQt5 UI — image view, badges, stats, error banner |
| `FailDialog` | Modal FAIL popup listing missing cells by IC and row/col |

### Exception hierarchy (inspection path)

```
InspectionError
├── MarkMissingError   ← expected fail: any ROI cell has no detection
├── _SystemError
│   ├── CameraError    ← capture failure
│   ├── ModelError     ← OpenVINO load / inference failure
│   ├── TemplateError  ← template missing, corrupt, or match score too low
│   └── GPIOError      ← pin init / write failure
└── ConfigError        ← invalid Config.json values
```

---

## Key Parameters (in code)

| Parameter | Value | Effect |
|---|---|---|
| `_CELL_SHRINK` | 0.95 | IC rect shrunk 5% before grid slicing |
| `_CELL_EXPAND` | 1.20 | Each cell expanded 20% for overlap with neighbours |
| `_COL_GAP_PCT` | 40% | Gap between left and right columns (% of IC width) |
| `_CELL_GUARD` | 0.05 | 5% guard band — boxes clipping only the cell border are not credited |
| `_INPUT_SIZE` | 640 px | YOLO letterbox input size |
| `CONF_THR` | 0.5 (config) | Minimum detection confidence |
| `NMS_IOU_THR` | 0.45 (config) | NMS IoU threshold |

---

## Current Status

- [x] Single-file implementation (`CLearIC.py`)
- [x] Auto Detect + template creation flow
- [x] YOLO-based IC localization + cell evaluation
- [x] TemplateMatcher (bilateral strip matching) built — currently disabled at runtime (YOLO active)
- [x] GPIO signaling with mock mode
- [x] Daily log rotation, image output
- [x] PyQt5 UI with live image, badges, stats
- [ ] Startup ACK_PIN pulse — confirm machine expects "ready" signal on boot
