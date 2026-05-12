# ClearIC Inspect

**Goal:** Classify laser-engraved marks on clear-mold ICs → PASS / FAIL per IC.

## Stack
- Camera: Basler via Pylon SDK
- Controller: Raspberry Pi 5, GPIO BCM
- UI: PyQt5 single `QMainWindow`
- Model: YOLO-cls via OpenVINO, 2-class (`0=NoText`, `1=Text`), 224×224 input
- Model path: `Text_cls-2/best_openvino_model/best.xml`

## Config Flags (top of CLearIC.py, not UI-changeable)
| Flag | Values | Effect |
|---|---|---|
| `DEBUG` | `True/False` | Verbose logs + annotated output |
| `CAMERA` | `"camera"/"directory"` | Live Basler or load from `Input/` |
| `IO` | `True/False` | Drive GPIO or mock (log `[IO MOCK] PIN→STATE`) |
| `MODE` | `RUN/DEBUG` | Production or dev |
| `MODEL_PATH` | path | OpenVINO `.xml` |

## Image Layout
Each image has 2 ICs. IC_A = anchor; IC_B = IC_A + `(offset_x, offset_y)`.  
Each IC: 3 rows × 2 columns = 6 ROI cells. Total: 12 cells/image.  
Left/right columns have a configurable horizontal offset.

## Inspection Logic
Per ROI cell: crop → resize 224×224 → classify → `Text=TRUE` / `NoText=FALSE`.  
Per IC: PASS if all 6 TRUE; FAIL if any FALSE → set pins + save images + log.

## GPIO (BCM)
| Signal | Pin | Dir | Description |
|---|---|---|---|
| `START_PIN` | 17 | IN↓ | Rising edge → start inspection |
| `DONE_PIN` | 27 | IN↓ | Rising edge → return to standby |
| `ACK_PIN` | 22 | OUT | Pulse HIGH when result ready (machine reads on this edge) |
| `FAIL_A_PIN` | 24 | OUT | HIGH = IC_A failed |
| `FAIL_B_PIN` | 25 | OUT | HIGH = IC_B failed |

IO flow: START → capture → inspect → set FAIL_A/B → pulse ACK → wait for DONE.  
DONE clears all output pins → standby. If BUSY on START: drop signal.

## Performance
Full inspection (both ICs) < 1000 ms on Pi 5.

## Setup Flow (one-time per product)
1. Capture/load reference image
2. Click to set IC_A anchor (top-left)
3. Set scale/ratio → auto-compute 6 ROI positions
4. Set column offset (Col_L ↔ Col_R horizontal shift)
5. Set IC_B offset (x, y from IC_A anchor)
6. Preview 12 ROI boxes → adjust → save template

## Inspection Loop
START_PIN (or Manual Trigger) → BUSY check → capture image →  
Phase 1: TemplateMatcher bilateral-strip → IC_A rect → IC_B by offset (fallback: fixed template coords) →  
Phase 2: classify 12 cells → PASS/FAIL per IC →  
set FAIL_A_PIN, FAIL_B_PIN → pulse ACK_PIN →  
on any FAIL: save `IMAGE_ID_R.jpg` (raw) + `IMAGE_ID.jpg` (annotated, red=fail) + log →  
update UI → DONE_PIN (DEBUG directory mode: auto-fire → next image).

`IMAGE_ID` format: `YYYYMMDD_HHMMSS_NNN`

## File Output (FAIL only)
- `IMAGE_ID_R.jpg` — raw
- `IMAGE_ID.jpg` — annotated (green=Text, red=NoText)

## Logging
JSON-lines, one record/inspection. Fields: `timestamp, image_id, ic_a_result, ic_a_missing, ic_b_result, ic_b_missing, cycle_time_ms, mode, io_mock`.  
DEBUG adds per-cell class + confidence.  
Rotation: daily → `logs/inspect_YYYYMMDD.log`, 365-file retention.

## Exceptions
```
InspectionError
├── MarkMissingError   .ic_position, .missing_cells  → FAIL path
├── CameraError                                       → ERROR, stop loop
├── ModelError                                        → ERROR, stop loop
├── TemplateError                                     → block RUN, force Setup
├── GPIOError          startup=fatal; runtime=flag unreliable, continue
└── ConfigError                                       → abort startup
```

## Startup Sequence
1. Validate config → `ConfigError`
2. Load template → `TemplateError`
3. Load OpenVINO model → `ModelError`
4. Init GPIO (if IO=True) → `GPIOError`
5. Open camera/directory → `CameraError`
6. Warm-up inference (1 dummy pass)
7. Log "System ready"
8. Pulse ACK_PIN HIGH (ready signal)
9. Enter loop

## Shutdown
Finish current cycle → all output pins LOW → flush log → release camera + GPIO → log counts → exit.  
On crash: log traceback → release GPIO → exit nonzero.

## UI
Single `QMainWindow`, no separate windows except modal dialogs.

**Colors:** window bg `#5465FF` · panel bg `#788BFF` · card bg `#9BB1FF` · PASS `#BFD7FF` · image area `#E2FDFF` · FAIL/error `#EF5350` · white `#FFFFFF`

**Style:** `QFrame border-radius:8px`, buttons `#5465FF` fill white text no hover, inputs `QLineEdit` white bg `#5465FF` border, 8px padding, flat/no gradients.

**Layout:** left = image view + IC_A/IC_B PASS/FAIL badges + stats; right panel = Setup inputs (exposure, scale, col offset, IC_B offset X/Y, Set Anchor, Preview ROIs, Save Template) + Controls (Manual Trigger).

**FAIL popup:** `QDialog` `#5465FF` bg, lists failed IC + missing cells, single Acknowledge button.

## Single-File Rule
All code in `CLearIC.py`. No modules.

## CLearIC.py Section Order
```
# Config Flags / Stage & Error Flags / Exceptions / Image / Camera
# RaspberryIO / Detector / Cell Grid / TemplateManager / TemplateMatcher
# Inspector / Logger / Stylesheet / FailDialog / ImageView
# SetupDialog / RunWorker / MainWindow / Entry Point
```

## File Structure
```
ClearIC_Inspect/
├── CLearIC.py
├── Text_cls-2/best_openvino_model/   # best.xml + best.bin
├── Output/                           # FAIL images
├── logs/
├── templates/
└── Input/                            # CAMERA="directory" source images
```

## Open Items
- Confirm machine expects ACK_PIN pulse on boot as "ready" signal
- Auto-detect IC position for Setup (currently manual anchor click)
