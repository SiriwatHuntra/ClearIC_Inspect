Weekly MD
15-06-26 to 18-06-26


Camera / USB reliability
 - Camera self-check pass
 - USB watchdog standalone + init improvement
 - Store detected port in lighting controller init

Lighting
 - Add light brightness adjust section
 - Fix light toggle bug
 - Hook lighting health check into run() loop
 - Debounce brightness UI: 80ms hardware-write timer, 400ms TOML-save timer (kills slider-drag stutter)
 - Replace brightness slide bar with Increase/Decrease arrow buttons, 5-unit step, range unchanged (0-255)

Inspection pipeline
 - Add "no-work" category for template-not-found (TemplateError) — record as NG, run keeps going instead of halting
 - Downscale contour search (_CONTOUR_DOWNSCALE 2->3), store save-time downscale in template.json so old templates keep matching
 - Replace sequential per-cell classifier inference with AsyncInferQueue, pre-allocated output slots (verified bit-identical to old sync path)

Diagnostics / UI
 - Show Pi IP address in window title
 - Re-detect Hardware status: one line per device (Camera/Light/CellCon) with red/green status dot, was single pipe-joined text line

Lot number / CellCon flow
 - Debug mode: skip CellCon probe entirely when no CellCon detected at hardware-detection time, go straight to manual lot popup (avoids ~5s blocking delay)
 - Production mode: distinct alert for "CellCon not found on any COM port" vs "Lot number not found"

Misc
 - Update .gitignore

IFWFOCR02.py (new OCR app, rewrite of IFWFOCR01.py)
 - Code cleanup: unused imports/globals marked "[UNUSED]" instead of deleted, dead commented-out blocks (old SQL lot lookup, old serial init) removed
 - Config: switched from CameraConfig.txt to config.json with JSON validation; invalid/missing config now shows an error dialog instead of silently exit()
 - Camera: resolution corrected to actual sensor native (640x480, was oversized 1280x960); camera init wrapped in try/except with error dialog instead of crashing; get_frame() returns None safely on failure instead of raising
 - COM port detection (checkComPort): rewritten so a flaky/slow device on one port can't block detection of the other (independent try/except per port); no longer hard-requires exactly ttyUSB0+ttyUSB1 both present; returns (cellconPort, lightPort) directly
 - Lighting control UI: replaced QDial with QSlider + dedicated Up/Down step buttons (+/-5 per click) — same brightness-step concept ClearIC.py later adopted
 - Lighting timeouts: lightOn/lightOff/dial_method busy-wait loops now cap at 3s with a logged warning instead of looping forever
 - Add NTP time sync on startup (restarts systemd-timesyncd, polls timedatectl up to 15s) — replaces shelling out to a separate SyncTime.py script
 - Add IP address display (hostname -I with UDP-socket fallback) in the UI
 - Network calls (ReadMark/CreateRecord API) wrapped in try/except with timeout=10 and error dialogs instead of unhandled exceptions; getLotNumFromCellcon guards on serial port not initialized
 - Add closeEvent() override so closing via the title-bar X also stops the lighting-read thread (previously only the Exit button did, leaving it running forever)
 - Add setFittedScene() to fit the view to actual frame size with aspect ratio preserved (camera resolution no longer matches the view's original fixed design size)
