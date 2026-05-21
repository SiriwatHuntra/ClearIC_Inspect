# WSOFOCR — Reference Index
> Machine: V-35 (IFLV) | Platform: Raspberry Pi | Created: 2026-05-21

---

## 1. Project Overview

| Item | Value |
|---|---|
| Purpose | OCR marking verification on semiconductor production line |
| Machine | IFLV, number V-35 |
| Camera | Basler Pylon, serial 25107895, 1280×960 Mono8 |
| Exposure | 25 000 µs |
| UI | PyQt5, fullscreen 1920×1050, UI file: `dialog1.ui` |
| Config | `config.json` → `{CamID, MCType, MCNum}` |
| Log | `log.log` (INFO level, format: `LEVEL datetime message`) |

---

## 2. File Layout

```
WSOFOCR/
├── IFWFOCR01.py        ← Main application
├── dialog1.ui          ← Qt Designer UI
├── config.json         ← Machine config (JSON)
├── lighting.dat        ← Pickle: lighting level 0-255
├── SyncTime.py         ← Time sync (called at startup)
├── log.log             ← Runtime log
├── cropimg.jpg         ← Last OCR image (overwritten each cycle)
├── Capture/            ← Manual captures: {ddmmyyyy-HHMMSS}.jpg
├── OCR/                ← Result screenshots: {lot_no}_{ddmmyyyy}_{HHMMSS}.jpg
├── Teaching/           ← Teaching mode ROI data
└── NGPIC/              ← NG (fail) images
```

---

## 3. Class Structure

### `MyApp(QMainWindow)`
Main application window. Handles all UI, camera display, OCR flow, lighting, and serial communication.

| Method | Trigger | Purpose |
|---|---|---|
| `__init__` | Startup | Init UI, load config, open camera, setup serial, load lighting |
| `live()` | btnLive click | Toggle live camera feed (QTimer 50 ms) |
| `capture()` | btnCapture click | Grab and save image to `Capture/` |
| `OCRread()` | btnOCR click | Start OCR cycle: light on, grab image, get lot from cellcon |
| `OCRsending()` | btnOCRSend click | Validate, compare mark, send to API, save result |
| `OCRCancel()` | btnOCRCancel click | Cancel OCR, hide panel, light off |
| `getLotNumFromCellcon()` | called by OCRread | Query cellcon reader for current lot |
| `checkComPort()` | Startup | Auto-detect which USB port is cellcon vs lighting |
| `lightOn()` | various | Send ON command to lighting controller |
| `lightOff()` | various | Send OFF command to lighting controller |
| `lightSetup(val)` | startup, dial | Set brightness 0-255 |
| `readSerialLighting()` | thread | Listen for lighting ACK, clears `lightBusy` flag |
| `keyboard(key)` | on-screen keys | Handle on-screen keyboard input to textboxes |
| `applicationClose()` | btnExit click | Stop threads, log close, exit |
| `updateImg()` | QTimer (5 sec) | Restore plain camera image after OCR result display |

### `MyVideoCapture`
Basler Pylon camera wrapper.

| Method | Purpose |
|---|---|
| `__init__(serial)` | Open camera, set resolution/exposure/format, start grabbing |
| `get_frame()` | Retrieve latest frame as numpy array (grayscale) |
| `__del__` | Close camera on object destroy |

---

## 4. OCR Feature — Full Flow

### Operator Steps
```
1. Scan lot barcode label with cellcon reader (physical action)
2. Press [OCR] button on screen
3. Type marking string + 6-digit operator number
4. Press [Send] button (appears when both fields are non-empty)
```

### Program Steps
```
OCRread()
  ├─ Stop live view if running
  ├─ Light ON → wait 300 ms
  ├─ Wait for lightBusy == False (serial ACK, max 3 sec)
  ├─ Grab camera frame × 3 (flush stale frames)
  ├─ Display image on screen
  └─ getLotNumFromCellcon()
       ├─ Send "LA\r\n" to self.ser (cellcon port)
       ├─ Read up to 5 lines, look for "LS" prefix
       ├─ Parse CSV: part = msg.split(',')
       ├─ SUCCESS → return part[]  (lot_no = part[1])
       └─ FAIL    → return 'err' → popup warning

  If lot found:
    ├─ self.currentLot = ret[1]
    ├─ Update lot label on screen
    ├─ Show OCR input panel (groupBox2)
    ├─ Hide Send button (visible only when both fields filled)
    └─ Set focus to marking textbox

OCRsending()
  ├─ Validate OP number: must be exactly 6 numeric digits
  ├─ Validate marking: must not be empty
  ├─ Resize image: 1280×960 → 640×480 (INTER_AREA)
  ├─ Save as cropimg.jpg
  ├─ POST /OCR/ReadMark → get expected mark
  ├─ Compare: input == expected ? PASS (green) : FAIL (red)
  ├─ Read cropimg.jpg → base64 encode
  ├─ POST /OCR/CreateRecord → save result + image
  ├─ Display result overlay on screen
  ├─ Save screen screenshot → OCR/{lot}_{timestamp}.jpg
  ├─ Light OFF
  └─ Start 5-sec timer → updateImg() restores camera image
```

---

## 5. API Reference

Base URL: `http://webserv.thematrix.net/ROHMApi/api/OCR`

### GET Expected Mark
```
POST /ReadMark
Content-Type: application/json

Request:
{
  "username": "123456",       // operator 6-digit number
  "lot_no":   "AB12345678"    // lot number from cellcon
}

Response (HTTP 200):
[
  {
    "lot_no": "AB12345678",
    "mark":   "BU28131029T75"   // expected marking string
  }
]
```

### Save OCR Result
```
POST /CreateRecord
Content-Type: application/json

Request:
{
  "username":      "123456",          // operator number
  "lot_no":        "AB12345678",      // lot number
  "mark":          "BU28131029T75",   // marking typed by operator
  "image":         "<base64 string>", // 640×480 JPG, base64 encoded
  "is_pass":       1,                 // 1 = pass, 0 = fail
  "recheck_count": 0,                 // always 0 (not used)
  "is_logo_pass":  0                  // always 0 (not used)
}

Response (HTTP 200): success
Response (other):    error → log "OCR record error"
```

---

## 6. Serial Port — Cellcon Reader

**Protocol:** 38400 baud, 8N1, timeout 1 sec  
**Port object:** `self.ser`

| Direction | Data | Meaning |
|---|---|---|
| Send → | `LA\r\n` | Lot Ask: request current lot number |
| ← Receive | `LS,{lot_no},…\r\n` | Lot Send: CSV, lot_no is index [1] |

**Port auto-detection logic** (`checkComPort()`):
1. Check `/dev/ttyUSB0` and `/dev/ttyUSB1` both exist (else exit)
2. Open each port, send `LA\r\n`, wait for `LS` response
3. Port that responds `LS` → cellcon (`self.ser`)
4. Other port → lighting controller (`self.ser1`)

---

## 7. Serial Port — Lighting Controller

**Protocol:** 38400 baud, 8N1, timeout 1 sec  
**Port object:** `self.ser1`

| Command | Bytes | Action |
|---|---|---|
| Light ON | `@00L1007D\r\n` | Turn light on |
| Light OFF | `@00L0007C\r\n` | Turn light off |
| Set brightness | `@00F{nnn}00{checksum}\r\n` | Set level 000-255 |

**Brightness command construction** (`lightSetup1()`):
```python
SendMes = '@00F' + f"{value:03}" + '00'
# Checksum: sum all ASCII bytes, keep lowest byte, hex uppercase
checksum = hex(sum(bytes(SendMes,'ascii')) & 0xFF)[2:].upper()
lightVal = (SendMes + checksum + '\r\n').encode('utf-8')
```

**Busy flag pattern:**
- `lightBusy = True` is set immediately after writing a command
- Background thread `readSerialLighting()` clears `lightBusy = False` when ACK arrives
- Main thread waits: `while lightBusy == True: time.sleep(0.01)`

---

## 8. Global State Variables

| Variable | Type | Default | Purpose |
|---|---|---|---|
| `liveBit` | bool | False | Live camera feed active |
| `lightOnbit` | bool | False | Light currently on |
| `lightBusy` | bool | False | Waiting for serial ACK |
| `serialLightingEnable` | bool | False | Lighting serial port available |
| `threadKill` | bool | False | Signal background threads to stop |
| `threadLiveAlive` | bool | False | Live auto-timeout thread running |
| `exposureTime` | int | 25000 | Camera exposure in µs |
| `Resolution` | tuple | (1280,960) | Camera frame size |
| `mcNo` | str | '' | Machine number (from config.json) |
| `mcType` | str | 'IFLR' | Machine type (from config.json) |
| `currentLot` | str | '' | Active lot number |
| `lightingValue` | int | 0 | Current brightness level |

---

## 9. Log Events Reference

Format: `INFO 2026-05-21 14:30:00,123 <message>`

| Event | Log message |
|---|---|
| App start | `Start application` |
| Config loaded | `Machine no:V-35` |
| Live ON | `Live image` |
| Live OFF | `Stop live image` |
| Capture | `Capture image` |
| OCR button pressed | `OCR menu select` |
| Lot found | `Get lot:{lot_no}` |
| Lot not found | `Get lot:Lot not found` |
| OCR submit start | `OCR start` |
| Submit empty marking | `OCR start: character empty` |
| Submit with marking | `OCR start by text:{string}` |
| Compare pass | `OCR compare correct:{mark}` |
| Compare fail | `OCR compare in-correct:{mark}` |
| DB record saved | `OCR record complete` |
| DB record failed | `OCR record error` |
| OCR cycle done | `OCR finished` |
| API error | `OCR start: result error` |
| Result image saved | `OCR image save` |
| OCR cancelled | `OCR cancel` |
| Serial port missing | `"/dev/ttyUSBx" not found` |
| App closed | `Close application` |
| Exit on port error | `Program exit` |

---

## 10. Key Behaviors for Porting

### Camera Grab Pattern
```python
# Always grab 3 times — first 2 flush stale buffer, 3rd is used
self.GrapImg_front()
self.GrapImg_front()
self.GrapImg_front()
# self.frameFront is now a fresh numpy grayscale array (1280×960)
```

### Light Stabilization Wait
```python
self.lightOn()
cv2.waitKey(300)   # minimum wait after ON command
# then wait for serial ACK:
stTime = time.time()
while True:
    if lightBusy == False:
        break
    time.sleep(0.01)
    if time.time() - stTime >= 3:   # 3 sec timeout
        break
```

### OCR Trigger (one record per lot)
- Cellcon holds last scanned lot until a new barcode is scanned
- Each OCR press re-queries cellcon — same lot returned if not re-scanned
- No lock-out: multiple records can be created for the same lot
- `recheck_count` field exists in API but is always sent as 0

### Display Result Pattern
```python
scene = QGraphicsScene()
pixmap = QPixmap(QImage(frame, w, h, QImage.Format_Grayscale8))
scene.addPixmap(pixmap)
scene.addRect(0, 0, 630, 280, pen=QPen(Qt.red), brush=QBrush(QColor('black')))
scene.addItem(self.dispText(x, y, 'text', fontsize, 'lime'))  # or 'red'
self.FgraphicsView.setScene(scene)
QCoreApplication.processEvents()   # force immediate screen update
```

### Image to Base64 for API
```python
imgBuf = cv2.resize(self.frameFront, [640, 480], interpolation=cv2.INTER_AREA)
cv2.imwrite('cropimg.jpg', imgBuf)
with open('cropimg.jpg', 'rb') as f:
    encodeImg = base64.b64encode(f.read())
# encodeImg is bytes — pass directly in JSON payload
```

---

## 11. Dependencies

| Package | Purpose |
|---|---|
| `pypylon` | Basler Pylon camera SDK |
| `cv2` (OpenCV) | Image resize, write, waitKey |
| `PyQt5` | UI framework |
| `pymssql` | MSSQL connection (currently unused, commented out) |
| `serial` (pyserial) | RS-232 for cellcon and lighting |
| `requests` | REST API calls |
| `RPi.GPIO` | Raspberry Pi GPIO (imported, minimal use) |
| `PIL` | Image utilities (imported, minimal use) |
| `pickle` | Persist lighting level, template point |
| `logging` | Application log |
| `subprocess` | Run SyncTime.py at startup |
