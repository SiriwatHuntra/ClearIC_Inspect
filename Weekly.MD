# Weekly Report

**Week:** W23 · **Period:** 2026-06-02 – 2026-06-06
**Prepared by:** Siriwat H.
**Date:** 2026-06-02

---

## 1. Production Summary

| Metric | Value |
|---|---|
| Total inspections | |
| PASS (`_G`) | |
| Suspect PASS (`_GS`) | |
| Suspect NG (`_NGS`) | |
| NG (`_NG`) | |
| Pass rate (%) | |
| Lots completed | |

---

## 2. Equipment & System Status

| Item | Status | Notes |
|---|---|---|
| Camera (Basler) | OK / Issue | |
| OpenVINO model | OK / Issue | |
| GPIO I/O | OK / Issue | |
| CellCon serial | OK / Issue | |
| Disk space | OK / Low | |

---

## 3. Issues & Incidents

| # | Date | Description | Root Cause | Resolution | Status |
|---|---|---|---|---|---|
| 1 | | | | | Open / Closed |
| 2 | | | | | Open / Closed |

---

## 4. Actions Taken This Week

- Reviewed `IFWFOCR01.py` imports — identified 11 unused imports (`multiprocessing`, `PIL`, duplicate `time`, `datetime`, `argparse`, `shutil`, `socket`, `RPi.GPIO`, `glob`, `pymssql`, `getch`)
- Confirmed `socket` import is unused (only referenced in commented-out IP-display code)
- Created `Weekly.MD` report template
- 

---

## 5. Planned Actions Next Week

- Remove 11 unused imports from `IFWFOCR01.py` to clean up codebase
- 
- 

---

## 6. Notes / Remarks

> _(Any additional observations, operator feedback, or quality concerns.)_

---

*ClearIC Inspect — Weekly Report Template*
