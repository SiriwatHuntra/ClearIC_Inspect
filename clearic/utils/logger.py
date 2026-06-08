import os
import csv
import sys
import glob
from datetime import datetime, timedelta


# LOGGER
class Logger:
    """
    Dual-CSV logging system.

    Operation log  — one file per calendar day, appended across all lots.
      File: logs/op_YYYYMMDD.csv
      Columns: timestamp, event, lot_number, detail, cycle_ms

    Result log — one file per calendar day, multiple lots appended.
      File: logs/result_YYYYMMDD.csv
      Each lot is bracketed by # --- LOT_START --- / # --- LOT_END --- markers.
      Header block: lot metadata rows. Data rows: one per inspection.
      Footer block: summary appended at lot end. Blank line between lots.

    Both files rotate by age: files older than LOG_RETENTION days are deleted.
    """

    _OP_HEADER   = ["timestamp", "event", "lot_number", "detail", "cycle_ms"]
    _RES_HEADER  = ["timestamp", "image_id", "ic_a_result",
                    "ic_b_result", "cycle_ms", "is_retry", "is_suspect"]

    def __init__(self, log_dir: str = "logs", log_retention: int = 365):
        self._dir        = log_dir
        self._retention  = log_retention
        self._lot        = ""
        self._package    = ""
        self._res_path:  str | None = None
        self._pass_ct    = 0
        self._fail_ct    = 0
        self._err_ct     = 0
        os.makedirs(log_dir, exist_ok=True)
        self._rotate()

    # internal helpers

    def _op_path(self) -> str:
        return os.path.join(self._dir, f"op_{datetime.now():%Y%m%d}.csv")

    def _rotate(self):
        cutoff = datetime.now() - timedelta(days=self._retention)
        for pattern, fmt in [
            ("op_*.csv",     "op_%Y%m%d.csv"),
            ("result_*.csv", "result_%Y%m%d.csv"),
        ]:
            for path in glob.glob(os.path.join(self._dir, pattern)):
                try:
                    if datetime.strptime(os.path.basename(path), fmt) < cutoff:
                        os.remove(path)
                except (ValueError, OSError):
                    pass  # skip legacy filenames or permission errors

    def _log_error(self, operation: str, exc: Exception) -> None:
        print(f"[Logger] {operation} failed: {exc}", file=sys.stderr)

    def _op_append(self, event: str, detail: str = "", cycle_ms: float = 0):
        path = self._op_path()
        write_header = not os.path.exists(path)
        try:
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(self._OP_HEADER)
                w.writerow([
                    datetime.now().isoformat(),
                    event,
                    self._lot,
                    detail,
                    round(cycle_ms, 1),
                ])
        except Exception as e:
            self._log_error("op write", e)

    def _res_write(self, row: list):
        if not self._res_path:
            return
        try:
            with open(self._res_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)
        except Exception as e:
            self._log_error("result write", e)

    def _write_result_header(self, lot: str, package: str, mode: str, hw_info: str = ""):
        if not self._res_path:
            return
        try:
            with open(self._res_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["# --- LOT_START ---"])
                w.writerow(["LOT_NUMBER", lot])
                w.writerow(["PACKAGE",    package])
                w.writerow(["START_TIME", datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                w.writerow(["MODE",       mode])
                if hw_info:
                    w.writerow(["HW_PORTS",  hw_info])
                w.writerow([])                       # blank separator
                w.writerow(self._RES_HEADER)
        except Exception as e:
            self._log_error("result header write", e)

    def _write_result_footer(self, pass_ct: int, fail_ct: int,
                             err_ct: int, elapsed_s: float):
        if not self._res_path:
            return
        total  = pass_ct + fail_ct
        yield_ = f"{pass_ct / total * 100:.1f}" if total else "N/A"
        try:
            with open(self._res_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([])
                w.writerow(["TOTAL",       total])
                w.writerow(["PASS",        pass_ct])
                w.writerow(["FAIL",        fail_ct])
                w.writerow(["ERRORS",      err_ct])
                w.writerow(["YIELD_PCT",   yield_])
                w.writerow(["END_TIME",    datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
                w.writerow(["DURATION_S",  round(elapsed_s, 1)])
                w.writerow(["# --- LOT_END ---"])
                w.writerow([])   # blank line between lots
        except Exception as e:
            self._log_error("result footer write", e)

    # public interface

    def start_lot(self, lot_number: str, package: str, mode: str, hw_info: str = ""):
        self._rotate()
        self._lot     = lot_number
        self._package = package
        self._pass_ct = self._fail_ct = self._err_ct = 0
        self._res_path = os.path.join(
            self._dir, f"result_{datetime.now():%Y%m%d}.csv")
        self._write_result_header(lot_number, package, mode, hw_info)
        detail = f"mode={mode}"
        if hw_info:
            detail += f" {hw_info}"
        self._op_append("SESSION_START", detail)

    def end_lot(self, reason: str,
                pass_ct: int, fail_ct: int, err_ct: int, elapsed_s: float):
        total  = pass_ct + fail_ct
        yield_ = f"{pass_ct / total * 100:.1f}%" if total else "N/A"
        self._op_append("SESSION_END",
                         f"reason={reason} pass={pass_ct} fail={fail_ct} "
                         f"error={err_ct} yield={yield_}")
        self._write_result_footer(pass_ct, fail_ct, err_ct, elapsed_s)
        self._res_path = None

    def log_inspection(self, image_id: str,
                       ic_a_result: str, ic_a_missing: list,
                       ic_b_result: str, ic_b_missing: list,
                       cycle_ms: float, is_retry: bool,
                       is_suspect: bool = False):
        passed = (ic_a_result == "PASS" and ic_b_result == "PASS")
        event  = "PASS" if passed else "FAIL"
        if is_suspect:
            event += "_SUSPECT"
        # Build detail: image filename + missing cells if any
        detail_parts = [image_id]
        if ic_a_missing:
            detail_parts.append(f"miss_a={ic_a_missing}")
        if ic_b_missing:
            detail_parts.append(f"miss_b={ic_b_missing}")
        detail_parts.append(f"is_retry={1 if is_retry else 0}")
        if is_suspect:
            detail_parts.append("suspect=1")
        self._op_append(event, " ".join(detail_parts), cycle_ms)
        # Result log row
        self._res_write([
            datetime.now().isoformat(),
            image_id,
            ic_a_result,
            ic_b_result,
            round(cycle_ms, 1),
            1 if is_retry else 0,
            1 if is_suspect else 0,
        ])
        if passed:
            self._pass_ct += 1
        else:
            self._fail_ct += 1

    def log_error(self, error_type: str, message: str, cycle_ms: float = 0):
        self._op_append("ERROR", f"{error_type}: {message}", cycle_ms)
        self._err_ct += 1

    def log_pause(self):
        self._op_append("PAUSE")

    def log_resume(self):
        self._op_append("RESUME")

    def log_ocr(self, operator: str, expect_mark: str):
        self._op_append("OCR_VERIFY", f"op={operator} expect={expect_mark}")
