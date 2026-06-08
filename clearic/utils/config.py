import os

from .exceptions import ConfigError


# CONFIG LOADER
class ConfigLoader:
    CONFIG_FILE = "Config.toml"
    #This defualt config is used as a template for the Config.toml file and as fallback for missing keys. It is not used directly in the code, but serves as a reference for the expected configuration parameters and their default values.
    DEFAULT_CONFIG = {
        "USE_CAMERA":           False,
        "CONF_THR":             0.5,
        "TEXT_MIN_CONF":        0.80,
        "TEXT_NG_THRESHOLD":    2,
        "BLANK_CELL_STD_THR":   0.0,
        "NMS_IOU_THR":          0.45,
        "CAMERA_SERIAL":        "",
        "EXPOSURE_US":          8000,
        "DEBUG":                True,
        "IO":                   False,
        "MODE":                 "DEBUG",
        "COLLECT_DATASET":      False,
        "DIR_INPUT":            "Input/",
        "OUT_DIR":              "Output/",
        "MODEL_PATH":           "Text_cls-2/best_openvino_model/best.xml",
        "CAMERA_WARMUP_FRAMES": 5,
        "CAMERA_RETRY_DELAY":   0.2,
        "CAMERA_RETRIES":       2,
        "RECONNECT_ATTEMPTS":   3,
        "RECONNECT_DELAY_S":    5.0,
        "RETRY_DELAY_MS":       10,
        "DISK_WARN_MB":         200,
        "GPIO_START_PIN":        17,
        "GPIO_BUSY_PIN":         23,
        "GPIO_END_PIN":          18,
        "GPIO_INSPEC_STAGE_PIN": 24,
        "CELL_SHRINK":          0.95,
        "CELL_EXPAND":          1.2,
        "COL_GAP_PCT":          40.0,
        "GRID_MARGIN_TOP":      0.0,
        "GRID_MARGIN_BOT":      15.0,
        "DATA_DIR":             "Dataset",
        "DATA_SPLIT":           "train",
        "LOG_DIR":              "logs",
        "LOG_RETENTION":        730,   # days to keep log files (2 years)
        "ANN_BORDER_PX":        1,
        "RESULT_OVERLAY":      True,
        "WARMUP_FRAMES":        5,
        "CELLCON_PORT":         "/dev/ttyUSB0",
        "IMAGE_W":              0,
        "IMAGE_H":              0,
        "CLS_N_PASSES":         1,   # deterministic model — multi-pass gives identical results
        "CLS_UNCERTAIN_THR":    0.50,
        "RETRY_W2":             0.7, # weight of Conf in retry decision (vs Text/NoText ratio)
        "RETRY_W1":             0.3, # weight of Conf in retry decision (vs Text/NoText ratio)
        "RETRY_PASS_THR":       0.90,   # weighted score threshold to call a retried cell PASS
        "BLOB_MIN_RATIO":       0.0,    # 0.0 = disabled; 0.2 removes small non-pin blobs from binary map
        "TEMPLATE_MATCH_THR":   0.6,    # minimum match score for IC_A template matching
        "TEMPLATE_FIND_CONF_THR": 0.4,  # minimum score to accept IC_B in auto-detection
        "LIGHTING_ENABLE":      True,
        "LIGHTING_USB_ID":      "Prolific_Technology_Inc._USB-Serial_Controller",
        "LIGHTING_PORT":        "/dev/ttyUSB1",
        "LIGHTING_VALUE":       100,
    }

    @classmethod
    def load(cls) -> dict:
        import tomlkit
        if not os.path.exists(cls.CONFIG_FILE):
            raise ConfigError("Config.toml not found — create it before running.")
        try:
            import re as _re
            with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                _raw = f.read()
            _raw = _re.sub(
                r'(=\s*)(True|TRUE|False|FALSE)(\s*(?:#.*)?$)',
                lambda m: m.group(1) + m.group(2).lower() + m.group(3),
                _raw,
                flags=_re.MULTILINE,
            )
            data = tomlkit.loads(_raw)
        except Exception as e:
            raise ConfigError(f"Config.toml unreadable: {e}")
        cfg = dict(cls.DEFAULT_CONFIG)
        data_upper = {k.upper(): v for k, v in data.items()}
        for k in cls.DEFAULT_CONFIG:
            if k in data_upper:
                cfg[k] = data_upper[k]
        for k in cls.DEFAULT_CONFIG:
            if isinstance(cls.DEFAULT_CONFIG[k], bool) and isinstance(cfg[k], str):
                if cfg[k].lower() in ("true", "yes", "1"):
                    cfg[k] = True
                elif cfg[k].lower() in ("false", "no", "0"):
                    cfg[k] = False
        if not isinstance(cfg["USE_CAMERA"], bool):
            raise ConfigError("USE_CAMERA must be true or false")
        cfg["CAMERA"] = "camera" if cfg["USE_CAMERA"] else "directory"
        if not (0.0 < cfg["CONF_THR"] <= 1.0):
            raise ConfigError("CONF_THR must be in (0, 1]")
        if not (0.0 < cfg["TEXT_MIN_CONF"] <= 1.0):
            raise ConfigError("TEXT_MIN_CONF must be in (0, 1]")
        if not (0.0 <= cfg["BLANK_CELL_STD_THR"] <= 255.0):
            raise ConfigError("BLANK_CELL_STD_THR must be in [0, 255]")
        if not isinstance(cfg["DEBUG"], bool):
            raise ConfigError("DEBUG must be a boolean")
        if not isinstance(cfg["IO"], bool):
            raise ConfigError("IO must be a boolean")
        if not isinstance(cfg["COLLECT_DATASET"], bool):
            raise ConfigError("COLLECT_DATASET must be a boolean")
        if not isinstance(cfg["LOG_RETENTION"], int) or cfg["LOG_RETENTION"] < 1:
            raise ConfigError("LOG_RETENTION must be a positive integer (days)")
        for pin_key in ("GPIO_START_PIN", "GPIO_BUSY_PIN",
                        "GPIO_END_PIN", "GPIO_INSPEC_STAGE_PIN"):
            if not isinstance(cfg[pin_key], int) or not (1 <= cfg[pin_key] <= 27):
                raise ConfigError(f"{pin_key} must be a BCM pin number (1–27)")
        if not isinstance(cfg["RESULT_OVERLAY"], bool):
            raise ConfigError("RESULT_OVERLAY must be true or false")
        for _k in ("NMS_IOU_THR", "RETRY_PASS_THR", "CLS_UNCERTAIN_THR"):
            if not (0.0 < cfg[_k] <= 1.0):
                raise ConfigError(f"{_k} must be in (0, 1]")
        if not (0.0 < cfg["CELL_SHRINK"] <= 1.0):
            raise ConfigError("CELL_SHRINK must be in (0, 1]")
        if not (cfg["CELL_EXPAND"] > 0.0):
            raise ConfigError("CELL_EXPAND must be > 0")
        if not (0.0 <= cfg["COL_GAP_PCT"] < 100.0):
            raise ConfigError("COL_GAP_PCT must be in [0, 100)")
        if cfg["GRID_MARGIN_TOP"] + cfg["GRID_MARGIN_BOT"] >= 100.0:
            raise ConfigError("GRID_MARGIN_TOP + GRID_MARGIN_BOT must be < 100")
        for _k in ("TEXT_NG_THRESHOLD", "EXPOSURE_US", "CLS_N_PASSES",
                   "CAMERA_WARMUP_FRAMES", "WARMUP_FRAMES"):
            if not isinstance(cfg[_k], int) or cfg[_k] < 1:
                raise ConfigError(f"{_k} must be a positive integer (>= 1)")
        for _k in ("IMAGE_W", "IMAGE_H", "CAMERA_RETRIES",
                   "RECONNECT_ATTEMPTS", "RETRY_DELAY_MS", "ANN_BORDER_PX"):
            if not isinstance(cfg[_k], int) or cfg[_k] < 0:
                raise ConfigError(f"{_k} must be a non-negative integer (>= 0)")
        for _k in ("CAMERA_RETRY_DELAY", "RECONNECT_DELAY_S"):
            if not (cfg[_k] >= 0.0):
                raise ConfigError(f"{_k} must be >= 0")
        if cfg["DATA_SPLIT"] not in ("train", "val"):
            raise ConfigError("DATA_SPLIT must be 'train' or 'val'")
        if not (0.0 <= cfg["BLOB_MIN_RATIO"] <= 1.0):
            raise ConfigError("BLOB_MIN_RATIO must be 0.0–1.0")
        if not (0.0 <= cfg["TEMPLATE_MATCH_THR"] <= 1.0):
            raise ConfigError("TEMPLATE_MATCH_THR must be 0.0–1.0")
        if not (0.0 <= cfg["TEMPLATE_FIND_CONF_THR"] <= 1.0):
            raise ConfigError("TEMPLATE_FIND_CONF_THR must be 0.0–1.0")
        _w_sum = cfg["RETRY_W2"] + cfg["RETRY_W1"]
        if abs(_w_sum - 1.0) > 0.001:
            print(f"[Config] Warning: RETRY_W2 + RETRY_W1 = {_w_sum:.3f} (expected 1.0)")
        _unknown = sorted(k for k in data_upper if k not in cls.DEFAULT_CONFIG)
        if _unknown:
            print(f"[Config] Unrecognised keys (possible typo): {_unknown}")
        if not os.path.exists(cfg["MODEL_PATH"]):
            print(f"[Config] Warning: MODEL_PATH not found: {cfg['MODEL_PATH']!r}")
        return cfg

    @classmethod
    def save(cls, updates: dict):
        import tomlkit
        try:
            with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                doc = tomlkit.load(f)
        except Exception:
            doc = tomlkit.document()
        for k, v in updates.items():
            if k in cls.DEFAULT_CONFIG:
                doc[k] = v
        with open(cls.CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))

    @classmethod
    def update(cls, updates: dict):
        """Merge partial updates into saved config. Only known keys are accepted."""
        import tomlkit
        try:
            with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                doc = tomlkit.load(f)
        except Exception:
            doc = tomlkit.document()
        for k, v in updates.items():
            if k in cls.DEFAULT_CONFIG:
                doc[k] = v
        with open(cls.CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(tomlkit.dumps(doc))
        return cls.load()
