import os
from .exceptions import ConfigError


class ConfigLoader:
    CONFIG_FILE = "Config.toml"
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
        "LOG_RETENTION":        365,
        "ANN_BORDER_PX":        1,
        "ANN_SHOW_LABELS":      True,
        "WARMUP_FRAMES":        5,
        "CELLCON_PORT":         "/dev/ttyUSB0",
        "IMAGE_W":              0,
        "IMAGE_H":              0,
        "CLS_N_PASSES":         1,
        "CLS_UNCERTAIN_THR":    0.50,
    }

    @classmethod
    def load(cls) -> dict:
        import tomlkit
        if not os.path.exists(cls.CONFIG_FILE):
            raise ConfigError("Config.toml not found — create it before running.")
        try:
            with open(cls.CONFIG_FILE, "r", encoding="utf-8") as f:
                data = tomlkit.load(f)
        except Exception as e:
            raise ConfigError(f"Config.toml unreadable: {e}")
        cfg = dict(cls.DEFAULT_CONFIG)
        for k in cls.DEFAULT_CONFIG:
            if k in data:
                cfg[k] = data[k]
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
            raise ConfigError("LOG_RETENTION must be a positive integer")
        for pin_key in ("GPIO_START_PIN", "GPIO_BUSY_PIN",
                        "GPIO_END_PIN", "GPIO_INSPEC_STAGE_PIN"):
            if not isinstance(cfg[pin_key], int) or not (1 <= cfg[pin_key] <= 27):
                raise ConfigError(f"{pin_key} must be a BCM pin number (1–27)")
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
