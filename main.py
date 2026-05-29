import sys
import os
import signal
import fcntl

from PyQt5 import QtWidgets, QtGui

from clearic.utils.config import ConfigLoader
from clearic.utils.exceptions import ConfigError
from clearic.ui.style import STYLE
from clearic.ui.main_window import MainWindow


def main():
    app = QtWidgets.QApplication(sys.argv)

    _lockfile = open("/tmp/clearic.lock", "w")
    try:
        fcntl.flock(_lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        QtWidgets.QMessageBox.critical(
            None, "Already Running",
            "ClearIC is already running.\nClose the existing window first.")
        sys.exit(1)

    try:
        cfg = ConfigLoader.load()
    except ConfigError as e:
        QtWidgets.QMessageBox.critical(
            None, "Configuration Error",
            f"Cannot start — Config.toml problem:\n\n{e}\n\n"
            "Contact your system administrator.")
        sys.exit(1)

    os.makedirs(cfg.get("LOG_DIR", "logs"), exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs(cfg.get("DIR_INPUT", "Input/"), exist_ok=True)
    if cfg.get("COLLECT_DATASET", False):
        _dd, _ds = cfg.get("DATA_DIR", "Dataset"), cfg.get("DATA_SPLIT", "train")
        os.makedirs(os.path.join(_dd, _ds, "Text"),   exist_ok=True)
        os.makedirs(os.path.join(_dd, _ds, "NoText"), exist_ok=True)
        print(f"[Dataset] Collection ON → {_dd}/{_ds}/")

    for _stale in ("cropimg.jpg",):
        try:
            os.remove(_stale)
        except OSError:
            pass

    app.setStyleSheet(STYLE)

    pal = QtGui.QPalette()
    for role, col in [
        (QtGui.QPalette.Window,          (84,  101, 255)),
        (QtGui.QPalette.WindowText,      (255, 255, 255)),
        (QtGui.QPalette.Base,            (120, 139, 255)),
        (QtGui.QPalette.Text,            (255, 255, 255)),
        (QtGui.QPalette.Button,          (84,  101, 255)),
        (QtGui.QPalette.ButtonText,      (255, 255, 255)),
        (QtGui.QPalette.Highlight,       (191, 215, 255)),
        (QtGui.QPalette.HighlightedText, ( 84, 101, 255)),
    ]:
        pal.setColor(role, QtGui.QColor(*col))
    app.setPalette(pal)

    win = MainWindow(cfg)
    win.show()
    signal.signal(signal.SIGTERM, lambda *_: app.quit())
    signal.signal(signal.SIGINT,  lambda *_: app.quit())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
