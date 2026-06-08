from datetime import datetime

from PyQt5 import QtWidgets


# LOT START DIALOG
class LotStartDialog(QtWidgets.QDialog):
    """
    Shown before a run starts. Operator enters a lot number.
    API hook: override get_lot_number_from_api() to inject from an external system;
    when it returns a non-empty string the dialog is skipped entirely.
    """

    @staticmethod
    def get_lot_number_from_api() -> str:
        """Plugin point: replace to inject lot number from an internal API."""
        return ""   # empty = show dialog; non-empty = skip dialog

    @classmethod
    def request(cls, parent=None, api_fn=None) -> str | None:
        """
        Returns lot number string, or None if operator cancelled.
        api_fn: optional callable → str; if it returns non-empty the dialog is skipped.
        Falls back to get_lot_number_from_api() for subclass overrides.
        """
        if api_fn is not None:
            lot = api_fn()
            if lot:
                return lot
        api_lot = cls.get_lot_number_from_api()   # kept as subclass plugin point
        if api_lot:
            return api_lot
        dlg = cls(parent)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            text = dlg._edit.text().strip()
            return text if text else datetime.now().strftime("%Y%m%d_%H%M%S")
        return None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Start Lot")
        self.setFixedWidth(300)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setSpacing(10)
        lay.addWidget(QtWidgets.QLabel("Enter Lot Number:"))
        self._edit = QtWidgets.QLineEdit()
        self._edit.setPlaceholderText("Leave blank for auto timestamp")
        lay.addWidget(self._edit)
        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)
        self._edit.returnPressed.connect(self.accept)
