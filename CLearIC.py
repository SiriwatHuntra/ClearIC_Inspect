import numpy as np
import os, sys
import cv2 as cv


# ─── Exceptions ───────────────────────────────────────────────────────────────

class InspectionError(Exception):
    pass


class MarkMissingError(InspectionError):
    def __init__(self, ic_position: str, missing_cells: list):
        self.ic_position = ic_position      # "A" or "B"
        self.missing_cells = missing_cells  # [[row, col], ...]
        super().__init__(f"IC_{ic_position}: mark missing at {missing_cells}")


class SystemError(InspectionError):
    pass

class CameraError(SystemError):
    pass

class ModelError(SystemError):
    pass

class TemplateError(SystemError):
    pass

class GPIOError(SystemError):
    pass


class ConfigError(InspectionError):
    pass


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    pass


if __name__ == "__main__":
    main()

