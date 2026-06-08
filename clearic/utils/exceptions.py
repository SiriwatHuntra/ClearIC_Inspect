from enum import Enum


class ErrorFlag(Enum):
    NONE     = "NONE"
    CAMERA   = "CAMERA"
    MODEL    = "MODEL"
    GPIO     = "GPIO"
    TEMPLATE = "TEMPLATE"


class InspectionError(Exception):
    pass


class MarkMissingError(InspectionError):
    def __init__(self, missing_a: list, missing_b: list,
                 annotated=None,
                 confs_a: list = None, confs_b: list = None):
        self.missing_a = missing_a
        self.missing_b = missing_b
        self.annotated = annotated
        self.confs_a   = confs_a or []   # per-cell Text confidence (6 floats) for IC_A
        self.confs_b   = confs_b or []   # per-cell Text confidence (6 floats) for IC_B
        parts = []
        if missing_a:
            parts.append(f"IC_A={missing_a}")
        if missing_b:
            parts.append(f"IC_B={missing_b}")
        super().__init__("Missing cells: " + ", ".join(parts))


class _SystemError(InspectionError):
    pass


class CameraError(_SystemError):
    pass


class ModelError(_SystemError):
    pass


class TemplateError(_SystemError):
    pass


class GPIOError(_SystemError):
    pass


class LowMatchError(InspectionError):
    """Template match score too low to trust — skip this frame, not fatal."""
    pass


class ConfigError(InspectionError):
    pass
