from .hardware import BaslerCamera, RaspberryIO, LightingController, CellCon, _detect_ports
from .source import DirectorySource


# CAMERA FACADE
class Camera:
    """
    Unified camera source.
    CAMERA='camera'    : Basler pypylon InstantCamera   (BaslerCamera, hardware.py)
    CAMERA='directory' : reads files from Input/ in sorted order, loops (DirectorySource, source.py)
    """

    def __init__(self, mode: str, serial: str = "",
                 exposure_us: int = 8000, input_dir: str = "Input",
                 retry_delay: float = 0.2, retries: int = 2,
                 warmup_frames: int = 5,
                 image_w: int = 0, image_h: int = 0):
        self._mode = mode
        if mode == "camera":
            self._impl = BaslerCamera(
                serial=serial, exposure_us=exposure_us,
                retry_delay=retry_delay, retries=retries,
                warmup_frames=warmup_frames)
        else:
            self._impl = DirectorySource(
                input_dir=input_dir, retry_delay=retry_delay, retries=retries,
                image_w=image_w, image_h=image_h)

    def open(self):
        self._impl.open()

    def grab(self):
        return self._impl.grab()

    def grab_first(self):
        return self._impl.grab_first()

    def warmup(self):
        self._impl.warmup()

    def set_exposure(self, us: int):
        self._impl.set_exposure(us)

    def close(self):
        self._impl.close()
        print("[Camera] Closed.")

    def is_open(self) -> bool:
        return self._impl.is_open()

    def has_more(self) -> bool:
        return self._impl.has_more()

    def reset(self):
        self._impl.reset()
