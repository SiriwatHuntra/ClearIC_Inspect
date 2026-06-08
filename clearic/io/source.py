import os
import glob
import time

import cv2
import numpy as np

from ..utils.exceptions import CameraError


# DIRECTORY IMAGE SOURCE
class DirectorySource:
    """
    File-based image source — reads files from a directory in sorted order,
    looping indefinitely. Used when CAMERA='directory' (USE_CAMERA=false).
    """

    def __init__(self, input_dir: str = "Input",
                 retry_delay: float = 0.2, retries: int = 2,
                 image_w: int = 0, image_h: int = 0):
        self._input_dir   = input_dir
        self._retry_delay = retry_delay
        self._retries     = retries
        self._image_w     = image_w
        self._image_h     = image_h

        self._files: list = []
        self._idx:   int  = 0

    def open(self):
        exts  = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(self._input_dir, ext)))
        files.sort()
        if not files:
            raise CameraError(
                f"No images found in '{self._input_dir}/'")
        self._files = files
        self._idx   = 0
        print(f"[Camera] Directory mode — {len(files)} image(s) in '{self._input_dir}/'")

    def grab(self) -> np.ndarray:
        """Return BGR ndarray or raise CameraError."""
        for attempt in range(self._retries + 1):
            try:
                img = self._grab_directory()
                if img is not None:
                    return img
            except CameraError:
                raise
            except Exception as e:
                if attempt < self._retries:
                    time.sleep(self._retry_delay)
                else:
                    raise CameraError(f"Grab failed after retries: {e}")
        raise CameraError("Grab returned None after retries")

    def _grab_directory(self) -> np.ndarray:
        if not self._files:
            raise CameraError("No files loaded")
        path = self._files[self._idx % len(self._files)]
        self._idx += 1
        img = cv2.imread(path)
        if img is None:
            raise CameraError(f"Cannot read image: {path}")
        if self._image_w > 0 and self._image_h > 0:
            img = cv2.resize(img, (self._image_w, self._image_h),
                             interpolation=cv2.INTER_AREA)
        return img

    def warmup(self):
        pass

    def set_exposure(self, us: int):
        pass

    def close(self):
        pass

    def is_open(self) -> bool:
        return bool(self._files)

    def has_more(self) -> bool:
        """True if there are still un-visited images this cycle."""
        return self._idx < len(self._files)

    def reset(self):
        """Reset directory index to beginning."""
        self._idx = 0

    def grab_first(self) -> np.ndarray:
        """Grab the first frame: rewinds index before and after grab."""
        self.reset()
        img = self.grab()
        self.reset()
        return img
