import os
import glob
import time

import cv2
import numpy as np

from ..utils.exceptions import CameraError


class Camera:
    """
    Unified camera source.
    CAMERA='camera'    : Basler pypylon InstantCamera
    CAMERA='directory' : reads files from Input/ in sorted order, loops
    """

    def __init__(self, mode: str, serial: str = "",
                 exposure_us: int = 8000, input_dir: str = "Input",
                 retry_delay: float = 0.2, retries: int = 2,
                 warmup_frames: int = 5,
                 image_w: int = 0, image_h: int = 0):
        self._mode        = mode
        self._serial      = serial
        self._exposure_us = exposure_us
        self._input_dir   = input_dir
        self._retry_delay = retry_delay
        self._retries     = retries
        self._warmup_frames = warmup_frames
        self._image_w     = image_w
        self._image_h     = image_h

        self._camera      = None
        self._pylon       = None
        self._basler_ok   = False

        self._files:  list = []
        self._idx:    int  = 0

        if mode == "camera":
            try:
                from pypylon import pylon
                self._pylon     = pylon
                self._basler_ok = True
            except ImportError:
                raise CameraError("pypylon not installed — cannot use CAMERA='camera'")

    def open(self):
        if self._mode == "camera":
            self._open_basler()
        else:
            self._open_directory()

    def _open_basler(self):
        try:
            pylon = self._pylon
            tl    = pylon.TlFactory.GetInstance()
            devs  = tl.EnumerateDevices()
            device = None
            for d in devs:
                if not self._serial or d.GetSerialNumber() == self._serial:
                    device = tl.CreateDevice(d)
                    break
            if device is None:
                raise CameraError(
                    f"Basler camera not found (serial='{self._serial}')")
            self._camera = pylon.InstantCamera(device)
            self._camera.Open()
            self._camera.ExposureAuto.SetValue("Off")
            try:
                self._camera.ExposureTimeAbs.SetValue(float(self._exposure_us))
            except Exception:
                self._camera.ExposureTime.SetValue(float(self._exposure_us))
            self._camera.PixelFormat.SetValue("Mono8")
            self._camera.TriggerSelector.SetValue("FrameStart")
            self._camera.TriggerMode.SetValue("On")
            self._camera.TriggerSource.SetValue("Software")
            self._camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            print(f"[Camera] Opened (software trigger). Exposure={self._exposure_us} µs")
        except CameraError:
            raise
        except Exception as e:
            raise CameraError(f"Camera open failed: {e}")

    def _open_directory(self):
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
                img = self._grab_once()
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

    def _grab_once(self) -> np.ndarray:
        if self._mode == "camera":
            return self._grab_basler()
        else:
            return self._grab_directory()

    def _grab_basler(self) -> np.ndarray:
        if self._camera is None:
            raise CameraError("Camera not open")
        self._camera.ExecuteSoftwareTrigger()
        result = self._camera.RetrieveResult(
            5000, self._pylon.TimeoutHandling_ThrowException)
        try:
            if result.GrabSucceeded():
                gray = result.GetArray().copy()
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            raise CameraError(f"Grab failed: code {result.ErrorCode}")
        finally:
            result.Release()

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
        if self._mode == "camera":
            for _ in range(self._warmup_frames):
                try:
                    self._grab_basler()
                except Exception:
                    pass
            print(f"[Camera] Warmup done ({self._warmup_frames} frames).")

    def set_exposure(self, us: int):
        self._exposure_us = int(us)
        if self._camera and self._camera.IsOpen():
            try:
                self._camera.ExposureTimeAbs.SetValue(float(us))
            except Exception as e:
                print(f"[Camera] Exposure set error: {e}")

    def close(self):
        if self._camera:
            try:
                self._camera.StopGrabbing()
                self._camera.Close()
            except Exception:
                pass
            self._camera = None
        print("[Camera] Closed.")

    def reconnect(self, attempts: int = 1, delay_s: float = 0.0) -> bool:
        """Close and re-open the Basler camera. Returns True on success."""
        if self._mode != "camera":
            return False
        for _ in range(max(1, attempts)):
            self.close()
            if delay_s > 0:
                time.sleep(delay_s)
            try:
                self._open_basler()
                self.warmup()
                print("[Camera] Reconnected.")
                return True
            except CameraError as e:
                print(f"[Camera] Reconnect attempt failed: {e}")
        return False

    def is_open(self) -> bool:
        if self._mode == "camera":
            return self._camera is not None and self._camera.IsOpen()
        return bool(self._files)

    def is_healthy(self) -> bool:
        """True if camera is open and ready to accept triggers."""
        return self.is_open()

    def has_more(self) -> bool:
        """Directory mode: True if there are still un-visited images this cycle."""
        if self._mode != "directory":
            return True
        return self._idx < len(self._files)

    def reset(self):
        """Reset directory index to beginning."""
        self._idx = 0

    def grab_first(self) -> np.ndarray:
        """Grab the first frame: rewinds directory index before and after grab."""
        if self._mode == "directory":
            self.reset()
        img = self.grab()
        if self._mode == "directory":
            self.reset()
        return img
