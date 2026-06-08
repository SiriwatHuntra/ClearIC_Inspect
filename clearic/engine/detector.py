import os

import cv2
import numpy as np

from ..utils.exceptions import ModelError


_CLS_INPUT_SIZE = 224   # YOLO-cls default input size
_TOTAL_CELLS    = 12    # 6 cells × 2 ICs


# DETECTOR  (OpenVINO Classifier — 2-class)
class Detector:
    """
    OpenVINO image classifier for ClearIC mark inspection.
    Each ROI cell crop is classified as Text (mark present) or NoText (absent).
    Output shape: [1, 2]  — index 0 = NoText, index 1 = Text
    """

    def __init__(self, conf_thr: float = 0.5, text_min_conf: float = 0.80,
                 blank_cell_std_thr: float = 0.0,
                 model_path: str = "Text_cls-2/best_openvino_model/best.xml",
                 n_passes: int = 3, uncertain_thr: float = 0.50,
                 debug: bool = False, **_):
        self._conf_thr           = conf_thr
        self._text_min_conf      = text_min_conf
        self._blank_cell_std_thr = blank_cell_std_thr
        self._n_passes           = max(1, int(n_passes))
        self._uncertain_thr      = float(uncertain_thr)
        self._debug              = debug
        self._compiled = None
        self._ready    = False
        try:
            import openvino as ov
            self._model_xml = model_path
            if not os.path.exists(self._model_xml):
                raise ModelError(f"Model not found: {self._model_xml}")
            core  = ov.Core()
            model = core.read_model(self._model_xml)
            self._compiled = core.compile_model(model, "CPU", {
                "INFERENCE_PRECISION_HINT": "f32",
                "PERFORMANCE_HINT":         "LATENCY",
            })
            self._ready = True
            print(f"[Detector] OpenVINO classifier loaded: {self._model_xml}")
        except ModelError:
            raise
        except Exception as e:
            raise ModelError(f"Model load failed: {e}")

    def is_ready(self) -> bool:
        return self._ready

    def warmup(self, frames: int = 5):
        blank = np.zeros((_CLS_INPUT_SIZE, _CLS_INPUT_SIZE, 3), dtype=np.uint8)
        for _ in range(frames):
            self.classify_crop(blank)
        print(f"[Detector] Warmup done ({frames} frames).")

    def classify_crop(self, crop_bgr: np.ndarray) -> tuple:
        """
        Classify one ROI cell crop.
        Returns (class_idx, confidence):
          class_idx 0 = NoText  (mark absent)
          class_idx 1 = Text    (mark present)
        """
        if not self._ready or self._compiled is None:
            return 0, 0.0
        try:
            sz = _CLS_INPUT_SIZE
            if crop_bgr.ndim == 2:
                crop_bgr = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
            if crop_bgr.size == 0:
                return 0, 0.0
            if self._blank_cell_std_thr > 0.0:
                _g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if crop_bgr.ndim == 3 else crop_bgr
                if float(_g.std()) < self._blank_cell_std_thr:
                    return 0, 1.0   # guard-triggered NoText; conf=1.0 marks it in logs
            resized = cv2.resize(crop_bgr, (sz, sz))
            blob    = resized[:, :, ::-1].astype(np.float32) / 255.0
            blob    = blob.transpose(2, 0, 1)[np.newaxis]   # [1, 3, sz, sz]
            text_probs = []
            for _ in range(self._n_passes):
                result = self._compiled(blob)
                text_probs.append(float(result[0][0][1]))   # P(Text) each pass
            text_prob   = sum(text_probs) / len(text_probs)
            notext_prob = 1.0 - text_prob
            # Require Text probability to clear TEXT_MIN_CONF; anything below → NoText.
            # Asymmetric on purpose: guards unmarked products without penalising NoText.
            if text_prob >= self._text_min_conf:
                return 1, text_prob
            if self._debug and text_prob >= self._uncertain_thr:
                print(f"[Detector] Uncertain cell: text_prob={text_prob:.3f} "
                      f"(gate={self._text_min_conf:.2f})")
            return 0, notext_prob
        except Exception as e:
            print(f"[Detector] Classify error: {e}")
            return 0, 0.0
