import threading
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


_img_counter  = 0
_counter_lock = threading.Lock()


def _next_image_id() -> str:
    global _img_counter
    with _counter_lock:
        _img_counter += 1
        return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{_img_counter:03d}"


def _reset_image_counter():
    global _img_counter
    with _counter_lock:
        _img_counter = 0


@dataclass
class Image:
    id:        str
    raw:       np.ndarray
    annotated: np.ndarray = field(default=None)
