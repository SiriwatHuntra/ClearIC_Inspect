import cv2
import numpy as np


def inspect_text_exist(
    roi_img,
    resize=(128, 128),
    blur_kernel=21,
    clahe_clip=2.0,
    clahe_grid=(8, 8),
    edge_threshold=40,
    edge_score_threshold=0.015,
):
    """
    Simple text existence inspection for laser-marked IC surface.

    INPUT:
        roi_img                 : sliced ROI image (numpy array)

    OUTPUT:
        result = {
            "label": "TEXT" or "NO_TEXT",
            "edge_score": float,
            "edge_pixel_count": int,
            "edge_ratio": float,
            "threshold": float,

            # debug images
            "gray": ...,
            "normalized": ...,
            "background_removed": ...,
            "gradient": ...,
            "binary_edge": ...
        }
    """

    # -------------------------------------------------
    # STEP 1 : grayscale
    # -------------------------------------------------
    roi_img = cv2.imread(roi_img) if isinstance(roi_img, str) else roi_img
    if len(roi_img.shape) == 3:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_img.copy()

    # -------------------------------------------------
    # STEP 2 : resize
    # -------------------------------------------------
    gray = cv2.resize(gray, resize)

    # -------------------------------------------------
    # STEP 3 : CLAHE normalize
    # -------------------------------------------------
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=clahe_grid
    )

    normalized = clahe.apply(gray)

    # -------------------------------------------------
    # STEP 4 : background suppression
    # remove slow-changing surface brightness
    # -------------------------------------------------
    blur = cv2.GaussianBlur(
        normalized,
        (blur_kernel, blur_kernel),
        0
    )

    background_removed = cv2.subtract(normalized, blur)

    # -------------------------------------------------
    # STEP 5 : edge extraction
    # -------------------------------------------------
    grad_x = cv2.Sobel(
        background_removed,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    grad_y = cv2.Sobel(
        background_removed,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = cv2.magnitude(grad_x, grad_y)

    gradient = cv2.normalize(
        gradient,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # -------------------------------------------------
    # STEP 6 : binary edge map
    # -------------------------------------------------
    _, binary_edge = cv2.threshold(
        gradient,
        edge_threshold,
        255,
        cv2.THRESH_BINARY
    )

    # -------------------------------------------------
    # STEP 7 : scoring
    # -------------------------------------------------
    edge_pixel_count = np.count_nonzero(binary_edge)

    total_pixels = binary_edge.shape[0] * binary_edge.shape[1]

    edge_ratio = edge_pixel_count / total_pixels

    edge_score = edge_ratio

    # -------------------------------------------------
    # STEP 8 : decision
    # -------------------------------------------------
    if edge_score >= edge_score_threshold:
        label = "TEXT"
    else:
        label = "NO_TEXT"

    # -------------------------------------------------
    # RETURN
    # -------------------------------------------------
    return {
        "label": label,
        "edge_score": float(edge_score),
        "edge_pixel_count": int(edge_pixel_count),
        "edge_ratio": float(edge_ratio),
        "threshold": float(edge_score_threshold),

        # debug images
        "gray": gray,
        "normalized": normalized,
        "background_removed": background_removed,
        "gradient": gradient,
        "binary_edge": binary_edge
    }

result = inspect_text_exist("Input/Font_Crop/6.jpg")

print(result["label"])
print(result["edge_score"])