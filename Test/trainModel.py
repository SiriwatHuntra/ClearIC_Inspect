import cv2
import numpy as np
from pathlib import Path
from imblearn.over_sampling import SMOTE
from ultralytics import YOLO

# Colab paths
PROJECT_DIR = Path("/kaggle/input/datasets/siriwat121")
DATASET_DIR = PROJECT_DIR / "ic-font/Dataset"

MODEL = str(PROJECT_DIR / "model-yolo/yolo26n-cls.pt")

IMG_SIZE   = 224
EPOCHS     = 100
BATCH_SIZE = 64
PROJECT = "/kaggle/working/Model_IC"
NAME       = "Text_cls"

# ── SMOTE balancing ──────────────────────────────────────────────────────────
# Images are downsampled to SMOTE_SIZE for k-NN interpolation (memory efficient),
# then synthetic crops are saved back at IMG_SIZE so YOLO trains at full resolution.
SMOTE_SIZE = 64

def _load_flat(folder: Path, sz: int) -> tuple:
    """Load all images from folder, resize to sz×sz, return (flat_array, paths)."""
    flat, paths = [], []
    for p in sorted(folder.glob("*.*")):
        img = cv2.imread(str(p))
        if img is not None:
            flat.append(cv2.resize(img, (sz, sz)).flatten().astype(np.float32))
            paths.append(p)
    return np.array(flat), paths

def apply_smote(dataset_dir: Path, minority: str, majority: str):
    maj_dir = dataset_dir / "train" / majority
    min_dir = dataset_dir / "train" / minority

    X_maj, _ = _load_flat(maj_dir, SMOTE_SIZE)
    X_min, _ = _load_flat(min_dir, SMOTE_SIZE)

    X = np.vstack([X_maj, X_min])
    y = np.array([1] * len(X_maj) + [0] * len(X_min))

    sm = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = sm.fit_resample(X, y)

    # Only the rows beyond the original dataset are synthetic
    synthetic  = X_res[len(X):]
    syn_labels = y_res[len(X):]
    count = 0
    for i, (flat, lbl) in enumerate(zip(synthetic, syn_labels)):
        if lbl == 0:   # minority class = NoText
            img     = flat.reshape(SMOTE_SIZE, SMOTE_SIZE, 3).astype(np.uint8)
            img_224 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(min_dir / f"_smote_{i:05d}.png"), img_224)
            count += 1
    print(f"[SMOTE] +{count} synthetic {minority} → total {len(X_min) + count} "
          f"(was {len(X_min)}, majority {len(X_maj)})")

apply_smote(DATASET_DIR, minority="NoText", majority="Text")
# ─────────────────────────────────────────────────────────────────────────────

model = YOLO(MODEL)

model.train(
    data=str(DATASET_DIR),
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    project=PROJECT,
    name=NAME,

    hsv_h=0.015,
    hsv_s=0.1,
    hsv_v=0.1,
    degrees=0.0,
    translate=0.1,
    scale=0.05,
    fliplr=0.0,
    flipud=0.0,

    lr0=0.01,
    optimizer="AdamW",
    dropout=0.2,           # increased from 0.1 — reduce Text-class overfitting
    label_smoothing=0.1,   # soften overconfident Text predictions

    save=True,
    pretrained=True,
    verbose=True,
    patience=20,           # increased from 15 — allow fuller convergence post-SMOTE
)

metrics = model.val()
print(metrics)
model.export(
    format="openvino",
    imgsz=IMG_SIZE,
    dynamic=False,
    half=False,
)
