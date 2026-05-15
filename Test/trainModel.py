import cv2
import numpy as np
from pathlib import Path
from imblearn.over_sampling import SMOTE
import albumentations as A
from ultralytics import YOLO

# Kaggle paths
PROJECT_DIR = Path("/kaggle/input/datasets/siriwat121")
DATASET_DIR = PROJECT_DIR / "ic-font/Dataset"

MODEL = str(PROJECT_DIR / "model-yolo/yolo26n-cls.pt")

IMG_SIZE   = 224
EPOCHS     = 100
BATCH_SIZE = 64
PROJECT    = "/kaggle/working/Model_IC"
NAME       = "Text_cls"

# ── SMOTE balancing ───────────────────────────────────────────────────────────
# Images are downsampled to SMOTE_SIZE for k-NN interpolation (memory efficient),
# then synthetic crops are saved back at IMG_SIZE so YOLO trains at full resolution.
SMOTE_SIZE = 64

def _load_flat(folder: Path, sz: int) -> tuple:
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

    synthetic  = X_res[len(X):]
    syn_labels = y_res[len(X):]
    count = 0
    for i, (flat, lbl) in enumerate(zip(synthetic, syn_labels)):
        if lbl == 0:
            img     = flat.reshape(SMOTE_SIZE, SMOTE_SIZE, 3).astype(np.uint8)
            img_224 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(str(min_dir / f"_smote_{i:05d}.png"), img_224)
            count += 1
    print(f"[SMOTE] +{count} synthetic {minority} → total {len(X_min) + count} "
          f"(was {len(X_min)}, majority {len(X_maj)})")

# ── Geometric augmentation ────────────────────────────────────────────────────
# Targets: stretch (aspect ratio), scale change, padding, shift offset.
#
# Pipeline:
#   1. PadIfNeeded  — adds ~40% replicated border so the crop can "see" IC
#                     background at cell edges (padding simulation)
#   2. RandomResizedCrop — combines stretch (ratio) + scale in one op:
#        ratio=(0.55, 1.82): portrait→landscape stretch of the crop window
#        scale=(0.50, 0.90): crop covers 50-90% of the padded area
#   3. ShiftScaleRotate   — translates content within the frame (shift offset),
#        border filled with BORDER_REPLICATE (realistic IC surface)
#
# Only original images are augmented (files not starting with '_'), so
# re-running is safe and doesn't compound generated copies.
AUG_COPIES = 2   # augmented variants per original image

_PAD_TO = int(IMG_SIZE * 1.4)   # 224 × 1.4 ≈ 313 px padded canvas

_TRANSFORM = A.Compose([
    A.PadIfNeeded(
        min_height=_PAD_TO, min_width=_PAD_TO,
        border_mode=cv2.BORDER_REPLICATE,
        p=1.0,
    ),
    A.RandomResizedCrop(
        size=(IMG_SIZE, IMG_SIZE),
        scale=(0.50, 0.90),   # zoom: content fills 50-90% of output frame
        ratio=(0.55, 1.82),   # stretch: 0.55:1 (tall) → 1.82:1 (wide)
        interpolation=cv2.INTER_LINEAR,
        p=1.0,
    ),
    A.ShiftScaleRotate(
        shift_limit=0.10,     # ±10% translation offset
        scale_limit=0.0,      # no additional scale (handled above)
        rotate_limit=0,       # no rotation
        border_mode=cv2.BORDER_REPLICATE,
        p=0.6,
    ),
])

def augment_dataset(dataset_dir: Path, classes: list, n_copies: int = AUG_COPIES):
    """Generate n_copies augmented images per original training image for each class."""
    for cls in classes:
        folder = dataset_dir / "train" / cls
        originals = [p for p in sorted(folder.glob("*.*"))
                     if not p.stem.startswith("_")]
        count = 0
        for p in originals:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            for k in range(n_copies):
                aug = _TRANSFORM(image=img)["image"]
                cv2.imwrite(str(folder / f"_aug_{p.stem}_{k:02d}.png"), aug)
                count += 1
        print(f"[Augment] {cls}: +{count} images → total {len(originals) + count}")

# ── Run preprocessing pipeline ────────────────────────────────────────────────
# Order: SMOTE first (balances NoText), then augment both classes so synthetic
# samples also receive geometric variation.
apply_smote(DATASET_DIR, minority="NoText", majority="Text")
augment_dataset(DATASET_DIR, classes=["Text", "NoText"])
# ─────────────────────────────────────────────────────────────────────────────

model = YOLO(MODEL)

model.train(
    data=str(DATASET_DIR),
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH_SIZE,
    project=PROJECT,
    name=NAME,

    # Colour jitter — unchanged
    hsv_h=0.015,
    hsv_s=0.1,
    hsv_v=0.1,

    # Geometric — YOLO's built-in kept minimal; heavy geometric variation is
    # handled by the pre-generated augmented images above.
    degrees=0.0,
    translate=0.05,
    scale=0.05,
    fliplr=0.0,
    flipud=0.0,

    lr0=0.01,
    optimizer="AdamW",
    dropout=0.2,
    label_smoothing=0.1,

    save=True,
    pretrained=True,
    verbose=True,
    patience=20,
)

metrics = model.val()
print(metrics)
model.export(
    format="openvino",
    imgsz=IMG_SIZE,
    dynamic=False,
    half=False,
)
