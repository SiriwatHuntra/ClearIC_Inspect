import shutil
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
import albumentations as A
from ultralytics import YOLO

# ── Paths ─────────────────────────────────────────────────────────────────────
# /kaggle/input is read-only — copy dataset to writable /kaggle/working first.
PROJECT_DIR = Path("/kaggle/input/datasets/siriwat121")
DATASET_DIR = PROJECT_DIR / "ic-font/Dataset"   # read-only source
WORK_DIR    = Path("/kaggle/working/Dataset")    # writable working copy

MODEL    = str(PROJECT_DIR / "model-yolo/yolo26n-cls.pt")
IMG_SIZE = 224
EPOCHS   = 100
BATCH    = 64
PROJECT  = "/kaggle/working/Model_IC"
NAME     = "Text_cls"

# ── Step 0: copy to writable workspace ───────────────────────────────────────
if WORK_DIR.exists():
    print(f"[Setup] {WORK_DIR} already exists, skipping copy.", flush=True)
else:
    print(f"[Setup] Copying dataset → {WORK_DIR} ...", flush=True)
    shutil.copytree(DATASET_DIR, WORK_DIR)
    print(f"[Setup] Done — {sum(1 for _ in WORK_DIR.rglob('*.*'))} files.", flush=True)

# ── SMOTE balancing ───────────────────────────────────────────────────────────
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

    print(f"[SMOTE] Loading images ...", flush=True)
    X_maj, _ = _load_flat(maj_dir, SMOTE_SIZE)
    X_min, _ = _load_flat(min_dir, SMOTE_SIZE)
    print(f"[SMOTE] {majority}={len(X_maj)}  {minority}={len(X_min)} — fitting ...", flush=True)

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
    print(f"[SMOTE] +{count} synthetic {minority} → total {len(X_min)+count} "
          f"(was {len(X_min)}, majority {len(X_maj)})", flush=True)

# ── Geometric augmentation ────────────────────────────────────────────────────
AUG_COPIES = 2
_PAD_TO    = int(IMG_SIZE * 1.4)

_TRANSFORM = A.Compose([
    A.PadIfNeeded(
        min_height=_PAD_TO, min_width=_PAD_TO,
        border_mode=cv2.BORDER_REPLICATE,
        p=1.0,
    ),
    A.RandomResizedCrop(
        size=(IMG_SIZE, IMG_SIZE),
        scale=(0.50, 0.90),
        ratio=(0.55, 1.82),
        interpolation=cv2.INTER_LINEAR,
        p=1.0,
    ),
    A.ShiftScaleRotate(
        shift_limit=0.10,
        scale_limit=0.0,
        rotate_limit=0,
        border_mode=cv2.BORDER_REPLICATE,
        p=0.6,
    ),
])

def augment_dataset(dataset_dir: Path, classes: list, n_copies: int = AUG_COPIES):
    for cls in classes:
        folder    = dataset_dir / "train" / cls
        originals = [p for p in sorted(folder.glob("*.*"))
                     if not p.stem.startswith("_")]
        count = 0
        for p in tqdm(originals, desc=f"[Augment] {cls}", unit="img"):
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            for k in range(n_copies):
                aug = _TRANSFORM(image=img)["image"]
                cv2.imwrite(str(folder / f"_aug_{p.stem}_{k:02d}.png"), aug)
                count += 1
        print(f"[Augment] {cls}: +{count} → total {len(originals)+count}", flush=True)

# ── Run preprocessing ─────────────────────────────────────────────────────────
apply_smote(WORK_DIR, minority="NoText", majority="Text")
augment_dataset(WORK_DIR, classes=["Text", "NoText"])

# ── Train ─────────────────────────────────────────────────────────────────────
model = YOLO(MODEL)

model.train(
    data=str(WORK_DIR),
    imgsz=IMG_SIZE,
    epochs=EPOCHS,
    batch=BATCH,
    project=PROJECT,
    name=NAME,

    # Colour jitter
    hsv_h=0.015,
    hsv_s=0.1,
    hsv_v=0.1,

    # Geometric (light — heavy variation handled by pre-augmented images)
    degrees=0.0,
    translate=0.05,
    scale=0.05,
    fliplr=0.0,
    flipud=0.0,

    # Regularisation — anti-overfit
    freeze=10,             # freeze backbone, train head only
    erasing=0.4,           # random erase patches — forces partial-mark robustness
    mixup=0.15,            # blend two images — prevents texture memorisation
    weight_decay=0.001,    # L2 regularisation (up from default 0.0005)
    dropout=0.2,
    label_smoothing=0.1,

    lr0=0.001,             # lower LR for fine-tuning pretrained backbone
    optimizer="AdamW",

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
