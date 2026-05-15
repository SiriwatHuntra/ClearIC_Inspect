from ultralytics import YOLO
from pathlib import Path

# Colab paths
PROJECT_DIR = Path("/kaggle/input/datasets/siriwat121")
DATASET_DIR = PROJECT_DIR / "ic-font/Dataset"

MODEL = str(PROJECT_DIR / "model-yolo/yolo26n-cls.pt")

IMG_SIZE   = 224
EPOCHS     = 100
BATCH_SIZE = 64
PROJECT = "/kaggle/working/Model_IC"
NAME       = "Text_cls"

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

    save=True,
    pretrained=True,
    dropout=0.1,
    verbose=True,
    patience=15,
)

metrics = model.val()
print(metrics)
model.export(
    format="openvino",
    imgsz=IMG_SIZE,
    dynamic=False,
    half=False,
)