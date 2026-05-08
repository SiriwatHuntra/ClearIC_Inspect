from pathlib import Path
from ultralytics import YOLO

# Paths resolved relative to this script so the file runs from any working directory
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent

if __name__ == "__main__":

    MODEL       = str(SCRIPT_DIR / "yolo26n-cls.pt")
    DATASET_DIR = str(PROJECT_DIR / "Dataset")

    IMG_SIZE   = 224     # standard size for classification
    EPOCHS     = 100
    BATCH_SIZE = 64
    DEVICE     = "cpu"
    PROJECT    = str(PROJECT_DIR / "ClearIC_Insp")
    NAME       = "Text_cls"

    model = YOLO(MODEL)

    # Train
    model.train(
        data=DATASET_DIR,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT,
        name=NAME,

        # augmentation — kept mild for IC mark images
        hsv_h=0.015,
        hsv_s=0.1,
        hsv_v=0.1,
        degrees=0.0,
        translate=0.1,
        scale=0.05,
        fliplr=0.0,   # no horizontal flip (text orientation matters)
        flipud=0.0,   # no vertical flip

        lr0=0.01,
        optimizer="AdamW",

        save=True,
        pretrained=True,
        dropout=0.1,
        verbose=True,
    )

    # Validate
    metrics = model.val()
    print("======== Validation Metrics =======")
    print(metrics)

    # Export to OpenVINO for deployment on Raspberry Pi
    print("==== Exporting model to OpenVINO format ====")
    model.export(
        format="openvino",
        imgsz=IMG_SIZE,
        dynamic=False,
        half=False,
    )
    print("Model exported to OpenVINO format successfully.")


