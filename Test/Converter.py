from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("IC_Search.pt")

# Export the model
model.export(format="openvino")  # creates 'yolo26n_openvino_model/'

# Load the exported OpenVINO model
# ov_model = YOLO("IC_Search_openvino_model/")  # Load the OpenVINO model

# Run inference
# results = ov_model("Input/test.jpg")

# print(results)