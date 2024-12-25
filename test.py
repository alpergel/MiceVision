from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("app/Models/YOLOV8_INTERACT.pt")

# Define path to video file
source = "Videos/Month_1/5.22.21-7-21-Testing.MOV"

# Run inference on the source
results = model(source, stream=True,show=True)  # generator of Results objects