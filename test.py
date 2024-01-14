from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
results = model.train(data='coco128.yaml', epochs=3)

# Export the model to ONNX format
success = model.export(format='onnx')
print(success)
with open(success, "rb") as rs:
    with open("yolov8n.onnx", "wb") as f:
        f.write(rs.read())
