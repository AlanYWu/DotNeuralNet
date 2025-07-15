from ultralytics import YOLO
import torch
import os

# Path to your data.yaml (update if you use a different path)
DATA_YAML = os.path.abspath('yolo_dataset/data.yaml')
# Choose a YOLOv8 model (nano, small, medium, large, xlarge)
MODEL = 'yolov8l.pt'  # or 'yolov8s.pt', 'yolov8m.pt', etc.

# Device selection: 'mps' for Apple Silicon, 'cuda' for NVIDIA, or 'cpu'
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"Using device: {device}")

# Load model
model = YOLO(MODEL)

# Train
model.train(
    data=DATA_YAML,
    epochs=100,           # Set as needed
    imgsz=640,            # Image size (can adjust)
    device=device,
    batch=4,              # Adjust for your RAM/VRAM
    workers=10,            # Adjust for your CPU
    project='yolo_braille',
    name='exp',
    exist_ok=True
)