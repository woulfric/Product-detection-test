from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")

# model.train(data= 'dataset.yaml', imgsz= 640, device= 0, batch= 8, epochs= 100, workers= 0)

model.train(
    data='dataset.yaml',
    imgsz=640,          # Image size
    device=0,           # GPU device
    batch=16,           # Batch size
    epochs=100,         # Number of epochs
    workers=0,          # Data loading workers
    optimizer='SGD',    # Optimizer
    lr0=0.01,           # Initial learning rate
    momentum=0.9,       # Momentum
    weight_decay=0.0005, # Weight decay
    patience=50         # Early stopping patience
)