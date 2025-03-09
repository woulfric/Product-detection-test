from ultralytics import YOLO

model = YOLO("runs/segment/train3/weights/best.pt")

metrics = model.val(data='dataset.yaml', split='test', workers= 0)

# Print the metrics
print("mAP@0.5:", metrics.box.map50)
print("mAP@0.5:0.95:", metrics.box.map)
print("Precision:", metrics.box.precision)
print("Recall:", metrics.box.recall)
print("F1 Score:", metrics.box.f1)