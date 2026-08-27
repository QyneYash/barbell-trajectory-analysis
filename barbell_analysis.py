from ultralytics import YOLO

model = YOLO("yolov8s-world.pt")

model.set_classes([
    "barbell",
    "weightlifting barbell",
    "gym barbell",
    "olympic barbell"
])

results = model("videos/20260715_170046.mp4", 
    conf=0.05,
    save=True
    )

for r in results:
    print("Classes:", r.boxes.cls)
    print("Conf:", r.boxes.conf)