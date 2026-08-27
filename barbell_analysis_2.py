from ultralytics import YOLO
import cv2

# Load YOLO-World model
model = YOLO("yolov8s-world.pt")

# Detect only barbells
model.set_classes(["barbell"])

video_path = "videos/20260715_170015.mp4"

cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "output/barbell_detected.avi",
    cv2.VideoWriter_fourcc(*'XVID'),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO-World inference
    results = model.predict(
        frame,
        conf=0.2,
        verbose=False
    )

    for r in results:
        boxes = r.boxes

        print("Detected:", len(boxes))

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "barbell",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    out.write(frame)

cap.release()
out.release()
print("Video saved: output/barbell_detected.avi")