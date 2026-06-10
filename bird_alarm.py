import cv2
import torch
import requests

PI_IP = "10.40.19.67"  # <-- CHANGE to your Pi IP

# Load YOLOv8
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

# Read stream from Pi
cap = cv2.VideoCapture(f"http://{PI_IP}:5000/video")

bird_active = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    bird_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "bird":
                bird_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, "Bird", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Send signal to Pi
    if bird_detected and not bird_active:
        requests.get(f"http://{PI_IP}:5000/bird")
        bird_active = True

    if not bird_detected and bird_active:
        requests.get(f"http://{PI_IP}:5000/stop")
        bird_active = False

    cv2.imshow("Bird Detection (Laptop)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()