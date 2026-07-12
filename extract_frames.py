import cv2
import torch
import math
from ultralytics import YOLO
import os

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8s.pt")
model.to(device)

VIDEO_PATH = "signal.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

print("Processing video to extract frames...")

captured = {
    "NORMAL": False,
    "HIGH": False,
    "VERY HIGH": False,
}

# Ensure we capture a frame with some vehicles for NORMAL, not just empty
min_vehicles_for_normal = 2

while cap.isOpened() and not all(captured.values()):
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    height, width = frame.shape[:2]

    center_x_lane = int(width * 0.35)
    center_y_lane = height // 2
    angle = -34
    length = height
    dx = int(length * math.sin(math.radians(angle)))
    dy = int(length * math.cos(math.radians(angle)))
    x1, y1 = center_x_lane - dx, center_y_lane - dy
    x2, y2 = center_x_lane + dx, center_y_lane + dy

    LINE_CRITICAL = 100
    LINE_HIGH = 350
    LINE_NORMAL = 550
    max_depth_right_lane = 0
    right_lane_vehicle_count = 0
    ambulance_detected = False

    results = model(frame, verbose=False)[0]

    for box in results.boxes:
        cls = int(box.cls[0])
        label = model.names[cls]

        if label in ["car", "truck", "bus", "motorcycle"] or "ambulance" in label.lower():
            if "ambulance" in label.lower():
                ambulance_detected = True

            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
            center_x_vehicle = (x_min + x_max) // 2
            bottom_y = y_max

            if (center_x_vehicle - x1) * (y2 - y1) - (bottom_y - y1) * (x2 - x1) > 0:
                right_lane_vehicle_count += 1
                if bottom_y > max_depth_right_lane:
                    max_depth_right_lane = bottom_y

                b_color = (0, 255, 0) if "ambulance" in label.lower() else (212, 182, 6) # Cyan in BGR
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), b_color, 2)
                conf_val = round(float(box.conf[0]) * 100, 1)
                cv2.rectangle(frame, (x_min, y_min - 20), (x_min + 100, y_min), b_color, -1)
                cv2.putText(frame, f"{label.upper()} {conf_val}%", (x_min + 5, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if ambulance_detected:
        density = "EMERGENCY"
        line_color = (0, 255, 0)
    elif max_depth_right_lane < LINE_CRITICAL:
        density = "NORMAL"
        line_color = (0, 255, 0)
    elif max_depth_right_lane < LINE_HIGH:
        density = "HIGH"
        line_color = (0, 255, 255)
    else:
        density = "VERY HIGH"
        line_color = (0, 0, 255)

    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
    cv2.line(frame, (0, LINE_CRITICAL), (width, LINE_CRITICAL), (0, 0, 255), 1)
    cv2.line(frame, (0, LINE_HIGH), (width, LINE_HIGH), (0, 255, 255), 1)
    cv2.line(frame, (0, LINE_NORMAL), (width, LINE_NORMAL), (0, 255, 0), 1)
    cv2.putText(frame, f"STATUS: {density}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 2)

    if density == "VERY HIGH" and not captured["VERY HIGH"]:
        cv2.imwrite("screenshot_very_high_density.jpg", frame)
        captured["VERY HIGH"] = True
        print("Captured Very High Density state.")
    elif density == "HIGH" and not captured["HIGH"]:
        cv2.imwrite("screenshot_high_density.jpg", frame)
        captured["HIGH"] = True
        print("Captured High Density state.")
    elif density == "NORMAL" and not captured["NORMAL"] and right_lane_vehicle_count >= min_vehicles_for_normal:
        cv2.imwrite("screenshot_normal_density.jpg", frame)
        captured["NORMAL"] = True
        print("Captured Normal Density state.")

cap.release()
print("Extraction complete! Images saved to disk.")
