import cv2
import torch
import math
from ultralytics import YOLO
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python process_image.py <image_path>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: File '{image_path}' not found.")
    sys.exit(1)

print("Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8s.pt")
model.to(device)

print(f"Processing image: {image_path}")
frame = cv2.imread(image_path)
if frame is None:
    print("Error: Could not read image.")
    sys.exit(1)

frame = cv2.resize(frame, (1280, 720))
height, width = frame.shape[:2]

# Same lane geometry as video
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

        # Tilted lane logic (same as video)
        if (center_x_vehicle - x1) * (y2 - y1) - (bottom_y - y1) * (x2 - x1) > 0:
            right_lane_vehicle_count += 1
            if bottom_y > max_depth_right_lane:
                max_depth_right_lane = bottom_y

            # Draw sci-fi bounding boxes
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

# Draw lane and depth lines
cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
cv2.line(frame, (0, LINE_CRITICAL), (width, LINE_CRITICAL), (0, 0, 255), 1)
cv2.line(frame, (0, LINE_HIGH), (width, LINE_HIGH), (0, 255, 255), 1)
cv2.line(frame, (0, LINE_NORMAL), (width, LINE_NORMAL), (0, 255, 0), 1)
cv2.putText(frame, f"STATUS: {density}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 2)
cv2.putText(frame, f"Vehicles: {right_lane_vehicle_count}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 2)

base_name = os.path.basename(image_path)
out_path = f"output_{base_name}"
cv2.imwrite(out_path, frame)
print(f"Done! Saved futuristic output to {out_path}")
