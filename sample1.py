"""
UrbanFlow AI - Lane-Aware Adaptive Traffic Optimization System
AI4Dev '26 Hackathon Submission

Author: Rohith Lakshman G

IMPORTANT NOTE:
Lane boundary lines in this prototype are manually configured
based on fixed camera positioning.

For real-world deployment, these lane boundaries can be:
- Calibrated per intersection
- Dynamically detected using computer vision
- Integrated with smart traffic infrastructure

Description:
This application performs:
- Real-time vehicle detection using YOLOv8
- Manual lane-aware vehicle classification
- Density estimation using queue depth
- Adaptive signal timing simulation
- Live dashboard monitoring using Streamlit
"""

"""
UrbanFlow AI - Lane-Aware Adaptive Traffic Optimization System
AI4Dev '26 Hackathon Submission

Author: G Rohith Lakshman 

Description:
- Real-time vehicle detection using YOLOv8 Nano
- Lane-aware vehicle classification (Tilted separator)
- Queue depth based density estimation
- Adaptive signal timing logic
- Live monitoring dashboard using Streamlit
"""

import streamlit as st
import cv2
import torch
import math
import time
from ultralytics import YOLO
import numpy as np

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(layout="wide")
st.title("🚦 UrbanFlow AI - Lane One Traffic Dashboard")

# =========================================================
# LOAD YOLO MODEL (AUTO DOWNLOAD IF NOT AVAILABLE)
# =========================================================
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")  # Auto-downloads if missing
    model.to(device)
    return model, device

model, device = load_model()

# =========================================================
# VIDEO SOURCE
# =========================================================
VIDEO_PATH = "signal.mp4"  # Replace with your sample video name

# =========================================================
# DASHBOARD LAYOUT
# =========================================================
col1, col2 = st.columns([3, 1])
video_placeholder = col1.empty()

with col2:
    st.subheader("📊 Lane One Metrics")
    vehicle_metric = st.empty()
    density_metric = st.empty()
    signal_metric = st.empty()
    fps_metric = st.empty()

start_button = st.button("▶ Start Monitoring")

# =========================================================
# MAIN EXECUTION
# =========================================================
if start_button:

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        st.error("Error: Unable to open video file.")
    else:

        frame_counter = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (1280, 720))
            height, width = frame.shape[:2]

            # =====================================================
            # LANE ONE GEOMETRY (TILTED RIGHT SEPARATOR)
            # =====================================================
            center_x_lane = int(width * 0.35)
            center_y_lane = height // 2
            angle = -34
            length = height

            dx = int(length * math.sin(math.radians(angle)))
            dy = int(length * math.cos(math.radians(angle)))

            x1 = center_x_lane - dx
            y1 = center_y_lane - dy
            x2 = center_x_lane + dx
            y2 = center_y_lane + dy

            # =====================================================
            # DENSITY THRESHOLDS (Queue Depth Based)
            # =====================================================
            LINE_CRITICAL = 100
            LINE_HIGH = 350
            LINE_NORMAL = 550

            max_depth_right_lane = 0
            right_lane_vehicle_count = 0

            # =====================================================
            # YOLO INFERENCE
            # =====================================================
            results = model(frame, verbose=False)[0]

            for box in results.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]

                if label in ["car", "truck", "bus", "motorcycle"]:

                    x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                    center_x_vehicle = (x_min + x_max) // 2
                    bottom_y = y_max

                    # Lane classification using cross-product
                    if (center_x_vehicle - x1) * (y2 - y1) - (bottom_y - y1) * (x2 - x1) > 0:

                        right_lane_vehicle_count += 1

                        if bottom_y > max_depth_right_lane:
                            max_depth_right_lane = bottom_y

                        cv2.rectangle(frame, (x_min, y_min),
                                      (x_max, y_max), (0, 255, 0), 2)

            # =====================================================
            # DENSITY DECISION ENGINE
            # =====================================================
            if max_depth_right_lane < LINE_CRITICAL:
                density = "NORMAL"
                signal_time = 30
                color = (0, 255, 0)

            elif max_depth_right_lane < LINE_HIGH:
                density = "HIGH"
                signal_time = 60
                color = (0, 255, 255)

            else:
                density = "VERY HIGH"
                signal_time = 90
                color = (0, 0, 255)

            # =====================================================
            # DRAW VISUAL ELEMENTS
            # =====================================================
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

            cv2.line(frame, (0, LINE_CRITICAL),
                     (width, LINE_CRITICAL), (0, 0, 255), 2)
            cv2.line(frame, (0, LINE_HIGH),
                     (width, LINE_HIGH), (0, 255, 255), 2)
            cv2.line(frame, (0, LINE_NORMAL),
                     (width, LINE_NORMAL), (0, 255, 0), 2)

            cv2.putText(frame, f"Density: {density}",
                        (40, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, color, 3)

            # =====================================================
            # FPS CALCULATION
            # =====================================================
            frame_counter += 1
            elapsed_time = time.time() - start_time
            fps = frame_counter / elapsed_time if elapsed_time > 0 else 0

            # Convert BGR → RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # =====================================================
            # UPDATE DASHBOARD
            # =====================================================
            video_placeholder.image(frame_rgb,
                                    channels="RGB",
                                    use_container_width=True)

            vehicle_metric.metric("🚗 Vehicles (Right Lane)",
                                  right_lane_vehicle_count)

            density_metric.metric("🚦 Density Level", density)

            signal_metric.metric("⏱ Signal Time",
                                 f"{signal_time} sec")

            fps_metric.metric("⚡ FPS", f"{fps:.2f}")

            time.sleep(0.02)

        cap.release()