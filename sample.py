"""
UrbanFlow AI - Lane-Aware Adaptive Traffic Optimization System
Fixed & Optimized Version
"""

import streamlit as st
import cv2
import torch
import math
import time
import tempfile
from ultralytics import YOLO

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(layout="wide")
st.title("🚦 UrbanFlow AI - Traffic Monitoring Dashboard")

# =========================================================
# LOAD MODEL (GPU ENABLED)
# =========================================================
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")  # replace with best.pt if needed
    model.to(device)
    return model, device

model, device = load_model()

# =========================================================
# FILE UPLOAD
# =========================================================
uploaded_file = st.file_uploader("📤 Upload Traffic Video", type=["mp4", "avi", "mov"])

# =========================================================
# UI LAYOUT
# =========================================================
col1, col2 = st.columns([3, 1])
video_placeholder = col1.empty()

with col2:
    st.subheader("📊 Traffic Metrics")
    vehicle_metric = st.empty()
    density_metric = st.empty()
    signal_metric = st.empty()
    fps_metric = st.empty()

start = st.button("▶ Start Monitoring")

# =========================================================
# MAIN EXECUTION
# =========================================================
if start:

    if uploaded_file is None:
        st.warning("⚠ Please upload a video first!")
        st.stop()

    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)

    if not cap.isOpened():
        st.error("❌ Error opening video")
        st.stop()

    frame_counter = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        height, width = frame.shape[:2]

        # =====================================================
        # LANE GEOMETRY
        # =====================================================
        center_x_lane = int(width * 0.37)
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
        # QUEUE THRESHOLDS
        # =====================================================
        LINE_CRITICAL = 100
        LINE_HIGH = 350
        LINE_NORMAL = 550

        max_depth = 0
        vehicle_count = 0

        # =====================================================
        # YOLO INFERENCE (GPU ENABLED)
        # =====================================================
        results = model(frame, conf=0.25, device=device, verbose=False)[0]

        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["car", "truck", "bus", "motorcycle"]:

                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                center_x = (x_min + x_max) // 2
                bottom_y = y_max

                # Lane check
                if (center_x - x1) * (y2 - y1) - (bottom_y - y1) * (x2 - x1) > 0:

                    vehicle_count += 1

                    if bottom_y > max_depth:
                        max_depth = bottom_y

                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # =====================================================
        # DENSITY LOGIC
        # =====================================================
        if max_depth < LINE_CRITICAL:
            density = "NORMAL"
            signal_time = 30
            color = (0, 255, 0)

        elif max_depth < LINE_HIGH:
            density = "HIGH"
            signal_time = 60
            color = (0, 255, 255)

        else:
            density = "VERY HIGH"
            signal_time = 90
            color = (0, 0, 255)

        # =====================================================
        # DRAWING
        # =====================================================
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)

        cv2.line(frame, (0, LINE_CRITICAL), (width, LINE_CRITICAL), (0, 0, 255), 2)
        cv2.line(frame, (0, LINE_HIGH), (width, LINE_HIGH), (0, 255, 255), 2)
        cv2.line(frame, (0, LINE_NORMAL), (width, LINE_NORMAL), (0, 255, 0), 2)

        cv2.putText(frame, f"Density: {density}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # =====================================================
        # FPS
        # =====================================================
        frame_counter += 1
        elapsed = time.time() - start_time
        fps = frame_counter / elapsed if elapsed > 0 else 0

        # Convert for Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # =====================================================
        # UPDATE UI
        # =====================================================
        video_placeholder.image(frame, channels="RGB", use_container_width=True)

        vehicle_metric.metric("🚗 Vehicles", vehicle_count)
        density_metric.metric("🚦 Density", density)
        signal_metric.metric("⏱ Signal Time", f"{signal_time} sec")
        fps_metric.metric("⚡ FPS", f"{fps:.2f}")

        time.sleep(0.01)

    cap.release()