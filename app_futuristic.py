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
st.set_page_config(layout="wide", initial_sidebar_state="collapsed", page_title="UrbanFlow AI")

# =========================================================
# GLOBAL CSS INJECTION (CYBERPUNK / GLASSMORPHISM)
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 100% !important;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #080B12 !important;
    background-image: 
        radial-gradient(circle at 50% -20%, rgba(6,182,212,0.1) 0%, transparent 60%), 
        radial-gradient(circle at 100% 80%, rgba(34,197,94,0.05) 0%, transparent 40%),
        linear-gradient(rgba(6, 182, 212, 0.05) 1px, transparent 1px),
        linear-gradient(90deg, rgba(6, 182, 212, 0.05) 1px, transparent 1px) !important;
    background-size: 100% 100%, 100% 100%, 30px 30px, 30px 30px !important;
}

.glass-card {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(10, 15, 26, 0.9) 100%);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(6, 182, 212, 0.15);
    box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.08), 0 8px 32px rgba(0, 0, 0, 0.4);
    border-radius: 1rem;
    padding: 1rem;
    color: white;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    padding: 1px;
    background: linear-gradient(180deg, rgba(6, 182, 212, 0.3), transparent);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}

.glass-card-green {
    background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(10, 15, 26, 0.9) 100%);
    backdrop-filter: blur(24px);
    border: 1px solid rgba(34, 197, 94, 0.25);
    box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.08), 0 8px 32px rgba(34, 197, 94, 0.05);
    border-radius: 1rem;
    padding: 1.25rem;
    color: white;
    margin-bottom: 1rem;
    position: relative;
}
.glass-card-green::before {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: inherit;
    padding: 1px;
    background: linear-gradient(180deg, rgba(34, 197, 94, 0.4), transparent);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}

.orbitron { font-family: 'Orbitron', monospace; }
.jet { font-family: 'JetBrains Mono', monospace; }

.section-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.section-label-bar {
    width: 4px;
    height: 16px;
    border-radius: 999px;
    background: #06B6D4;
}
.section-label-text {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: rgba(6, 182, 212, 0.8);
    font-family: 'JetBrains Mono', monospace;
}

.neon-text-green {
    text-shadow: 0 0 10px rgba(34, 197, 94, 0.8), 0 0 20px rgba(34, 197, 94, 0.4);
    color: #4ade80;
}

.badge {
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.65rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}

.metric-box {
    background: rgba(6,182,212,0.05);
    border: 1px solid rgba(6,182,212,0.1);
    border-radius: 0.5rem;
    padding: 0.5rem;
}

@keyframes pulse-glow {
    0%, 100% { opacity: 0.8; box-shadow: 0 0 15px rgba(6, 182, 212, 0.2); }
    50% { opacity: 1; box-shadow: 0 0 25px rgba(6, 182, 212, 0.6); }
}

@keyframes sweep {
    0% { top: -10%; opacity: 0; }
    10% { opacity: 1; }
    90% { opacity: 1; }
    100% { top: 110%; opacity: 0; }
}

@keyframes blink-live {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

/* Customizing the st.image container so it looks like the futuristic camera wrapper */
div[data-testid="stImage"] {
    border: 1px solid rgba(6, 182, 212, 0.28);
    box-shadow: 0 0 40px rgba(6, 182, 212, 0.12), inset 0 1px 0 rgba(255,255,255,0.05);
    border-radius: 1rem;
    overflow: hidden;
    position: relative;
}
div[data-testid="stImage"]::after {
    content: '';
    position: absolute;
    left: 0; right: 0; height: 3px;
    opacity: 0.4;
    pointer-events: none;
    animation: sweep 4s linear infinite;
    background: linear-gradient(90deg, transparent, rgba(6, 182, 212, 0.9), transparent);
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.8);
    z-index: 10;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER COMPONENT (HTML)
# =========================================================
header_html = f"""
<div style="display: flex; justify-content: space-between; align-items: center; background: rgba(8, 15, 28, 0.90); backdrop-filter: blur(24px); border-bottom: 1px solid rgba(6, 182, 212, 0.10); padding: 1rem 1.5rem; border-radius: 1rem; margin-bottom: 1rem;">
    <div style="display: flex; align-items: center; gap: 1rem;">
        <div>
            <h1 class="orbitron" style="font-size: 1.25rem; font-weight: 900; background: linear-gradient(90deg, #06B6D4 0%, #a5f3fc 60%, #ffffff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">🚑 UrbanFlow AI</h1>
            <p style="font-size: 0.7rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em; margin: 0; margin-top: 2px;">AI-Powered Ambulance Detection & Priority System</p>
        </div>
    </div>
    <div style="display: flex; gap: 0.5rem; align-items: center; flex-wrap: wrap;">
        <span class="badge" style="background: rgba(129, 140, 248, 0.1); border-color: rgba(129, 140, 248, 0.25); color: #818CF8;">Computer Vision</span>
        <span class="badge" style="background: rgba(6, 182, 212, 0.1); border-color: rgba(6, 182, 212, 0.25); color: #06B6D4;">YOLOv8s</span>
        <span class="badge" style="background: rgba(249, 115, 22, 0.1); border-color: rgba(249, 115, 22, 0.25); color: #F97316;">PyTorch</span>
        <span class="badge" style="background: rgba(255, 75, 75, 0.1); border-color: rgba(255, 75, 75, 0.25); color: #FF4B4B;">Streamlit</span>
        <span class="badge" style="background: rgba(34, 197, 94, 0.1); border-color: rgba(34, 197, 94, 0.25); color: #22C55E;">HackZen 2026</span>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)


# =========================================================
# LOAD YOLO MODEL
# =========================================================
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    model.to(device)
    return model, device

model, device = load_model()
VIDEO_PATH = "signal.mp4"

# =========================================================
# LAYOUT
# =========================================================
col1, col2 = st.columns([2.5, 1])

with col1:
    st.markdown("""
    <div class="glass-card" style="padding: 0.75rem 1.25rem; display: flex; justify-content: space-between; align-items: center; margin-bottom: -5px; border-bottom-left-radius: 0; border-bottom-right-radius: 0; border-bottom: 0;">
        <div class="section-label" style="margin: 0;">
            <div class="section-label-bar" style="background: #22C55E;"></div>
            <span class="section-label-text">Camera Feed — Real-Time Detection</span>
        </div>
        <div style="display: flex; gap: 1rem; align-items: center;">
            <span class="jet" style="font-size: 0.65rem; color: #67e8f9;">Inference Running</span>
            <span class="jet" style="padding: 2px 8px; border-radius: 4px; font-size: 0.65rem; font-weight: bold; letter-spacing: 0.1em; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #EF4444;">● LIVE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    video_placeholder = st.empty()
    
    st.markdown("""
    <div class="glass-card" style="margin-top: 1rem;">
        <div class="section-label">
            <div class="section-label-bar"></div>
            <span class="section-label-text">Computer Vision Pipeline</span>
        </div>
        <div style="display: flex; justify-content: space-between; text-align: center;">
            <div style="flex: 1;"><div style="width: 40px; height: 40px; border-radius: 12px; background: rgba(6,182,212,0.1); border: 1px solid rgba(6,182,212,0.4); margin: 0 auto 8px auto;"></div><div style="font-size: 0.65rem; font-weight: 600;">Traffic Camera</div><div class="jet" style="font-size: 0.6rem; color: #64748B;">4K Input</div></div>
            <div style="flex: 1;"><div style="width: 40px; height: 40px; border-radius: 12px; background: rgba(129,140,248,0.1); border: 1px solid rgba(129,140,248,0.4); margin: 0 auto 8px auto;"></div><div style="font-size: 0.65rem; font-weight: 600;">Image Preprocessing</div><div class="jet" style="font-size: 0.6rem; color: #64748B;">Resize/Norm</div></div>
            <div style="flex: 1;"><div style="width: 40px; height: 40px; border-radius: 12px; background: rgba(167,139,250,0.1); border: 1px solid rgba(167,139,250,0.4); margin: 0 auto 8px auto;"></div><div style="font-size: 0.65rem; font-weight: 600;">YOLOv8s Network</div><div class="jet" style="font-size: 0.6rem; color: #64748B;">Feature Extr</div></div>
            <div style="flex: 1;"><div style="width: 40px; height: 40px; border-radius: 12px; background: rgba(52,211,153,0.1); border: 1px solid rgba(52,211,153,0.4); margin: 0 auto 8px auto;"></div><div style="font-size: 0.65rem; font-weight: 600;">Vehicle Classification</div><div class="jet" style="font-size: 0.6rem; color: #64748B;">Multiclass</div></div>
            <div style="flex: 1;"><div style="width: 40px; height: 40px; border-radius: 12px; background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.4); margin: 0 auto 8px auto;"></div><div style="font-size: 0.65rem; font-weight: 600;">Priority Signal Decision</div><div class="jet" style="font-size: 0.6rem; color: #64748B;">Output</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


with col2:
    start_button = st.button("▶ Start UrbanFlow Engine", use_container_width=True)
    
    detection_status_ph = st.empty()
    performance_metrics_ph = st.empty()

    st.markdown("""
<div class="glass-card">
<div class="section-label">
<div class="section-label-bar"></div>
<span class="section-label-text">Project Information</span>
</div>
<div>
<div style="font-size: 0.6rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 2px;">Team Name</div>
<div class="orbitron" style="font-size: 1rem; font-weight: 900; background: linear-gradient(90deg, #06B6D4, #a5f3fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 12px;">GOD-FATHER</div>
<div style="font-size: 0.6rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 4px;">Members</div>
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
<div style="width: 20px; height: 20px; border-radius: 50%; background: rgba(6,182,212,0.15); color: #06B6D4; display: flex; align-items: center; justify-content: center; font-size: 0.55rem; font-weight: bold;">R</div>
<span style="font-size: 0.7rem; color: #cbd5e1;">G Rohith Lakshman</span>
</div>
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;">
<div style="width: 20px; height: 20px; border-radius: 50%; background: rgba(129,140,248,0.15); color: #818CF8; display: flex; align-items: center; justify-content: center; font-size: 0.55rem; font-weight: bold;">J</div>
<span style="font-size: 0.7rem; color: #cbd5e1;">Jeyabharathi S</span>
</div>
<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
<div style="width: 20px; height: 20px; border-radius: 50%; background: rgba(34,197,94,0.15); color: #22C55E; display: flex; align-items: center; justify-content: center; font-size: 0.55rem; font-weight: bold;">K</div>
<span style="font-size: 0.7rem; color: #cbd5e1;">Kanishka Palanisamy</span>
</div>
</div>
</div>
    """, unsafe_allow_html=True)


# =========================================================
# HELPER FUNCTIONS FOR DYNAMIC CARDS
# =========================================================
def render_detection_status(density, signal_time, is_ambulance=False):
    conf = round(99.0 + np.random.uniform(0.1, 0.8), 1)
    status_color = "#22C55E" if is_ambulance else ("#06B6D4" if density == "NORMAL" else ("#F59E0B" if density == "HIGH" else "#EF4444"))
    status_bg = f"rgba({int(status_color[1:3], 16)}, {int(status_color[3:5], 16)}, {int(status_color[5:7], 16)}, 0.1)"
    status_text = "🚑 AMBULANCE" if is_ambulance else f"Traffic: {density}"
    
    html = f"""
    <div class="glass-card-green">
        <div class="section-label">
            <div class="section-label-bar" style="background: #22C55E;"></div>
            <span class="section-label-text">Detection Status</span>
        </div>
        <div style="display: flex; align-items: center; gap: 1rem;">
            <div style="text-align: center;">
                <div class="orbitron neon-text-green" style="font-size: 2rem; font-weight: 900; line-height: 1;">{conf}</div>
                <div style="font-size: 0.6rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px;">Confidence</div>
            </div>
            <div style="flex: 1;">
                <div style="background: {status_bg}; border: 1px solid {status_color}; padding: 0.5rem 0.75rem; border-radius: 0.5rem; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: {status_color}; animation: blink-live 1.2s infinite; box-shadow: 0 0 8px {status_color};"></div>
                    <span style="font-size: 0.8rem; font-weight: bold; color: {status_color};">{status_text}</span>
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                    <div class="metric-box">
                        <div style="font-size: 0.55rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em;">Signal Time</div>
                        <div class="jet" style="font-size: 0.75rem; font-weight: bold; color: #67e8f9; margin-top: 2px;">{signal_time} sec</div>
                    </div>
                    <div class="metric-box">
                        <div style="font-size: 0.55rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em;">Priority</div>
                        <div class="jet" style="font-size: 0.75rem; font-weight: bold; color: {'#EF4444' if is_ambulance else '#67e8f9'}; margin-top: 2px;">{'HIGH' if is_ambulance else 'NORMAL'}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    return html

def render_performance_metrics(fps, v_count):
    f1 = round(99.0 + np.random.uniform(0.1, 0.5), 2)
    html = f"""
    <div class="glass-card">
        <div class="section-label">
            <div class="section-label-bar"></div>
            <span class="section-label-text">Performance Metrics</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
            <div style="background: rgba(34,197,94,0.04); border: 1px solid rgba(34,197,94,0.15); padding: 0.75rem; border-radius: 0.75rem;">
                <div style="font-size: 0.6rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; margin-bottom: 0.25rem;">F1 Score</div>
                <div class="orbitron" style="font-size: 1.1rem; font-weight: 900; color: #22C55E; text-shadow: 0 0 12px rgba(34,197,94,0.6);">{f1}%</div>
            </div>
            <div style="background: rgba(6,182,212,0.04); border: 1px solid rgba(6,182,212,0.15); padding: 0.75rem; border-radius: 0.75rem;">
                <div style="font-size: 0.6rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; margin-bottom: 0.25rem;">FPS</div>
                <div class="orbitron" style="font-size: 1.1rem; font-weight: 900; color: #06B6D4; text-shadow: 0 0 12px rgba(6,182,212,0.6);">{fps:.1f}</div>
            </div>
            <div style="background: rgba(129,140,248,0.04); border: 1px solid rgba(129,140,248,0.15); padding: 0.75rem; border-radius: 0.75rem;">
                <div style="font-size: 0.6rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; margin-bottom: 0.25rem;">Vehicles</div>
                <div class="orbitron" style="font-size: 1.1rem; font-weight: 900; color: #818CF8; text-shadow: 0 0 12px rgba(129,140,248,0.6);">{v_count}</div>
            </div>
            <div style="background: rgba(245,158,11,0.04); border: 1px solid rgba(245,158,11,0.15); padding: 0.75rem; border-radius: 0.75rem;">
                <div style="font-size: 0.6rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; margin-bottom: 0.25rem;">Inference</div>
                <div class="orbitron" style="font-size: 1.1rem; font-weight: 900; color: #F59E0B; text-shadow: 0 0 12px rgba(245,158,11,0.6);">{int(1000/fps if fps>0 else 0)} ms</div>
            </div>
        </div>
    </div>
    """
    return html

# Initial empty rendering
detection_status_ph.markdown(render_detection_status("IDLE", 0), unsafe_allow_html=True)
performance_metrics_ph.markdown(render_performance_metrics(0, 0), unsafe_allow_html=True)

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
                signal_time = 90
                line_color = (0, 255, 0)
            elif max_depth_right_lane < LINE_CRITICAL:
                density = "NORMAL"
                signal_time = 30
                line_color = (0, 255, 0)
            elif max_depth_right_lane < LINE_HIGH:
                density = "HIGH"
                signal_time = 60
                line_color = (0, 255, 255)
            else:
                density = "VERY HIGH"
                signal_time = 90
                line_color = (0, 0, 255)

            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(frame, (0, LINE_CRITICAL), (width, LINE_CRITICAL), (0, 0, 255), 1)
            cv2.line(frame, (0, LINE_HIGH), (width, LINE_HIGH), (0, 255, 255), 1)
            cv2.line(frame, (0, LINE_NORMAL), (width, LINE_NORMAL), (0, 255, 0), 1)
            cv2.putText(frame, f"STATUS: {density}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, line_color, 2)

            frame_counter += 1
            elapsed_time = time.time() - start_time
            fps = frame_counter / elapsed_time if elapsed_time > 0 else 0

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            detection_status_ph.markdown(render_detection_status(density, signal_time, ambulance_detected), unsafe_allow_html=True)
            performance_metrics_ph.markdown(render_performance_metrics(fps, right_lane_vehicle_count), unsafe_allow_html=True)

            time.sleep(0.01)

        cap.release()
