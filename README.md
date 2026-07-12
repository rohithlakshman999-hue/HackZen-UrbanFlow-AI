# 🚦 UrbanFlow AI
**Lane-Aware Adaptive Traffic Optimization System**  
*AI4Dev '26 Hackathon Submission*

## 📌 Project Overview
UrbanFlow AI is an AI-powered traffic optimization system designed to dynamically adjust signal timings based on real-time lane-wise vehicle density.

The system performs:
- 🚗 Real-time vehicle detection using YOLOv8
- 🛣 Lane-aware vehicle classification using geometric lane separation
- 📊 Queue depth estimation for density calculation
- 🚦 Adaptive signal timing logic
- 📈 Live monitoring dashboard using Streamlit

## 🎯 Problem Statement
Urban intersections often use fixed signal timers that fail to adapt to real-time traffic density. UrbanFlow AI introduces a lane-aware, adaptive signal control mechanism to reduce congestion and optimize traffic flow.

## 🧠 Key Features
- Real-time object detection (YOLOv8)
- Tilted lane separation logic
- Density classification (Normal / High / Very High / Emergency)
- Dynamic signal time adjustment
- Futuristic Streamlit dashboard UI (Glassmorphism & Live Animations)
- GPU support (CUDA enabled)

## 🛠 Tech Stack
- Python 3.12
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Streamlit
- NumPy

## 🔄 System Workflow
**CCTV Feed** → **YOLOv8 Detection** → **Lane Separation** → **Queue Depth Estimation** → **Signal Decision Engine** → **Live Dashboard**

## 🚀 Installation Guide

**Step 1: Clone Repository**
```bash
git clone https://github.com/rohithlakshman999-hue/HackZen-UrbanFlow-AI.git
cd HackZen-UrbanFlow-AI
```

**Step 2: Create Virtual Environment**
```bash
python -m venv venv 
venv\Scripts\activate
```

**Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 4: Run Application**
```bash
streamlit run app_futuristic.py
```

## 📦 Model Information
This project uses the YOLOv8 model for real-time inference. The model will automatically download during the first run if not available.

## 🎥 Demo Video
Watch working prototype here:
*(Insert Google Drive / YouTube link here)*

## 📁 Repository Structure
```
UrbanFlow-AI-Traffic-Optimization/
│
├── app_futuristic.py
├── process_image.py
├── extract_frames.py
├── requirements.txt
├── README.md
├── .gitignore
├── tested_image/
└── utils/
```

## ⚠️ Notes
- Virtual environment (venv) not included.
- Model weights (.pt) not included.
- Sample videos not included.
- Designed for demonstration purposes.

## 👤 Author
**G Rohith Lakshman**  
*AI4Dev '26 Hackathon Participant*

## 📜 License
For academic and hackathon demonstration purposes.
