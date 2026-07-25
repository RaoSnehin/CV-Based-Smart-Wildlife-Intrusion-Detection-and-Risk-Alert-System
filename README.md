<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:11998e,100:38ef7d&height=200&section=header&text=CV-Based%20Smart%20Wildlife%20Intrusion%20Detection&fontSize=28&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=YOLOv8%20%7C%20Real-time%20Risk%20Assessment%20%7C%20SMS%20Alerts%20%7C%20Streamlit%20Dashboard&descAlignY=58&descSize=15" />

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/YOLOv8-00FFFF?style=for-the-badge&logo=python&logoColor=black"/>
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Twilio-F22F46?style=for-the-badge&logo=twilio&logoColor=white"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-active-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/python-3.8%2B-yellow?style=flat-square"/>
  <img src="https://img.shields.io/badge/domain-Computer%20Vision%20%7C%20Wildlife%20Safety-green?style=flat-square"/>
</p>

</div>

---

## 📌 Project Overview

**CV-Based Smart Wildlife Intrusion Detection and Risk Alert System** is a real-time computer vision application that protects human settlements from dangerous wildlife encounters. Using a **custom-trained YOLOv8 model**, the system detects 5 wildlife species and assesses encounter risk based on animal-human proximity — triggering instant **SMS alerts** to forest rangers and residents.

> 🎓 **Application Domain:** Conservation Technology, Smart Village Safety, Wildlife Management  
> 🏆 **Key Achievement:** Custom YOLOv8 model with temporal consistency filtering to reduce false positives

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🦁 **Multi-Species Detection** | Custom YOLOv8 trained on 5 dangerous wildlife species + humans |
| 🎯 **Risk Assessment Engine** | HIGH / MEDIUM / LOW classification based on animal-human proximity distance |
| 📱 **Multi-Channel SMS Alerts** | Twilio, Textbelt, Fast2SMS integration for instant ranger notifications |
| 🎥 **Multi-Source Input** | Analyze uploaded videos, images, or live webcam feeds |
| 🔧 **Adjustable Sensitivity** | Confidence threshold slider (1–10) for environment tuning |
| 📊 **Evaluation Dashboard** | Precision, Recall, F1-Score, and per-class accuracy charts |
| 📸 **Auto Screenshot** | Captures frames automatically when intrusion is detected |
| 🕐 **Session History** | Persistent detection logs across analysis runs |
| 🔄 **Temporal Consistency** | Requires consecutive-frame confirmation to eliminate false alarms |

---

## 🐾 Detectable Species

| Species | Risk Potential | Detection Priority |
|---------|---------------|-------------------|
| 🐘 Elephant | 🔴 CRITICAL | Highest |
| 🐯 Tiger | 🔴 CRITICAL | Highest |
| 🐆 Leopard | 🔴 HIGH | High |
| 🐻 Bear | 🟠 HIGH | High |
| 🐗 Wild Boar | 🟡 MEDIUM | Medium |
| 🧍 Person | ℹ️ Reference | Context |

---

## 🏗️ System Architecture

```
Input Source (Image / Video / Webcam)
          │
          ▼
┌─────────────────────────────────────┐
│        YOLOv8 Detection Engine       │
│  - Custom wildlife model (best.pt)   │
│  - YOLOv8n for person detection      │
│  - Confidence threshold filtering    │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│       Risk Assessment Engine         │
│  - Bounding box distance calc        │
│  - Temporal consistency check        │
│  - HIGH / MEDIUM / LOW classification│
└─────────────┬───────────────────────┘
              │
         ┌────┴────┐
         ▼         ▼
  ┌──────────┐ ┌──────────────────┐
  │ Streamlit│ │   Alert System    │
  │Dashboard │ │ Twilio / SMS APIs │
  └──────────┘ └──────────────────┘
```

---

## 📁 Project Structure

```
Wildlife_GUI/
├── app.py                    # Main Streamlit application
├── best (5).pt               # Primary custom YOLOv8 wildlife model
├── best (6).pt               # Alternative/backup model weights
├── yolov8n.pt                # YOLOv8n for person detection
├── Test Images/              # Sample test images for demo
├── Test_Videos/              # Sample test videos
└── requirements.txt          # Python dependencies
```

---

## 🧰 Tech Stack

| Technology | Role |
|-----------|------|
| **Python 3.8+** | Primary language |
| **YOLOv8 (Ultralytics)** | Object detection — custom wildlife + person models |
| **OpenCV** | Video frame processing, bounding box rendering |
| **Streamlit** | Interactive web dashboard UI |
| **Twilio / Fast2SMS** | Multi-provider SMS alert delivery |
| **NumPy / PIL** | Image processing and manipulation |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

```bash
# Clone the repository
git clone https://github.com/RaoSnehin/CV-Based-Smart-Wildlife-Intrusion-Detection-and-Risk-Alert-System.git
cd CV-Based-Smart-Wildlife-Intrusion-Detection-and-Risk-Alert-System

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run app.py
```

### Configure SMS Alerts (Optional)
```python
# In app.py, set your credentials:
TWILIO_SID = "your_account_sid"
TWILIO_TOKEN = "your_auth_token"
ALERT_PHONE = "+91XXXXXXXXXX"
```

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| mAP@50 | ~85%+ |
| Inference Speed | Real-time (≥25 FPS on GPU) |
| Species Accuracy | High per-class F1 on test set |
| False Positive Rate | Reduced via temporal consistency filtering |

---

## 🌍 Real-World Impact

This system directly addresses **human-wildlife conflict** — a critical conservation challenge in India's forest fringe communities. By providing early warning via SMS alerts, it enables:
- Faster ranger response times
- Reduced human casualties and livestock losses
- Better wildlife movement tracking data

---

## 👨‍💻 Author

**Snehin Rao** — Final Year CSE, Amrita School of Engineering, Bangalore

[![GitHub](https://img.shields.io/badge/GitHub-RaoSnehin-181717?style=flat-square&logo=github)](https://github.com/RaoSnehin)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-7C3AED?style=flat-square&logo=github)](https://raosnehin.github.io/Portfolio/)
[![Email](https://img.shields.io/badge/Email-raosnehin56%40gmail.com-EA4335?style=flat-square&logo=gmail)](mailto:raosnehin56@gmail.com)

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:11998e,100:38ef7d&height=100&section=footer" />

*⭐ Star this repo if you find it helpful!*
</div>
