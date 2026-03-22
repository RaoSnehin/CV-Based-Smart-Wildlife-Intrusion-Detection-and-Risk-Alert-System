# 🐘 CV-Based Smart Wildlife Intrusion Detection and Risk Alert System

A real-time computer vision system to detect wildlife intrusions near human settlements using YOLOv8-based object detection, risk assessment, and multi-channel alerting.

---

## 📋 Project Overview

This system uses a custom-trained YOLOv8 model to detect dangerous wildlife (elephants, leopards, tigers, bears, wild boars) near human habitats. When a potential intrusion is detected, the system evaluates the **risk level** based on proximity between animals and people, triggers **visual alerts**, and optionally sends **SMS notifications** to rangers or residents.

---

## 🔍 Key Features

- **Multi-Source Detection**: Analyze uploaded videos, images, or live webcam feeds
- **Custom YOLOv8 Model**: Trained on wildlife-specific dataset for accurate detection of:
  - 🐘 Elephant
  - 🐆 Leopard
  - 🐯 Tiger
  - 🐻 Bear
  - 🐗 Wild Boar
  - 🧍 Person
- **Risk Assessment Engine**: Classifies encounters as HIGH / MEDIUM / LOW risk based on animal-person proximity
- **Adjustable Sensitivity**: Slider from 1–10 to tune detection confidence threshold
- **SMS Alerts**: Integrates with Twilio, Textbelt, and Fast2SMS for real-time alerts
- **Evaluation Metrics Dashboard**: Precision, Recall, F1-Score, and per-class accuracy charts
- **Screenshot Capture**: Auto-captures frames when intrusion is detected
- **Session History**: Tracks detections across multiple analysis runs

---

## 🏗️ System Architecture

```
Wildlife_GUI/
├── app.py                  # Main Streamlit application
├── best (5).pt             # Custom YOLOv8 wildlife detection model
├── best (6).pt             # Alternative/backup model weights
├── yolov8n.pt              # YOLOv8n pretrained model (person detection)
├── Test Images/            # Sample test images
├── Test_Videos/            # Sample test videos
└── requirements.txt        # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/RaoSnehin/CV-Based-Smart-Wildlife-Intrusion-Detection-and-Risk-Alert-System.git
cd CV-Based-Smart-Wildlife-Intrusion-Detection-and-Risk-Alert-System

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📱 SMS Alert Configuration

The system supports three SMS providers:

| Provider | Description |
|----------|-------------|
| **Twilio** | Full-featured commercial SMS API |
| **Textbelt** | Free tier (1 SMS/day), no signup needed |
| **Fast2SMS** | India-focused SMS gateway |

Configure your preferred provider in the sidebar under **SMS Alert Settings**.

---

## 🎛️ Detection Settings

| Sensitivity | Confidence Threshold | Use Case |
|-------------|---------------------|----------|
| 1–3 (Low) | ~0.55–0.60 | High precision, fewer false alarms |
| 4–7 (Medium) | ~0.35–0.55 | Balanced — recommended for most scenarios |
| 8–10 (High) | ~0.20–0.35 | Maximum recall, more detections |

---

## 📊 Evaluation Metrics

The **Evaluation Metrics** tab computes and visualizes:
- Precision, Recall, F1-Score per class
- Confusion matrix
- Detection confidence distribution
- Animal vs. person encounter statistics

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Computer Vision**: OpenCV, Ultralytics YOLOv8
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib
- **Image Handling**: Pillow (PIL)
- **SMS Integration**: Twilio API, Textbelt, Fast2SMS

---

## 📸 Screenshots

> Test images and videos are available in `Test Images/` and `Test_Videos/` directories.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is for academic and research purposes.

---

## 👨‍💻 Author

**Rao Snehin** — Final Year Project, 2026

# Image analysis module initialized - see app.py tab2
