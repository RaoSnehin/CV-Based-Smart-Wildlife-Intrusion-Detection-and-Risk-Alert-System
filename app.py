import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import pandas as pd
import math
from datetime import datetime
from collections import Counter, deque
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image
try:
    import winsound
except ImportError:
    winsound = None
import threading
import io
import requests

def send_sms_alert(message, force=False):
    current_time = time.time()
    # 2 minute cooldown (120 seconds), bypass if force=True
    if not force and current_time - st.session_state.get('last_sms_time', 0) < 120:
        return
        
    if not st.session_state.get('sms_enabled', False):
        return
        
    st.session_state.last_sms_time = current_time
    
    # Extract values before entering the thread to avoid Streamlit Context Error
    provider = st.session_state.sms_provider
    target_phone = st.session_state.target_phone
    twilio_sid = st.session_state.get('twilio_sid', '')
    twilio_token = st.session_state.get('twilio_token', '')
    twilio_from = st.session_state.get('twilio_from', '')
    fast2sms_key = st.session_state.get('fast2sms_key', '')
    
    # Visual confirmation that the trigger worked
    if not force:
        st.toast("🚨 Intrusion Detected! Attempting to send SMS...", icon="📱")
    
    def send_task():
        try:
            if provider == "Twilio":
                url = f"https://api.twilio.com/2010-04-01/Accounts/{twilio_sid}/Messages.json"
                data = {
                    "To": target_phone,
                    "From": twilio_from,
                    "Body": message
                }
                res = requests.post(url, data=data, auth=(twilio_sid, twilio_token))
                print(f"Twilio SMS response: {res.status_code} - {res.text}")
            elif provider == "Textbelt (Free 1/day)":
                res = requests.post('https://textbelt.com/text', data={
                    'phone': target_phone,
                    'message': message,
                    'key': 'textbelt',
                })
                print(f"Textbelt SMS response: {res.json()}")
            elif provider == "Fast2SMS (India)":
                url = "https://www.fast2sms.com/dev/bulkV2"
                querystring = {
                    "authorization": fast2sms_key,
                    "message": message,
                    "language": "english",
                    "route": "q",
                    "numbers": target_phone
                }
                res = requests.get(url, headers={"cache-control": "no-cache"}, params=querystring)
                print(f"Fast2SMS response: {res.json()}")
        except Exception as e:
            print(f"SMS Error: {e}")
            
    threading.Thread(target=send_task, daemon=True).start()

# Page config
st.set_page_config(page_title="Wildlife Intrusion Detection", layout="wide")

# Title
st.title("🐘 Wildlife Intrusion Detection System")
st.markdown("Detect Elephant, Leopard, Tiger, Bear, Wild Boar, and Person")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📹 Video Detection", "🖼️ Image Analysis", "🎥 Live Webcam", "📊 Evaluation Metrics"])

# ============================================
# SENSITIVITY SLIDER
# ============================================
with st.sidebar:
    st.header("⚙️ Detection Settings")
    
    sensitivity = st.slider(
        "Detection Sensitivity",
        1, 10, 5,
        help="1-3: Low (detects only clear animals)\n4-7: Medium (balanced)\n8-10: High (detects more)"
    )
    
    if sensitivity <= 3:
        st.info("🔵 LOW Sensitivity - Fewer false alarms, use when animals are close")
    elif sensitivity <= 7:
        st.success("🟢 MEDIUM Sensitivity - Balanced detection (Recommended)")
    else:
        st.warning("🟠 HIGH Sensitivity - Detects more, may have false alarms")
    
    st.caption(f"Current setting: {'Low' if sensitivity <=3 else 'Medium' if sensitivity <=7 else 'High'}")
    st.caption("💡 Higher sensitivity = more detections")

    st.divider()
    st.header("📱 SMS Alert Settings")
    st.session_state.sms_provider = st.selectbox("SMS Provider", ["None", "Twilio", "Textbelt (Free 1/day)", "Fast2SMS (India)"])
    
    st.session_state.sms_enabled = False
    if st.session_state.sms_provider == "Twilio":
        st.session_state.twilio_sid = st.text_input("Twilio Account SID", type="password")
        st.session_state.twilio_token = st.text_input("Twilio Auth Token", type="password")
        st.session_state.twilio_from = st.text_input("Twilio Phone Number")
        st.session_state.target_phone = st.text_input("Target Phone Number")
        if st.session_state.twilio_sid and st.session_state.twilio_token and st.session_state.twilio_from and st.session_state.target_phone:
            st.session_state.sms_enabled = True
    elif st.session_state.sms_provider == "Textbelt (Free 1/day)":
        st.session_state.target_phone = st.text_input("Target Phone Number", help="Include country code, e.g., +91 for India")
        if st.session_state.target_phone:
            st.session_state.sms_enabled = True
    elif st.session_state.sms_provider == "Fast2SMS (India)":
        st.session_state.fast2sms_key = st.text_input("Fast2SMS API Key", type="password")
        st.session_state.target_phone = st.text_input("Target Phone Number", help="10-digit mobile number without +91")
        if st.session_state.fast2sms_key and st.session_state.target_phone:
            st.session_state.sms_enabled = True

    if st.session_state.sms_enabled:
        if st.button("🔔 Send Test SMS"):
            send_sms_alert("Test message from Wildlife Intrusion Detection System!", force=True)
            st.success("Test SMS sent! Check your phone.")

# Convert sensitivity to confidence threshold
CONFIDENCE_THRESHOLD = 0.6 - ((sensitivity - 1) * 0.05)
CONFIDENCE_THRESHOLD = max(0.2, min(0.6, CONFIDENCE_THRESHOLD))

# Fixed values
ALERT_GAP_FRAMES = 15
INTRUSION_DISTANCE = 10

# Colors
RISK_COLORS = {"HIGH": (0, 0, 255), "MEDIUM": (0, 165, 255), "LOW": (0, 255, 0)}
ANIMAL_COLOR = (0, 0, 255)      # RED for animals
PERSON_COLOR = (0, 255, 0)      # GREEN for persons
LINE_COLOR = (0, 255, 255)      # YELLOW for connection line

# Define which classes are animals and person
ANIMAL_CLASSES = ["Tiger", "Elephant", "Leopard", "Bear", "Wild Boar"]
PERSON_CLASS = "Person"
PERSON_CLASS_ID = 0  # COCO dataset person class ID

# Initialize session state
if 'webcam_running' not in st.session_state:
    st.session_state.webcam_running = False
if 'webcam_results' not in st.session_state:
    st.session_state.webcam_results = None
if 'video_alerts' not in st.session_state:
    st.session_state.video_alerts = None
if 'video_screenshots' not in st.session_state:
    st.session_state.video_screenshots = None
if 'video_metrics' not in st.session_state:
    st.session_state.video_metrics = None
if 'image_history' not in st.session_state:
    st.session_state.image_history = []
if 'video_history' not in st.session_state:
    st.session_state.video_history = []
if 'eval_metrics_history' not in st.session_state:
    st.session_state.eval_metrics_history = []
if 'last_sms_time' not in st.session_state:
    st.session_state.last_sms_time = 0

# Pre-load models when app starts
@st.cache_resource
def load_custom_model():
    model_path = "best (5).pt"
    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st.stop()
    return YOLO(model_path)

@st.cache_resource
def load_pretrained_model():
    return YOLO("yolov8n.pt")

# Load models immediately when app starts
custom_model = load_custom_model()
pretrained_model = load_pretrained_model()

# Sound alert function
def play_alert_sound(risk_level):
    def beep():
        if winsound is None:
            print(f"BEEP: Risk {risk_level}")
            return
        if risk_level == "HIGH":
            for _ in range(4):
                winsound.Beep(1000, 150)
                time.sleep(0.08)
        elif risk_level == "MEDIUM":
            for _ in range(3):
                winsound.Beep(800, 150)
                time.sleep(0.08)
        elif risk_level == "LOW":
            for _ in range(2):
                winsound.Beep(600, 150)
                time.sleep(0.08)
    threading.Thread(target=beep, daemon=True).start()

def get_centroid(x1, y1, x2, y2):
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))

def get_distance_meters(animal_centroid, person_centroid, person_height_px):
    if person_height_px > 0:
        pixel_dist = math.dist(animal_centroid, person_centroid)
        meters_per_pixel = 1.7 / person_height_px
        return pixel_dist * meters_per_pixel
    return 999

def get_risk_level(distance_m, class_name):
    species_risk_map = {
        "Tiger": 3, "Elephant": 3, "Leopard": 3, "Bear": 3, "Wild Boar": 2
    }
    species_risk = species_risk_map.get(class_name, 1)
    
    if distance_m < 3:
        distance_risk = 3
    elif distance_m < 7:
        distance_risk = 2
    elif distance_m < 15:
        distance_risk = 1
    else:
        distance_risk = 0
    
    total = species_risk + distance_risk
    if total >= 6:
        return "HIGH"
    elif total >= 4:
        return "MEDIUM"
    return "LOW"

def draw_text_with_bg(frame, text, position, color, font_scale=0.5):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
    x, y = position
    cv2.rectangle(frame, (x-2, y-th-2), (x+tw+2, y+2), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

def merge_overlapping_detections(detections, merge_distance=50):
    """Merge overlapping detections based on centroid distance"""
    if len(detections) <= 1:
        return detections
    
    merged = []
    used = set()
    
    for i, det1 in enumerate(detections):
        if i in used:
            continue
        group = [det1]
        for j, det2 in enumerate(detections):
            if j != i and j not in used:
                dist = math.dist(det1.get('centroid', (0,0)), det2.get('centroid', (0,0)))
                if dist < merge_distance:
                    group.append(det2)
                    used.add(j)
        
        avg_centroid = (int(sum(d['centroid'][0] for d in group) / len(group)),
                        int(sum(d['centroid'][1] for d in group) / len(group)))
        
        best = max(group, key=lambda x: x.get('conf', 0))
        
        result = {
            'centroid': avg_centroid,
            'conf': best.get('conf', 0.5)
        }
        
        if 'height' in best:
            result['height'] = best['height']
        if 'class' in best:
            result['class'] = best['class']
        
        merged.append(result)
        used.add(i)
    
    return merged

def apply_nms(detections, distance_threshold=60):
    """Apply Non-Maximum Suppression to remove overlapping detections"""
    if len(detections) <= 1:
        return detections
    
    detections = sorted(detections, key=lambda x: x.get('conf', 0), reverse=True)
    
    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)
        
        filtered = []
        for det in detections:
            dist = math.dist(best.get('centroid', (0,0)), det.get('centroid', (0,0)))
            if dist > distance_threshold:
                filtered.append(det)
        
        detections = filtered
    
    return keep

# ============================================
# VIDEO DETECTION (CUSTOM MODEL ONLY - WORKING PERFECTLY)
# ============================================
def detect_objects_video(image):
    """Detect animals and persons using custom model only (for videos)"""
    results = custom_model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    
    animals = []
    persons = []
    
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            class_name = results.names[class_id]
            conf = float(box.conf[0])
            centroid = get_centroid(x1, y1, x2, y2)
            bbox_height = y2 - y1
            
            if class_name in ANIMAL_CLASSES:
                animals.append({
                    'class': class_name,
                    'centroid': centroid,
                    'conf': conf,
                    'height': bbox_height
                })
            elif class_name == PERSON_CLASS or class_name.lower() == "person":
                persons.append({
                    'centroid': centroid,
                    'height': bbox_height,
                    'conf': conf
                })
    
    animals = apply_nms(animals, distance_threshold=60)
    persons = apply_nms(persons, distance_threshold=60)
    
    return animals, persons

# ============================================
# IMAGE/WEBCAM DETECTION (CUSTOM + PRETRAINED FOR PERSONS)
# ============================================
def detect_objects_with_pretrained(image):
    """Detect animals using custom model, persons using pretrained YOLO"""
    animals = []
    persons = []
    
    # 1. Detect ANIMALS using custom model
    animal_results = custom_model(image, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    
    if animal_results.boxes is not None:
        for box in animal_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])
            class_name = animal_results.names[class_id]
            conf = float(box.conf[0])
            centroid = get_centroid(x1, y1, x2, y2)
            bbox_height = y2 - y1
            
            if class_name in ANIMAL_CLASSES:
                animals.append({
                    'class': class_name,
                    'centroid': centroid,
                    'conf': conf,
                    'height': bbox_height
                })
    
    # 2. Detect PERSONS using pretrained YOLO (better for images)
    person_threshold = 0.4
    person_results = pretrained_model(image, conf=person_threshold, verbose=False)[0]
    
    if person_results.boxes is not None:
        for box in person_results.boxes:
            class_id = int(box.cls[0])
            if class_id == PERSON_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                centroid = get_centroid(x1, y1, x2, y2)
                bbox_height = y2 - y1
                bbox_width = x2 - x1
                
                # Filter: Person should be reasonably sized
                if bbox_height > 40 and bbox_width > 20:
                    persons.append({
                        'centroid': centroid,
                        'height': bbox_height,
                        'conf': conf
                    })
    
    animals = apply_nms(animals, distance_threshold=60)
    persons = apply_nms(persons, distance_threshold=60)
    
    return animals, persons

# ============================================
# EVALUATION METRICS
# ============================================
def calculate_video_metrics(frame_data, alerts, processing_time, fps, total_frames):
    metrics = {}
    
    frames_with_detections = sum(1 for f in frame_data if f['has_animals'] or f['has_persons'])
    metrics['DCR'] = round((frames_with_detections / total_frames) * 100, 2) if total_frames > 0 else 0
    
    stable_frames = 0
    prev_animal = None
    for f in frame_data:
        current_animal = f['animal_class'] if f['animal_class'] else None
        if current_animal and prev_animal and current_animal == prev_animal:
            stable_frames += 1
        prev_animal = current_animal
    metrics['TSS'] = round((stable_frames / max(1, frames_with_detections)) * 100, 2)
    
    duplicate_alerts = 0
    for i in range(1, len(alerts)):
        if alerts[i]['frame'] - alerts[i-1]['frame'] < 15:
            duplicate_alerts += 1
    metrics['ANR'] = round((duplicate_alerts / max(1, len(alerts))) * 100, 2)
    
    frames_with_both = sum(1 for f in frame_data if f['has_animals'] and f['has_persons'])
    metrics['HAIR'] = round((frames_with_both / total_frames) * 100, 2) if total_frames > 0 else 0
    
    risk_sequence = [f['risk'] for f in frame_data if f['risk']]
    stable_risk = 0
    for i in range(1, len(risk_sequence)):
        if risk_sequence[i] == risk_sequence[i-1]:
            stable_risk += 1
    metrics['TSI'] = round((stable_risk / max(1, len(risk_sequence)-1)) * 100, 2) if len(risk_sequence) > 1 else 100
    
    events = []
    current_event = []
    for f in frame_data:
        if f['has_animals']:
            current_event.append(f['animal_class'])
        else:
            if current_event:
                events.append(current_event)
                current_event = []
    if current_event:
        events.append(current_event)
    
    consistent_events = 0
    for event in events:
        if len(set(event)) == 1:
            consistent_events += 1
    metrics['ECS'] = round((consistent_events / max(1, len(events))) * 100, 2)
    
    if alerts and frame_data:
        first_animal_frame = next((f['frame'] for f in frame_data if f['has_animals']), None)
        if first_animal_frame is not None:
            metrics['AD'] = alerts[0]['frame'] - first_animal_frame
        else:
            metrics['AD'] = -1
    else:
        metrics['AD'] = -1
    
    if alerts:
        alert_durations = []
        for i in range(len(alerts)):
            start_frame = alerts[i]['frame']
            end_frame = alerts[i+1]['frame'] if i+1 < len(alerts) else total_frames
            alert_durations.append(end_frame - start_frame)
        metrics['AAD'] = round(sum(alert_durations) / len(alert_durations), 2)
    else:
        metrics['AAD'] = 0
    
    confidences = [f['confidence'] for f in frame_data if f['confidence'] > 0]
    if confidences:
        std_conf = np.std(confidences)
        metrics['CSS'] = round(1 - std_conf, 3)
    else:
        metrics['CSS'] = 1.0
    
    metrics['FPS'] = round(fps, 2)
    metrics['total_frames'] = total_frames
    metrics['frames_with_detections'] = frames_with_detections
    metrics['total_alerts'] = len(alerts)
    metrics['processing_time'] = round(processing_time, 2)
    
    return metrics

def calculate_image_metrics(animals, persons, confidence_threshold=0.5):
    metrics = {}
    
    all_confidences = [a['conf'] for a in animals] + [p['conf'] for p in persons]
    metrics['DCS'] = round(np.mean(all_confidences), 3) if all_confidences else 0
    
    low_conf_animals = sum(1 for a in animals if a['conf'] < confidence_threshold)
    low_conf_persons = sum(1 for p in persons if p['conf'] < confidence_threshold)
    metrics['FDC'] = low_conf_animals + low_conf_persons
    
    has_animals = len(animals) > 0
    has_persons = len(persons) > 0
    metrics['HAPS'] = 100 if (has_animals and has_persons) else (50 if (has_animals or has_persons) else 0)
    
    if has_animals and has_persons:
        metrics['SUS'] = 100
    elif has_animals or has_persons:
        metrics['SUS'] = 50
    else:
        metrics['SUS'] = 0
    
    metrics['animal_count'] = len(animals)
    metrics['person_count'] = len(persons)
    metrics['avg_animal_conf'] = round(np.mean([a['conf'] for a in animals]), 3) if animals else 0
    metrics['avg_person_conf'] = round(np.mean([p['conf'] for p in persons]), 3) if persons else 0
    
    return metrics

# ============================================
# VIDEO PROCESSING (CUSTOM MODEL ONLY)
# ============================================
def process_video(video_path, progress_bar, status_text):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = time.time()
    
    alerts = []
    all_detections = []
    all_confidences = []
    frame_count = 0
    frame_data = []
    
    screenshot_dir = tempfile.mkdtemp()
    screenshots = []
    last_alert_frame = -ALERT_GAP_FRAMES
    
    all_animals_detected = set()
    all_persons_detected = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 2nd frame for faster processing
        if frame_count % 2 != 0:
            frame_count += 1
            status_text.text(f"Processing: {frame_count}/{total_frames} frames... [Alerts: {len(alerts)}]")
            continue
        
        animals, persons = detect_objects_video(frame)
        
        animal_class = animals[0]['class'] if animals else None
        has_animals = len(animals) > 0
        has_persons = len(persons) > 0
        
        for animal in animals:
            all_animals_detected.add(animal['class'])
        if persons:
            all_persons_detected = True
        
        for animal in animals:
            all_detections.append({'class': animal['class'], 'confidence': animal['conf'], 'frame': frame_count})
            all_confidences.append(animal['conf'])
        
        # Draw ANIMALS - RED circles
        for animal in animals:
            cx, cy = animal['centroid']
            cv2.circle(frame, (cx, cy), 25, ANIMAL_COLOR, 3)
            cv2.circle(frame, (cx, cy), 5, ANIMAL_COLOR, -1)
            draw_text_with_bg(frame, animal['class'], (cx-30, cy-28), ANIMAL_COLOR, 0.5)
        
        # Draw PERSONS - GREEN circles
        for person in persons:
            cx, cy = person['centroid']
            cv2.circle(frame, (cx, cy), 25, PERSON_COLOR, 3)
            cv2.circle(frame, (cx, cy), 5, PERSON_COLOR, -1)
            draw_text_with_bg(frame, "Person", (cx-30, cy-28), PERSON_COLOR, 0.5)
        
        # Find closest animal-person pair
        closest_distance = None
        closest_animal = None
        closest_person = None
        frame_risk = None
        
        for animal in animals:
            for person in persons:
                distance = get_distance_meters(animal['centroid'], person['centroid'], person['height'])
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance
                    closest_animal = animal
                    closest_person = person
        
        if closest_animal and closest_person and closest_distance:
            frame_risk = get_risk_level(closest_distance, closest_animal['class'])
        
        frame_data.append({
            'frame': frame_count,
            'has_animals': has_animals,
            'has_persons': has_persons,
            'animal_class': animal_class,
            'confidence': animals[0]['conf'] if animals else 0,
            'risk': frame_risk
        })
        
        # Draw line and risk meter if alert
        if closest_animal and closest_person and closest_distance and closest_distance < INTRUSION_DISTANCE:
            risk = get_risk_level(closest_distance, closest_animal['class'])
            risk_color = RISK_COLORS.get(risk, (0, 0, 255))
            
            cv2.line(frame, closest_animal['centroid'], closest_person['centroid'], LINE_COLOR, 3)
            mid = ((closest_animal['centroid'][0] + closest_person['centroid'][0]) // 2,
                   (closest_animal['centroid'][1] + closest_person['centroid'][1]) // 2)
            draw_text_with_bg(frame, f"{closest_distance:.1f}m", mid, LINE_COLOR, 0.5)
            
            cv2.rectangle(frame, (width-120, 10), (width-10, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"RISK: {risk}", (width-115, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2)
            
            if frame_count - last_alert_frame > ALERT_GAP_FRAMES:
                last_alert_frame = frame_count
                play_alert_sound(risk)
                send_sms_alert(f"ALERT: {closest_animal['class']} detected near Person! Distance: {closest_distance:.1f}m. Risk: {risk}")
                
                screenshot_path = os.path.join(screenshot_dir, f"alert_{len(alerts)+1}.jpg")
                ss_frame = frame.copy()
                draw_text_with_bg(ss_frame, f"ALERT: {closest_animal['class']}", (10, 45), ANIMAL_COLOR, 0.55)
                draw_text_with_bg(ss_frame, f"Distance: {closest_distance:.1f}m", (10, 75), risk_color, 0.5)
                draw_text_with_bg(ss_frame, datetime.now().strftime("%H:%M:%S"), (10, height-18), (200, 200, 200), 0.4)
                cv2.imwrite(screenshot_path, ss_frame)
                screenshots.append(screenshot_path)
                
                alerts.append({
                    'frame': frame_count,
                    'animal': closest_animal['class'],
                    'distance': closest_distance,
                    'risk': risk,
                    'confidence': closest_animal['conf']
                })
        
        cv2.rectangle(frame, (5, 5), (180, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        frame_count += 1
        
        status_text.text(f"Processing: {frame_count}/{total_frames} frames... [Alerts: {len(alerts)}]")
        
        if frame_count % 50 == 0:
            progress_bar.progress(min(frame_count / total_frames, 0.99))
    
    cap.release()
    progress_bar.progress(1.0)
    status_text.text(f"✅ Complete! {len(alerts)} alerts triggered")
    
    processing_time = time.time() - start_time
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    actual_fps = frame_count / processing_time if processing_time > 0 else 0
    
    final_animal, scoring_details = get_final_animal_verdict(all_detections)
    eval_metrics = calculate_video_metrics(frame_data, alerts, processing_time, actual_fps, total_frames)
    
    return alerts, screenshots, {
        'total_frames': total_frames,
        'processed_frames': frame_count,
        'processing_time': processing_time,
        'fps': actual_fps,
        'avg_confidence': avg_confidence,
        'alert_count': len(alerts),
        'animal_counts': Counter([a['animal'] for a in alerts]),
        'risk_counts': Counter([a['risk'] for a in alerts]),
        'distances': [a['distance'] for a in alerts],
        'final_animal': final_animal,
        'scoring_details': scoring_details,
        'animals_detected': list(all_animals_detected),
        'persons_detected': all_persons_detected,
        'eval_metrics': eval_metrics,
        'frame_data': frame_data
    }

# ============================================
# ENSEMBLE SCORING
# ============================================
def calculate_animal_scores(all_detections):
    if not all_detections:
        return {}
    
    animals = set([d['class'] for d in all_detections])
    total_detections = len(all_detections)
    
    scores = {}
    
    for animal in animals:
        animal_dets = [d for d in all_detections if d['class'] == animal]
        detection_count = len(animal_dets)
        
        frequency_score = (detection_count / total_detections) * 100
        avg_confidence = sum([d['confidence'] for d in animal_dets]) / detection_count
        confidence_score = avg_confidence * 100
        
        current_streak = 0
        max_streak = 0
        
        for det in all_detections:
            if det['class'] == animal:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        temporal_score = (max_streak / total_detections) * 100 if total_detections > 0 else 0
        unique_frames = set([d['frame'] for d in animal_dets])
        coverage_score = (len(unique_frames) / total_detections) * 100 if total_detections > 0 else 0
        
        final_score = (
            (frequency_score * 0.25) + 
            (confidence_score * 0.15) + 
            (temporal_score * 0.35) + 
            (coverage_score * 0.25)
        )
        
        scores[animal] = {
            'final_score': final_score,
            'detection_count': detection_count,
            'avg_confidence': avg_confidence,
            'max_streak': max_streak,
            'coverage_score': coverage_score
        }
    
    return scores

def get_final_animal_verdict(all_detections):
    if not all_detections:
        return "None", {}
    
    scores = calculate_animal_scores(all_detections)
    
    if not scores:
        return "None", {}
    
    final_animal = max(scores, key=lambda x: scores[x]['final_score'])
    
    animals = set([d['class'] for d in all_detections])
    total_detections = len(all_detections)
    
    temporal_streaks = {}
    coverage_scores = {}
    
    for animal in animals:
        animal_dets = [d for d in all_detections if d['class'] == animal]
        
        current_streak = 0
        max_streak = 0
        for det in all_detections:
            if det['class'] == animal:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        temporal_streaks[animal] = max_streak
        
        unique_frames = set([d['frame'] for d in animal_dets])
        coverage_scores[animal] = round((len(unique_frames) / total_detections) * 100, 1)
    
    details = {
        'scores': {animal: round(data['final_score'], 2) for animal, data in scores.items()},
        'detection_counts': {animal: data['detection_count'] for animal, data in scores.items()},
        'avg_confidences': {animal: round(data['avg_confidence'], 3) for animal, data in scores.items()},
        'temporal_streaks': temporal_streaks,
        'coverage': coverage_scores
    }
    
    return final_animal, details

# ============================================
# IMAGE ANALYSIS (WITH PRETRAINED FOR PERSONS)
# ============================================
def analyze_image(image):
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    animals, persons = detect_objects_with_pretrained(image)
    
    result_image = image.copy()
    
    # Draw ANIMALS - RED circles
    for animal in animals:
        cx, cy = animal['centroid']
        cv2.circle(result_image, (cx, cy), 25, ANIMAL_COLOR, 3)
        cv2.circle(result_image, (cx, cy), 5, ANIMAL_COLOR, -1)
        draw_text_with_bg(result_image, f"{animal['class']} ({animal['conf']:.2f})", (cx-30, cy-28), ANIMAL_COLOR, 0.5)
    
    # Draw PERSONS - GREEN circles (from pretrained model)
    for person in persons:
        cx, cy = person['centroid']
        cv2.circle(result_image, (cx, cy), 25, PERSON_COLOR, 3)
        cv2.circle(result_image, (cx, cy), 5, PERSON_COLOR, -1)
        draw_text_with_bg(result_image, f"Person ({person['conf']:.2f})", (cx-30, cy-28), PERSON_COLOR, 0.5)
    
    closest_distance = None
    closest_animal = None
    closest_person = None
    
    if animals and persons:
        for animal in animals:
            for person in persons:
                distance = get_distance_meters(animal['centroid'], person['centroid'], person['height'])
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance
                    closest_animal = animal
                    closest_person = person
    
    if closest_animal and closest_person and closest_distance:
        cv2.line(result_image, closest_animal['centroid'], closest_person['centroid'], LINE_COLOR, 3)
        mid = ((closest_animal['centroid'][0] + closest_person['centroid'][0]) // 2,
               (closest_animal['centroid'][1] + closest_person['centroid'][1]) // 2)
        draw_text_with_bg(result_image, f"{closest_distance:.1f}m", mid, LINE_COLOR, 0.5)
        
        risk = get_risk_level(closest_distance, closest_animal['class'])
        risk_color = RISK_COLORS.get(risk, (0, 0, 255))
        
        cv2.rectangle(result_image, (result_image.shape[1]-120, 10), (result_image.shape[1]-10, 60), (0, 0, 0), -1)
        cv2.putText(result_image, f"RISK: {risk}", (result_image.shape[1]-115, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2)
    
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    
    return result_image_rgb, animals, persons, closest_distance

# ============================================
# WEBCAM PROCESSING (WITH PRETRAINED FOR PERSONS)
# ============================================
def process_webcam(frame_placeholder, status_text, result_placeholder):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_text.error("Cannot open webcam")
        return []
    
    frame_count = 0
    alert_count = 0
    alerts_data = []
    last_alert_animal = None
    last_alert_frame = -ALERT_GAP_FRAMES
    
    while st.session_state.webcam_running:
        ret, frame = cap.read()
        if not ret:
            break
        
        animals, persons = detect_objects_with_pretrained(frame)
        
        # Draw ANIMALS - RED circles
        for animal in animals:
            cx, cy = animal['centroid']
            cv2.circle(frame, (cx, cy), 25, ANIMAL_COLOR, 3)
            cv2.circle(frame, (cx, cy), 5, ANIMAL_COLOR, -1)
            draw_text_with_bg(frame, animal['class'], (cx-30, cy-28), ANIMAL_COLOR, 0.5)
        
        # Draw PERSONS - GREEN circles (from pretrained model)
        for person in persons:
            cx, cy = person['centroid']
            cv2.circle(frame, (cx, cy), 25, PERSON_COLOR, 3)
            cv2.circle(frame, (cx, cy), 5, PERSON_COLOR, -1)
            draw_text_with_bg(frame, "Person", (cx-30, cy-28), PERSON_COLOR, 0.5)
        
        closest_distance = None
        closest_animal = None
        for animal in animals:
            for person in persons:
                distance = get_distance_meters(animal['centroid'], person['centroid'], person['height'])
                if closest_distance is None or distance < closest_distance:
                    closest_distance = distance
                    closest_animal = animal
        
        if closest_animal and persons and closest_distance and closest_distance < INTRUSION_DISTANCE:
            risk = get_risk_level(closest_distance, closest_animal['class'])
            risk_color = RISK_COLORS.get(risk, (0, 0, 255))
            
            for person in persons:
                cv2.line(frame, closest_animal['centroid'], person['centroid'], LINE_COLOR, 3)
                mid = ((closest_animal['centroid'][0] + person['centroid'][0]) // 2,
                       (closest_animal['centroid'][1] + person['centroid'][1]) // 2)
                draw_text_with_bg(frame, f"{closest_distance:.1f}m", mid, LINE_COLOR, 0.5)
            
            cv2.rectangle(frame, (frame.shape[1]-120, 10), (frame.shape[1]-10, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"RISK: {risk}", (frame.shape[1]-115, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2)
            
            if last_alert_animal != closest_animal['class'] and frame_count - last_alert_frame > ALERT_GAP_FRAMES:
                last_alert_animal = closest_animal['class']
                last_alert_frame = frame_count
                alert_count += 1
                play_alert_sound(risk)
                send_sms_alert(f"ALERT: {closest_animal['class']} detected near Person! Distance: {closest_distance:.1f}m. Risk: {risk}")
                
                alerts_data.append({
                    'animal': closest_animal['class'],
                    'distance': closest_distance,
                    'risk': risk,
                    'confidence': closest_animal['conf']
                })
                
                status_text.text(f"Webcam running | Alerts: {alert_count}")
        
        cv2.rectangle(frame, (5, 5), (180, 45), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        status_text.text(f"Webcam running | Frames: {frame_count} | Alerts: {alert_count}")
        
        if alerts_data:
            result_placeholder.dataframe(pd.DataFrame(alerts_data[-5:]), use_container_width=True)
        
        frame_count += 1
        time.sleep(0.03)
    
    cap.release()
    status_text.text("Webcam stopped")
    return alerts_data

# ============================================
# TAB 1: VIDEO DETECTION (CUSTOM MODEL ONLY)
# ============================================
with tab1:
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"], key="video")
    
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        
        col_video, col_results = st.columns([1, 1])
        
        with col_video:
            st.subheader("📹 Input Video")
            st.video(video_path)
            
            if st.button("🚀 Start Detection", type="primary", use_container_width=True):
                st.session_state.video_alerts = None
                st.session_state.video_screenshots = None
                st.session_state.video_metrics = None
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                alerts, screenshots, metrics = process_video(video_path, progress_bar, status_text)
                
                st.session_state.video_alerts = alerts
                st.session_state.video_screenshots = screenshots
                st.session_state.video_metrics = metrics
                
                st.session_state.video_history.append({
                    'timestamp': datetime.now(),
                    'filename': uploaded_file.name,
                    'total_frames': metrics['total_frames'],
                    'alert_count': metrics['alert_count'],
                    'animals_detected': metrics.get('animals_detected', []),
                    'persons_detected': metrics.get('persons_detected', False),
                    'final_animal': metrics.get('final_animal', 'None'),
                    'eval_metrics': metrics.get('eval_metrics', {})
                })
                
                st.session_state.eval_metrics_history.append({
                    'timestamp': datetime.now(),
                    'filename': uploaded_file.name,
                    'type': 'video',
                    'metrics': metrics.get('eval_metrics', {})
                })
                
                progress_bar.empty()
                status_text.empty()
        
        with col_results:
            st.subheader("📊 Detection Results")
            
            if st.session_state.video_metrics is not None:
                metrics = st.session_state.video_metrics
                alerts = st.session_state.video_alerts
                screenshots = st.session_state.video_screenshots
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Total Frames", metrics['total_frames'])
                col_b.metric("Processing Time", f"{metrics['processing_time']:.1f}s")
                col_c.metric("Total Alerts", metrics['alert_count'])
                
                if metrics.get('final_animal') and metrics['final_animal'] != "None":
                    st.success(f"🎯 **FINAL VERDICT: {metrics['final_animal'].upper()}**")
                    st.caption("Based on ensemble scoring (Temporal persistence 35% + Frequency 25% + Coverage 25% + Confidence 15%)")
                    
                    if 'scoring_details' in metrics and metrics['scoring_details']:
                        scoring = metrics['scoring_details']
                        total_dets = sum(scoring['detection_counts'].values())
                        
                        st.info(f"📊 Total detections: {total_dets} across {len(scoring['detection_counts'])} animal types")
                        
                        with st.expander("📊 View Ensemble Scoring Details"):
                            st.write("**Animal Scores Breakdown:**")
                            for animal, score in scoring['scores'].items():
                                det_count = scoring['detection_counts'][animal]
                                st.write(f"• **{animal}**: {score} points ({det_count} detections)")
                            
                            st.write(f"\n**Detection frequencies:** {scoring['detection_counts']}")
                            st.write(f"**Average confidences:** {scoring['avg_confidences']}")
                            
                            if 'temporal_streaks' in scoring:
                                st.write(f"**Temporal streaks (consecutive frames):** {scoring['temporal_streaks']}")
                            if 'coverage' in scoring:
                                st.write(f"**Frame coverage (%):** {scoring['coverage']}")
                
                if alerts:
                    st.success(f"✅ {len(alerts)} alert(s) triggered")
                    
                    df = pd.DataFrame([{
                        'Animal': a['animal'],
                        'Distance (m)': round(a['distance'], 1),
                        'Risk': a['risk'],
                        'Confidence': round(a['confidence'], 2)
                    } for a in alerts])
                    st.dataframe(df, use_container_width=True)
                    
                    if metrics['risk_counts']:
                        rc1, rc2, rc3 = st.columns(3)
                        rc1.metric("🔴 HIGH", metrics['risk_counts'].get("HIGH", 0))
                        rc2.metric("🟠 MEDIUM", metrics['risk_counts'].get("MEDIUM", 0))
                        rc3.metric("🟢 LOW", metrics['risk_counts'].get("LOW", 0))
                    
                    st.download_button("📥 Download CSV", df.to_csv(index=False), "alerts.csv", "text/csv")
                    
                    if len(metrics['distances']) >= 2:
                        st.subheader("📈 Distance Progression")
                        fig, ax = plt.subplots(figsize=(10, 3))
                        alert_nums = list(range(1, len(metrics['distances']) + 1))
                        ax.plot(alert_nums, metrics['distances'], 'o-', color='red', linewidth=2, markersize=8)
                        ax.axhline(y=INTRUSION_DISTANCE, color='orange', linestyle='--')
                        ax.set_xlabel('Alert Number')
                        ax.set_ylabel('Distance (m)')
                        ax.set_title('Distance Progression')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    
                    if screenshots:
                        st.subheader("📸 Alert Screenshots")
                        for idx, screenshot in enumerate(screenshots[:5]):
                            st.image(screenshot, caption=f"Alert {idx+1}", use_container_width=True)
                else:
                    # Filter out false positive animals (less than 5 detections or low confidence)
                    if metrics.get('animals_detected'):
                        # Only show animals with significant presence
                        significant_animals = []
                        for animal in metrics['animals_detected']:
                            if metrics['scoring_details']['detection_counts'].get(animal, 0) >= 5:
                                significant_animals.append(animal)
                        if significant_animals:
                            st.info(f"🐾 Animals detected: {', '.join(significant_animals)} - No alerts triggered")
                        else:
                            st.info("No significant animal detections (possible false positives filtered)")
                    elif metrics.get('persons_detected'):
                        st.info("👤 Persons detected - No animals found")
                    else:
                        st.info("No detections in this video")

# ============================================
# TAB 2: IMAGE ANALYSIS (WITH PRETRAINED FOR PERSONS)
# ============================================
with tab2:
    st.subheader("🖼️ Single Image Analysis")
    st.caption("🔴 Red circles = Animals | 🟢 Green circles = Persons (using pretrained model for better detection)")
    
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image")
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        col_img, col_results = st.columns([1, 1])
        
        with col_img:
            st.image(image, use_container_width=True, caption="Original Image")
            
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing..."):
                    result_image, animals, persons, closest_distance = analyze_image(image)
                    
                    img_metrics = calculate_image_metrics(animals, persons)
                    
                    st.session_state['img_result'] = result_image
                    st.session_state['img_animals'] = animals
                    st.session_state['img_persons'] = persons
                    st.session_state['img_distance'] = closest_distance
                    st.session_state['img_metrics'] = img_metrics
                    
                    st.session_state.image_history.append({
                        'timestamp': datetime.now(),
                        'filename': uploaded_image.name,
                        'animals': len(animals),
                        'persons': len(persons),
                        'distance': closest_distance,
                        'animal_names': [a['class'] for a in animals],
                        'metrics': img_metrics
                    })
                    
                    if closest_distance and animals:
                        risk = get_risk_level(closest_distance, animals[0]['class'])
                        if risk == "HIGH":
                            play_alert_sound("HIGH")
                            send_sms_alert(f"ALERT: {animals[0]['class']} detected near Person! Distance: {closest_distance:.1f}m. Risk: HIGH")
        
        with col_results:
            if 'img_result' in st.session_state:
                st.image(st.session_state['img_result'], use_container_width=True, caption="Detection Result")
                
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("🐾 Animals", len(st.session_state.get('img_animals', [])))
                col_m2.metric("👤 Persons", len(st.session_state.get('img_persons', [])))
                
                if st.session_state.get('img_distance'):
                    col_m3.metric("📏 Distance", f"{st.session_state['img_distance']:.1f}m")
                else:
                    col_m3.metric("📏 Distance", "N/A")
                
                st.markdown("---")
                
                if st.session_state.get('img_animals'):
                    st.markdown("**🐾 Detected Animals:**")
                    for a in st.session_state['img_animals']:
                        st.write(f"- **{a['class']}** (confidence: {a['conf']:.2f})")
                else:
                    st.info("No animals detected")
                
                if st.session_state.get('img_persons'):
                    st.markdown("**👤 Detected Persons:**")
                    for p in st.session_state['img_persons']:
                        st.write(f"- Person (confidence: {p['conf']:.2f})")
                else:
                    if not st.session_state.get('img_animals'):
                        st.info("No persons detected")
                    else:
                        st.info("No persons detected in this image")
                
                if 'img_metrics' in st.session_state:
                    st.markdown("---")
                    st.markdown("### 📊 Image Evaluation Metrics")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("DCS", f"{st.session_state['img_metrics']['DCS']:.3f}")
                    col_m2.metric("FDC", st.session_state['img_metrics']['FDC'])
                    col_m3.metric("HAPS", f"{st.session_state['img_metrics']['HAPS']:.0f}%")
                    
                    col_m1.metric("SUS", f"{st.session_state['img_metrics']['SUS']:.0f}%")
                    col_m2.metric("Animals", st.session_state['img_metrics']['animal_count'])
                    col_m3.metric("Persons", st.session_state['img_metrics']['person_count'])
                
                if st.session_state.get('img_distance') and st.session_state.get('img_animals') and st.session_state.get('img_persons'):
                    distance = st.session_state['img_distance']
                    animal_name = st.session_state['img_animals'][0]['class']
                    risk = get_risk_level(distance, animal_name)
                    
                    st.markdown("---")
                    st.markdown("### ⚠️ Risk Assessment")
                    
                    risk_pct = {"HIGH": 1.0, "MEDIUM": 0.6, "LOW": 0.3}.get(risk, 0)
                    st.progress(risk_pct)
                    
                    if risk == "HIGH":
                        st.error(f"🚨 HIGH RISK - {animal_name} at {distance:.1f}m!")
                    elif risk == "MEDIUM":
                        st.warning(f"⚠️ MEDIUM RISK - {animal_name} at {distance:.1f}m")
                    else:
                        st.success(f"✅ LOW RISK - {animal_name} at {distance:.1f}m")
                
                if 'img_result' in st.session_state:
                    st.markdown("---")
                    result_pil = Image.fromarray(st.session_state['img_result'])
                    
                    buf = io.BytesIO()
                    result_pil.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📥 Download Result",
                        data=byte_im,
                        file_name=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )
            else:
                st.info("Click 'Analyze Image' to start")

# ============================================
# TAB 3: LIVE WEBCAM (WITH PRETRAINED FOR PERSONS)
# ============================================
with tab3:
    st.markdown("### Live Webcam Feed")
    st.caption("🔴 Animals | 🟢 Persons (using pretrained model for better detection)")
    
    col_start, col_stop = st.columns(2)
    
    with col_start:
        if st.button("▶ Start Webcam", use_container_width=True):
            st.session_state.webcam_running = True
            st.session_state.webcam_results = None
            
            frame_placeholder = st.empty()
            status_text = st.empty()
            result_placeholder = st.empty()
            
            results = process_webcam(frame_placeholder, status_text, result_placeholder)
            st.session_state.webcam_results = results
    
    with col_stop:
        if st.button("⏹ Stop Webcam", use_container_width=True):
            st.session_state.webcam_running = False
            st.rerun()
    
    if st.session_state.webcam_results is not None and len(st.session_state.webcam_results) > 0:
        st.subheader("📊 Webcam Results")
        df = pd.DataFrame(st.session_state.webcam_results)
        st.dataframe(df, use_container_width=True)
        
        risk_counts = Counter([a['risk'] for a in st.session_state.webcam_results])
        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("🔴 HIGH", risk_counts.get("HIGH", 0))
        rc2.metric("🟠 MEDIUM", risk_counts.get("MEDIUM", 0))
        rc3.metric("🟢 LOW", risk_counts.get("LOW", 0))

# ============================================
# TAB 4: EVALUATION METRICS
# ============================================
with tab4:
    st.subheader("📊 System Evaluation Metrics")
    st.caption("Performance metrics calculated without ground truth labels")
    
    if st.session_state.video_history:
        st.markdown("### 🎥 Video Processing Metrics")
        
        video_metrics_data = []
        for h in st.session_state.video_history[-10:]:
            if 'eval_metrics' in h and h['eval_metrics']:
                m = h['eval_metrics']
                video_metrics_data.append({
                    'Video': h['filename'][:20],
                    'DCR%': m.get('DCR', 0),
                    'TSS%': m.get('TSS', 0),
                    'ANR%': m.get('ANR', 0),
                    'HAIR%': m.get('HAIR', 0),
                    'TSI%': m.get('TSI', 0),
                    'ECS%': m.get('ECS', 0),
                    'CSS': m.get('CSS', 0),
                    'FPS': m.get('FPS', 0)
                })
        
        if video_metrics_data:
            st.dataframe(pd.DataFrame(video_metrics_data), use_container_width=True)
            
            st.markdown("### 📈 Average Performance")
            
            avg_metrics = {
                'DCR': np.mean([d['DCR%'] for d in video_metrics_data]),
                'TSS': np.mean([d['TSS%'] for d in video_metrics_data]),
                'ANR': np.mean([d['ANR%'] for d in video_metrics_data]),
                'HAIR': np.mean([d['HAIR%'] for d in video_metrics_data]),
                'TSI': np.mean([d['TSI%'] for d in video_metrics_data]),
                'ECS': np.mean([d['ECS%'] for d in video_metrics_data]),
                'CSS': np.mean([d['CSS'] for d in video_metrics_data]),
                'FPS': np.mean([d['FPS'] for d in video_metrics_data])
            }
            
            col1, col2, col3, col4 = st.columns(4)
            items = list(avg_metrics.items())
            for i, (name, value) in enumerate(items):
                col = [col1, col2, col3, col4][i % 4]
                with col:
                    if 'CSS' in name:
                        st.metric(name, f"{value:.3f}")
                    else:
                        st.metric(name, f"{value:.1f}")
            
            st.markdown("### 📊 Metrics Radar Chart")
            fig, ax = plt.subplots(figsize=(8, 8))
            
            categories = ['DCR', 'TSS', 'HAIR', 'TSI', 'ECS']
            values = [avg_metrics['DCR'], avg_metrics['TSS'], avg_metrics['HAIR'], avg_metrics['TSI'], avg_metrics['ECS']]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, color='red')
            ax.fill(angles, values, alpha=0.25, color='red')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, 100)
            ax.set_title('System Performance Radar')
            st.pyplot(fig)
    
    if st.session_state.image_history:
        st.markdown("---")
        st.markdown("### 🖼️ Image Analysis Metrics")
        
        image_metrics_data = []
        for h in st.session_state.image_history[-10:]:
            if 'metrics' in h and h['metrics']:
                m = h['metrics']
                image_metrics_data.append({
                    'Image': h['filename'][:20],
                    'DCS': m.get('DCS', 0),
                    'FDC': m.get('FDC', 0),
                    'HAPS%': m.get('HAPS', 0),
                    'SUS%': m.get('SUS', 0),
                    'Animals': m.get('animal_count', 0),
                    'Persons': m.get('person_count', 0)
                })
        
        if image_metrics_data:
            st.dataframe(pd.DataFrame(image_metrics_data), use_container_width=True)
            
            st.markdown("### 📈 Average Image Metrics")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Avg DCS", f"{np.mean([d['DCS'] for d in image_metrics_data]):.3f}")
            col2.metric("Avg FDC", f"{np.mean([d['FDC'] for d in image_metrics_data]):.1f}")
            col3.metric("Avg HAPS", f"{np.mean([d['HAPS%'] for d in image_metrics_data]):.1f}%")
            col4.metric("Avg SUS", f"{np.mean([d['SUS%'] for d in image_metrics_data]):.1f}%")
    
    with st.expander("📖 Understanding the Metrics"):
     st.markdown("""
    ### Video Metrics (No Ground Truth Required)
    
    | Metric | Formula | What it measures | Why it matters |
    |--------|---------|------------------|----------------|
    | **DCR** | (Frames with detections / Total frames) × 100 | Coverage - how often system detects anything | Higher = System is actively monitoring |
    | **TSS** | (Stable class frames / Detected frames) × 100 | Class consistency - animal label doesn't flip | Higher = Reliable detection (Elephant→Elephant, not Elephant→Bear) |
    | **ANR** | (Duplicate alerts / Total alerts) × 100 | Alert quality - too many repetitive alerts | Lower = Less annoyance for forest officers |
    | **HAIR** | (Frames with both animal+person / Total frames) × 100 | Conflict detection - human-animal proximity | Higher = More dangerous coexistence scenes detected |
    | **TSI** | (Stable risk frames / Total threat frames) × 100 | Risk reasoning - risk level doesn't flicker | Higher = Consistent threat assessment (HIGH stays HIGH) |
    | **ECS** | (Consistent events / Total events) × 100 | Intrusion integrity - same animal per event | Higher = One animal, one alert (not mixed) |
    | **CSS** | 1 - Standard deviation(confidence) | Detection stability - confidence doesn't jump | Higher = Model is certain and consistent |
    | **FPS** | Total frames / Processing time | Real-time capability | Higher = Faster processing, better for live feed |
    
    ### Image Metrics (No Ground Truth Required)
    
    | Metric | Formula | What it measures | Why it matters |
    |--------|---------|------------------|----------------|
    | **DCS** | Average of all detection confidences | Overall detection reliability | Higher = Model is confident in detections |
    | **FDC** | Count of low-confidence detections (<0.5) | Potential false positives | Lower = Fewer wrong detections |
    | **HAPS** | 100% if both animal+person present, 50% if one present, 0% if none | Coexistence detection | Higher = System identifies human-animal conflicts |
    | **SUS** | 100% if both or none detected logically, 50% if partial | Scene understanding quality | Higher = Detections make logical sense |
    
    ### Ensemble Scoring (For FINAL VERDICT)
    
    | Component | Weight | Formula |
    |-----------|--------|---------|
    | **Frequency Score** | 25% | (Animal detections / Total detections) × 100 |
    | **Confidence Score** | 15% | Average confidence of animal × 100 |
    | **Temporal Persistence** | 35% | (Longest consecutive streak / Total detections) × 100 |
    | **Coverage Score** | 25% | (Unique frames with animal / Total detections) × 100 |
    
    **Final Score = (Freq×0.25) + (Conf×0.15) + (Temp×0.35) + (Cov×0.25)**
    
    > The animal with highest FINAL SCORE wins the verdict. Temporal persistence (35% weight) is most important - it measures how consistently the animal appears in consecutive frames, which is the most reliable indicator without ground truth.
    """)

st.markdown("---")
st.caption(f"⚙️ Sensitivity: {sensitivity} ({'Low' if sensitivity<=3 else 'Medium' if sensitivity<=7 else 'High'}) | Confidence: {CONFIDENCE_THRESHOLD:.2f} | Alert Distance: {INTRUSION_DISTANCE}m")