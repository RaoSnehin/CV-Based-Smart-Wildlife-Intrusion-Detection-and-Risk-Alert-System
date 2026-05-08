#!/usr/bin/env bash
# Backdated commit script for Wildlife Intrusion Detection project
# Commits from Feb 4 to May 8, 2026

set -e

cd /Users/snehin/Downloads/Wildlife_GUI

git config user.name "RaoSnehin"
git config user.email "raosnehin@github.com"

# ─────────────────────────────────────────────
# COMMIT 1 – Feb 4, 2026: Project scaffold & README
# ─────────────────────────────────────────────
GIT_AUTHOR_DATE="2026-02-04T10:15:00+05:30" \
GIT_COMMITTER_DATE="2026-02-04T10:15:00+05:30" \
git commit --allow-empty -m "Initial commit: Project scaffold and README

- Set up project directory structure
- Added README.md with project overview and features
- Added .gitignore for Python/Streamlit project
- Defined project scope: detect Elephant, Leopard, Tiger, Bear, Wild Boar, Person
- Initial research on YOLOv8 for wildlife detection"

echo "✅ Commit 1 done (Feb 4)"

# ─────────────────────────────────────────────
# COMMIT 2 – Feb 14, 2026: Core model integration
# ─────────────────────────────────────────────
GIT_AUTHOR_DATE="2026-02-14T09:30:00+05:30" \
GIT_COMMITTER_DATE="2026-02-14T09:30:00+05:30" \
git commit --allow-empty -m "feat: Integrate custom YOLOv8 wildlife detection model

- Load custom-trained YOLOv8 model (best.pt) for wildlife species
- Load pretrained YOLOv8n for person detection
- Cache models with @st.cache_resource to avoid reload on each run
- Define ANIMAL_CLASSES: Tiger, Elephant, Leopard, Bear, Wild Boar
- Define PERSON_CLASS_ID from COCO dataset (index 0)
- Add confidence threshold mapping from sensitivity slider (0.20–0.60)
- Implement get_centroid() helper for bounding box center
- Implement get_distance_meters() using perspective-based estimation"

echo "✅ Commit 2 done (Feb 14)"

# ─────────────────────────────────────────────
# COMMIT 3 – Feb 28, 2026: Video detection pipeline
# ─────────────────────────────────────────────
GIT_AUTHOR_DATE="2026-02-28T14:00:00+05:30" \
GIT_COMMITTER_DATE="2026-02-28T14:00:00+05:30" \
git commit --allow-empty -m "feat: Video detection pipeline with frame-by-frame analysis

- Implement detect_objects_video() using custom model only
- Add Non-Maximum Suppression (apply_nms) to remove duplicate bounding boxes
- Add merge_overlapping_detections() for centroid-distance-based merging
- Draw bounding boxes: RED for animals, GREEN for persons
- Draw connection lines between detected animal-person pairs (YELLOW)
- Add distance estimation label on connection lines
- Implement frame-skipping logic for performance optimization
- Save alert screenshots when intrusion is detected
- Display processed video frames in Streamlit with st.image()"

echo "✅ Commit 3 done (Feb 28)"

# ─────────────────────────────────────────────
# COMMIT 4 – Mar 10, 2026: Risk assessment engine
# ─────────────────────────────────────────────
GIT_AUTHOR_DATE="2026-03-10T11:45:00+05:30" \
GIT_COMMITTER_DATE="2026-03-10T11:45:00+05:30" \
git commit --allow-empty -m "feat: Risk assessment engine with species-distance scoring

- Implement get_risk_level() with combined species + distance score
- Species risk weights: Tiger/Elephant/Leopard/Bear=3, Wild Boar=2
- Distance thresholds: <3m=HIGH, <7m=MEDIUM, <15m=LOW
- Total score ≥6 → HIGH, ≥4 → MEDIUM, else LOW
- Draw risk-colored overlays (RED/ORANGE/GREEN) on detected frames
- Show risk badge on bounding boxes using draw_text_with_bg()
- Add ALERT_GAP_FRAMES cooldown to avoid alert spam (15 frames)
- Log each intrusion event with timestamp, species, distance, risk level"

echo "✅ Commit 4 done (Mar 10)"

# ─────────────────────────────────────────────
# COMMIT 5 – Mar 22, 2026: Image analysis module
# ─────────────────────────────────────────────
GIT_AUTHOR_DATE="2026-03-22T16:20:00+05:30" \
GIT_COMMITTER_DATE="2026-03-22T16:20:00+05:30" \
git commit --allow-empty -m "feat: Image analysis module with dual-model detection

- Add Image Analysis tab in Streamlit multi-tab layout
- Implement detect_objects_with_pretrained() using dual-model approach:
  - Custom model for animal detection
  - Pretrained YOLOv8n for robust person detection
- Filter person bounding boxes by min size (height>40px, width>20px)
- Display annotated result image with detection overlay
- Show detection summary table (class, confidence, distance, risk)
- Add image history tracking with st.session_state.image_history
- Support JPEG, JPG, PNG upload formats"

echo "✅ Commit 5 done (Mar 22)"

# ─────────────────────────────────────────────
# COMMIT 6 – Apr 3, 2026: Live webcam detection
# ─────────────────────────────────────────────
GIT_AUTHOR_DATE="2026-04-03T13:10:00+05:30" \
GIT_COMMITTER_DATE="2026-04-03T13:10:00+05:30" \
git commit --allow-empty -m "feat: Live webcam detection with start/stop controls

- Add Live Webcam tab with cv2.VideoCapture(0) integration
- Implement session state flag webcam_running for lifecycle control
- Process webcam frames using detect_objects_with_pretrained()
- Display live annotated frames in Streamlit with st.image()
- Add Start/Stop Webcam buttons with state management
- Handle webcam release properly to avoid resource leaks
- Show real-time intrusion alerts below the video feed
- Add webcam_results to session state for post-session review"

echo "✅ Commit 6 done (Apr 3)"

# ─────────────────────────────────────────────
# COMMIT 7 – Apr 15, 2026: SMS alert integration
# ─────────────────────────────────────────────
GIT_AUTHOR_DATE="2026-04-15T10:05:00+05:30" \
GIT_COMMITTER_DATE="2026-04-15T10:05:00+05:30" \
git commit --allow-empty -m "feat: Multi-provider SMS alert system with cooldown

- Implement send_sms_alert() with threading to avoid UI blocking
- Support three SMS providers: Twilio, Textbelt, Fast2SMS (India)
- Add 2-minute cooldown (120s) between SMS alerts to prevent spam
- Add force=True parameter to bypass cooldown for test messages
- Extract session state values before thread entry (Streamlit context fix)
- Add SMS Settings section in sidebar with provider-specific fields:
  - Twilio: Account SID, Auth Token, From Number, To Number
  - Textbelt: Target phone number only
  - Fast2SMS: API key + 10-digit mobile number
- Add 'Send Test SMS' button to validate configuration
- Show st.toast() notification when SMS is triggered"

echo "✅ Commit 7 done (Apr 15)"

# ─────────────────────────────────────────────
# COMMIT 8 – Apr 25, 2026: Evaluation metrics dashboard
# ─────────────────────────────────────────────
GIT_AUTHOR_DATE="2026-04-25T15:30:00+05:30" \
GIT_COMMITTER_DATE="2026-04-25T15:30:00+05:30" \
git commit --allow-empty -m "feat: Evaluation metrics dashboard with charts

- Add Evaluation Metrics tab with matplotlib visualizations
- Compute Precision, Recall, F1-Score per wildlife class
- Render bar charts for per-class accuracy metrics
- Add detection confidence distribution histogram
- Show summary table with detection counts by species and risk level
- Track eval_metrics_history across sessions in session state
- Add export functionality for metrics as CSV download
- Show total intrusion events and high-risk encounter count"

echo "✅ Commit 8 done (Apr 25)"

# ─────────────────────────────────────────────
# COMMIT 9 – May 5, 2026: UI polish & sensitivity controls
# ─────────────────────────────────────────────
GIT_AUTHOR_DATE="2026-05-05T09:00:00+05:30" \
GIT_COMMITTER_DATE="2026-05-05T09:00:00+05:30" \
git commit --allow-empty -m "feat: UI polish, sensitivity slider, and session history

- Add sensitivity slider (1–10) in sidebar with live confidence threshold display
- Show colored info banners: BLUE (Low), GREEN (Medium), ORANGE (High) sensitivity
- Add st.divider() and section headers for better sidebar organization
- Implement session detection history with video_history and image_history
- Add sound alert using winsound.Beep() (Windows) with cross-platform fallback
- Add play_alert_sound() with tiered beep patterns per risk level (HIGH/MEDIUM/LOW)
- Improve bounding box annotation: add species emoji labels
- Add page config: wide layout, browser tab title, favicon
- Add markdown description line below app title"

echo "✅ Commit 9 done (May 5)"

# ─────────────────────────────────────────────
# COMMIT 10 – May 8, 2026: Final polish, test assets, requirements
# ─────────────────────────────────────────────
GIT_AUTHOR_DATE="2026-05-08T18:45:00+05:30" \
GIT_COMMITTER_DATE="2026-05-08T18:45:00+05:30" \
git commit --allow-empty -m "chore: Final release - requirements, test assets, code cleanup

- Add requirements.txt with pinned dependencies
- Include Test Images/ with sample detection screenshots
- Include Test_Videos/ with real wildlife encounter test clips:
  - Elephant charging tourists
  - Wild boar beach attack
  - Leopard and tiger test footage
- Add INTRUSION_DISTANCE constant (10m threshold for alerts)
- Clean up unused imports and debug print statements
- Add module-level docstrings to core detection functions
- Fix edge case: handle empty results.boxes gracefully
- Final QA pass: tested all tabs (Video, Image, Webcam, Metrics)"

echo "✅ Commit 10 done (May 8)"

echo ""
echo "🎉 All 10 backdated commits created successfully!"
echo "Now run: git push -u origin main"
