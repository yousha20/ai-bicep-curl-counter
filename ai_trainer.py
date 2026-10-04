"""
AI Personal Trainer — Bicep Curl Counter
Works with mediapipe 0.10.30+ (new Tasks API)
-------------------------------------------------
Install dependencies:
    python -m pip install opencv-python mediapipe numpy

First run will auto-download two small model files (~10 MB total).

Controls:
    Q  → Quit
    R  → Reset rep counters
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import urllib.request
import os


# ─────────────────────────────────────────────────────────────
# STEP 1 — Auto-download model files (only on first run)
# ─────────────────────────────────────────────────────────────
POSE_MODEL_PATH = "pose_landmarker.task"
HAND_MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(POSE_MODEL_PATH):
    print("Downloading pose model (first run only)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
        POSE_MODEL_PATH
    )
    print("Pose model downloaded.")

if not os.path.exists(HAND_MODEL_PATH):
    print("Downloading hand model (first run only)...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task",
        HAND_MODEL_PATH
    )
    print("Hand model downloaded.")


# ─────────────────────────────────────────────────────────────
# HELPER: angle at joint B between A → B → C
# ─────────────────────────────────────────────────────────────
def calculate_angle(A, B, C):
    A, B, C = np.array(A), np.array(B), np.array(C)
    BA = A - B
    BC = C - B
    cosine = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# ─────────────────────────────────────────────────────────────
# HELPER: semi-transparent dark box for UI panels
# ─────────────────────────────────────────────────────────────
def draw_box(img, x, y, w, h, color=(20, 20, 20), alpha=0.65):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ─────────────────────────────────────────────────────────────
# STEP 2 — Create Pose Landmarker (VIDEO mode)
# ─────────────────────────────────────────────────────────────
pose_options = mp_vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)


# ─────────────────────────────────────────────────────────────
# STEP 3 — Create Hand Landmarker (VIDEO mode)
# ─────────────────────────────────────────────────────────────
hand_options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)


# ─────────────────────────────────────────────────────────────
# CONNECTIONS to draw for body skeleton and hands
# ─────────────────────────────────────────────────────────────
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),   # left arm
    (12, 14), (14, 16),             # right arm
    (11, 23), (12, 24), (23, 24),   # torso
    (23, 25), (25, 27),             # left leg
    (24, 26), (26, 28),             # right leg
]

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index
    (9,10),(10,11),(11,12),            # middle
    (13,14),(14,15),(15,16),           # ring
    (17,18),(18,19),(19,20),           # pinky
    (0,17),(5,9),(9,13),(13,17),       # palm
]

# Arm landmark indices (same in new & old API)
ARM_LANDMARKS = {
    "Left":  {"shoulder": 11, "elbow": 13, "wrist": 15},
    "Right": {"shoulder": 12, "elbow": 14, "wrist": 16},
}


# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────
reps  = {"Left": 0, "Right": 0}
stage = {"Left": None, "Right": None}   # "up" or "down"
frame_count = 0


# ─────────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Camera on — start curling!  Q = quit  |  R = reset reps")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)          # mirror so it feels natural
    h, w  = frame.shape[:2]
    frame_count += 1

    # MediaPipe needs an mp.Image object
    mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    timestamp  = int(frame_count * 1000 / 30)   # milliseconds (assume 30 fps)

    # ── Run both detectors ──────────────────────────────────────
    pose_result = pose_landmarker.detect_for_video(mp_image, timestamp)
    hand_result = hand_landmarker.detect_for_video(mp_image, timestamp)

    # ── Draw pose skeleton ──────────────────────────────────────
    if pose_result.pose_landmarks:
        lm = pose_result.pose_landmarks[0]     # first person detected

        # Draw bone connections
        for a, b in POSE_CONNECTIONS:
            if a < len(lm) and b < len(lm):
                pt1 = (int(lm[a].x * w), int(lm[a].y * h))
                pt2 = (int(lm[b].x * w), int(lm[b].y * h))
                cv2.line(frame, pt1, pt2, (255, 255, 255), 2)

        # Draw joint dots
        for point in lm:
            cx, cy = int(point.x * w), int(point.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 180), -1)

        # ── Curl counting logic ─────────────────────────────────
        for side, ids in ARM_LANDMARKS.items():
            shoulder = [lm[ids["shoulder"]].x * w, lm[ids["shoulder"]].y * h]
            elbow    = [lm[ids["elbow"]].x    * w, lm[ids["elbow"]].y    * h]
            wrist    = [lm[ids["wrist"]].x    * w, lm[ids["wrist"]].y    * h]

            angle = calculate_angle(shoulder, elbow, wrist)

            # Show angle next to the elbow
            cv2.putText(frame, f"{int(angle)}°",
                        (int(elbow[0]) - 20, int(elbow[1]) - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Arm straight → "down", then arm curled → "up" = 1 rep
            if angle > 155:
                stage[side] = "down"
            if angle < 40 and stage[side] == "down":
                stage[side] = "up"
                reps[side] += 1

    # ── Draw hands ──────────────────────────────────────────────
    if hand_result.hand_landmarks:
        for hand_lm in hand_result.hand_landmarks:
            for a, b in HAND_CONNECTIONS:
                pt1 = (int(hand_lm[a].x * w), int(hand_lm[a].y * h))
                pt2 = (int(hand_lm[b].x * w), int(hand_lm[b].y * h))
                cv2.line(frame, pt1, pt2, (200, 200, 255), 2)
            for point in hand_lm:
                cx, cy = int(point.x * w), int(point.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 128, 255), -1)

    # ── UI Panels ───────────────────────────────────────────────

    # Left arm (top-left)
    draw_box(frame, 10, 10, 210, 125)
    cv2.putText(frame, "LEFT ARM",               (22, 42),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (0, 255, 180), 2)
    cv2.putText(frame, f"Reps: {reps['Left']}",  (22, 85),  cv2.FONT_HERSHEY_SIMPLEX, 1.2,  (255, 255, 255), 3)
    cv2.putText(frame, stage["Left"] or "-",     (22, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    # Right arm (top-right)
    draw_box(frame, w - 220, 10, 210, 125)
    cv2.putText(frame, "RIGHT ARM",               (w - 208, 42),  cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (0, 255, 180), 2)
    cv2.putText(frame, f"Reps: {reps['Right']}", (w - 208, 85),  cv2.FONT_HERSHEY_SIMPLEX, 1.2,  (255, 255, 255), 3)
    cv2.putText(frame, stage["Right"] or "-",    (w - 208, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

    # Bottom hint bar
    draw_box(frame, 0, h - 40, w, 40, alpha=0.65)
    cv2.putText(frame, "Q: Quit   |   R: Reset reps",
                (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1)

    cv2.imshow("AI Trainer — Bicep Curl Counter", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        reps  = {"Left": 0, "Right": 0}
        stage = {"Left": None, "Right": None}
        print("Reps reset!")

# ── Cleanup ─────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
pose_landmarker.close()
hand_landmarker.close()

