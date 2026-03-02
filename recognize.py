"""
Step 3: Real-Time Gesture Recognition
========================================
Run this script after training is complete.
It opens your webcam, detects your hand in real time, and displays
the predicted digit on screen.

Controls:
    Q  →  Quit

Usage:
    python recognize.py
"""

import cv2
import mediapipe as mp
import numpy as np
import pickle
import os

MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "gesture_model.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

# Confidence below this threshold shows "?" instead of a digit
MIN_CONFIDENCE = 0.6

# Colour palette (BGR)
CLR_GREEN  = (50,  220,  80)
CLR_YELLOW = (0,   220, 220)
CLR_RED    = (50,   50, 220)
CLR_WHITE  = (240, 240, 240)
CLR_DARK   = (20,   20,  20)


# ── Load model & encoder ──────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("ERROR: Trained model not found.")
        print("       Please run  python train_model.py  first.")
        return None, None
    with open(MODEL_PATH,   "rb") as f:
        clf = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le  = pickle.load(f)
    return clf, le


# ── Helper: extract normalised landmark vector (same as collect_data.py) ──────
def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


# ── Draw a rounded rectangle (pill button background) ────────────────────────
def draw_rounded_rect(img, x, y, w, h, r, color):
    cv2.rectangle(img, (x + r, y),     (x + w - r, y + h),     color, -1)
    cv2.rectangle(img, (x,     y + r), (x + w,     y + h - r), color, -1)
    cv2.circle(img, (x + r,     y + r),     r, color, -1)
    cv2.circle(img, (x + w - r, y + r),     r, color, -1)
    cv2.circle(img, (x + r,     y + h - r), r, color, -1)
    cv2.circle(img, (x + w - r, y + h - r), r, color, -1)


# ── Main recognition loop ─────────────────────────────────────────────────────
def recognize():
    clf, le = load_model()
    if clf is None:
        return

    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles
    hands    = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    # Smooth the prediction with a short history to avoid flickering
    pred_history = []
    HISTORY_LEN  = 8

    print("✅ Recognizer running – press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed – retrying …")
            continue

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Default display values
        digit_text = "?"
        conf_text  = ""
        box_color  = CLR_RED
        hand_found = False

        if result.multi_hand_landmarks:
            hand_found = True
            for hl in result.multi_hand_landmarks:
                # Draw styled landmarks
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_style.get_default_hand_landmarks_style(),
                    mp_style.get_default_hand_connections_style(),
                )

                vec   = extract_landmarks(hl).reshape(1, -1)
                probs = clf.predict_proba(vec)[0]
                conf  = float(probs.max())
                pred  = int(le.inverse_transform([probs.argmax()])[0])

                pred_history.append(pred)
                if len(pred_history) > HISTORY_LEN:
                    pred_history.pop(0)

                # Majority vote over history
                stable = max(set(pred_history), key=pred_history.count)

                if conf >= MIN_CONFIDENCE:
                    digit_text = str(stable)
                    conf_text  = f"{conf * 100:.0f}%"
                    box_color  = CLR_GREEN
                else:
                    digit_text = "?"
                    conf_text  = "low conf"
                    box_color  = CLR_YELLOW
        else:
            pred_history.clear()

        # ── HUD overlay ──────────────────────────────────────────────────────
        # Large digit box (top-right)
        box_x, box_y, box_w, box_h = w - 180, 20, 160, 160
        overlay = frame.copy()
        draw_rounded_rect(overlay, box_x, box_y, box_w, box_h, 20, CLR_DARK)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Digit
        digit_scale = 4.5 if len(digit_text) == 1 else 2.5
        digit_size, _ = cv2.getTextSize(digit_text, cv2.FONT_HERSHEY_DUPLEX,
                                        digit_scale, 5)
        digit_ox = box_x + (box_w - digit_size[0]) // 2
        digit_oy = box_y + box_h - 30
        cv2.putText(frame, digit_text, (digit_ox, digit_oy),
                    cv2.FONT_HERSHEY_DUPLEX, digit_scale, box_color, 5, cv2.LINE_AA)

        # Confidence
        if conf_text:
            conf_size, _ = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX,
                                           0.7, 2)
            cv2.putText(frame, conf_text,
                        (box_x + (box_w - conf_size[0]) // 2, box_y + box_h + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR_WHITE, 2, cv2.LINE_AA)

        # Status bar (bottom)
        status = "Hand detected – predicting" if hand_found else "Show your hand …"
        cv2.rectangle(frame, (0, h - 40), (w, h), CLR_DARK, -1)
        cv2.putText(frame, status, (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, CLR_WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, "Q: Quit", (w - 110, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (120, 120, 120), 1, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Recognition (1-9)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Bye!")


if __name__ == "__main__":
    recognize()
