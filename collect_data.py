"""
Step 1: Collect Training Data
==============================
Run this script first. It will open your webcam and guide you to show
each hand gesture (1-9). Press SPACE to capture a sample, and the
script will save 100 samples per digit automatically.

Usage:
    python collect_data.py
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ── Configuration ────────────────────────────────────────────────────────────
SAMPLES_PER_CLASS = 100          # Number of frames to capture per digit
CLASSES = list(range(1, 10))     # Digits 1 through 9
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ── MediaPipe setup ──────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# ── Helper: extract normalised landmark vector ────────────────────────────────
def extract_landmarks(hand_landmarks):
    """Return a flat numpy array of 63 normalised landmark values (x, y, z × 21)."""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)


# ── Main collection loop ──────────────────────────────────────────────────────
def collect():
    all_X, all_y = [], []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam. Check your camera connection.")
        return

    print("\n=== Hand Gesture Data Collection ===")
    print("Hold up the indicated gesture and keep it steady.")
    print("Collection starts automatically after a 3-second countdown.\n")

    for digit in CLASSES:
        samples = []
        countdown_done = False
        start_time = None

        print(f"─── Gesture: {digit}  ({SAMPLES_PER_CLASS} samples needed) ───")
        print("      Press ENTER in the terminal when ready, then show the gesture.")
        input("      → Press ENTER to start countdown for digit " + str(digit) + " … ")
        start_time = time.time()

        while len(samples) < SAMPLES_PER_CLASS:
            ret, frame = cap.read()
            if not ret:
                print("  Camera read failed – retrying …")
                continue

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            elapsed = time.time() - start_time
            remaining = max(0, 3 - int(elapsed))

            # Draw countdown
            if remaining > 0:
                cv2.putText(frame, f"Get ready: {remaining}", (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 165, 255), 4)
            else:
                # Collecting phase
                cv2.putText(frame, f"Gesture: {digit}", (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 200, 0), 3)
                cv2.putText(frame, f"Collected: {len(samples)}/{SAMPLES_PER_CLASS}",
                            (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

                if result.multi_hand_landmarks:
                    for hl in result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                        vec = extract_landmarks(hl)
                        samples.append(vec)
                        # Small pause so we don't grab duplicate identical frames
                        time.sleep(0.03)
                else:
                    cv2.putText(frame, "No hand detected", (30, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Collecting Data", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Aborted by user.")
                cap.release()
                cv2.destroyAllWindows()
                return

        all_X.extend(samples[:SAMPLES_PER_CLASS])
        all_y.extend([digit] * SAMPLES_PER_CLASS)
        print(f"  ✓ Collected {SAMPLES_PER_CLASS} samples for digit {digit}\n")

    cap.release()
    cv2.destroyAllWindows()

    # Save
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)
    print(f"✅ Data saved to '{DATA_DIR}/'  →  X shape: {X.shape}, y shape: {y.shape}")
    print("   Next step: run  python train_model.py")


if __name__ == "__main__":
    collect()
