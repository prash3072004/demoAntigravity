# 🖐️ Hand Gesture Recognition (Numbers 1–9)

A beginner-friendly ML project that uses your **webcam** + **MediaPipe** to detect hand landmarks in real time, trains a **Random Forest** classifier on YOUR gestures, and predicts digits 1–9 live.

---

## 📁 Project Structure

```
demo project/
├── requirements.txt   # Dependencies
├── collect_data.py    # Step 1 – gather training data
├── train_model.py     # Step 2 – train the classifier
├── recognize.py       # Step 3 – live recognition
├── data/              # Created automatically (gesture data)
└── model/             # Created automatically (saved model)
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage – Three Simple Steps

### Step 1 – Collect training data

```bash
python collect_data.py
```

- A window opens showing your webcam feed.
- Press **ENTER** in the terminal for each digit.
- Wait 3 seconds for the countdown, then **hold up the corresponding number of fingers** steadily.
- 100 frames are captured automatically per digit.
- Repeat for all 9 digits.

**Suggested gestures:**

| Digit | Gesture |
|-------|---------|
| 1     | Index finger only |
| 2     | Index + Middle |
| 3     | Index + Middle + Ring |
| 4     | Index + Middle + Ring + Pinky |
| 5     | All 5 fingers open |
| 6     | Thumb + Pinky (hang-loose 🤙) |
| 7     | Thumb + Index + Middle |
| 8     | Thumb + Index + Middle + Ring |
| 9     | All fingers except thumb (4 fingers) |

> **Tip:** Keep your hand clearly visible and well-lit. The more consistent your gesture, the better the accuracy.

---

### Step 2 – Train the model

```bash
python train_model.py
```

- Trains a Random Forest classifier on your collected data.
- Prints a classification report — expect **>90% accuracy** with consistent gestures.
- Saves the model to `model/gesture_model.pkl`.

---

### Step 3 – Real-time recognition

```bash
python recognize.py
```

- Opens the webcam with a live prediction overlay.
- The predicted digit is shown in large text (top-right corner).
- Confidence % is shown below the digit.
- Press **Q** to quit.

---

## 🛠️ Tips for Best Results

- **Lighting** – make sure your hand is well-lit with no harsh shadows.
- **Background** – a plain background helps MediaPipe detect your hand faster.
- **Distance** – keep your hand 30–60 cm from the camera.
- **Retrain** – if accuracy is low, re-run `collect_data.py` with more consistent gestures.

---

## 📦 Dependencies

| Library | Purpose |
|---------|---------|
| `opencv-python` | Webcam access & image display |
| `mediapipe` | Hand landmark detection |
| `scikit-learn` | Random Forest classifier |
| `numpy` | Numerical arrays |
