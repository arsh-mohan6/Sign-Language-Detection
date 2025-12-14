import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque

# -----------------------------
# Configuration
# -----------------------------
IMG_SIZE = 128
CLASSES = ["Bye", "Hello", "No", "Perfect", "Thank You", "Yes"]

#  Load the trained model (now stored in ../model/)
MODEL_PATH = "../model/sign_language_cnn_model_full.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Smooth prediction history
history = deque(maxlen=7)

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess(frame):
    """Crop / pad image to square and normalize."""
    h, w, _ = frame.shape
    size = max(h, w)
    square = np.zeros((size, size, 3), dtype=np.uint8)
    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = frame
    frame_resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE))
    frame_normalized = frame_resized / 255.0
    return np.expand_dims(frame_normalized, axis=0)

# -----------------------------
# Initialize MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

#  Make window full screen
cv2.namedWindow("Sign Language Detection (MediaPipe)", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Sign Language Detection (MediaPipe)",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)

print(" Webcam started — press 'q' or 'ESC' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror image
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Compute bounding box
            x_min, y_min = w, h
            x_max = y_max = 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            pad = 30
            x_min = max(0, x_min - pad)
            y_min = max(0, y_min - pad)
            x_max = min(w, x_max + pad)
            y_max = min(h, y_max + pad)

            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size == 0:
                continue

            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            # Predict
            input_data = preprocess(hand_roi)
            pred = model.predict(input_data, verbose=0)[0]
            pred_class = np.argmax(pred)
            confidence = pred[pred_class]

            history.append(pred_class)
            class_id = max(set(history), key=history.count)

            # Display prediction
            if confidence > 0.6:
                text = f"{CLASSES[class_id]} ({confidence*100:.1f}%)"
                color = (0, 255, 0)
            else:
                text = "Uncertain"
                color = (0, 0, 255)

            cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    else:
        cv2.putText(frame, "Show hand in view", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Display the frame (now full screen)
    cv2.imshow("Sign Language Detection (MediaPipe)", frame)

    # Quit on 'q' or 'ESC'
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
print(" Webcam closed.")
