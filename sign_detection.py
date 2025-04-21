import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import Counter

# — NEW IMPORTS —
import requests
import pyttsx3

# Configuration
WEIGHT_HAND = 1.0
WEIGHT_FACE = 0.1
WEIGHT_POSE = 0.3
FACE_LANDMARKS_TO_USE = [1, 4, 10]
BUFFER_LENGTH = 15
CONFIDENCE_THRESHOLD = 0.6
VOTE_THRESHOLD = 0.6

# Load model and scaler
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
scaler = model_dict['scaler']

# Label mapping - update this with your actual labels
labels_dict = {
    0: 'ok/correct', 1: 'good', 2: 'two', 3: 'engineer',
    4: 'Nice to meet you', 5: 'alright', 6: 'God', 7: 'Walk',
    8: 'sorry', 9: 'call', 10: 'here', 11: 'light', 12: 'namaste/pray', 
    13: 'read', 14: 'loss/headache', 15: 'bond/love', 16: 'strength/powerful',
    17: 'may i go to washroom', 18: 'flower', 19: 'waiting for your action',
    20: 'drink/water', 21: 'food', 22: 'unknown'
}

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Custom finger colors
FINGER_COLORS = [(255, 0, 0), (0, 255, 0), (255, 255, 0),
                 (255, 0, 255), (0, 255, 255)]

# Draw colored finger connections
def draw_fingers(hand_landmarks, frame, handedness='Left'):
    fingers = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20]
    ]
    h, w, _ = frame.shape
    for i, finger in enumerate(fingers):
        color = FINGER_COLORS[i]
        for j in range(len(finger) - 1):
            pt1 = hand_landmarks.landmark[finger[j]]
            pt2 = hand_landmarks.landmark[finger[j+1]]
            x1, y1 = int(pt1.x * w), int(pt1.y * h)
            x2, y2 = int(pt2.x * w), int(pt2.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (x1, y1), 3, color, -1)

# Extract weighted features from landmarks
def get_weighted_features(results):
    features = []
    # Left hand
    if results.left_hand_landmarks:
        xs = [lm.x for lm in results.left_hand_landmarks.landmark]
        ys = [lm.y for lm in results.left_hand_landmarks.landmark]
        minx, miny = min(xs), min(ys)
        for lm in results.left_hand_landmarks.landmark:
            features += [(lm.x - minx) * WEIGHT_HAND, (lm.y - miny) * WEIGHT_HAND]
    else:
        features += [0.0] * 42
    # Right hand
    if results.right_hand_landmarks:
        xs = [lm.x for lm in results.right_hand_landmarks.landmark]
        ys = [lm.y for lm in results.right_hand_landmarks.landmark]
        minx, miny = min(xs), min(ys)
        for lm in results.right_hand_landmarks.landmark:
            features += [(lm.x - minx) * WEIGHT_HAND, (lm.y - miny) * WEIGHT_HAND]
    else:
        features += [0.0] * 42
    # Face landmarks
    if results.face_landmarks:
        sel_x = [results.face_landmarks.landmark[i].x for i in FACE_LANDMARKS_TO_USE]
        sel_y = [results.face_landmarks.landmark[i].y for i in FACE_LANDMARKS_TO_USE]
        minx, miny = min(sel_x), min(sel_y)
        for i in FACE_LANDMARKS_TO_USE:
            lm = results.face_landmarks.landmark[i]
            features += [(lm.x - minx) * WEIGHT_FACE, (lm.y - miny) * WEIGHT_FACE]
    else:
        features += [0.0] * (len(FACE_LANDMARKS_TO_USE) * 2)
    # Pose landmarks
    if results.pose_landmarks:
        xs = [lm.x for lm in results.pose_landmarks.landmark]
        ys = [lm.y for lm in results.pose_landmarks.landmark]
        minx, miny = min(xs), min(ys)
        for lm in results.pose_landmarks.landmark:
            features += [(lm.x - minx) * WEIGHT_POSE, (lm.y - miny) * WEIGHT_POSE]
    else:
        features += [0.0] * 66
    return features

# Draw a semi-transparent overlay and text
def draw_overlay(frame, text, ypos=80):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = 20, ypos
    overlay = frame.copy()
    cv2.rectangle(overlay, (x-10, y-h-10), (x+w+10, y+10), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

# Main detection function
def signDetection():
    cap = cv2.VideoCapture(0)
    buffer = []
    recent_predictions = []
    final_predictions = []
    tts = pyttsx3.init()

    with mp_holistic.Holistic(
        static_image_mode=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Draw landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.face_landmarks:
                h, w, _ = frame.shape
                for i in FACE_LANDMARKS_TO_USE:
                    pt = results.face_landmarks.landmark[i]
                    cv2.circle(frame, (int(pt.x*w), int(pt.y*h)), 3, (0,255,255), -1)
            if results.left_hand_landmarks:
                draw_fingers(results.left_hand_landmarks, frame)
            if results.right_hand_landmarks:
                draw_fingers(results.right_hand_landmarks, frame)

            # Feature buffering
            features = get_weighted_features(results)
            buffer.append(features)
            if len(buffer) > BUFFER_LENGTH:
                buffer = buffer[-BUFFER_LENGTH:]

            # Prediction and voting
            if len(buffer) >= BUFFER_LENGTH:
                mean_feats = np.mean(buffer, axis=0).reshape(1, -1)
                try:
                    scaled = scaler.transform(mean_feats)
                    prediction = model.predict(scaled)
                    probabilities = model.predict_proba(scaled)[0]
                    pred_idx = np.argmax(probabilities)
                    confidence = probabilities[pred_idx]

                    if confidence >= CONFIDENCE_THRESHOLD:
                        recent_predictions.append(int(prediction[0]))
                        if len(recent_predictions) > 5:
                            recent_predictions.pop(0)
                        if recent_predictions:
                            top_class, vote_count = Counter(recent_predictions).most_common(1)[0]
                            if vote_count / len(recent_predictions) >= VOTE_THRESHOLD:
                                label = labels_dict.get(top_class, "Unknown")
                                final_predictions.append(label)
                                cv2.putText(frame, f"{label} ({confidence:.2f})", 
                                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                           1.2, (0, 0, 255), 3, cv2.LINE_AA)
                    else:
                        cv2.putText(frame, "Waiting for clear gesture...", 
                                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                   1.0, (0, 255, 255), 2, cv2.LINE_AA)
                except Exception as e:
                    print("Prediction error:", e)
                    cv2.putText(frame, f"Error: {str(e)}", 
                               (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 0, 255), 2, cv2.LINE_AA)

            # Instruction overlay
            cv2.putText(frame, "Press G to interpret sequence ↓", 
                        (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Sign Detection", frame)
            key = cv2.waitKey(1) & 0xFF

            # On 'g', call Gemini
            if key == ord('g') and final_predictions:
                prompt = (
                    "You are an expert at converting sequences of raw sign-language labels into fluent, human-readable English. You are helping a poor person who cant speak or hear communicate with normal people"
                    f"Given this list of words: {final_predictions}, return one single paragraph adding all the necessary context relatable to the words capturing their meaning. you are supposed to put emotiions into the words , since they cant be captured by the words."
                )
                try:
                    resp = requests.post(
                        "http://localhost:8000/gemini",
                        json={"message": prompt}
                    )
                    resp.raise_for_status()
                    ai_text = resp.json().get("response", "")
                    draw_overlay(frame, ai_text, ypos=80)
                    tts.say(ai_text)
                    tts.runAndWait()
                except Exception as e:
                    print("Error calling Gemini API:", e)
                finally:
                    final_predictions.clear()

            # Quit
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    signDetection()
