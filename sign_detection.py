import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import Counter

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

def draw_fingers(hand_landmarks, frame, handedness='Left'):
    fingers = [
        [0, 1, 2, 3, 4],     # Thumb
        [0, 5, 6, 7, 8],     # Index
        [0, 9, 10, 11, 12],  # Middle
        [0, 13, 14, 15, 16], # Ring
        [0, 17, 18, 19, 20]  # Pinky
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

def get_weighted_features(results):
    features = []
    
    # Process left hand landmarks
    if results.left_hand_landmarks:
        left = results.left_hand_landmarks
        left_x = [lm.x for lm in left.landmark]
        left_y = [lm.y for lm in left.landmark]
        min_left_x, min_left_y = min(left_x), min(left_y)
        for lm in left.landmark:
            features.append((lm.x - min_left_x) * WEIGHT_HAND)
            features.append((lm.y - min_left_y) * WEIGHT_HAND)
    else:
        features.extend([0.0] * (21 * 2))

    # Process right hand landmarks
    if results.right_hand_landmarks:
        right = results.right_hand_landmarks
        right_x = [lm.x for lm in right.landmark]
        right_y = [lm.y for lm in right.landmark]
        min_right_x, min_right_y = min(right_x), min(right_y)
        for lm in right.landmark:
            features.append((lm.x - min_right_x) * WEIGHT_HAND)
            features.append((lm.y - min_right_y) * WEIGHT_HAND)
    else:
        features.extend([0.0] * (21 * 2))
        
    # Process face landmarks
    if results.face_landmarks:
        face = results.face_landmarks
        selected_face_x = [face.landmark[i].x for i in FACE_LANDMARKS_TO_USE]
        selected_face_y = [face.landmark[i].y for i in FACE_LANDMARKS_TO_USE]
        min_face_x, min_face_y = min(selected_face_x), min(selected_face_y)
        for i in FACE_LANDMARKS_TO_USE:
            lm = face.landmark[i]
            features.append((lm.x - min_face_x) * WEIGHT_FACE)
            features.append((lm.y - min_face_y) * WEIGHT_FACE)
    else:
        features.extend([0.0] * (len(FACE_LANDMARKS_TO_USE) * 2))
        
    # Process pose landmarks
    if results.pose_landmarks:
        pose = results.pose_landmarks
        pose_x = [lm.x for lm in pose.landmark]
        pose_y = [lm.y for lm in pose.landmark]
        min_pose_x, min_pose_y = min(pose_x), min(pose_y)
        for lm in pose.landmark:
            features.append((lm.x - min_pose_x) * WEIGHT_POSE)
            features.append((lm.y - min_pose_y) * WEIGHT_POSE)
    else:
        features.extend([0.0] * (33 * 2))

    return features

def signDetection():
    cap = cv2.VideoCapture(0)
    buffer = []
    recent_predictions = []

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

            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

            # Draw simple face dots
            if results.face_landmarks:
                h, w, _ = frame.shape
                for i in FACE_LANDMARKS_TO_USE:
                    pt = results.face_landmarks.landmark[i]
                    cx, cy = int(pt.x * w), int(pt.y * h)
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 255), -1)

            # Draw custom colored fingers
            if results.left_hand_landmarks:
                draw_fingers(results.left_hand_landmarks, frame, 'Left')
            if results.right_hand_landmarks:
                draw_fingers(results.right_hand_landmarks, frame, 'Right')

            # Process features
            features = get_weighted_features(results)
            buffer.append(features)
            if len(buffer) > BUFFER_LENGTH:
                buffer = buffer[-BUFFER_LENGTH:]

            if len(buffer) >= BUFFER_LENGTH:
                mean_features = np.mean(buffer, axis=0).reshape(1, -1)
                try:
                    # Scale the features
                    scaled = scaler.transform(mean_features)
                    
                    # Get prediction and probability
                    prediction = model.predict(scaled)
                    probabilities = model.predict_proba(scaled)[0]
                    pred_idx = np.argmax(probabilities)
                    confidence = probabilities[pred_idx]
                    
                    # Add to recent predictions if confidence is high enough
                    if confidence >= CONFIDENCE_THRESHOLD:
                        recent_predictions.append(int(prediction[0]))
                        if len(recent_predictions) > 5:
                            recent_predictions.pop(0)
                        
                        # Use voting to determine the final prediction
                        if recent_predictions:
                            vote_counts = Counter(recent_predictions)
                            top_vote = vote_counts.most_common(1)[0]
                            top_class, vote_count = top_vote
                            
                            # Only display if there's strong consensus
                            if vote_count / len(recent_predictions) >= VOTE_THRESHOLD:
                                label = labels_dict.get(top_class, "Unknown")
                                cv2.putText(frame, f"{label} ({confidence:.2f})", 
                                           (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                           1.2, (0, 0, 255), 3, cv2.LINE_AA)
                    else:
                        # Low confidence
                        cv2.putText(frame, "Waiting for clear gesture...", 
                                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                   1.0, (0, 255, 255), 2, cv2.LINE_AA)
                        
                except Exception as e:
                    print("Prediction error:", e)
                    cv2.putText(frame, f"Error: {str(e)}", 
                               (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                               0.7, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    signDetection()