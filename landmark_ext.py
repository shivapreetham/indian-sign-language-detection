import cv2
import mediapipe as mp
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Configuration values must match those used during training!
WEIGHT_HAND = 1.0
WEIGHT_FACE = 0.1
WEIGHT_POSE = 0.3
FACE_LANDMARKS_TO_USE = [1, 4, 10]
BUFFER_LENGTH = 10

# Load the trained model and scaler
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
scaler = model_dict['scaler']  # ✅ load pre-fitted scaler

# Mapping from class indices to text labels (adjust as needed)
labels_dict = {
    0: 'ok', 1: 'good', 2: 'two', 3: 'engineer',
    4: 'Nice to meet you', 5: 'alright', 6: 'God', 7: 'Walk',
    8: 'sorry', 9: 'call', 10: 'here ', 11: 'light'
}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def get_weighted_features(results):
    features = []

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.append(lm.x * WEIGHT_HAND)
            features.append(lm.y * WEIGHT_HAND)
    else:
        features.extend([0.0] * (21 * 2))

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.append(lm.x * WEIGHT_HAND)
            features.append(lm.y * WEIGHT_HAND)
    else:
        features.extend([0.0] * (21 * 2))

    if results.face_landmarks:
        face = results.face_landmarks
        selected_face_x = [face.landmark[i].x for i in FACE_LANDMARKS_TO_USE]
        selected_face_y = [face.landmark[i].y for i in FACE_LANDMARKS_TO_USE]
        min_face_x = min(selected_face_x)
        min_face_y = min(selected_face_y)
        for i in FACE_LANDMARKS_TO_USE:
            lm = face.landmark[i]
            features.append((lm.x - min_face_x) * WEIGHT_FACE)
            features.append((lm.y - min_face_y) * WEIGHT_FACE)
    else:
        features.extend([0.0] * (len(FACE_LANDMARKS_TO_USE) * 2))

    if results.pose_landmarks:
        pose = results.pose_landmarks
        pose_x = [lm.x for lm in pose.landmark]
        pose_y = [lm.y for lm in pose.landmark]
        for lm in pose.landmark:
            features.append((lm.x - min(pose_x)) * WEIGHT_POSE)
            features.append((lm.y - min(pose_y)) * WEIGHT_POSE)
    else:
        features.extend([0.0] * (33 * 2))

    return features

def signDetection():
    cap = cv2.VideoCapture(0)
    feature_buffer = []

    with mp_holistic.Holistic(
            static_image_mode=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6) as holistic:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)

            # Draw landmarks
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Feature extraction
            features = get_weighted_features(results)
            feature_buffer.append(features)
            if len(feature_buffer) > BUFFER_LENGTH:
                feature_buffer = feature_buffer[-BUFFER_LENGTH:]

            if len(feature_buffer) >= BUFFER_LENGTH:
                aggregated_features = np.mean(feature_buffer, axis=0).reshape(1, -1)

                try:
                    aggregated_features_scaled = scaler.transform(aggregated_features)  
                    prediction = model.predict(aggregated_features_scaled)
                    predicted_action = labels_dict.get(int(prediction[0]), "Unknown")
                except Exception as e:
                    predicted_action = "Error"
                    print("Prediction error:", e)

                cv2.putText(frame, predicted_action, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.imshow('Action Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    signDetection()
