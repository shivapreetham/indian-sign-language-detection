import cv2
import mediapipe as mp
import numpy as np
import pickle

# Set feature weights (face weight reduced)
WEIGHT_HAND = 1.0
WEIGHT_FACE = 0.1   # Reduced face weight
WEIGHT_POSE = 0.3

BUFFER_LENGTH = 10

# Load the trained model, scaler (and PCA if used)
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
scaler = model_dict['scaler']
# If PCA was used during training, load it here too:
# pca = model_dict['pca']

# Mapping numerical labels to action names.
labels_dict = {
    0: 'hello', 1: 'balle balle', 2: 'thinking', 3: 'i love this',
    4: 'had you food?', 5: 'i am strong', 6: 'shut up', 7: 'attack on titan',
    8: 'namaste', 9: 'perfect', 10: 'how are you', 11: 'i am good/agree/ok',
    12: 'a student', 13: 'wait', 14: 'i am',
    15: 'At a university', 16: 'Computer Student', 17: 'Pursuing Engineering',
    18: 'Passionate about Innovation', 19: 'Could you please repeat?', 20: 'NO',
    21: 'I got it', 22: 'Thank you', 23: 'Nice to meet you',
    24: 'Good bye', 25: 'See you soon!',
    26: 'God', 27: 'walk', 28: 'sleep',
    29: 'Time', 30: 'Hearing Aid', 31: 'Sick',
    32: 'Drink', 33: 'sorry', 34: 'call',
    35: 'here', 36: '', 37: '',
    38: '', 39: '', 40: '',
}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def get_weighted_features(results):
    """Extract weighted features from landmarks."""
    features = []

    # Left hand landmarks
    if results.left_hand_landmarks:
        left = results.left_hand_landmarks
        left_x = [lm.x for lm in left.landmark]
        left_y = [lm.y for lm in left.landmark]
        for lm in left.landmark:
            features.append((lm.x - min(left_x)) * WEIGHT_HAND)
            features.append((lm.y - min(left_y)) * WEIGHT_HAND)
    else:
        features.extend([0.0] * (21 * 2))

    # Right hand landmarks
    if results.right_hand_landmarks:
        right = results.right_hand_landmarks
        right_x = [lm.x for lm in right.landmark]
        right_y = [lm.y for lm in right.landmark]
        for lm in right.landmark:
            features.append((lm.x - min(right_x)) * WEIGHT_HAND)
            features.append((lm.y - min(right_y)) * WEIGHT_HAND)
    else:
        features.extend([0.0] * (21 * 2))

    # Face landmarks with reduced weight
    if results.face_landmarks:
        face = results.face_landmarks
        face_x = [lm.x for lm in face.landmark]
        face_y = [lm.y for lm in face.landmark]
        for lm in face.landmark:
            features.append((lm.x - min(face_x)) * WEIGHT_FACE)
            features.append((lm.y - min(face_y)) * WEIGHT_FACE)
    else:
        features.extend([0.0] * (468 * 2))

    # Pose landmarks
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

    # Use higher confidence thresholds for detection and tracking.
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
            
            # Visualize landmarks for clarity
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 200), thickness=2))
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2))
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2))
            
            # Extract features from the current frame and update buffer
            features = get_weighted_features(results)
            feature_buffer.append(features)
            if len(feature_buffer) > BUFFER_LENGTH:
                feature_buffer = feature_buffer[-BUFFER_LENGTH:]

            if len(feature_buffer) >= BUFFER_LENGTH:
                aggregated_features = np.mean(feature_buffer, axis=0)
                aggregated_features = np.asarray(aggregated_features).reshape(1, -1)
                # Scale the features using the saved scaler
                aggregated_features_scaled = scaler.transform(aggregated_features)
                # If PCA was used during training, apply it as well:
                # aggregated_features_scaled = pca.transform(aggregated_features_scaled)
                try:
                    prediction = model.predict(aggregated_features_scaled)
                    predicted_action = labels_dict.get(int(prediction[0]), "Unknown")
                except Exception as e:
                    predicted_action = "Error"
                    print("Prediction error:", e)
                
                cv2.putText(frame, predicted_action, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
            
            cv2.imshow('Action Detection (Holistic Tracking)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    signDetection()
