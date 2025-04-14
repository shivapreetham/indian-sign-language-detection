import cv2
import mediapipe as mp
import numpy as np
import pickle

# Configuration values must match those used during training!
WEIGHT_HAND = 1.0
WEIGHT_FACE = 0.1    # Ensure same lower weight is used here
WEIGHT_POSE = 0.3
FACE_LANDMARKS_TO_USE = [1, 4, 10]  # Use the same indices

BUFFER_LENGTH = 10

# Load the trained model
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Mapping from class indices to text labels (adjust as needed)
labels_dict = {
    0: 'hello', 1: 'balle balle', 2: 'thinking', 3: 'i love this',
    4: 'had you food?', 5: 'i am strong', 6: 'shut up', 7: 'attack on titan',
    8: 'namaste', 9: 'perfect', 10: 'how are you ', 11: 'i am good/agree/ok',
    12: 'a student', 13: 'wait', 14: 'i am',  
    15: 'At a university', 16: 'Computer Student', 17: 'Pursuing Engineering',
    18: 'Passionate about Innovation', 19: 'Could you please repeat?', 20: 'NO',
    21: 'I got it', 22: 'Thank you', 23: 'Nice to meet you',
    24: 'Good bye', 25: 'See you soon!', 26: 'God', 27: 'walk',
    28: 'sleep', 29: 'Time', 30: 'Hearing Aid', 31: 'Sick',
    32: 'Drink', 33: 'sorry', 34: 'call', 35: 'here' , 36 : 'Light'
}

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def get_weighted_features(results):
    features = []
    
    # For both hands, use positions relative to face anchor (if desired)
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
        
    # For the face, use only selected landmarks
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
        
    # Process pose landmarks (unchanged)
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
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3) as holistic:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            
            # Draw landmarks for better visualization
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

            # Extract features and buffer them to smooth out predictions
            features = get_weighted_features(results)
            feature_buffer.append(features)
            if len(feature_buffer) > BUFFER_LENGTH:
                feature_buffer = feature_buffer[-BUFFER_LENGTH:]
            
            if len(feature_buffer) >= BUFFER_LENGTH:
                aggregated_features = np.mean(feature_buffer, axis=0)
                try:
                    prediction = model.predict([np.asarray(aggregated_features)])
                    predicted_action = labels_dict[int(prediction[0])]
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
