import cv2
import mediapipe as mp
import numpy as np
import pickle

# Configuration
WEIGHT_HAND = 3.0
WEIGHT_FACE = 0.3
WEIGHT_POSE = 0.4
FACE_LANDMARKS_TO_USE = [1, 4, 10]
BUFFER_LENGTH = 10

# Load model and scaler
with open('model.p', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']
scaler = model_dict['scaler']

labels_dict = {
    0: 'ok/correct', 1: 'good', 2: 'two', 3: 'engineer',
    4: 'Nice to meet you', 5: 'alright', 6: 'God', 7: 'Walk',
    8: 'sorry', 9: 'call', 10: 'here ', 11: 'light', 12: 'namaste/pray', 
    13: 'read', 14:'loss/headache',15: 'bond/love', 16:'strength/powerful' , 17: 'may i go to washroom' ,
    18:'flower' , 19:'waiting for your action' , 20: 'drink/water', 21: 'food'
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
    
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            features.extend([lm.x * WEIGHT_HAND, lm.y * WEIGHT_HAND])
    else:
        features.extend([0.0] * (21 * 2))

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            features.extend([lm.x * WEIGHT_HAND, lm.y * WEIGHT_HAND])
    else:
        features.extend([0.0] * (21 * 2))
        
    if results.face_landmarks:
        face = results.face_landmarks
        selected_x = [face.landmark[i].x for i in FACE_LANDMARKS_TO_USE]
        selected_y = [face.landmark[i].y for i in FACE_LANDMARKS_TO_USE]
        min_x, min_y = min(selected_x), min(selected_y)
        for i in FACE_LANDMARKS_TO_USE:
            lm = face.landmark[i]
            features.extend([(lm.x - min_x) * WEIGHT_FACE, (lm.y - min_y) * WEIGHT_FACE])
    else:
        features.extend([0.0] * (len(FACE_LANDMARKS_TO_USE) * 2))
        
    if results.pose_landmarks:
        pose = results.pose_landmarks
        pose_x = [lm.x for lm in pose.landmark]
        pose_y = [lm.y for lm in pose.landmark]
        min_px, min_py = min(pose_x), min(pose_y)
        for lm in pose.landmark:
            features.extend([(lm.x - min_px) * WEIGHT_POSE, (lm.y - min_py) * WEIGHT_POSE])
    else:
        features.extend([0.0] * (33 * 2))

    return features

def signDetection():
    cap = cv2.VideoCapture(0)
    buffer = []

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
                    scaled = scaler.transform(mean_features)
                    prediction = model.predict(scaled)
                    label = labels_dict.get(int(prediction[0]), "Unknown")
                except Exception as e:
                    print("Prediction error:", e)
                    label = "Error"
                cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 0, 0), 3, cv2.LINE_AA)

            cv2.imshow("Sign Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    signDetection()
