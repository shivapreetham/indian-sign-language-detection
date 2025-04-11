import os
import pickle
import cv2
import mediapipe as mp

# Feature weights: reduced face weight to avoid overpowering hand features.
WEIGHT_HAND = 1.0
WEIGHT_FACE = 0.1   # Reduced face weight
WEIGHT_POSE = 0.3

DATA_DIR = './data'
data = []
labels = []

mp_holistic = mp.solutions.holistic

# Use a higher confidence threshold for static image mode
with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.6) as holistic:
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue

        for img_path in os.listdir(dir_path):
            img_file = os.path.join(dir_path, img_path)
            img = cv2.imread(img_file)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = holistic.process(img_rgb)
            feature_vector = []

            # Left hand landmarks
            if results.left_hand_landmarks:
                left = results.left_hand_landmarks
                left_x = [lm.x for lm in left.landmark]
                left_y = [lm.y for lm in left.landmark]
                for lm in left.landmark:
                    feature_vector.append((lm.x - min(left_x)) * WEIGHT_HAND)
                    feature_vector.append((lm.y - min(left_y)) * WEIGHT_HAND)
            else:
                feature_vector.extend([0.0] * (21 * 2))

            # Right hand landmarks
            if results.right_hand_landmarks:
                right = results.right_hand_landmarks
                right_x = [lm.x for lm in right.landmark]
                right_y = [lm.y for lm in right.landmark]
                for lm in right.landmark:
                    feature_vector.append((lm.x - min(right_x)) * WEIGHT_HAND)
                    feature_vector.append((lm.y - min(right_y)) * WEIGHT_HAND)
            else:
                feature_vector.extend([0.0] * (21 * 2))

            # Face landmarks with reduced weight
            if results.face_landmarks:
                face = results.face_landmarks
                face_x = [lm.x for lm in face.landmark]
                face_y = [lm.y for lm in face.landmark]
                for lm in face.landmark:
                    feature_vector.append((lm.x - min(face_x)) * WEIGHT_FACE)
                    feature_vector.append((lm.y - min(face_y)) * WEIGHT_FACE)
            else:
                feature_vector.extend([0.0] * (468 * 2))

            # Pose landmarks
            if results.pose_landmarks:
                pose = results.pose_landmarks
                pose_x = [lm.x for lm in pose.landmark]
                pose_y = [lm.y for lm in pose.landmark]
                for lm in pose.landmark:
                    feature_vector.append((lm.x - min(pose_x)) * WEIGHT_POSE)
                    feature_vector.append((lm.y - min(pose_y)) * WEIGHT_POSE)
            else:
                feature_vector.extend([0.0] * (33 * 2))

            data.append(feature_vector)
            labels.append(dir_)

# Save features and labels for training
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Feature extraction complete. Data saved to 'data.pickle'")
