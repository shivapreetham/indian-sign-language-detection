import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Configuration for weights and selected landmarks
WEIGHT_HAND = 1.0
WEIGHT_FACE = 0.1
WEIGHT_POSE = 0.3

FACE_LANDMARKS_TO_USE = [1, 4, 10]  

# Choose the data directory - original or augmented
DATA_DIR = './augmented_data'  # or './data' if you're not using augmentation
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

            # Process left hand landmarks relative to their min values
            if results.left_hand_landmarks:
                left = results.left_hand_landmarks
                left_x = [lm.x for lm in left.landmark]
                left_y = [lm.y for lm in left.landmark]
                min_left_x = min(left_x)
                min_left_y = min(left_y)
                for lm in left.landmark:
                    feature_vector.append((lm.x - min_left_x) * WEIGHT_HAND)
                    feature_vector.append((lm.y - min_left_y) * WEIGHT_HAND)
            else:
                feature_vector.extend([0.0] * (21 * 2))

            # Process right hand landmarks
            if results.right_hand_landmarks:
                right = results.right_hand_landmarks
                right_x = [lm.x for lm in right.landmark]
                right_y = [lm.y for lm in right.landmark]
                min_right_x = min(right_x)
                min_right_y = min(right_y)
                for lm in right.landmark:
                    feature_vector.append((lm.x - min_right_x) * WEIGHT_HAND)
                    feature_vector.append((lm.y - min_right_y) * WEIGHT_HAND)
            else:
                feature_vector.extend([0.0] * (21 * 2))

            # Process only a selected subset of face landmarks
            if results.face_landmarks:
                face = results.face_landmarks
                selected_face_x = [face.landmark[i].x for i in FACE_LANDMARKS_TO_USE]
                selected_face_y = [face.landmark[i].y for i in FACE_LANDMARKS_TO_USE]
                min_face_x = min(selected_face_x)
                min_face_y = min(selected_face_y)
                for i in FACE_LANDMARKS_TO_USE:
                    lm = face.landmark[i]
                    feature_vector.append((lm.x - min_face_x) * WEIGHT_FACE)
                    feature_vector.append((lm.y - min_face_y) * WEIGHT_FACE)
            else:
                feature_vector.extend([0.0] * (len(FACE_LANDMARKS_TO_USE) * 2))

            # Process pose landmarks
            if results.pose_landmarks:
                pose = results.pose_landmarks
                pose_x = [lm.x for lm in pose.landmark]
                pose_y = [lm.y for lm in pose.landmark]
                min_pose_x = min(pose_x)
                min_pose_y = min(pose_y)
                for lm in pose.landmark:
                    feature_vector.append((lm.x - min_pose_x) * WEIGHT_POSE)
                    feature_vector.append((lm.y - min_pose_y) * WEIGHT_POSE)
            else:
                feature_vector.extend([0.0] * (33 * 2))

            data.append(feature_vector)
            labels.append(dir_)

# Save the processed data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Extracted features from {len(data)} images across {len(set(labels))} classes")
print("Data saved to data.pickle")