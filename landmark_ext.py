import os
import pickle
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
# Configuration for weights and selected landmarks
WEIGHT_HAND = 1.0
WEIGHT_FACE = 0.1
WEIGHT_POSE = 0.3

FACE_LANDMARKS_TO_USE = [1, 4, 10]

# Choose the data directory - original or augmented
DATA_DIR = './augmented_data'  

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
            fv = []

            # Process left hand landmarks relative to their min values
            if results.left_hand_landmarks:
                lh = results.left_hand_landmarks
                xs = [lm.x for lm in lh.landmark]
                ys = [lm.y for lm in lh.landmark]
                mx, my = min(xs), min(ys)
                for lm in lh.landmark:
                    fv.append((lm.x - mx) * WEIGHT_HAND)
                    fv.append((lm.y - my) * WEIGHT_HAND)
            else:
                fv.extend([0.0] * 42)

            # Process right hand landmarks
            if results.right_hand_landmarks:
                rh = results.right_hand_landmarks
                xs = [lm.x for lm in rh.landmark]
                ys = [lm.y for lm in rh.landmark]
                mx, my = min(xs), min(ys)
                for lm in rh.landmark:
                    fv.append((lm.x - mx) * WEIGHT_HAND)
                    fv.append((lm.y - my) * WEIGHT_HAND)
            else:
                fv.extend([0.0] * 42)

            # Process only selected face landmarks
            if results.face_landmarks:
                f = results.face_landmarks
                sel_x = [f.landmark[i].x for i in FACE_LANDMARKS_TO_USE]
                sel_y = [f.landmark[i].y for i in FACE_LANDMARKS_TO_USE]
                mx, my = min(sel_x), min(sel_y)
                for i in FACE_LANDMARKS_TO_USE:
                    lm = f.landmark[i]
                    fv.append((lm.x - mx) * WEIGHT_FACE)
                    fv.append((lm.y - my) * WEIGHT_FACE)
            else:
                fv.extend([0.0] * (len(FACE_LANDMARKS_TO_USE) * 2))

            # Process pose landmarks
            if results.pose_landmarks:
                p = results.pose_landmarks
                xs = [lm.x for lm in p.landmark]
                ys = [lm.y for lm in p.landmark]
                mx, my = min(xs), min(ys)
                for lm in p.landmark:
                    fv.append((lm.x - mx) * WEIGHT_POSE)
                    fv.append((lm.y - my) * WEIGHT_POSE)
            else:
                fv.extend([0.0] * 66)

            data.append(fv)
            labels.append(dir_)

# Save the processed data to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"Extracted features from {len(data)} images across {len(set(labels))} classes")
print("Data saved to data.pickle")

# Visualize class distribution using matplotlib
cnt = Counter(labels)
classes = list(cnt.keys())
counts = list(cnt.values())
plt.figure(figsize=(8, 4))
plt.bar(classes, counts)
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Samples per Class Distribution')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
