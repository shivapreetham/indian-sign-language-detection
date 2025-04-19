# import os
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report
# import seaborn as sns

# # Load model and configuration
# with open('model.p', 'rb') as f:
#     model_dict = pickle.load(f)
#     model = model_dict['model']
#     scaler = model_dict['scaler']

# # Test data directory
# TEST_DIR = './test_data'  # Create this folder and put some test images
# WEIGHT_HAND = 1.0
# WEIGHT_FACE = 0.1
# WEIGHT_POSE = 0.3
# FACE_LANDMARKS_TO_USE = [1, 4, 10]

# # Label mapping (make sure this matches your training labels)
# labels_dict = {
#     0: 'ok/correct', 1: 'good', 2: 'two', 3: 'engineer',
#     4: 'Nice to meet you', 5: 'alright', 6: 'God', 7: 'Walk',
#     8: 'sorry', 9: 'call', 10: 'here', 11: 'light', 12: 'namaste/pray', 
#     13: 'read', 14:'loss/headache',15: 'bond/love', 16:'strength/powerful', 
#     17: 'may i go to washroom', 18:'flower', 19:'waiting for your action', 
#     20: 'drink/water', 21: 'food', 22: 'unknown'
# }

# # MediaPipe setup
# mp_holistic = mp.solutions.holistic

# def get_features(results):
#     features = []
    
#     # Process left hand landmarks
#     if results.left_hand_landmarks:
#         left = results.left_hand_landmarks
#         left_x = [lm.x for lm in left.landmark]
#         left_y = [lm.y for lm in left.landmark]
#         min_left_x, min_left_y = min(left_x), min(left_y)
#         for lm in left.landmark:
#             features.append((lm.x - min_left_x) * WEIGHT_HAND)
#             features.append((lm.y - min_left_y) * WEIGHT_HAND)
#     else:
#         features.extend([0.0] * (21 * 2))

#     # Process right hand landmarks
#     if results.right_hand_landmarks:
#         right = results.right_hand_landmarks
#         right_x = [lm.x for lm in right.landmark]
#         right_y = [lm.y for lm in right.landmark]
#         min_right_x, min_right_y = min(right_x), min(right_y)
#         for lm in right.landmark:
#             features.append((lm.x - min_right_x) * WEIGHT_HAND)
#             features.append((lm.y - min_right_y) * WEIGHT_HAND)
#     else:
#         features.extend([0.0] * (21 * 2))
        
#     # Process face landmarks
#     if results.face_landmarks:
#         face = results.face_landmarks
#         selected_face_x = [face.landmark[i].x for i in FACE_LANDMARKS_TO_USE]
#         selected_face_y = [face.landmark[i].y for i in FACE_LANDMARKS_TO_USE]
#         min_face_x, min_face_y = min(selected_face_x), min(selected_face_y)
#         for i in FACE_LANDMARKS_TO_USE:
#             lm = face.landmark[i]
#             features.append((lm.x - min_face_x) * WEIGHT_FACE)
#             features.append((lm.y - min_face_y) * WEIGHT_FACE)
#     else:
#         features.extend([0.0] * (len(FACE_LANDMARKS_TO_USE) * 2))
        
#     # Process pose landmarks
#     if results.pose_landmarks:
#         pose = results.pose_landmarks
#         pose_x = [lm.x for lm in pose.landmark]
#         pose_y = [lm.y for lm in pose.landmark]
#         min_pose_x, min_pose_y = min(pose_x), min(pose_y)
#         for lm in pose.landmark:
#             features.append((lm.x - min_pose_x) * WEIGHT_POSE)
#             features.append((lm.y - min_pose_y) * WEIGHT_POSE)
#     else:
#         features.extend([0.0] * (33 * 2))

#     return features

# def test_model_on_images():
#     true_labels = []
#     predicted_labels = []
#     confidences = []
    
#     with mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.6) as holistic:
#         # Process each class folder
#         for class_folder in os.listdir(TEST_DIR):
#             class_path = os.path.join(TEST_DIR, class_folder)
#             if not os.path.isdir(class_path):
#                 continue
                
#             print(f"Testing class {class_folder}...")
#             class_label = int(class_folder)
            
#             # Process each image in the class folder
#             for img_file in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_file)
#                 if not os.