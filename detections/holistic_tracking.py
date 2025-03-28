import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Custom drawing specs for different components:
# For face landmarks, use a smaller circle radius and thinner lines.
face_landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
face_connection_spec = mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=1)

# For hand landmarks, you might want a slightly different style.
hand_landmark_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
hand_connection_spec = mp_drawing.DrawingSpec(color=(200, 0, 0), thickness=2)

# For pose landmarks, you can also define a custom spec.
pose_landmark_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
pose_connection_spec = mp_drawing.DrawingSpec(color=(0, 0, 200), thickness=2)

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame for natural selfie view
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = holistic.process(image_rgb)
        image_rgb.flags.writeable = True
        output_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        # Draw face landmarks with custom drawing specs
        if results.face_landmarks:
            mp_drawing.draw_landmarks(
                output_frame, 
                results.face_landmarks, 
                mp_holistic.FACEMESH_TESSELATION,
                landmark_drawing_spec=face_landmark_spec,
                connection_drawing_spec=face_connection_spec)
        
        # Draw left hand landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(
                output_frame, 
                results.left_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_spec,
                connection_drawing_spec=hand_connection_spec)
        
        # Draw right hand landmarks
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(
                output_frame, 
                results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=hand_landmark_spec,
                connection_drawing_spec=hand_connection_spec)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                output_frame, 
                results.pose_landmarks, 
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=pose_landmark_spec,
                connection_drawing_spec=pose_connection_spec)
        
        cv2.imshow('Holistic Tracking', output_frame)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit on pressing 'Esc'
            break

cap.release()
cv2.destroyAllWindows()
