import cv2
import mediapipe as mp

# Initialize MediaPipe solutions and drawing utilities
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define drawing specifications for visualization
face_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
hand_drawing_spec = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

# Open the default webcam
cap = cv2.VideoCapture(0)

# Initialize both Face Mesh and Hands simultaneously
with mp_face_mesh.FaceMesh(
    static_image_mode=False,      # Process as a video stream
    max_num_faces=1,              # Detect one face
    refine_landmarks=True,        # Refine landmarks for eyes and lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh, \
     mp_hands.Hands(
         static_image_mode=False,
         max_num_hands=2,         # Detect up to 2 hands
         min_detection_confidence=0.5,
         min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for a natural selfie-view
        frame = cv2.flip(frame, 1)

        # Convert the image to RGB as required by MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False  # Optimize performance

        # Process the image for face and hand landmarks
        face_results = face_mesh.process(image_rgb)
        hands_results = hands.process(image_rgb)

        image_rgb.flags.writeable = True
        # Convert back to BGR for OpenCV display
        output_frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw face landmarks if detected
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=face_drawing_spec
                )

        # Draw hand landmarks if detected
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=output_frame,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_drawing_spec,
                    connection_drawing_spec=hand_drawing_spec
                )

        # Display the output frame
        cv2.imshow('MediaPipe Face Mesh and Hands', output_frame)
        # Exit on pressing the 'Esc' key
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
