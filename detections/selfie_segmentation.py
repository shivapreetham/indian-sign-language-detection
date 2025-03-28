# selfie_segmentation.py
import cv2
import mediapipe as mp
import numpy as np

mp_selfie_segmentation = mp.solutions.selfie_segmentation

cap = cv2.VideoCapture(0)
with mp_selfie_segmentation.SelfieSegmentation(model_selection=1) as segmenter:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmenter.process(image_rgb)
        mask = results.segmentation_mask
        # Create a condition mask where pixels with probability > 0.5 are considered foreground.
        condition = np.stack((mask,)*3, axis=-1) > 0.5
        # Define a background image (here: blue background)
        bg_image = np.zeros(frame.shape, dtype=np.uint8)
        bg_image[:] = (255, 0, 0)
        output_frame = np.where(condition, frame, bg_image)
        cv2.imshow('Selfie Segmentation', output_frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
