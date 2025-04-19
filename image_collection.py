import os
import cv2
import time

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 23
dataset_size = 100
batch_size = 10  # Number of images per batch

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_folder = os.path.join(DATA_DIR, str(j))
    
    # If folder doesn't exist, create it and set counter to 0.
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
        counter = 0
    else:
        # Count the existing images
        existing_images = len([name for name in os.listdir(class_folder)
                               if os.path.isfile(os.path.join(class_folder, name))])
        if existing_images >= dataset_size:
            print(f"Class {j} already has {existing_images} images. Skipping...")
            continue
        else:
            counter = existing_images  # Start from the number of images already captured
    
    print(f'Collecting data for class {j} (starting at image {counter})')
    
    # Loop until 100 images are captured for the current class
    while counter < dataset_size:
        # Ready loop: display a message and wait for 'Q' to start a batch capture
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.putText(frame, 'Ready? Press "Q" to capture 10 images!',
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                        (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)
            key = cv2.waitKey(25)
            if key == ord('q'):
                break

        if not ret:
            break

        # Delay 2 seconds after pressing Q to allow for repositioning
        time.sleep(2)
        print(f'Start capturing a batch. Total images so far: {counter}')

        # Capture a batch of images (10 images per batch)
        batch_counter = 0
        while batch_counter < batch_size and counter < dataset_size:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow('frame', frame)
            key = cv2.waitKey(25)
            img_filename = os.path.join(class_folder, f'{counter}.jpg')
            cv2.imwrite(img_filename, frame)
            counter += 1
            batch_counter += 1

        print(f'Captured a batch of {batch_counter} images. Total images captured: {counter}')

cap.release()
cv2.destroyAllWindows()
print('Done!')