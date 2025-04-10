import os
import cv2
import time  # Import time to allow for the sleep delay

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 36
dataset_size = 100

cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_folder = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    else:
        existing_images = len([name for name in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, name))])
        if existing_images >= dataset_size:
            print(f"Class {j} already has {existing_images} images. Skipping...")
            continue 
    print('Collecting data for class {}'.format(j))

    # Ready loop: display a message before capture begins
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Delay 2 seconds after pressing q
    time.sleep(2)

    print(f'Start capturing images for class {j}')
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('frame', frame)
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        cv2.imwrite(os.path.join(class_folder, '{}.jpg'.format(counter)), frame)
        counter += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print('Done!')
