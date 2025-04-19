import os
import cv2
import numpy as np
from albumentations import (
    Compose, RandomBrightnessContrast, GaussianBlur, 
    HorizontalFlip, ShiftScaleRotate, CLAHE
)

# Augmentation settings
AUGMENTATIONS_PER_IMAGE = 3  # Number of augmented versions per original image
OUTPUT_DIR = './augmented_data'

# Create augmentation pipeline
transform = Compose([
    RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    GaussianBlur(blur_limit=3, p=0.3),
    CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
    ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.7),
    HorizontalFlip(p=0.3),
])

def augment_dataset(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if not os.path.isdir(class_path):
            continue
            
        # Create output class folder
        class_output_path = os.path.join(output_dir, class_folder)
        if not os.path.exists(class_output_path):
            os.makedirs(class_output_path)
            
        print(f"Processing class {class_folder}...")
        
        # Copy original images
        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            if not os.path.isfile(img_path):
                continue
                
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # Save original
            orig_output_path = os.path.join(class_output_path, img_file)
            cv2.imwrite(orig_output_path, img)
            
            # Create augmented versions
            for i in range(AUGMENTATIONS_PER_IMAGE):
                augmented = transform(image=img)['image']
                aug_filename = f"{os.path.splitext(img_file)[0]}_aug{i+1}.jpg"
                aug_output_path = os.path.join(class_output_path, aug_filename)
                cv2.imwrite(aug_output_path, augmented)
                
    print("Augmentation complete!")

if __name__ == "__main__":
    augment_dataset('./data', OUTPUT_DIR)
    print(f"Original and augmented data saved to {OUTPUT_DIR}")