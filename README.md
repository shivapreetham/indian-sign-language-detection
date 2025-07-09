Sign Language Assistant
Overview
The Sign Language Assistant is a Python-based application designed to facilitate communication for individuals with hearing impairments. It offers two primary functionalities:

Voice to Sign: Converts spoken words into visual sign language representations (GIFs or individual letter signs) using speech recognition.
Sign Detection: Detects and interprets sign language gestures in real-time from video input using a trained machine learning model and MediaPipe Holistic.

The application leverages computer vision, machine learning, and natural language processing to bridge communication gaps, supporting 35 distinct sign classes with a focus on Indian Sign Language (ISL) phrases and letters.
Features

Data Collection: Captures images for 35 sign classes using a webcam, with each class containing 100 images, organized in batches of 10.
Data Augmentation: Generates augmented versions of images to enhance dataset diversity using techniques like brightness/contrast adjustment, Gaussian blur, and horizontal flips.
Feature Extraction: Extracts weighted landmarks (hand, face, pose) using MediaPipe Holistic for model training.
Model Training: Trains a RandomForestClassifier with hyperparameter tuning to classify signs based on extracted features.
Voice to Sign: Uses Google Speech Recognition to convert spoken input into text, matching it to predefined ISL phrases or displaying individual letter signs.
Sign Detection: Real-time gesture recognition with a voting mechanism for stable predictions, featuring text-to-speech output and Gemini API integration for sequence interpretation.
User Interface: A Tkinter-based GUI for selecting modes and displaying results, with visual feedback for detected signs.

Prerequisites
To run the Sign Language Assistant, ensure you have the following installed:

Python: Version 3.8 or higher
Dependencies: Install required Python packages using:pip install opencv-python numpy albumentations mediapipe scikit-learn matplotlib pickle5 pyttsx3 speechrecognition pillow requests python-Levenshtein


Webcam: For data collection and real-time sign detection.
Gemini API: Access to a running Gemini API instance (e.g., at http://localhost:8000/gemini) for sign sequence interpretation. Replace with your API endpoint if hosted differently.
Image/GIF Assets: 
logo.png for the main menu.
letters/ directory containing .jpg files for individual alphabet signs (a-z) and empty.jpg.
ISL_Gifs/ directory containing .gif files for predefined phrases (e.g., good morning.gif).


Hardware: A system with sufficient processing power for real-time video processing and machine learning inference.

Setup Instructions

Clone the Repository:
git clone <repository-url>
cd sign-language-assistant


Install Dependencies:
pip install -r requirements.txt

Alternatively, install individual packages listed in the Prerequisites section.

Prepare Asset Directories:

Place logo.png in the project root.
Create a letters/ directory with .jpg files for each alphabet (e.g., a.jpg, b.jpg, ..., z.jpg) and empty.jpg.
Create an ISL_Gifs/ directory with .gif files for supported phrases (e.g., good morning.gif, hello.gif).


Set Up Gemini API:

Ensure the Gemini API is accessible at http://localhost:8000/gemini or update the endpoint in the signDetection function.
If using a remote service, replace with the appropriate URL (e.g., https://sign-language-3-5vax.onrender.com/gemini).


Directory Structure:Ensure the following structure:
sign-language-assistant/
├── data/                # Raw data collected from webcam
├── augmented_data/      # Augmented dataset
├── letters/             # Alphabet sign images (*.jpg)
├── ISL_Gifs/            # Phrase GIFs (*.gif)
├── logo.png             # Main menu logo
├── data.pickle          # Extracted features and labels
├── model.p              # Trained model and scaler
├── confusion_matrix.png # Generated confusion matrix plot
├── feature_importances.png # Generated feature importance plot
├── image_gallery.html   # Generated HTML for letter displays
├── *.py                 # Python scripts (provided files)
└── requirements.txt     # Dependency list



Usage

Run the Application:
python main.py

This launches the Tkinter GUI with three options: "Voice To Sign," "Sign Detection," and "Exit."

Voice to Sign:

Select "Voice To Sign" from the main menu.
Speak a phrase or word when prompted.
The system will:
Recognize the speech using Google Speech Recognition.
Match it to a predefined ISL phrase and display the corresponding GIF.
If no match is found, display individual letter signs in an HTML gallery.


Press "Back to Main Menu" to return.


Sign Detection:

Select "Sign Detection" from the main menu.
Perform sign language gestures in front of the webcam.
The system will:
Detect gestures using MediaPipe Holistic and the trained RandomForestClassifier.
Display detected signs with confidence scores.
Buffer predictions for stability (using a 15-frame buffer and 60% voting threshold).
Press 'G' to interpret a sequence of signs via the Gemini API, with text-to-speech output.
Press 'B' to return to the main menu or 'Q' to quit.




Data Collection (Optional):

Run the data collection script (data_collection.py) to capture images for training:python data_collection.py


Captures 100 images per class (35 classes) in batches of 10.
Press 'Q' to start capturing each batch.




Data Augmentation (Optional):

Run the augmentation script (data_augmentation.py) to generate augmented images:python data_augmentation.py


Creates 3 augmented versions per original image, stored in augmented_data/.




Feature Extraction and Training (Optional):

Run the feature extraction script (feature_extraction.py):python feature_extraction.py


Extracts landmarks using MediaPipe Holistic and saves to data.pickle.


Run the training script (model_training.py):python model_training.py


Trains a RandomForestClassifier with hyperparameter tuning and saves to model.p.





Technical Details

Data Collection:

Captures 100 images per class (35 classes) using OpenCV.
Images are stored in ./data/<class_id>/ as .jpg files.
Batch size: 10 images per capture session.


Data Augmentation:

Uses the albumentations library for transformations (e.g., RandomBrightnessContrast, GaussianBlur, HorizontalFlip).
Generates 3 augmented versions per image, stored in ./augmented_data/<class_id>/.


Feature Extraction:

Uses MediaPipe Holistic to extract landmarks (left hand, right hand, selected face landmarks, pose).
Applies weights: hand (1.0), face (0.1), pose (0.3).
Normalizes landmarks relative to their minimum x/y values.
Saves features and labels to data.pickle.


Model Training:

Uses scikit-learn’s RandomForestClassifier with GridSearchCV for hyperparameter tuning.
Parameters tuned: n_estimators, max_depth, min_samples_leaf, max_features.
Scales features using StandardScaler.
Saves trained model and scaler to model.p.
Generates visualization plots: confusion matrix and feature importances.


Voice to Sign:

Uses speech_recognition with Google Speech Recognition API.
Matches input text to a predefined list of ISL phrases using Levenshtein distance (threshold: 0.4).
Displays GIFs for matched phrases or individual letter signs in an HTML gallery.


Sign Detection:

Processes webcam video using OpenCV and MediaPipe Holistic.
Buffers 15 frames for stable predictions (60% voting threshold, 0.6 confidence threshold).
Displays landmarks (pose, face, colored hand fingers) on the video feed.
Integrates with Gemini API for sequence interpretation, with pyttsx3 for text-to-speech output.
Supports key commands: 'G' (interpret sequence), 'B' (back to menu), 'Q' (quit).



Limitations

Speech Recognition: Requires internet access for Google Speech Recognition and may fail in noisy environments.
Gemini API: Requires a running API instance; errors occur if the endpoint is unavailable.
Sign Detection: Limited to 35 predefined classes; may struggle with ambiguous or untrained gestures.
Asset Dependency: Missing GIFs or letter images will cause errors in the "Voice to Sign" module.
Performance: Real-time detection may lag on low-end hardware due to video processing and model inference.

Future Improvements

Expand the sign language vocabulary beyond 35 classes.
Implement offline speech recognition for better accessibility.
Optimize video processing for low-end devices using lightweight models.
Add support for dynamic sign sequences (e.g., sentence construction).
Enhance GUI with more interactive features and accessibility options.

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m 'Add feature').
Push to the branch (git push origin feature-name).
Open a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

MediaPipe: For robust landmark detection.
scikit-learn: For machine learning capabilities.
OpenCV: For video and image processing.
albumentations: For data augmentation.
Google Speech Recognition: For speech-to-text functionality.
pyttsx3: For text-to-speech output.
