# Ishara: Indian Sign-Language Recognition

A comprehensive application that captures, augments, and recognizes Indian Sign Language (ISL) gestures in real-time using MediaPipe, OpenCV, and a machine learning pipeline. It also provides a GUI for voice-to-sign conversion and context-aware interpretation via the Gemini API.

## Technology Stack Overview

### 🧠 Machine Learning & AI

* **Random Forest Classifier**: Multi-class gesture classification, robust to overfitting.
* **GridSearchCV**: Hyperparameter tuning across parameter grid for optimized performance.
* **Scikit-learn**: Modeling, training, evaluation utilities.

### 🖐 Computer Vision

* **MediaPipe (Hands, Face, Pose)**: Real-time landmark extraction.
* **OpenCV**: Video frame capture and preprocessing.
* **Albumentations**: Data augmentation—brightness/contrast, blur, flip, geometric transforms.

### 🗣 Voice Interaction

* **SpeechRecognition**: Converts voice commands to text.
* **pyttsx3 / gTTS**: Text-to-speech feedback for visually impaired users.

### 📄 Document Processing

* **PyMuPDF / PDFPlumber**: PDF parsing and extraction.
* **Custom Summarizer (NLP)**: Concise document summarization.

### 💻 Backend / Development Tools

* **Python**: Core language.
* **NumPy / Pandas**: Data manipulation.
* **Matplotlib**: Visualization (feature importance, accuracy plots).

### 🧪 Testing & Evaluation

* **5-Fold Cross-Validation**: Model reliability across splits.
* **Classification Report & Confusion Matrix**: Detailed performance metrics.

### 🖼 System Requirements

* Webcam, microphone, and basic laptop/PC for offline use.

## System Architecture

```text
User Interaction
├─ Gesture Input → Sign Detection Module → Random Forest Model → Text Output
└─ Voice Command → Voice Interface Module → PDF Reader & Summarizer → TTS Output
```

* **Sign Detection Module**: Processes hand/face/pose landmarks to predict ISL gestures.
* **Voice Interface Module**: Reads documents, summarizes, and speaks them aloud.

## Model Training & Evaluation

* **Dataset**: 35 classes, 400 samples each (14,000 total frames).
* **Preprocessing**: Landmark extraction via MediaPipe → normalized feature vectors (\~200 features).
* **Training**: Random Forest tuned with GridSearchCV over 24 parameter sets.
* **Results**:

  * Average cross-validation accuracy: 97.65%.
  * Test macro-F1: 0.98.
* **Visualizations**: Validation accuracy per fold, confusion matrix, top 20 feature importances.

## Gemini AI Integration

* **Use Case**: Convert detected gesture sequences into coherent sentences.
* **Flow**:

  1. Capture gestures → buffer until user presses 'G'.
  2. Send sequence to Gemini API.
  3. Receive and display context-aware sentence.
  4. Speak sentence via TTS.
* **Impact**: Enhances clarity and conversational feel for deaf users.

## Getting Started

### Prerequisites

```bash
pip install opencv-python mediapipe albumentations scikit-learn pyttsx3 SpeechRecognition pillow requests pymupdf pdfplumber
```

### Project Structure

```
├── data/                   # Raw captured images (35 classes)
├── augmented_data/         # Augmented dataset
├── letters/                # Letter images
├── ISL_Gifs/               # Sign GIFs for voice mode
├── data.pickle             # Extracted features
├── model.p                 # Trained model & scaler
├── image_gallery.html      # Letter gallery
├── collect_data.py         # Data capture
├── augment_dataset.py      # Augmentation
├── extract_features.py     # Feature extraction
├── train_model.py          # Training & evaluation
├── app.py                  # GUI application
└── README.md
```

### Usage

1. `python collect_data.py`
2. `python augment_dataset.py`
3. `python extract_features.py`
4. `python train_model.py`
5. `python app.py`

## Contributing

Pull requests welcome!

## License

MIT License
