# Sign Language Assistant

## 🧠 Overview

The **Sign Language Assistant** is a Python-based application aimed at bridging the communication gap for individuals with hearing impairments. It provides two core functionalities:

* **Voice to Sign**: Converts spoken language to visual sign language using GIFs or alphabet signs.
* **Sign Detection**: Detects and interprets real-time sign gestures from webcam input using a trained ML model and MediaPipe.

> The app supports 35 classes focused on **Indian Sign Language (ISL)** using techniques from **Computer Vision**, **Machine Learning**, and **Natural Language Processing**.

---

## 🚀 Features

* **📸 Data Collection**: Captures 100 images/class for 35 signs using webcam (in 10-image batches).
* **🧪 Data Augmentation**: Enhances dataset using brightness/contrast, blur, flips (3x augmentations).
* **🧬 Feature Extraction**: Uses **MediaPipe Holistic** landmarks (weighted hands/face/pose) for model input.
* **🧠 Model Training**: Trains a `RandomForestClassifier` with hyperparameter tuning.
* **🗣️ Voice to Sign**: Google Speech Recognition to ISL GIFs or letters.
* **🤟 Sign Detection**: Real-time prediction with buffer-based voting and **Gemini API** interpretation.
* **🖥️ UI**: Built with **Tkinter**, interactive and easy to use.

---

## 🔧 Prerequisites

* **Python**: Version 3.8+
* **Webcam**: Required for real-time tasks
* **Gemini API**: Must be running at `http://localhost:8000/gemini` or update in code
* **Image/GIF Assets**:

  * `logo.png`
  * `letters/` folder with `a.jpg` ... `z.jpg`, `empty.jpg`
  * `ISL_Gifs/` with predefined phrases (e.g., `good morning.gif`)

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Directory Structure

```
sign-language-assistant/
├── data/                      # Collected raw sign images
├── augmented_data/           # Augmented images
├── letters/                  # Alphabet signs (JPGs)
├── ISL_Gifs/                 # Phrase signs (GIFs)
├── logo.png
├── data.pickle               # Extracted features
├── model.p                   # Trained model
├── confusion_matrix.png
├── feature_importances.png
├── image_gallery.html
├── *.py                      # Python scripts
└── requirements.txt
```

---

## 💻 Usage

### ▶ Launch App

```bash
python main.py
```

Provides 3 GUI options: `Voice To Sign`, `Sign Detection`, `Exit`

### 🗣 Voice to Sign

* Converts spoken phrases → ISL GIFs or letters
* Displays gallery if no phrase match

### ✋ Sign Detection

* Live gesture prediction via webcam
* Buffered voting (15 frames, 60% confidence)
* Press:

  * `G` – Interpret buffered sequence via **Gemini API**
  * `B` – Back
  * `Q` – Quit

---

## 🛠 Optional Scripts

### 📸 Data Collection

```bash
python data_collection.py
```

Captures 100 images per sign (10/batch)

### 🧪 Data Augmentation

```bash
python data_augmentation.py
```

Generates 3 augmented images per original

### 📊 Feature Extraction

```bash
python feature_extraction.py
```

Uses MediaPipe Holistic landmarks, saves as `data.pickle`

### 🧠 Model Training

```bash
python model_training.py
```

* Trains RandomForest with `GridSearchCV`
* Outputs `model.p`, plots

---

## 🧬 Technical Details

### 🖼️ Image Handling

* Stored in `./data/<class>` or `./augmented_data/<class>`
* JPG format

### ⚙️ Feature Extraction

* Weights: hand = 1.0, face = 0.1, pose = 0.3
* Normalized using min(x, y)

### 🧠 Classifier

* **RandomForestClassifier** + `StandardScaler`
* Parameters: `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`
* Outputs: `confusion_matrix.png`, `feature_importances.png`

### 🗣 Speech Module

* Uses `speech_recognition` with Google API
* Levenshtein threshold = 0.4 for matching

### 📹 Detection Module

* Real-time with OpenCV & MediaPipe
* 15-frame buffer, 60% voting threshold
* Gemini integration for phrase interpretation

---

## ⚠️ Limitations

* 🧏 Only 35 classes supported currently
* 🌐 Speech recognition needs internet
* 🧠 No dynamic sequence classification yet
* 📉 Real-time inference may lag on low-end systems

---

## 🔮 Future Scope

* Expand sign vocabulary & phrases
* Add offline speech recognition
* Optimize inference on low-resource devices
* Dynamic sign sequences and sentence formation
* Improved accessibility & GUI UX

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature-name`
3. Commit: `git commit -m 'Add feature'`
4. Push: `git push origin feature-name`
5. Open a Pull Request

---

## 📜 License

MIT License – see [LICENSE](./LICENSE)

---

## 🙏 Acknowledgments

* [MediaPipe](https://google.github.io/mediapipe/)
* [scikit-learn](https://scikit-learn.org/)
* [OpenCV](https://opencv.org/)
* [albumentations](https://albumentations.ai/)
* [Google Speech Recognition](https://pypi.org/project/SpeechRecognition/)
* [pyttsx3](https://pypi.org/project/pyttsx3/)
