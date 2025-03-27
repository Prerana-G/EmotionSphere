# EmotionSphere 😃

EmotionSphere is a Facial Emotion Recognition (FER) project that detects human emotions from images and real-time video streams. It utilizes a **Convolutional Neural Network (CNN)** trained on facial expression datasets to classify emotions such as happiness, sadness, anger, and surprise.

## Features ✨
- **Facial Emotion Recognition**: Detects emotions from images, uploaded files, or real-time webcam feed.
- **Deep Learning Model**: Trained using **CNN (Convolutional Neural Network)** with TensorFlow/Keras.
- **Multiple Output Modes**:
  - **Tkinter-based GUI** 🖼️: Upload an image and detect emotions in a pop-up window.
  - **Flask-based Web App** 🌍: Run a web server for emotion recognition.
  - **Real-time Detection** 🎥: Uses OpenCV to analyze facial expressions via webcam.

## Technologies Used 🛠️
- **Python**
- **TensorFlow/Keras** (Deep Learning Model)
- **OpenCV** (Image Processing & Face Detection)
- **Tkinter** (GUI Application)
- **Flask** (Web Application for Testing)
- **Numpy & Pandas** (Data Handling)

## Test Images 🖼️

The images/ folder contains some test images that were downloaded for experimentation and model testing.

## Uploaded Images Storage 📂

The uploads/ folder (included in .gitignore) stores images that were uploaded and detected at least once in the web-based application.

---
## Installation
```bash
pip install -r requirements.txt
```

## Running the Project 🚀
1. **Load Data**: Prepares the dataset and processes images.
   ```bash
   python src/load_data.py
   ```
2. **Train Model**: Trains the CNN model for facial emotion recognition.
   ```bash
   python src/model.py
   python src/train.py
   ```
3. **Evaluate Model**: Tests the model on validation data.
   ```bash
   python src/evaluate.py
   ```
4. **Run Detection**:
   - **Tkinter GUI** 🖼️: Upload an image for detection.
     ```bash
     python src/dycheck.py
     ```
   - **Web App (Flask-based)** 🌍:
     ```bash
     python src/app.py
     ```
     - Open your browser and go to http://127.0.0.1:5000/ to use the web-based emotion detection system.
   - **Real-time Webcam Recognition** 🎥:
     ```bash
     python src/real_time.py
     ```

## How It Works 🔍
1. **Preprocessing**: Images are loaded, resized, and normalized.
2. **Model Training**: A **CNN** is trained on facial expression data.
3. **Prediction**: The trained model classifies the emotions of detected faces.
4. **Deployment Options**: Users can detect emotions via GUI, Web App, or real-time video.

---
I hope you find this project useful! Feel free to use, modify, or improve it. I'm continuously learning and will work on more exciting projects in the future.