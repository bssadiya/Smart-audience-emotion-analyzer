# Smart Audience Emotion Analyzer

##  Overview

Smart Audience Emotion Analyzer is an AI-powered system that detects human emotions in real-time using facial expressions and converts them into meaningful engagement insights such as **“Interested”** and **“Not Interested.”**

This project combines **Computer Vision**, **Deep Learning**, and **Web Deployment** to create an end-to-end intelligent system that can be used in classrooms, presentations, and business environments.

---

##  Features

*  Real-time face detection using MTCNN
*  Emotion recognition using ResNet18 (Deep Learning)
*  Multi-face detection and batch processing
*  Face tracking across frames
*  Emotion → Engagement mapping (Interested / Not Interested)
*  Live video streaming using Flask
*  Final audience analytics (percentage output)

---

##  Tech Stack

###  Computer Vision

* OpenCV → video capture & image processing

###  Deep Learning

* PyTorch → model development
* ResNet18 → emotion classification

###  Face Detection

* MTCNN → robust multi-face detection

###  Backend

* Flask → web server & streaming

###  Data Handling

* NumPy, Collections → analytics & tracking

---

##  Project Architecture

```
Camera Input
     ↓
Frame Capture (OpenCV)
     ↓
Face Detection (MTCNN)
     ↓
Preprocessing (Resize, Normalize)
     ↓
Emotion Prediction (ResNet18)
     ↓
Emotion Mapping (Interested / Not Interested)
     ↓
Face Tracking + Majority Voting
     ↓
Final Analytics (Percentage Output)
```

---

##  Installation

### 1. Clone Repository

```
git clone https://github.com/your-username/smart-audience-emotion-analyzer.git
cd smart-audience-emotion-analyzer
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Add Trained Model

Place your trained model file:

```
emotion_cnn.pth
```

inside:

```
/model/
```

---

##  Run the Project

```
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

---

##  How It Works

1. Webcam captures live video
2. MTCNN detects faces in each frame
3. Faces are preprocessed and passed to ResNet18
4. Model predicts emotions (happy, sad, etc.)
5. Emotions are mapped to engagement levels
6. Faces are tracked across frames
7. Final output shows audience engagement percentage

---

##  Emotion Mapping Logic

| Emotion  | Engagement     |
| -------- | -------------- |
| Happy    | Interested     |
| Surprise | Interested     |
| Neutral  | Interested     |
| Sad      | Not Interested |
| Angry    | Not Interested |
| Fear     | Not Interested |
| Disgust  | Not Interested |

---

##  Model Details

* Model: ResNet18
* Framework: PyTorch
* Input Size: 128 × 128
* Output Classes: 7 emotions
* Loss Function: CrossEntropyLoss
* Optimizer: Adam

---

##  Key Innovations

*  Uses **ResNet (skip connections)** for better accuracy
*  Batch processing of multiple faces for speed
*  Majority voting per face for stable predictions
*  Converts raw emotions into **business insights**

---



---

##  Future Improvements

*  Dashboard for analytics visualization
*  Voice-based emotion detection
*  Deploy as web/mobile application
*  Use advanced models (ResNet50, EfficientNet)
*  Temporal emotion tracking (video-based trends)

---

##  Use Cases

*  Classroom engagement monitoring
*  Business presentations
*  Customer reaction analysis
*  Mental health monitoring

---


