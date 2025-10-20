# Smart-audience-emotion-analyzer
Real-time audience emotion tracking and engagement feedback

# Live Emotion / Interest Detection App

A web application to detect emotions from live webcam feed, classify users as **Interested** or **Not Interested**, and provide real-time summaries.

---

## Features

- Real-time face detection using OpenCV Haar cascades.
- Emotion recognition using a custom **CNN** trained on 7 emotion classes:  
  `angry, disgust, fear, happy, neutral, sad, surprise`.
- Maps emotions to **Interested / Not Interested** categories.
- Dynamic face tracking with per-face majority voting.
- Displays live video with bounding boxes and emotion/interest labels.
- Computes overall percentage of **Interested** and **Not Interested** users on stop.

---

## Demo

- Start camera → live emotion overlay.
- Stop camera → get final summary percentages.

---

## Requirements

- Python 3.10+
- Packages:

```bash
pip install flask opencv-python torch torchvision pillow numpy
