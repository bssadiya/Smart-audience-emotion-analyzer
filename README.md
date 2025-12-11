Here is a **clean, aesthetic, modern GitHub-style README** — no emojis, but with beautiful spacing, visual hierarchy, and professional formatting.
You can paste this directly into your repository.

---

# Real-Time Emotion & Interest Detection System

*A deep learning system that measures audience engagement in real time.*

---

## 1. Overview

This project began with a simple question:
**How can we understand audience engagement without asking them to speak?**

To answer this, the system uses a combination of **Flask**, **PyTorch**, **OpenCV**, **MTCNN**, and a fine-tuned **ResNet18** model to:

* Detect multiple faces from a webcam stream
* Classify seven human emotions
* Convert those emotions into two categories: *Interested* or *Not Interested*
* Display results live through a browser interface
* Provide a final engagement summary at the end of the session

This creates a seamless way to measure attention and involvement during classes, presentations, training sessions, and live events.

---

## 2. Dataset

The model is trained on a balanced dataset with:

* Seven emotion classes
* 2000 images per class
* Augmentation techniques such as rotation, flip, jitter, and affine transforms

This ensures strong performance in varied lighting and real-time conditions.



## 3. Approach

**Face Detection**

* MTCNN is used to detect multiple faces accurately in each frame.

**Preprocessing**

* Detected faces are cropped, resized to 128×128, normalized, and transformed.

**Emotion Classification**

* A ResNet18 model fine-tuned on emotion data predicts one of seven classes.

**Interest Mapping**

* Happy, Neutral, Surprise → Interested
* Angry, Disgust, Fear, Sad → Not Interested

**Live Rendering**

* Flask delivers real-time bounding boxes and emotion/interest labels to the browser.

**Session Summary**

* After stopping the stream, the system calculates the percentage distribution of Interested vs Not Interested participants.

---

## 4. Results

* Model accuracy: **80–85%**
* Strong multi-face detection and tracking
* Consistent performance across different lighting and facial orientations
* Example summary output:

  * Interested: 72.40%
  * Not Interested: 27.60%

The evaluation report also provides class-wise accuracy and overall performance metrics.

---

## 5. Demo

```
![Demo](demo.gif)
```

(Replace the file after uploading your GIF.)

---

## 6. Project Structure

```
app.py                     # Flask application
train.py                   # Model training
evaluate.py                # Model evaluation
model/
    emotion_cnn.pth        # Trained model
    model_evaluation_report.txt
templates/
    index.html             # Web interface
dataset/
    train_reduced
    test
```

## 7. How to Run

**Step 1. Install dependencies**

```
pip install torch torchvision facenet-pytorch opencv-python scikit-learn flask Pillow numpy
```

**Step 2. Train the model**

```
python train.py
```

**Step 3. Evaluate the model**

```
python evaluate.py
```

**Step 4. Start the application**
python app.py

**Step 5. Open the browser**
```
http://127.0.0.1:5000
```

Click **Start** to begin the live detection, and **Stop** to view the engagement summary.



## 8. Conclusion

This system combines deep learning and computer vision to provide a meaningful and practical measure of audience engagement.
By interpreting emotions and mapping them into interest levels, it offers real-time insights that can support educators, presenters, and anyone who depends on audience feedback.


