
**Project Title: Real-Time Emotion and Interest Detection System**
**Technology Stack: Flask, PyTorch, OpenCV, MTCNN, ResNet18**



**Overview:**
This project is a real-time emotion recognition and audience interest detection system developed using Flask and PyTorch.
It detects multiple faces from a live webcam feed, classifies each person’s emotion using a fine-tuned ResNet18 model, and then determines whether the person is Interested or Not Interested.
The system provides instant feedback through the browser interface and also gives a summary of the audience’s engagement level.



**Key Features:**
• Detects multiple faces in real time using MTCNN (deep learning based face detection).
• Classifies seven basic human emotions using a fine-tuned ResNet18 model.
• Maps emotions into two interest levels: Interested or Not Interested.
• Runs on both CPU and GPU automatically.
• Web interface built with Flask to stream the webcam feed and interact with the model.
• Start and Stop buttons to control the camera and generate a summary report.
• Saves a detailed evaluation report with model accuracy and per-class results.

---

**Emotion Classes and Mapping:**
The model predicts seven different emotions and maps them into interest levels as follows:

1. Angry → Not Interested
2. Disgust → Not Interested
3. Fear → Not Interested
4. Sad → Not Interested
5. Happy → Interested
6. Neutral → Interested
7. Surprise → Interested

---

**Model Details:**
• Base Model: ResNet18 pre-trained on ImageNet
• Input Image Size: 128 x 128
• Optimizer: Adam with learning rate 0.0001
• Loss Function: Cross Entropy Loss
• Training Data: Balanced dataset with 2000 images per class
• Data Augmentation: Random rotation, horizontal flip, color jitter, and affine transformations
• Output: Emotion label and Interest category for each detected face

---

**Folder Structure:**
Multimedia
├── app.py (Flask web app)
├── train.py (Model training script)
├── evaluate.py (Model evaluation script)
├── model
│     ├── emotion_cnn.pth (Trained model file)
│     └── model_evaluation_report.txt (Evaluation results)
├── templates
│     └── index.html (Webpage for live stream)
└── images
└── dataset
├── train_reduced (Balanced training dataset)
└── test (Testing dataset)

---

**Steps to Run the Project:**

1. Install Python and create a virtual environment.
2. Install all required libraries such as torch, torchvision, facenet-pytorch, opencv-python, scikit-learn, flask, Pillow, and numpy.
3. Train the model by running train.py. This will save the trained model file in the “model” folder.
4. Evaluate the model by running evaluate.py. It will print the accuracy and create a text report.
5. Start the Flask web app by running app.py.
6. Open the link shown in the terminal (usually [http://127.0.0.1:5000](http://127.0.0.1:5000)).
7. Click “Start” to activate the webcam and begin live emotion detection.
8. Click “Stop” to stop the camera and view the final Interested vs Not Interested percentages.

**Example Output:**
During the live demo, the application draws bounding boxes around detected faces and displays each person’s emotion label and interest level.
Green boxes indicate Interested faces, and red boxes indicate Not Interested faces.

After stopping the stream, the summary looks like:
Interested = 72.40%
Not Interested = 27.60%

The evaluation script also reports:
Overall Model Accuracy: 81.4%



**Results and Performance:**
• The model achieved around 80 to 85 percent accuracy on the test dataset after fine-tuning.
• The system performs well in real-time and can track multiple faces simultaneously.
• It is robust to lighting variations and facial orientations due to data augmentation and MTCNN detection.



**Future Enhancements:**
• Integrate a Vision Transformer (ViT) model for improved accuracy.
• Deploy the Flask application on a cloud server for remote access.
• Add confidence and FPS (frames per second) display on the live video feed.
• Use larger emotion datasets such as FER2013 or AffectNet to enhance generalization.


Would you like me to make a short “About Project” paragraph (3–4 lines) that you can paste on your project report cover page or inside your PowerPoint presentation?
