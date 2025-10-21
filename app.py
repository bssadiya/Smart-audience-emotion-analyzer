from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from collections import Counter
import numpy as np
import time
import os

#  Model Definition
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*16*16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

#  Setup 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
model_path = "model/emotion_cnn.pth"  # your trained model path
model = EmotionCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

idx_to_emotion = {0:'angry',1:'disgust',2:'fear',3:'happy',4:'neutral',5:'sad',6:'surprise'}
emotion_map = {
    'happy': 'Interested','surprise': 'Interested','neutral': 'Interested',
    'sad': 'Not Interested','angry': 'Not Interested','fear': 'Not Interested','disgust': 'Not Interested'
}

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

#  Flask App
app = Flask(__name__)

camera = None
face_track = {}
face_id_counter = 0
running = False

# Helper Functions 
def get_face_id(face_bbox):
    global face_track, face_id_counter
    x, y, w, h = face_bbox
    cx, cy = x + w//2, y + h//2
    for fid, data in face_track.items():
        fx, fy, fw, fh = data['bbox']
        fcx, fcy = fx + fw//2, fy + fh//2
        dist = np.sqrt((cx-fcx)**2 + (cy-fcy)**2)
        if dist < max(w,h):
            return fid
    fid = face_id_counter
    face_id_counter += 1
    return fid

# Video Generator 
def gen_frames():
    global face_track
    while camera is not None:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))

        face_imgs, face_bboxes = [], []
        for (x, y, w, h) in faces:
            min_dim = min(w,h)
            face_img = frame[y:y+min_dim, x:x+min_dim]
            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_img_rgb)
            face_tensor = transform(face_pil)
            face_imgs.append(face_tensor)
            face_bboxes.append((x, y, min_dim, min_dim))

        if running and face_imgs:
            batch = torch.stack(face_imgs).to(device)
            with torch.no_grad():
                outputs = model(batch)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()

            for i, bbox in enumerate(face_bboxes):
                emotion = idx_to_emotion[preds[i]]
                interest = emotion_map[emotion]

                fid = get_face_id(bbox)
                if fid not in face_track:
                    face_track[fid] = {'bbox': bbox, 'preds': []}
                face_track[fid]['bbox'] = bbox
                face_track[fid]['preds'].append(interest)

                color = (0,255,0) if interest=="Interested" else (0,0,255)
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{emotion} | {interest}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

#  Flask Routes 
@app.route('/')
def index():
    return render_template('index.html')  # create index.html in templates folder

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start', methods=['POST'])
def start_stream():
    global running, camera, face_track, face_id_counter
    if not running:
        camera = cv2.VideoCapture(0)
        time.sleep(0.5)
        for _ in range(5):
            ret, _ = camera.read()
        running = True
        face_track = {}
        face_id_counter = 0
    return jsonify({"status":"started"})

@app.route('/stop', methods=['POST'])
def stop_stream():
    global running, camera, face_track, face_id_counter
    running = False
    if camera is not None:
        camera.release()
        camera = None
        time.sleep(0.5)

    final_counts = []
    for f_id, data in face_track.items():
        count = Counter(data['preds'])
        majority = count.most_common(1)[0][0]
        final_counts.append(majority)

    total = len(final_counts)
    interested_pct = final_counts.count("Interested") / total * 100 if total>0 else 0
    not_interested_pct = final_counts.count("Not Interested") / total * 100 if total>0 else 0

    face_track.clear()
    face_id_counter = 0

    return jsonify({
        "Interested": f"{interested_pct:.2f}%",
        "Not Interested": f"{not_interested_pct:.2f}%"
    })

#  Run App 
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
