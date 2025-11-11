from flask import Flask, render_template, Response, request, jsonify
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from collections import Counter
from facenet_pytorch import MTCNN
import numpy as np
import time
import os

#  MODEL SETUP 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7

#  Use ResNet18 (same as train.py)
model = resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

model_path = r"C:\Users\bssad\Pictures\Projects\Multimedia\model\emotion_cnn.pth"
if not os.path.exists(model_path):
    raise FileNotFoundError(f" Trained model not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(" ResNet18 model loaded successfully!")

#  MTCNN for robust face detection
mtcnn = MTCNN(keep_all=True, device=device)

# Label mappings (same order as training dataset)
idx_to_emotion = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'neutral', 5: 'sad', 6: 'surprise'
}

emotion_map = {
    'happy': 'Interested', 'surprise': 'Interested', 'neutral': 'Interested',
    'sad': 'Not Interested', 'angry': 'Not Interested',
    'fear': 'Not Interested', 'disgust': 'Not Interested'
}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

#  FLASK APP SETUP
app = Flask(__name__)
camera = None
face_track = {}
face_id_counter = 0
running = False

#  HELPER FUNCTIONS 
def get_face_id(face_bbox):
    global face_track, face_id_counter
    x1, y1, x2, y2 = face_bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    for fid, data in face_track.items():
        fx1, fy1, fx2, fy2 = data['bbox']
        fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
        dist = np.sqrt((cx - fcx) ** 2 + (cy - fcy) ** 2)
        if dist < max(x2 - x1, y2 - y1):
            return fid
    fid = face_id_counter
    face_id_counter += 1
    return fid

#  VIDEO GENERATOR
def gen_frames():
    global face_track
    while camera is not None:
        success, frame = camera.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces using MTCNN
        boxes, _ = mtcnn.detect(img_rgb)

        face_imgs, face_bboxes = [], []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil)
                face_imgs.append(face_tensor)
                face_bboxes.append((x1, y1, x2, y2))

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

                color = (0, 255, 0) if interest == 'Interested' else (0, 0, 255)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{emotion} | {interest}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# FLASK ROUTES 
@app.route('/')
def index():
    return render_template('index.html')

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
    return jsonify({"status": "started"})

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
    interested_pct = final_counts.count("Interested") / total * 100 if total > 0 else 0
    not_interested_pct = final_counts.count("Not Interested") / total * 100 if total > 0 else 0

    face_track.clear()
    face_id_counter = 0

    return jsonify({
        "Interested": f"{interested_pct:.2f}%",
        "Not Interested": f"{not_interested_pct:.2f}%"
    })

#  RUN APP 
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
