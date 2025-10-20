# # ===============================
# # app.py
# # ===============================
# import streamlit as st
# import cv2
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# from collections import Counter
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings


# # =========================
# # CNN Model Definition
# # =========================
# class EmotionCNN(nn.Module):
#     def __init__(self, num_classes=7):
#         super(EmotionCNN, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128*16*16, 256),
#             nn.ReLU(),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x

# # =========================
# # Setup
# # =========================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes = 7
# model = EmotionCNN(num_classes=num_classes).to(device)
# model.load_state_dict(torch.load("model/emotion_cnn.pth", map_location=device))
# model.eval()

# # Dataset mapping (replace with your dataset mapping)
# dataset_class_to_idx = {
#     'angry':0,'disgust':1,'fear':2,'happy':3,'neutral':4,'sad':5,'surprise':6
# }
# idx_to_emotion = {v:k for k,v in dataset_class_to_idx.items()}

# # Emotion → Interested/Not Interested
# emotion_map = {
#     'happy': 'Interested',
#     'surprise': 'Interested',
#     'neutral': 'Interested',
#     'sad': 'Not Interested',
#     'angry': 'Not Interested',
#     'fear': 'Not Interested',
#     'disgust': 'Not Interested'
# }
# idx_to_interest = {idx: emotion_map[idx_to_emotion[idx]] for idx in idx_to_emotion.keys()}

# # Haar cascade
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Transform
# transform = transforms.Compose([
#     transforms.Resize((128,128)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
# ])

# # =========================
# # Streamlit Interface
# # =========================
# st.title("😊 Live Emotion / Interest Detection")
# start = st.button("Start Webcam")
# stop = st.button("Stop Webcam")
# stframe = st.empty()

# if "running" not in st.session_state:
#     st.session_state.running = False
# if "face_track" not in st.session_state:
#     st.session_state.face_track = {}

# # =========================
# # Start / Stop Webcam
# # =========================
# if start:
#     st.session_state.running = True
# if stop:
#     st.session_state.running = False

# # cap = cv2.VideoCapture(0)
# # face_id = 0

# # while st.session_state.running:
# #     ret, frame = cap.read()
# #     if not ret:
# #         st.warning("Failed to capture frame")
# #         break

# #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# #     for (x, y, w, h) in faces:
# #         face_img = frame[y:y+h, x:x+w]
# #         face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
# #         face_tensor = transform(face_pil).unsqueeze(0).to(device)

# #         # Predict
# #         with torch.no_grad():
# #             output = model(face_tensor)
# #             _, pred = output.max(1)
# #             pred = pred.item()
# #             emotion_label = idx_to_emotion[pred]
# #             interest_label = idx_to_interest[pred]

# #         # Track face
# #         st.session_state.face_track[face_id] = interest_label
# #         face_id += 1

# #         # Draw rectangle + labels
# #         color = (0,255,0) if interest_label=="Interested" else (0,0,255)
# #         cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
# #         cv2.putText(frame, f"{emotion_label} | {interest_label}", (x, y-10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# #     # Show frame
# #     stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# # cap.release()
# # cv2.destroyAllWindows()

# class EmotionTransformer(VideoTransformerBase):
#     face_id = 0

#     def transform(self, frame):
#         img = frame.to_ndarray(format="bgr24")
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#         for (x, y, w, h) in faces:
#             face_img = img[y:y+h, x:x+w]
#             face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
#             face_tensor = transform(face_pil).unsqueeze(0).to(device)

#             with torch.no_grad():
#                 output = model(face_tensor)
#                 _, pred = output.max(1)
#                 pred = pred.item()
#                 emotion_label = idx_to_emotion[pred]
#                 interest_label = idx_to_interest[pred]

#             st.session_state.face_track[self.face_id] = interest_label
#             self.face_id += 1

#             color = (0,255,0) if interest_label=="Interested" else (0,0,255)
#             cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
#             cv2.putText(img, f"{emotion_label} | {interest_label}", (x, y-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

#         return img




# # =========================
# # Final Percentages
# # =========================
# if st.session_state.face_track:
#     counts = Counter(st.session_state.face_track.values())
#     total_people = sum(counts.values())
#     interested_pct = counts.get('Interested',0) / total_people * 100
#     not_interested_pct = counts.get('Not Interested',0) / total_people * 100
#     st.success(f"📊 Overall Summary:")
#     st.info(f"Interested: {interested_pct:.2f}%")
#     st.warning(f"Not Interested: {not_interested_pct:.2f}%")
#     st.session_state.face_track = {}
# ===============================
# app.py
# ===============================
import streamlit as st
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings

# =========================
# CNN Model Definition
# =========================
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
            nn.MaxPool2d(2),
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

# =========================
# Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
model = EmotionCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("model/emotion_cnn.pth", map_location=device))
model.eval()

# Dataset mapping
dataset_class_to_idx = {
    'angry':0,'disgust':1,'fear':2,'happy':3,'neutral':4,'sad':5,'surprise':6
}
idx_to_emotion = {v:k for k,v in dataset_class_to_idx.items()}

# Emotion → Interested/Not Interested
emotion_map = {
    'happy': 'Interested',
    'surprise': 'Interested',
    'neutral': 'Interested',
    'sad': 'Not Interested',
    'angry': 'Not Interested',
    'fear': 'Not Interested',
    'disgust': 'Not Interested'
}
idx_to_interest = {idx: emotion_map[idx_to_emotion[idx]] for idx in idx_to_emotion.keys()}

# Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# =========================
# Streamlit Interface
# =========================
st.title("😊 Live Emotion / Interest Detection")

if "face_track" not in st.session_state:
    st.session_state.face_track = {}

# Reset Summary Button
if st.button("Reset Summary"):
    st.session_state.face_track = {}
    st.success("✅ Summary reset!")

# =========================
# WebRTC Video Transformer
# =========================
class EmotionTransformer(VideoTransformerBase):
    face_id = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face_img = img[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(face_tensor)
                _, pred = output.max(1)
                pred = pred.item()
                emotion_label = idx_to_emotion[pred]
                interest_label = idx_to_interest[pred]

            st.session_state.face_track[self.face_id] = interest_label
            self.face_id += 1

            color = (0,255,0) if interest_label=="Interested" else (0,0,255)
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, f"{emotion_label} | {interest_label}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return img

# =========================
# Run WebRTC
# =========================
webrtc_streamer(
    key="emotion-stream",
    video_transformer_factory=EmotionTransformer,
    client_settings=ClientSettings(
        media_stream_constraints={"video": True, "audio": False}
    )
)

# =========================
# Display Summary
# =========================
if st.session_state.face_track:
    counts = Counter(st.session_state.face_track.values())
    total_people = sum(counts.values())
    interested_pct = counts.get('Interested',0) / total_people * 100
    not_interested_pct = counts.get('Not Interested',0) / total_people * 100

    st.success(f"📊 Overall Summary:")
    st.info(f"Interested: {interested_pct:.2f}%")
    st.warning(f"Not Interested: {not_interested_pct:.2f}%")
