import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os


#  Paths 

model_path = "../model/emotion_cnn_epoch5.pth"
test_dir = "../images/dataset/test"
save_dir = "./"   # Saves inside evaluation folder

os.makedirs(save_dir, exist_ok=True)


#  Load Model Architecture (same as train_model.py)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7

model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

if not os.path.exists(model_path):
    raise FileNotFoundError(f" Model file not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print("Model loaded successfully!")


#  Load Test Dataset


transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f" Loaded test dataset with {len(test_dataset)} images "
      f"across {len(test_dataset.classes)} classes.")


#  Evaluate Model


all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


#  Compute Metrics


acc = accuracy_score(all_labels, all_preds) * 100

print(f"\n Overall Model Accuracy: {acc:.2f}%")
report = classification_report(all_labels, all_preds, target_names=test_dataset.classes)
print("\n Classification Report:\n")
print(report)

# Save classification report
with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
    f.write(f"Overall Accuracy: {acc:.2f}%\n\n")
    f.write(report)

print(" Saved: classification_report.txt")


#  Confusion Matrix 


cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=test_dataset.classes,
            yticklabels=test_dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Emotion Classification")

plt.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=300)
plt.close()

print("Saved: confusion_matrix.png")


# Save Raw Predictions for Dashboard


df = pd.DataFrame({
    "true": all_labels,
    "predicted": all_preds
})
df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

print(" Saved: predictions.csv")


#print Class Distribution 


pred_counts = Counter(all_preds)
true_counts = Counter(all_labels)

print("\n Prediction distribution:")
for cls_idx, count in pred_counts.items():
    print(f"{test_dataset.classes[cls_idx]}: {count}")

print("\n True label distribution:")
for cls_idx, count in true_counts.items():
    print(f"{test_dataset.classes[cls_idx]}: {count}")

print("\n Evaluation complete!")
