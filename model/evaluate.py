
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import os

#  Load Model Architecture (same as train.py)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7

# Recreate same ResNet18 structure
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Load trained weights
model_path = r"C:\Users\bssad\Pictures\Projects\Multimedia\model\emotion_cnn.pth"  # update if needed

if not os.path.exists(model_path):
    raise FileNotFoundError(f" Model file not found at {model_path}. Run train.py first.")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

print(" Model loaded successfully!")


#  Load Test Dataset


test_dir = r"C:\Users\bssad\Pictures\Projects\Multimedia\images\dataset\test"  # update path if needed

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f" Loaded test dataset with {len(test_dataset)} images across {len(test_dataset.classes)} classes.")


#  Evaluate Model


all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


# Compute Metrics


acc = accuracy_score(all_labels, all_preds) * 100
print(f"\n Overall Model Accuracy: {acc:.2f}%")
print("\n Detailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

#  Save Evaluation Report


with open("model_evaluation_report.txt", "w") as f:
    f.write(f"Overall Model Accuracy: {acc:.2f}%\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report(all_labels, all_preds, target_names=test_dataset.classes))

print("\n Evaluation complete! Results saved to model_evaluation_report.txt")




