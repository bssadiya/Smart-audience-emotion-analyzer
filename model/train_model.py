# ===============================
# train_model.py
# ===============================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# =========================
# Step 1: Define CNN Model
# =========================
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
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
# Step 2: Dataset + Dataloader
# =========================
data_dir = "images/dataset/train_cleaned/merged"

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# =========================
# Step 3: Model Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
model = EmotionCNN(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 10

# =========================
# Step 4: Training Loop (Batch-wise logging)
# =========================
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        #  Batch-wise print
        print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx+1}/{len(dataloader)}] | "
              f"Loss: {loss.item():.4f} | Accuracy: {100*correct/total:.2f}%")
    
    # Epoch summary
    print(f" Epoch [{epoch+1}] Complete | Avg Loss: {running_loss/len(dataloader):.4f} | "
          f"Epoch Accuracy: {100*correct/total:.2f}%\n")

# =========================
# Step 5: Save Model
# =========================
os.makedirs("/model", exist_ok=True)
torch.save(model.state_dict(),"model/emotion_cnn.pth")
print(" Model trained and saved at model/emotion_cnn.pth")
