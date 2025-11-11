
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

#  Dataset & Dataloader


data_dir = r"C:\Users\bssad\Pictures\Projects\Multimedia\images\dataset\train_reduced"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f" Loaded balanced dataset with {len(dataset)} images across {len(dataset.classes)} classes.")
print("Class mapping:", dataset.class_to_idx)


#  Model Setup (Fine-Tuned ResNet18)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7

# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Unfreeze deeper layers for better emotion-specific learning
for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Replace classifier layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
num_epochs = 10


#  Training Loop


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

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(dataloader):
            print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{batch_idx+1}/{len(dataloader)}] "
                  f"| Loss: {loss.item():.4f} | Accuracy: {100*correct/total:.2f}%")

    print(f" Epoch [{epoch+1}] Complete | Avg Loss: {running_loss/len(dataloader):.4f} "
          f"| Epoch Accuracy: {100*correct/total:.2f}%\n")


#  Save Trained Model


os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/emotion_cnn.pth")
print("Model trained and saved at model/emotion_cnn.pth")

