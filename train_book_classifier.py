import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import json

# === Dataset folder ===
DATASET_DIR = "book_dataset"

TRAIN_DIR = f"{DATASET_DIR}/train"
VAL_DIR = f"{DATASET_DIR}/val"

# === Check dataset folders exist ===
if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError("Dataset folders not found. Make sure dataset/train and dataset/val exist.")

# === Auto load & generate class list ===
train_data = datasets.ImageFolder(TRAIN_DIR)
class_labels = train_data.classes  # auto-detected folder names
num_classes = len(class_labels)

print("\n✅ Classes detected in dataset:")
for c in class_labels:
    print(" -", c)

# === Write labels.txt (used by ClickVision) ===
with open("labels.txt", "w", encoding="utf-8") as f:
    for label in class_labels:
        f.write(label + "\n")

print("\n✅ labels.txt updated successfully.")

# === Data transforms (resize and normalize) ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Reload datasets with transform
train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
val_data = datasets.ImageFolder(VAL_DIR, transform=transform)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# === Model Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")  # updated pretrained syntax
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# === Training ===
EPOCHS = 10
print("\n🚀 Starting Training...\n")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"✅ Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"🎯 Validation Accuracy: {acc:.2f}%")

# === Save classifier for ClickVision ===
torch.save(model.state_dict(), "classifier_model.pth")
print("\n✅ Model saved successfully as classifier_model.pth")
print("✅ Training complete.\n")
