import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms

import medmnist
from medmnist import INFO


# we can use: 'pathmnist', 'pneumoniamnist',chestmnist, 
DATA_FLAG = 'pathmnist'      # first try PathMNIST
BATCH_SIZE = 128
NUM_EPOCHS = 5               # no of epochs
LR = 1e-3
USE_PRETRAINED = True
FREEZE_BACKBONE = False       

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# loading the data

info = INFO[DATA_FLAG]
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

# Transform: 28x28 -> 224x224.
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = DataClass(split='train', transform=transform, download=True)
val_dataset   = DataClass(split='val',   transform=transform, download=True)
test_dataset  = DataClass(split='test',  transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Dataset: {DATA_FLAG}")
print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

# ResNet18
if USE_PRETRAINED:
    # use ImageNet pre-trained weights
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    print("Loaded ImageNet pre-trained ResNet18.")
else:
    # no pre-training data
    model = models.resnet18(weights=None)
    print("Loaded ResNet18 with random weights (no pre-training).")

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, n_classes)

if USE_PRETRAINED and FREEZE_BACKBONE:
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True
    print("Backbone frozen. Only final layer will be trained.")
else:
    print("All layers will be trained (fine-tuning).")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                       lr=LR)


# training and evalution 
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).squeeze().long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch}: Train Loss = {epoch_loss:.4f}, Train Acc = {epoch_acc:.4f}")

def evaluate(loader, split_name="Val"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).squeeze().long()

            outputs = model(images)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"{split_name} Accuracy = {acc:.4f}")
    return acc

# training loop
best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    train_one_epoch(epoch)
    val_acc = evaluate(val_loader, split_name="Val")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f"best_{DATA_FLAG}_resnet18.pth")
        print("  --> New best model saved.")

print("Training finished. Best Val Acc:", best_val_acc)
print("Evaluating on test set with best model...")

#m model testing
model.load_state_dict(torch.load(f"best_{DATA_FLAG}_resnet18.pth", map_location=device))
model.to(device)
test_acc = evaluate(test_loader, split_name="Test")
print("Final Test Accuracy:", test_acc)

