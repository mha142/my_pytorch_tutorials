import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

# Load the pretrained VGG16 model
vgg16 = models.vgg16(pretrained=True)

# Freeze the feature extraction layers
for param in vgg16.features.parameters():
    param.requires_grad = False

# Replace the classifier with a custom classifier for your dataset
num_classes = 10  # Example: 10 classes for CIFAR-10
vgg16.classifier[6] = nn.Linear(4096, num_classes)  # Change output layer


# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to VGG16 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Pretrained normalization
])

# Load datasets
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(vgg16.classifier.parameters(), lr=0.001)  # Optimizer for the new classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg16 = vgg16.to(device)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    vgg16.train()  # Set to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = vgg16(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


# Evaluation loop
vgg16.eval()  # Set to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')



