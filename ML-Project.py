import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from PIL import Image
import os

# Define the transformation for preprocessing CIFAR-10 images
transform = transforms.Compose([
    transforms.Resize(256),  # Resize image for ResNet input size
    transforms.CenterCrop(224),  # Crop to 224x224 for ResNet
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize ResNet18 model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)  # Adjust final layer for CIFAR-10 (10 classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Clear gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}], Loss: {loss.item():.4f}")

# Save the model after training
torch.save(model.state_dict(), 'model.pth')
print("Model saved!")

# Loading the model in future runs 

# Check if the model exists and load it
if os.path.exists('model.pth'):
    model.load_state_dict(torch.load('model.pth'))  # Load the saved model weights
    model.to(device)  # Ensure the model is on the correct device (GPU/CPU)
    print("Model loaded!")
else:
    print("Model file not found. Training from scratch.")

# Image Prediction 

# Load and preprocess the image (ensuring it's in RGB format)
image_path = 'Image-to-text.png'
image = Image.open(image_path).convert('RGB')  # Ensuring image is in RGB format
image = transform(image).unsqueeze(0)  # Add batch dimension (1, C, H, W)
image = image.to(device)

# Make prediction
model.eval()  # Set model to evaluation mode
with torch.no_grad():  # Disable gradient tracking for inference
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)

# Map the predicted class index to the corresponding label
class_names = train_dataset.classes  # List of CIFAR-10 class names
predicted_class = class_names[predicted.item()]
print(f"Predicted class for {image_path}: {predicted_class}")

