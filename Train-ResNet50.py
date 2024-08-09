import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import multiprocessing as mp

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self._load_data()
    
    def _load_data(self):
        for idx, class_name in enumerate(os.listdir(self.data_dir)):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                for img_file in os.listdir(class_dir):
                    self.image_paths.append(os.path.join(class_dir, img_file))
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")  # Ensure image is grayscale
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 expects 224x224 input size
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Normalization for single channel
])

train_dir = 'OriginalImages-split/train-new'
val_dir = 'OriginalImages-split/val'
test_dir = 'OriginalImages-split/test'

train_dataset = CustomImageDataset(data_dir=train_dir, transform=transform)
val_dataset = CustomImageDataset(data_dir=val_dir, transform=transform)
test_dataset = CustomImageDataset(data_dir=test_dir, transform=transform)

class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        # Load pre-trained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)
        # Modify the first convolutional layer to accept grayscale images (1 channel)
        self.resnet50.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # Modify the final fully connected layer to match the number of classes
        self.resnet50.fc = nn.Linear(in_features=self.resnet50.fc.in_features, out_features=num_classes)
    
    def forward(self, x):
        return self.resnet50(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = ResNet50(num_classes=len(train_dataset.class_to_idx)).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # Create the data loaders with num_workers=0 to avoid multiprocessing issues on Windows
    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Training loop
    num_epochs = 100
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss}')

        # Validation step
        model.eval()
        val_running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_loader)
        val_losses.append(val_epoch_loss)
        print(f'Validation Loss: {val_epoch_loss}')
        print(f'Validation Accuracy: {100 * correct / total}%')

    # Plot and save the loss
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot-ResNet50.pdf')
    plt.close()

    # Save the trained model
    torch.save(model.state_dict(), 'resnet50_model.pth')

    print("Model saved as resnet50_model.pth")
