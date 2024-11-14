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

train_dir = 'OriginalImages-split/train'
val_dir = 'OriginalImages-split/val'
test_dir = 'OriginalImages-split/test'

train_dataset = CustomImageDataset(data_dir=train_dir, transform=transform)
val_dataset = CustomImageDataset(data_dir=val_dir, transform=transform)
test_dataset = CustomImageDataset(data_dir=test_dir, transform=transform)

class ResNet50(nn.Module):
    def __init__(self, num_classes=10, ks=3, stride=2):
        super(ResNet50, self).__init__()
        self.resnet50 = models.resnet50(weights=None)

        # Adjust the first convolution layer to accommodate grayscale images
        self.resnet50.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(ks, ks),
                                        stride=(stride, stride), padding=(1, 1), bias=False)
        self.resnet50.bn1 = nn.BatchNorm2d(32)
        
        # Introducing a max pooling layer with large kernel size and stride
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        
        self._adjust_initial_blocks()
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def _adjust_initial_blocks(self):
        # Adjust the channels in the first block
        self.resnet50.layer1[0].conv1 = nn.Conv2d(32, 64, kernel_size=(9, 9), stride=(3, 3), bias=False)
        self.resnet50.layer1[0].bn1 = nn.BatchNorm2d(64)
        self.resnet50.layer1[0].downsample = nn.Sequential(
            nn.Conv2d(32, 256, kernel_size=(9, 9), stride=(3, 3), bias=False),
            nn.BatchNorm2d(256)
        )
        #11 5
        #7 3
    def forward(self, x):
        x = self.resnet50.conv1(x)
        x = self.resnet50.bn1(x)
        x = self.pooling(x)  # Applying the added pooling layer
        x = self.resnet50.relu(x)
        x = self.resnet50.maxpool(x)
        x = self.resnet50.layer1(x)  # Ensure layer1 is processed
        x = self.resnet50.layer2(x)
        x = self.resnet50.layer3(x)
        x = self.resnet50.layer4(x)
        x = self.resnet50.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet50.fc(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')





if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    # Create the data loaders with num_workers=0 to avoid multiprocessing issues on Windows
    bs = 32
    model = ResNet50(num_classes=len(train_dataset.class_to_idx),ks=5,stride=2).to(device)
    train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=bs, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False, num_workers=0)
    # Define the loss function and optimizer
    class_weights = torch.tensor([1.0, 2.0, 2.0, 1.0, 1.0], dtype=torch.float32)  # Higher weights to class 1 and 2
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # Training loop
    num_epochs = 400
    train_losses = []
    val_losses = []
    best_test, best_val = 0, 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

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
        # print(f'Validation Loss: {val_epoch_loss:.4f}')
        if correct/total > best_val:
            best_val = correct/total
            torch.save(model.state_dict(), f'./Res.pth')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

