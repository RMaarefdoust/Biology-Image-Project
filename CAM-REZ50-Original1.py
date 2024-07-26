import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import cv2
import numpy as np
from sklearn.metrics import recall_score, f1_score
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels // self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max_pool = F.adaptive_max_pool2d(x, output_size=1)
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        max_linear = self.linear(max_pool.view(b, c)).view(b, c, 1, 1)
        avg_linear = self.linear(avg_pool.view(b, c)).view(b, c, 1, 1)
        output = max_linear + avg_linear
        output = torch.sigmoid(output) * x
        return output

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
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def train_model(model, cam_module, criterion, optimizer, train_loader):
    model.train()
    cam_module.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Forward pass through ResNet-50
        outputs = model(images)

        # Apply CAM module to model output
        cam_outputs = cam_module(outputs)

        # Calculate loss
        loss = criterion(cam_outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate_model(model, cam_module, criterion, data_loader):
    model.eval()
    cam_module.eval()
    predictions = []
    ground_truths = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass through ResNet-50
            outputs = model(images)

            # Apply CAM module to model output
            cam_outputs = cam_module(outputs)

            # Calculate loss
            loss = criterion(cam_outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(cam_outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(data_loader)
    return avg_loss, np.array(predictions), np.array(ground_truths)


def test_model(model, cam_module, test_loader):
    model.eval()
    cam_module.eval()
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass through ResNet-50
            outputs = model(images)

            # Apply CAM module to model output
            cam_outputs = cam_module(outputs)

            _, predicted = torch.max(cam_outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            ground_truths.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(ground_truths)


def calculate_metrics(predictions, ground_truths):
    # Convert predictions and ground_truths to numpy arrays if not already
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # Calculate metrics
    recall = recall_score(ground_truths, predictions, average='macro')
    f1 = f1_score(ground_truths, predictions, average='macro')

    return recall, f1


def plot_metrics(train_losses, val_losses, recalls, f1_scores):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, recalls, label='Recall')
    plt.plot(epochs, f1_scores, label='F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Recall and F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('metrics_plot.png')
    plt.show()

def main():
    # Define transforms and datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dir = 'OriginalImages2/train'
    val_dir = 'OriginalImages2/val'
    test_dir = 'OriginalImages2/test'

    train_dataset = CustomImageDataset(data_dir=train_dir, transform=transform)
    val_dataset = CustomImageDataset(data_dir=val_dir, transform=transform)
    test_dataset = CustomImageDataset(data_dir=test_dir, transform=transform)

    batch_size = 16
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load ResNet-50 model with CAM module
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_dataset.class_to_idx))

    cam_module = CAM(channels=model.layer4[2].conv3.out_channels, r=16)  # Example layer from ResNet-50

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(cam_module.parameters()), lr=0.001)

    # Move model to GPU if available
    model = model.to(device)
    cam_module = cam_module.to(device)

    # Training loop
    num_epochs = 20
    train_losses = []
    val_losses = []
    recalls = []
    f1_scores = []

    for epoch in range(num_epochs):
        # Train
        train_loss = train_model(model, cam_module, criterion, optimizer, train_loader)
        train_losses.append(train_loss)

        # Validate
        val_loss, _, _ = evaluate_model(model, cam_module, criterion, val_loader)
        val_losses.append(val_loss)

        # Test
        predictions, ground_truths = test_model(model, cam_module, test_loader)

        # Calculate metrics
        recall, f1 = calculate_metrics(predictions, ground_truths)
        recalls.append(recall)
        f1_scores.append(f1)

        # Print metrics
        print(f'Epoch [{epoch+1}/{num_epochs}], Test Recall: {recall:.4f}, Test F1 Score: {f1:.4f}')

    # Plot metrics
    plot_metrics(train_losses, val_losses, recalls, f1_scores)


if __name__ == '__main__':
    main()
