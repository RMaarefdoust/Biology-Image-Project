import torch
from sklearn.metrics import average_precision_score, f1_score, recall_score
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np  # Import numpy

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
        image = Image.open(img_path).convert("L")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Assuming grayscale normalization
])

test_dir = 'OriginalImages-split/test'
test_dataset = CustomImageDataset(data_dir=test_dir, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=len(test_dataset.class_to_idx)).to(device)
model.load_state_dict(torch.load('simple_cnn.pth'))
model.eval()

# Testing loop
correct = 0
total = 0
all_labels = []
all_probs = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Apply softmax to get probabilities for each class
        probs = torch.softmax(outputs, dim=1)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

print(f'Test Accuracy: {100 * correct / total}%')

# Manually calculate mAP for each class and average them
num_classes = len(test_dataset.class_to_idx)
class_wise_aps = []

for i in range(num_classes):
    # Binary labels for the current class
    class_labels = np.array(all_labels) == i
    # Probabilities for the current class
    class_probs = np.array(all_probs)[:, i]
    # Calculate average precision score
    class_ap = average_precision_score(class_labels, class_probs)
    class_wise_aps.append(class_ap)

test_mAP = np.mean(class_wise_aps)
test_f1 = f1_score(all_labels, all_preds, average='macro')
test_recall = recall_score(all_labels, all_preds, average='macro')

print(f'Test mAP: {test_mAP}')
print(f'Test F1 Score: {test_f1}')
print(f'Test Recall: {test_recall}')
