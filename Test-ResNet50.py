import torch
from sklearn.metrics import average_precision_score, f1_score, recall_score
from torchvision import transforms, models
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
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Normalization for single channel
])


test_dir = 'OriginalImages-split/test'
test_dataset = CustomImageDataset(data_dir=test_dir, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)

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

# The rest of your testing code remains the same


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet50(num_classes=len(test_dataset.class_to_idx)).to(device)
model.load_state_dict(torch.load('custom_resnet50_model.pth'))
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
