import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score
import numpy as np


# Define the CBAM module
class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels // self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels // self.r, out_features=self.channels, bias=True)
        )

    def forward(self, x):
        max_pool = F.adaptive_max_pool2d(x, output_size=1)
        avg_pool = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max_pool.view(b, c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg_pool.view(b, c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x
        return output

class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max_pool = torch.max(x, 1)[0].unsqueeze(1)
        avg_pool = torch.mean(x, 1).unsqueeze(1)
        concat = torch.cat((max_pool, avg_pool), dim=1)
        output = self.conv(concat)
        output = torch.sigmoid(output) * x
        return output

class CBAM(nn.Module):
    def __init__(self, channels, r=16):
        super(CBAM, self).__init__()
        self.cam = CAM(channels=channels, r=r)
        self.sam = SAM()

    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        return x

# Define the CBAM-augmented ResNet blocks and ResNet50 model
class BasicBlockCBAM(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlockCBAM, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.cbam = CBAM(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = F.relu(out)
        return out

class ResNetCBAM(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNetCBAM, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes * block.expansion
        layers.extend([block(self.in_planes, planes) for _ in range(1, blocks)])
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet50_cbam(num_classes):
    return ResNetCBAM(BasicBlockCBAM, [3, 4, 6, 3], num_classes)

# Custom dataset class
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
        label = self.labels[idx]  # Should be a single integer for multiclass classification
        if self.transform:
            image = self.transform(image)
        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # RGB normalization
])

# Directories
test_dir = 'OriginalImages-split/test'

# Dataset and DataLoader
test_dataset = CustomImageDataset(data_dir=test_dir, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# Load model
model = resnet50_cbam(num_classes=len(test_dataset.class_to_idx)).to(device)
model.load_state_dict(torch.load('resnet50_cbam.pth'))
model.eval()

# DataLoader
batch_size = 32
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Evaluation
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        outputs = torch.sigmoid(outputs)  # For multi-label classification
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(outputs.cpu().numpy())

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_predictions = np.array(all_predictions)

# Debugging prints
print(f'all_labels shape: {all_labels.shape}')
print(f'all_predictions shape: {all_predictions.shape}')
print(f'First few labels: {all_labels[:5]}')
print(f'First few predictions: {all_predictions[:5]}')

# Apply threshold to convert predictions to binary
threshold = 0.5
binary_predictions = (all_predictions > threshold).astype(int)


# Convert single integer labels to binary matrix
def labels_to_binary_matrix(labels, num_classes):
    binary_matrix = np.zeros((len(labels), num_classes), dtype=int)
    for i, label in enumerate(labels):
        binary_matrix[i, label] = 1
    return binary_matrix

# Convert all_labels to binary matrix
num_classes = binary_predictions.shape[1]
binary_labels = labels_to_binary_matrix(all_labels, num_classes)

# Debugging prints
print(f'binary_labels shape: {binary_labels.shape}')
print(f'First few binary labels: {binary_labels[:5]}')

# Calculate precision, recall, and F1 score for multi-label classification
precision, recall, f1, _ = precision_recall_fscore_support(binary_labels, binary_predictions, average='samples')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

# Compute average precision for each class
average_precisions = []
for i in range(num_classes):  # Iterate over each class
    ap = average_precision_score(binary_labels[:, i], all_predictions[:, i])
    average_precisions.append(ap)

# Compute mean Average Precision
mAP = np.mean(average_precisions)
print(f'mAP: {mAP:.4f}')
