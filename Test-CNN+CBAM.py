import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import f1_score, recall_score, average_precision_score
from sklearn.preprocessing import label_binarize

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

# Define the transform to apply to the test images
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Assuming grayscale normalization
])
class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.r = r
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.channels, out_features=self.channels//self.r, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.channels//self.r, out_features=self.channels, bias=True))

    def forward(self, x):
        max = F.adaptive_max_pool2d(x, output_size=1)
        avg = F.adaptive_avg_pool2d(x, output_size=1)
        b, c, _, _ = x.size()
        linear_max = self.linear(max.view(b,c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b,c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x
        return output
    
class SAM(nn.Module):
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        output = torch.sigmoid(output) * x 
        return output 
    
class CBAM(nn.Module):
    def __init__(self, channels, r):
        super(CBAM, self).__init__()
        self.channels = channels
        self.r = r
        self.cam = CAM(channels=self.channels, r=self.r)
        self.sam = SAM()

    def forward(self, x):
        x = self.cam(x)
        x = self.sam(x)
        return x + x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels, r=16) if cbam else None

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.cbam:
            out = self.cbam(out)
        out = F.relu(out)
        return out

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.layer1 = BasicBlock(1, 32, cbam=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = BasicBlock(32, 64, cbam=True)
        self.fc1 = nn.Linear(64 * 32 * 32, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.pool(self.layer1(x))
        x = self.pool(self.layer2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load the model
model = SimpleCNN(num_classes=5).to(device)  # Adjust num_classes based on your dataset
model.load_state_dict(torch.load('simple_cnn_cbam.pth'))
model.eval()

# Define the test dataset and loader
test_dataset = CustomImageDataset(data_dir='OriginalImages-split/test', transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Lists to store the true labels and predicted probabilities
all_labels = []
all_preds = []

# Evaluate the model on the test set
# Lists to store the true labels and predicted probabilities
all_labels = []
all_probs = []  # Store probabilities, not predictions

# Evaluate the model on the test set
correct = 0
total = 0
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

# Convert lists to numpy arrays
all_labels = np.array(all_labels)
all_probs = np.array(all_probs)


# Convert labels to binary format for multi-class average precision
num_classes = len(np.unique(all_labels))
all_labels_bin = label_binarize(all_labels, classes=range(num_classes))

# Calculate metrics
# Calculate metrics
test_accuracy = 100 * correct / total
f1 = f1_score(all_labels, np.argmax(all_probs, axis=1), average='macro')
recall = recall_score(all_labels, np.argmax(all_probs, axis=1), average='macro')
mAP = average_precision_score(all_labels_bin, all_probs, average='macro')

# Print the results
print(f'Test Accuracy: {test_accuracy:.2f}%')
print(f'F1 Score: {f1:.4f}')
print(f'Recall: {recall:.4f}')
print(f'mAP: {mAP:.4f}')

