import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import multiprocessing as mp
import torch
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        self.model.eval()
        input_image = input_image.unsqueeze(0).to(device)

        output = self.model(input_image)
        self.model.zero_grad()
        class_loss = F.cross_entropy(output, torch.tensor([target_class]).to(device))
        class_loss.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_image.shape[2], input_image.shape[3]))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def show_cam_on_image_gray(img, mask):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    
    cam = heatmap * 0.5 + img_rgb * 0.5
    
    cam = cam / np.max(cam)
    
    cam = np.uint8(255 * cam)
    
    return cam



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
    transforms.Normalize(mean=[0.485], std=[0.229]), 
])

train_dir = 'dataset/train'
val_dir = 'dataset/valid'
test_dir = 'dataset/test'

train_dataset = CustomImageDataset(data_dir=train_dir, transform=transform)
val_dataset = CustomImageDataset(data_dir=val_dir, transform=transform)
test_dataset = CustomImageDataset(data_dir=test_dir, transform=transform)

class CAM(nn.Module):
    def __init__(self, channels, r):
        super(CAM, self).__init__()
        self.channels = channels
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // r, bias=False),
            nn.ReLU(),
            nn.Linear(channels // r, channels, bias=False)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        avg_pool = self.avg_pool(x).view(x.size(0), -1)
        max_pool = self.max_pool(x).view(x.size(0), -1)
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        out = avg_out + max_out
        scale = torch.sigmoid(out).view(x.size(0), self.channels, 1, 1)
        return x * scale

class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN(num_classes=len(train_dataset.class_to_idx)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Training loop
    num_epochs = 10
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
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

        # Validation 
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Validation Accuracy: {100 * correct / total}%')

    # Test 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total}%')
    # Grad-CAM Visualization
    grad_cam = GradCAM(model=model, target_layer=model.layer2)
    save_dir="CBAM_Images"
    os.makedirs(save_dir, exist_ok=True)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(len(images)):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = img - np.min(img)
            img = img / np.max(img)
            img = np.uint8(255 * img)
            
            mask = grad_cam.generate_cam(images[i], predicted[i].item())
            cam_image = show_cam_on_image_gray(img, mask)
            
                
            # Save the cam_image
            image_name = f'cam_image_{i}_predicted_{predicted[i].item()}_true_{labels[i].item()}.png'
            image_path = os.path.join(save_dir, image_name)
            cv2.imwrite(image_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))  
           