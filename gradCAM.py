import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import os

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
        self.target_layer.register_full_backward_hook(backward_hook)

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
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
        return cam

def show_cam_on_image_gray(img, mask):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    cam = heatmap * 0.5 + img_rgb * 0.5
    if np.max(cam) != 0:
        cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, img_file) for img_file in os.listdir(data_dir)]
        self.labels = [i % 2 for i in range(len(self.image_paths))]  # Alternating dummy labels (0 and 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label, img_path

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to a smaller, more manageable size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),
])

data_dir = 'Dataset-test-original-image'
dataset = CustomImageDataset(data_dir=data_dir, transform=transform)

# Split dataset into training and testing sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)  # Adjust batch size as needed
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out

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
        linear_max = self.linear(max.view(b, c)).view(b, c, 1, 1)
        linear_avg = self.linear(avg.view(b, c)).view(b, c, 1, 1)
        output = linear_max + linear_avg
        output = torch.sigmoid(output) * x
        return output

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.layer1 = BasicBlock(1, 32)
        self.cam1 = CAM(32, 16)  # Add CAM to layer1
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = BasicBlock(32, 64)
        self.cam2 = CAM(64, 16)  # Add CAM to layer2
        self.layer3 = BasicBlock(64, 128)
        self.cam3 = CAM(128, 16)  # Add CAM to layer3
        self.fc1 = nn.Linear(128 * 32 * 32, 1000)  # Adjust the input size based on pooling
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.cam1(x)  # Apply CAM after layer1
        x = self.pool(x)
        x = self.layer2(x)
        x = self.cam2(x)  # Apply CAM after layer2
        x = self.pool(x)
        x = self.layer3(x)
        x = self.cam3(x)  # Apply CAM after layer3
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=2).to(device)  # Adjust number of classes based on your dataset

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}')

# Grad-CAM Visualization
grad_cam = GradCAM(model=model, target_layer=model.layer2)  # Choose the appropriate layer for Grad-CAM
save_dir = "GradCAM_Images"
os.makedirs(save_dir, exist_ok=True)

for images, labels, img_paths in test_loader:
    images = images.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    img = images[0].cpu().numpy().transpose(1, 2, 0)
    img = img - np.min(img)
    img = img / np.max(img)
    img = np.uint8(255 * img)

    mask = grad_cam.generate_cam(images[0], predicted[0].item())
    cam_image = show_cam_on_image_gray(img, mask)

    image_name = os.path.basename(img_paths[0])
    cam_image_name = f'cam_image_{image_name}'
    original_image_name = f'original_{image_name}'
    
    cam_image_path = os.path.join(save_dir, cam_image_name)
    original_image_path = os.path.join(save_dir, original_image_name)
    
    # Save CAM image
    cv2.imwrite(cam_image_path, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    
    # Save original image
    original_img = cv2.imread(img_paths[0])
    cv2.imwrite(original_image_path, original_img)
