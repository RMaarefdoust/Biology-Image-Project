from PIL import Image
import matplotlib.pyplot as plt
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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
        # Load image as grayscale
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

train_dir = 'dataset/train'
val_dir = 'dataset/valid'
test_dir = 'dataset/test'

train_dataset = CustomImageDataset(data_dir=train_dir, transform=transform)
val_dataset = CustomImageDataset(data_dir=val_dir, transform=transform)
test_dataset = CustomImageDataset(data_dir=test_dir, transform=transform)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    # Create the data loaders with num_workers=0 to avoid multiprocessing issues on Windows
    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for images, labels in train_loader:
        print(f"Images batch shape: {images.size()}")
        print(f"Labels batch shape: {labels.size()}")
        print(f"Labels: {labels}")
        break  # Break after the first batch to prevent printing too much data


images, labels = next(iter(train_loader))
image=images[10]

# Convert tensor to numpy array and reshape if necessary
image_np = image.squeeze().numpy()  # Squeeze to remove channel dimension if grayscale

# Display the image using matplotlib
plt.imshow(image_np, cmap='gray')
plt.title(f'Label: {labels[1]}')  # Assuming labels are provided as integers
plt.axis('off')
plt.show()
