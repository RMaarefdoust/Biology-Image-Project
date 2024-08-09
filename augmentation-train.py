import os
import random
from PIL import Image
from torchvision import transforms
import torch
import cv2
import numpy as np


input_folder = 'OriginalImages-split/train/uM0.5'#1.0
output_folder = 'OriginalImages-split/train-new/uM0.5'#1.0
os.makedirs(output_folder, exist_ok=True)


# Define the data augmentation transformations
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

# Process images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.tif')):
        # Load image using OpenCV
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Convert to PIL Image to apply augmentation
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Apply augmentation and save images
        augmented_image = augment_transform(pil_image)
        augmented_image_np = np.array(augmented_image)

        # Convert back to OpenCV format
        augmented_image_cv = cv2.cvtColor(augmented_image_np, cv2.COLOR_RGB2BGR)

        # Save augmented image
        base, ext = os.path.splitext(filename)
        new_filename = f"{base}_aug_{1}{ext}"
        new_image_path = os.path.join(output_folder, new_filename)
        cv2.imwrite(new_image_path, augmented_image_cv)

print("Augmentation complete.")

