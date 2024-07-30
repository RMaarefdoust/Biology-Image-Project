import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(input_folder, output_folder, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    classes = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    
    for class_name in classes:
        class_dir = os.path.join(input_folder, class_name)
        images = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.tif', '.tiff'))]
        
        if len(images) == 0:
            print(f"No images found in class '{class_name}', skipping.")
            continue

        train_images, temp_images = train_test_split(images, test_size=1 - train_ratio)
        val_images, test_images = train_test_split(temp_images, test_size=test_ratio / (val_ratio + test_ratio))
        
        save_images(train_images, os.path.join(output_folder, 'train', class_name))
        save_images(val_images, os.path.join(output_folder, 'val', class_name))
        save_images(test_images, os.path.join(output_folder, 'test', class_name))
        print(f"Class '{class_name}': {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images")

def save_images(images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for image_path in images:
        shutil.copy(image_path, output_dir)

# Example usage
input_folder = 'Original Images/'
output_folder = 'OriginalImages-split/'
split_data(input_folder, output_folder)
