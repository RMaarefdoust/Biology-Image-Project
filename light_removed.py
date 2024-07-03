import cv2
import numpy as np
import os

input_dir = '1'
output_dir = 'light_removed_images'

os.makedirs(output_dir, exist_ok=True)

blur_kernel_size = (51, 51)  # Must be an odd number


for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        file_path = os.path.join(input_dir, filename)
        
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            # Apply Gaussian blur to estimate the background
            blurred_image = cv2.GaussianBlur(image, blur_kernel_size, 0)
            
            # Subtract the blurred image from the original image
            light_removed_image = cv2.subtract(image, blurred_image)
            
            # Normalize the result to the range 0-255
            light_removed_image = cv2.normalize(light_removed_image, None, 0, 255, cv2.NORM_MINMAX)
            
            output_file_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(output_file_path, light_removed_image)
            
            print(f"Processed and saved: {output_file_path}")
        else:
            print(f"Failed to load: {file_path}")

print("Processing completed.")
