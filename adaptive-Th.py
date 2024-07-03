import cv2
import os

input_dir = '1'
output_dir = 'adaptive_thresholded_images'

os.makedirs(output_dir, exist_ok=True)

# Parameters
max_value = 255
adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
threshold_type = cv2.THRESH_BINARY
block_size = 11  # Size of a pixel neighborhood used to calculate the threshold value
C = 2  # Constant subtracted from the mean or weighted mean


for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):

        file_path = os.path.join(input_dir, filename)
        
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            thresholded_image = cv2.adaptiveThreshold(image, max_value, adaptive_method, threshold_type, block_size, C)
            
            output_file_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(output_file_path, thresholded_image)
            
            print(f"Processed and saved: {output_file_path}")
        else:
            print(f"Failed to load: {file_path}")

print("Processing completed.")
