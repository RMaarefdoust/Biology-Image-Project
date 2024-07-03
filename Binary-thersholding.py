import cv2
import os

input_dir = '1'
output_dir = 'Binary-thresholded_images'

os.makedirs(output_dir, exist_ok=True)

threshold_value = 150  
max_value = 255

for filename in os.listdir(input_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        file_path = os.path.join(input_dir, filename)
        
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        if image is not None:
            # Smooth the image using Gaussian blur
            blurred_image = cv2.GaussianBlur(image, (15, 15), 0)
            
            # Create a structuring element (kernel) for morphological operations based on image size
            kernel_size = (image.shape[1] // 30, image.shape[0] // 30)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
            
            # Apply morphological opening to estimate the background light
            background = cv2.morphologyEx(blurred_image, cv2.MORPH_OPEN, kernel)
            
            # Subtract the background light from the original image
            light_removed_image = cv2.subtract(image, background)
            
            # Normalize the result to the range 0-255
            light_removed_image = cv2.normalize(light_removed_image, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply histogram equalization to enhance contrast
            equalized_image = cv2.equalizeHist(light_removed_image)
            
            # Apply binary thresholding
            _, thresholded_image = cv2.threshold(equalized_image, threshold_value, max_value, cv2.THRESH_BINARY)
            
            output_file_path = os.path.join(output_dir, filename)
            
            cv2.imwrite(output_file_path, thresholded_image)
            
            print(f"Processed and saved: {output_file_path}")
        else:
            print(f"Failed to load: {file_path}")

print("Processing completed.")
