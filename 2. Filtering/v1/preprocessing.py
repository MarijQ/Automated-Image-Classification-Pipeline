import cv2
import numpy as np
import os

def process_images(input_folder):
    # List all files in the input folder
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    
    # Create an output folder if it doesn't exist
    output_folder = os.path.join(input_folder, 'filtered')
    os.makedirs(output_folder, exist_ok=True)

    for file in files:
        file_path = os.path.join(input_folder, file)
        
        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to load {file}")
            continue

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Detect edges using Canny
        edges = cv2.Canny(blur, 50, 150)

        # Find contours based on edges detected
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out images with too few contours (simple shapes, possibly text or not detailed)
        if len(contours) > 5:
            output_path = os.path.join(output_folder, file)
            cv2.imwrite(output_path, image)
            print(f"Processed and saved {file} to {output_folder}")

if __name__ == "__main__":
    input_folder = './input'  # Adjust as necessary
    process_images(input_folder)
