import os
import cv2
import numpy as np
from PIL import Image
import easyocr

# Initialize easyocr reader
reader = easyocr.Reader(['en'])

# Define filters
def ocr_filter(image):
    text = reader.readtext(np.array(image), detail=0)
    return len(text) > 0  # Returns True if text is present

def color_filter(image):
    image_array = np.array(image)
    if len(image_array.shape) == 2:  # Grayscale image
        return False  # No unusual colors in grayscale images
    # Define thresholds for colors that are not typical for fingernails
    # Example: Unusual colors like bright purple, green, or blue
    unusual_colors = np.sum(
        ((image_array[:, :, 0] > 150) & (image_array[:, :, 1] < 50) & (image_array[:, :, 2] > 150)) |  # Bright purple
        ((image_array[:, :, 0] < 50) & (image_array[:, :, 1] > 150) & (image_array[:, :, 2] < 50)) |   # Bright green
        ((image_array[:, :, 0] < 50) & (image_array[:, :, 1] < 50) & (image_array[:, :, 2] > 150))     # Bright blue
    )
    return unusual_colors > 1000  # Returns True if there are many unusual color pixels

def quality_filter(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY) if len(np.array(image).shape) == 3 else np.array(image)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return laplacian_var < 100  # Returns True if image is blurry

def edge_detection_filter(image):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY) if len(np.array(image).shape) == 3 else np.array(image)
    edges = cv2.Canny(gray_image, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours) < 5  # Returns True if there are too few contours

# Combine filters to calculate relevance score
def combined_filter(image):
    quality_score = 5  # Start with the highest quality score
    # if ocr_filter(image):
    #     quality_score -= 1
    if color_filter(image):
        quality_score -= 1
    if quality_filter(image):
        quality_score -= 1
    if edge_detection_filter(image):
        quality_score -= 1
    return quality_score


# Process images
input_folder = 'input'
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path)
        score = combined_filter(image)
        new_filename = f"{score}_{filename}"
        output_path = os.path.join(output_folder, new_filename)

        # Convert image to RGB mode if it's in 'P' mode
        if image.mode == 'P':
            image = image.convert('RGB')

        image.save(output_path)
        print(f"Processed {filename} -> {new_filename}")

