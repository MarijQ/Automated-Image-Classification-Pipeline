import cv2
import pytesseract
import numpy as np
import os
from skimage import filters
from skimage.measure import shannon_entropy
import shutil
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Parameters
TEXT_DETECTION_THRESHOLD = 0.1  # Threshold for OCR text detection
BLUR_THRESHOLD = 100.0  # Threshold for detecting blur (higher is less blurry)
BRIGHTNESS_THRESHOLD = (50, 200)  # Acceptable brightness range
COLOR_FILTER_THRESHOLD = 0.3  # Threshold for color filtering (0 to 1)
EDGE_THRESHOLD = 0.1  # Threshold for edge detection
COLOR_RANGES = [
    (np.array([0, 20, 70]), np.array([20, 255, 255])),  # Skin tones
    (np.array([0, 10, 100]), np.array([20, 255, 255])),  # Light nail colors
]

# Weights for each filter in the final score
WEIGHTS = {
    'ocr': 0.25,
    'color': 0.25,
    'quality': 0.25,
    'edges': 0.25,
}

# Create output directory
output_dir = 'output'

# Delete the output directory if it exists
if os.path.exists(output_dir):
    # Remove all files and subdirectories in the output directory
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove file
            elif os.path.isdir(file_path):
                os.rmdir(file_path)  # Remove directory (only if empty)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    # Now remove the output directory itself
    try:
        os.rmdir(output_dir)  # Remove the output directory
    except Exception as e:
        print(f"Error deleting output directory: {e}")

os.makedirs(output_dir, exist_ok=True)


def ocr_filter(image):
    """Detects the presence of text in the image using OCR and calculates the area occupied by the text at the letter level."""
    # Get the bounding box information for each detected character
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    total_text_area = 0
    total_image_area = image.shape[0] * image.shape[1]  # Total number of pixels in the image

    # Iterate through detected characters
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0 and data['text'][i].strip() != "":  # Only consider characters with a positive confidence score and non-empty text
            # Get the bounding box coordinates for each character
            x = data['left'][i]
            y = data['top'][i]
            w = data['width'][i]
            h = data['height'][i]
            # Calculate the area of the bounding box for the character
            text_area = w * h
            total_text_area += text_area

    # Calculate text density as the ratio of text area to total image area
    text_density = total_text_area / total_image_area

    # Return a score based on the text density
    return 1 - min(text_density / TEXT_DETECTION_THRESHOLD, 1)


def color_filter(image):
    """Filters out images with a high presence of unwanted colors for fingernails."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Initialize combined mask
    combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

    # Create masks for each color range defined in COLOR_RANGES
    for lower_bound, upper_bound in COLOR_RANGES:
        mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Calculate color density of relevant colors
    color_density = np.sum(combined_mask) / (combined_mask.shape[0] * combined_mask.shape[1])

    # Normalize the score
    return 1 - min(color_density / COLOR_FILTER_THRESHOLD, 1)


def quality_filter(image):
    """Evaluates the quality of the image based on blur and brightness."""
    # Blur detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    blur_score = min(laplacian_var / BLUR_THRESHOLD, 1)

    # Brightness detection
    brightness = np.mean(gray_image)
    brightness_score = 1 if BRIGHTNESS_THRESHOLD[0] <= brightness <= BRIGHTNESS_THRESHOLD[1] else 0

    return (blur_score * brightness_score) ** 0.5


def edge_filter(image):
    """Detects the presence of edges in the image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = filters.sobel(gray_image)
    edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
    return min(edge_density / EDGE_THRESHOLD, 1)


def calculate_suitability_score(image):
    """Calculates the overall suitability score for an image."""
    ocr_score = ocr_filter(image)
    color_score = color_filter(image)
    quality_score = quality_filter(image)
    edge_score = edge_filter(image)

    # Weighted sum of the scores
    total_score = (
            WEIGHTS['ocr'] * ocr_score +
            WEIGHTS['color'] * color_score +
            WEIGHTS['quality'] * quality_score +
            WEIGHTS['edges'] * edge_score
    )

    return total_score * 100


def process_images(input_dir):
    """Processes all images in the input directory and saves them with a suitability score."""
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Calculate suitability score
            score = calculate_suitability_score(image)

            # Save image with prepended score
            new_filename = f"{int(score)}_{filename}"
            cv2.imwrite(os.path.join(output_dir, new_filename), image)
            print(f"Processed {filename}: Score = {score:.2f}")


if __name__ == "__main__":
    input_dir = 'input'
    process_images(input_dir)
