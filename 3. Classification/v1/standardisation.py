import os
import cv2
import numpy as np
from skimage import exposure
import shutil

def clean_image(image, target_size=(128, 128)):
    """Cleans the image by applying resizing, normalization, mean subtraction, grayscale conversion, histogram equalization, and noise reduction."""

    # Resize the image to the target size
    resized_image = cv2.resize(image, target_size)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # Normalize the pixel values to the range [0, 1]
    normalized_image = gray_image / 255.0

    # Subtract the mean pixel value
    mean_value = np.mean(normalized_image)
    mean_subtracted_image = normalized_image - mean_value

    # Apply histogram equalization to improve contrast
    equalized_image = exposure.equalize_hist(mean_subtracted_image)

    # Apply Gaussian blur to reduce noise
    cleaned_image = cv2.GaussianBlur(equalized_image, (5, 5), 0)

    return cleaned_image

def prepare_output_directory():
    """Clears the contents of the output directory if it exists."""
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
    else:
        os.makedirs(output_dir, exist_ok=True)

def clean_images_in_output(input_dir, output_dir, target_size=(128, 128)):
    """Cleans all images in the input directory and saves them to the output directory."""

    prepare_output_directory()

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)

        # Skip subdirectories
        if os.path.isdir(file_path):
            continue

        # Read the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Failed to read {filename}. Skipping...")
            continue

        # Clean the image
        cleaned_image = clean_image(image, target_size)

        # Save the cleaned image
        cleaned_filename = os.path.join(output_dir, filename)
        cv2.imwrite(cleaned_filename, (cleaned_image * 255).astype(np.uint8))  # Convert back to 0-255 range for saving

        print(f"Cleaned and saved {filename} to {cleaned_filename}")


if __name__ == "__main__":
    input_dir = '../sample_labelled_20'
    output_dir = 'output'
    clean_images_in_output(input_dir, output_dir)
