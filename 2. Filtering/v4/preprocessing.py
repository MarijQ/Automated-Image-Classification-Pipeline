import cv2
import pytesseract
import numpy as np
import os
from skimage import filters
from skimage.measure import shannon_entropy
import shutil

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.output_dir = config['output_dir']
        self._prepare_output_directory()

    def _prepare_output_directory(self):
        """Clears the contents of the output directory if it exists."""
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        else:
            os.makedirs(self.output_dir, exist_ok=True)

    def process_images(self, input_dir):
        """Processes all images in the input directory and saves them with a suitability score."""
        metrics = {
            'ocr': [],
            'color': [],
            'quality': [],
            'edges': [],
            'overall': []
        }

        # Create subdirectories for each metric
        for metric in metrics.keys():
            metric_dir = os.path.join(self.output_dir, metric)
            os.makedirs(metric_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)
                image = cv2.imread(image_path)

                # Calculate scores
                ocr_score = self.ocr_filter(image)
                color_score = self.color_filter(image)
                quality_score = self.quality_filter(image)
                edge_score = self.edge_filter(image)
                overall_score = self.calculate_suitability_score(image)

                # Save images with metric scores
                ocr_filename = f"{int(ocr_score * 100)}_{filename}"
                cv2.imwrite(os.path.join(self.output_dir, 'ocr', ocr_filename), image)
                metrics['ocr'].append((ocr_score, ocr_filename))

                color_filename = f"{int(color_score * 100)}_{filename}"
                cv2.imwrite(os.path.join(self.output_dir, 'color', color_filename), image)
                metrics['color'].append((color_score, color_filename))

                quality_filename = f"{int(quality_score * 100)}_{filename}"
                cv2.imwrite(os.path.join(self.output_dir, 'quality', quality_filename), image)
                metrics['quality'].append((quality_score, quality_filename))

                edge_filename = f"{int(edge_score * 100)}_{filename}"
                cv2.imwrite(os.path.join(self.output_dir, 'edges', edge_filename), image)
                metrics['edges'].append((edge_score, edge_filename))

                # Save image with overall score
                overall_filename = f"{int(overall_score)}_{filename}"
                cv2.imwrite(os.path.join(self.output_dir, overall_filename), image)
                metrics['overall'].append((overall_score, overall_filename))

                print(
                    f"Processed {filename}: Overall Score = {overall_score:.2f}, OCR Score = {ocr_score:.2f}, Color Score = {color_score:.2f}, Quality Score = {quality_score:.2f}, Edge Score = {edge_score:.2f}")

        # No need to copy files again since they are already saved in the correct locations

    def calculate_suitability_score(self, image):
        """Calculates the overall suitability score for an image."""
        ocr_score = self.ocr_filter(image)
        color_score = self.color_filter(image)
        quality_score = self.quality_filter(image)
        edge_score = self.edge_filter(image)

        # Weighted sum of the scores
        total_score = (
                self.config['weights']['ocr'] * ocr_score +
                self.config['weights']['color'] * color_score +
                self.config['weights']['quality'] * quality_score +
                self.config['weights']['edges'] * edge_score
        )

        return total_score * 100

    def ocr_filter(self, image):
        """Detects the presence of text in the image using OCR and calculates the area occupied by the text at the letter level."""
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        total_text_area = 0
        total_image_area = image.shape[0] * image.shape[1]

        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0 and data['text'][i].strip() != "":
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                text_area = w * h
                total_text_area += text_area

        text_density = total_text_area / total_image_area
        return 1 - min(text_density / self.config['text_detection_threshold'], 1)

    def color_filter(self, image):
        """Filters out images with a high presence of unwanted colors for fingernails."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv_image.shape[:2], dtype=np.uint8)

        for lower_bound, upper_bound in self.config['color_ranges']:
            mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        color_density = np.sum(combined_mask) / (combined_mask.shape[0] * combined_mask.shape[1])
        return 1 - min(color_density / self.config['color_filter_threshold'], 1)

    def quality_filter(self, image):
        """Evaluates the quality of the image based on blur and brightness."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        blur_score = min(laplacian_var / self.config['blur_threshold'], 1)

        brightness = np.mean(gray_image)
        brightness_score = 1 if self.config['brightness_threshold'][0] <= brightness <= self.config['brightness_threshold'][1] else 0

        return (blur_score * brightness_score) ** 0.5

    def edge_filter(self, image):
        """Detects the presence of edges in the image."""
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = filters.sobel(gray_image)
        edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
        return min(edge_density / self.config['edge_threshold'], 1)


if __name__ == "__main__":
    config = {
        'text_detection_threshold': 0.1,
        'blur_threshold': 100.0,
        'brightness_threshold': (50, 200),
        'color_filter_threshold': 0.3,
        'edge_threshold': 0.1,
        'color_ranges': [
            (np.array([0, 20, 70]), np.array([20, 255, 255])),
            (np.array([0, 10, 100]), np.array([20, 255, 255])),
        ],
        'weights': {
            'ocr': 0.25,
            'color': 0.25,
            'quality': 0.25,
            'edges': 0.25,
        },
        'output_dir': 'output'
    }

    input_dir = 'input'
    processor = ImageProcessor(config)
    processor.process_images(input_dir)