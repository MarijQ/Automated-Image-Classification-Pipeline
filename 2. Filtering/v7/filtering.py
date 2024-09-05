import cv2
import pytesseract
import numpy as np
import os
import pandas as pd
import shutil
import tkinter as tk
from tkinter import filedialog
from skimage import filters
from skimage.measure import shannon_entropy
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from bayes_opt import BayesianOptimization
import warnings

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

    def process_image(self, image_path):
        """Processes a single image and returns the filter scores."""
        image = cv2.imread(image_path)
        ocr_score = self.text_filter(image)
        color_score = self.color_filter(image)
        quality_score = self.quality_filter(image)
        edge_score = self.edge_filter(image)

        return {
            'ocr_score': ocr_score,
            'color_score': color_score,
            'quality_score': quality_score,
            'edge_score': edge_score
        }

    def text_filter(self, image):
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


class BayesianOptimizer:
    def __init__(self, config, processor):
        self.config = config
        self.processor = processor
        self.data = pd.DataFrame(columns=['filename', 'ocr_score', 'color_score', 'quality_score', 'edge_score', 'user_ocr', 'user_color', 'user_quality', 'user_edge'])
        self.optimizer = BayesianOptimization(
            f=self._objective_function,
            pbounds=self._get_pbounds(),
            random_state=42,
            verbose=2
        )

    def _get_pbounds(self):
        """Defines the bounds for the Bayesian Optimization process."""
        return {
            'text_detection_threshold': (0.01, 0.5),
            'blur_threshold': (10.0, 200.0),
            'brightness_threshold_low': (0, 100),
            'brightness_threshold_high': (100, 255),
            'color_filter_threshold': (0.1, 0.5),
            'edge_threshold': (0.01, 0.5)
        }

    def _objective_function(self, text_detection_threshold, blur_threshold, brightness_threshold_low, brightness_threshold_high, color_filter_threshold, edge_threshold):
        """Objective function for Bayesian Optimization."""
        self.config['text_detection_threshold'] = text_detection_threshold
        self.config['blur_threshold'] = blur_threshold
        self.config['brightness_threshold'] = (brightness_threshold_low, brightness_threshold_high)
        self.config['color_filter_threshold'] = color_filter_threshold
        self.config['edge_threshold'] = edge_threshold

        errors = []
        for _, row in self.data.iterrows():
            image_path = row['filename']
            scores = self.processor.process_image(image_path)
            errors.append(mean_absolute_error(
                [row['user_ocr'], row['user_color'], row['user_quality'], row['user_edge']],
                [scores['ocr_score'], scores['color_score'], scores['quality_score'], scores['edge_score']]
            ))

        return -np.mean(errors)  # Negative because BayesianOptimization maximizes by default

    def update_data(self, image_path, user_ratings):
        """Updates the dataset with new user ratings and optimizes thresholds."""
        scores = self.processor.process_image(image_path)
        new_data = {
            'filename': image_path,
            'ocr_score': scores['ocr_score'],
            'color_score': scores['color_score'],
            'quality_score': scores['quality_score'],
            'edge_score': scores['edge_score'],
            'user_ocr': user_ratings['ocr'],
            'user_color': user_ratings['color'],
            'user_quality': user_ratings['quality'],
            'user_edge': user_ratings['edge']
        }
        self.data = self.data.append(new_data, ignore_index=True)
        self.data.to_csv(os.path.join(self.config['output_dir'], 'labeled_data.csv'), index=False)
        self.optimizer.maximize(init_points=2, n_iter=5)

    def get_average_error(self):
        """Calculates the current average error across all labeled images."""
        errors = []
        for _, row in self.data.iterrows():
            errors.append(mean_absolute_error(
                [row['user_ocr'], row['user_color'], row['user_quality'], row['user_edge']],
                [row['ocr_score'], row['color_score'], row['quality_score'], row['edge_score']]
            ))
        return np.mean(errors) if errors else None


class UserInterface:
    def __init__(self, config, optimizer):
        self.config = config
        self.optimizer = optimizer
        self.root = tk.Tk()
        self.root.title("Image Suitability Grader")
        self.image_path = None

        # UI Elements
        self.label = tk.Label(self.root, text="Select an image to label")
        self.label.pack()

        self.select_button = tk.Button(self.root, text="Select Image", command=self.select_image)
        self.select_button.pack()

        self.ocr_label = tk.Label(self.root, text="OCR Score:")
        self.ocr_label.pack()
        self.ocr_scale = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
        self.ocr_scale.pack()

        self.color_label = tk.Label(self.root, text="Color Score:")
        self.color_label.pack()
        self.color_scale = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
        self.color_scale.pack()

        self.quality_label = tk.Label(self.root, text="Quality Score:")
        self.quality_label.pack()
        self.quality_scale = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
        self.quality_scale.pack()

        self.edge_label = tk.Label(self.root, text="Edge Score:")
        self.edge_label.pack()
        self.edge_scale = tk.Scale(self.root, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
        self.edge_scale.pack()

        self.submit_button = tk.Button(self.root, text="Submit Ratings", command=self.submit_ratings)
        self.submit_button.pack()

        self.error_label = tk.Label(self.root, text="")
        self.error_label.pack()

    def select_image(self):
        """Opens a file dialog to select an image."""
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if self.image_path:
            self.label.config(text=f"Selected: {self.image_path}")

    def submit_ratings(self):
        """Submits the user ratings and updates the optimization process."""
        if not self.image_path:
            self.label.config(text="Please select an image first.")
            return

        user_ratings = {
            'ocr': self.ocr_scale.get(),
            'color': self.color_scale.get(),
            'quality': self.quality_scale.get(),
            'edge': self.edge_scale.get()
        }

        self.optimizer.update_data(self.image_path, user_ratings)
        avg_error = self.optimizer.get_average_error()
        if avg_error is not None:
            self.error_label.config(text=f"Average Error: {avg_error:.4f}")
        else:
            self.error_label.config(text="No data yet.")

    def run(self):
        """Runs the Tkinter main loop."""
        self.root.mainloop()


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
        'output_dir': 'output'
    }

    processor = ImageProcessor(config)
    optimizer = BayesianOptimizer(config, processor)
    ui = UserInterface(config, optimizer)
    ui.run()
