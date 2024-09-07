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
        """Clears the contents of the output directory if it exists and removes duplicate images."""
        if os.path.exists(self.output_dir):
            image_hashes = {}
            for filename in os.listdir(self.output_dir):
                file_path = os.path.join(self.output_dir, filename)
                try:
                    if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_hash = self._compute_image_hash(file_path)
                        if img_hash in image_hashes:
                            # Duplicate found, remove the file
                            os.remove(file_path)
                            print(f"Removed duplicate image: {file_path}")
                        else:
                            image_hashes[img_hash] = file_path
                    elif os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
        else:
            os.makedirs(self.output_dir, exist_ok=True)

    def _compute_image_hash(self, image_path):
        """Computes the dHash of an image to detect duplicates."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(image, (9, 8))  # Resize to 9x8 to compute the hash
        diff = resized[:, 1:] > resized[:, :-1]  # Compute the difference between adjacent pixels
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

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
