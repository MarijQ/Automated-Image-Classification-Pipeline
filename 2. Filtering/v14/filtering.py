import os
import time
import shutil
import random
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

import cv2
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, recall_score,
                             f1_score, roc_curve, roc_auc_score)
from sklearn.preprocessing import MinMaxScaler

import pytesseract
from skimage import feature
from skimage.feature import hog
import pandas as pd


# ------------------ Feature Extraction Classes ------------------

class InterpretableFilters:
    """Handles interpretable filter feature extraction for images."""

    def __init__(self, device='cpu'):
        """Initializes the object detector and OCR engine."""
        self.device = device
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.object_detector = fasterrcnn_resnet50_fpn(weights=weights, progress=False).to(self.device)
        self.object_detector.eval()  # Set model to evaluation mode
        self.pytesseract = pytesseract
        self.transform = transforms.Compose([transforms.ToTensor()])  # Define image transformation

    def compute_text_presence(self, img):
        """Calculates the percentage of image area covered by detected text."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        custom_config = r'--oem 3 --psm 6'  # Configure pytesseract
        data = self.pytesseract.image_to_data(
            img_rgb, output_type=self.pytesseract.Output.DICT, config=custom_config)  # Detect text
        n_boxes = len(data['level'])  # Number of detected text boxes
        img_area = img.shape[0] * img.shape[1]  # Total image area
        text_area = 0
        for i in range(n_boxes):
            width = data['width'][i]
            height = data['height'][i]
            confidence = int(data['conf'][i])  # Get confidence score
            # Only count text areas with confidence > 80 (stricter threshold) and ignore small areas
            if confidence > 80 and width * height > 100:  # Adjust area threshold as needed
                text_area += width * height  # Accumulate text area
        text_presence = min(text_area / img_area, 1.0)  # Calculate text presence ratio
        return text_presence

    def compute_image_type(self, img):
        """Determines the edge density to classify image type."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # Further reduce thresholds to capture more edges
        edges = cv2.Canny(gray, threshold1=30, threshold2=100)  # Perform Canny edge detection
        edges_binary = edges / 255  # Normalize edges to binary values (0 or 1)
        edge_density = np.sum(edges_binary) / (edges.shape[0] * edges.shape[1])  # Calculate edge density
        return edge_density

    def compute_dominant_color(self, img):
        """Extracts the dominant color hue from the image."""
        img_small = cv2.resize(img, (80, 80))  # Resize to reduce computation
        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # Convert to RGB
        pixels = img_rgb.reshape((-1, 3)).astype(np.float32)  # Reshape for clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)  # Define KMeans criteria
        k = 5  # Increase the number of clusters to capture more colors
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)  # Apply KMeans
        counts = np.bincount(labels.flatten())  # Count cluster occurrences
        dominant_color = centers[np.argmax(counts)]  # Select dominant color
        dominant_color_hsv = cv2.cvtColor(
            np.uint8([[dominant_color]]), cv2.COLOR_RGB2HSV)[0][0][0]  # Convert to HSV and get Hue
        return dominant_color_hsv

    def compute_foreign_objects(self, img):
        """Calculates the probability that one or more irrelevant classes are present in the image."""
        img_tensor = self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(self.device)
        with torch.no_grad():
            predictions = self.object_detector([img_tensor])
        pred_classes = predictions[0]['labels'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()
        relevant_class_indices = [1]  # Define relevant classes (e.g., 'person')
        # Collect scores of irrelevant class detections
        irrelevant_scores = [score for cls, score in zip(pred_classes, pred_scores) if cls not in relevant_class_indices]
        if irrelevant_scores:
            # Assuming independence, calculate the probability of at least one irrelevant object being present
            prob_irrelevant = 1 - np.prod([1 - score for score in irrelevant_scores])
        else:
            prob_irrelevant = 0.0
        return prob_irrelevant

    def compute_image_clarity(self, img):
        """Calculates image clarity based on the variance of Laplacian."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()  # Compute variance of Laplacian
        # Adjust normalization factor based on expected variance range
        clarity_metric = min(variance / 2000.0, 1.0)  # Normalize variance (increased factor)
        return clarity_metric

    def compute_aspect_ratio(self, img):
        """Computes the normalized aspect ratio of the image."""
        h, w = img.shape[:2]  # Get height and width
        aspect_ratio = w / h  # Calculate aspect ratio
        return aspect_ratio

    def extract_features(self, img):
        """Extracts all interpretable features from the image."""
        text_presence = self.compute_text_presence(img)
        image_type = self.compute_image_type(img)
        dominant_color = self.compute_dominant_color(img)
        foreign_objects = self.compute_foreign_objects(img)
        image_clarity = self.compute_image_clarity(img)
        aspect_ratio = self.compute_aspect_ratio(img)
        features = [
            text_presence,
            image_type,
            dominant_color,
            foreign_objects,
            image_clarity,
            aspect_ratio
        ]  # List of feature values
        return np.array(features, dtype=np.float32)  # Return as numpy array


class FeatureExtractor:
    """Handles detailed feature extraction using histograms, LBP, and HOG."""

    def extract_color_histogram(self, img):
        """Extracts and normalizes color histograms for each channel."""
        hist = []
        for i in range(3):
            hist_channel = cv2.calcHist([img], [i], None, [32], [0, 256])  # Compute histogram
            hist.append(hist_channel)
        hist = np.concatenate(hist).flatten()  # Flatten and concatenate histograms
        hist /= (hist.sum() + 1e-7)  # Normalize histogram
        return hist

    def extract_texture_features(self, img):
        """Extracts and normalizes texture features using Local Binary Patterns."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        lbp = feature.local_binary_pattern(gray, P=24, R=8, method='uniform')  # Compute LBP
        hist, _ = np.histogram(
            lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))  # Compute histogram of LBP
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)  # Normalize histogram
        return hist

    def extract_edge_features(self, img):
        """Extracts HOG features from the grayscale resized image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        gray = cv2.resize(gray, (128, 128))  # Resize for consistency
        hog_features = hog(
            gray, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True)  # Compute HOG features
        return hog_features

    def extract_features(self, img):
        """Extracts and concatenates all detailed features from the image."""
        features = np.concatenate([
            self.extract_color_histogram(img),
            self.extract_texture_features(img),
            self.extract_edge_features(img)
        ])  # Concatenate all features
        return features  # Return combined feature vector


# ------------------ Main Application Class ------------------

class ImageClassifierApp:
    """Main application class for image classification with GUI."""

    def __init__(self, master):
        """Initializes the GUI and application state."""
        self.master = master
        master.title("Image Classifier")  # Set window title
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
        print(self.device)
        self.approach = tk.IntVar(value=2)  # Default approach F2_filters_ML
        self.setup_model()  # Initialize the selected model
        self.build_gui()  # Build GUI components
        self.reset_state()  # Initialize application state variables

    def reset_state(self):
        """Resets the application state variables."""
        self.image_paths = []  # List of image file paths
        self.unlabeled_indices = []  # Indices of unlabeled images
        self.current_image_index = None  # Currently displayed image index
        self.labels = []  # List of labels for images
        self.features = []  # List of feature vectors
        self.trained = False  # Flag indicating if the model is trained
        self.scaler = MinMaxScaler()  # Scaler for feature normalization
        self.label_count = 0  # Counter for labeled images
        self.count_label.config(text=f"Images Labelled: {self.label_count}")  # Update label count in GUI
        self.current_accuracy = None  # Current model accuracy
        self.iteration_results = []  # Results for benchmark iterations
        self.current_iteration = 1  # Current benchmark iteration
        # Only reset optimal_feature_values if not using Approach F1
        if self.approach.get() != 1:
            self.optimal_feature_values = None  # Reset optimal feature values

    # ------------------ Model Setup Functions ------------------

    def setup_model(self):
        """Sets up the model based on the selected approach."""
        if self.approach.get() == 1:
            # Approach F1: Threshold-based filters
            self.feature_extractor = InterpretableFilters(device=self.device)
            self.optimal_feature_values = [0.00, 0.1, 55, 0.00, 0.2, 1]  # Ideal values for features
            self.acceptable_ranges = [0.4, 0.3, 30, 2, 0.4, 0.5]  # Acceptable deviation ranges
        elif self.approach.get() == 2:
            # Approach F2: Interpretable filters with Logistic Regression
            self.feature_extractor = InterpretableFilters(device=self.device)
            self.model = LogisticRegression(max_iter=1000)  # Initialize Logistic Regression model
        elif self.approach.get() == 3:
            # Approach F3: Detailed features with Random Forest
            self.feature_extractor = FeatureExtractor()
            self.model = RandomForestClassifier()  # Initialize Random Forest model
        else:
            raise ValueError("Invalid approach selected")  # Handle invalid approach selection

    def change_approach(self):
        """Handles the change of approach selection by the user."""
        self.setup_model()  # Re-initialize the model based on new approach
        self.reset_state()  # Reset application state
        self.display_image()  # Update image display

    # ------------------ GUI Functions ------------------

    def build_gui(self):
        """Builds the graphical user interface components."""
        # Frame for approach selection
        self.approach_frame = tk.Frame(self.master)
        self.approach_frame.pack(pady=5)
        tk.Label(self.approach_frame, text="Select Approach:").pack(side=tk.LEFT)
        # Create radio buttons for each approach
        for i, name in enumerate(["F1_filters_threshold", "F2_filters_ML",
                                  "F3_feature_extraction"], 1):
            tk.Radiobutton(self.approach_frame, text=name, variable=self.approach,
                           value=i, command=self.change_approach).pack(side=tk.LEFT)

        # Canvas to display images
        self.canvas = tk.Canvas(self.master, width=600, height=400)
        self.canvas.pack()

        # Frame for control buttons
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(pady=5)

        # Button to mark image as relevant
        self.relevant_button = tk.Button(
            self.button_frame, text="Relevant", command=lambda: self.process_feedback(1))
        self.relevant_button.pack(side=tk.LEFT, padx=5)

        # Button to mark image as irrelevant
        self.irrelevant_button = tk.Button(
            self.button_frame, text="Irrelevant", command=lambda: self.process_feedback(0))
        self.irrelevant_button.pack(side=tk.LEFT, padx=5)

        # Button to load training images
        self.load_button = tk.Button(
            self.button_frame, text="Load Training Images", command=self.load_images)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Button to classify all images
        self.classify_button = tk.Button(
            self.button_frame, text="Classify All Images", command=self.classify_images)
        self.classify_button.pack(side=tk.LEFT, padx=5)

        # Button to save labeled data
        self.save_labels_button = tk.Button(
            self.button_frame, text="Save Labeled Data", command=self.save_labeled_data)
        self.save_labels_button.pack(side=tk.LEFT, padx=5)

        # Button to run benchmark simulation
        self.benchmark_button = tk.Button(
            self.button_frame, text="Run Benchmark", command=self.run_benchmark)
        self.benchmark_button.pack(side=tk.LEFT, padx=5)

        # Button to simulate max performance
        self.simulate_max_button = tk.Button(
            self.button_frame, text="Simulate Max", command=self.simulate_max)
        self.simulate_max_button.pack(side=tk.LEFT, padx=5)

        # Label to display model accuracy
        self.accuracy_label = tk.Label(self.master, text="Accuracy: N/A")
        self.accuracy_label.pack(pady=5)

        # Label to display number of labeled images
        self.count_label = tk.Label(self.master, text="Images Labelled: 0")
        self.count_label.pack(pady=5)

    def display_image(self):
        """Displays the current image in the GUI and prints its features if using Approach F1."""
        if self.current_image_index is not None:
            img_path = self.image_paths[self.current_image_index]
            img = Image.open(img_path)  # Open image
            img.thumbnail((600, 400))  # Resize for display
            self.photo = ImageTk.PhotoImage(img)  # Convert to PhotoImage
            self.canvas.delete("all")  # Clear previous image
            self.canvas.create_image(300, 200, anchor=tk.CENTER, image=self.photo)  # Display image

            # Extract and print features if Approach F1 is selected
            if self.approach.get() == 1:
                img_cv = cv2.imread(img_path)
                if img_cv is None:
                    print(f"Failed to read image: {img_path}")
                    return
                features = self.feature_extractor.extract_features(img_cv)
                feature_names = [
                    "Text Presence",
                    "Image Type",
                    "Dominant Color",
                    "Foreign Objects",
                    "Image Clarity",
                    "Aspect Ratio"
                ]
                print(f"\nFeatures for {os.path.basename(img_path)}:")
                for name, value in zip(feature_names, features):
                    print(f"  {name}: {value:.4f}")
        else:
            # Display message when no images are left
            self.canvas.delete("all")
            self.canvas.create_text(300, 200, text="No more images", font=("Arial", 24))

    # ------------------ Data Loading Functions ------------------

    def load_images(self):
        """Loads images from a selected directory for training."""
        print("Loading Training Images...")
        directory = filedialog.askdirectory(title="Select Training Image Folder")
        if directory:
            filenames = sorted(os.listdir(directory))
            # Filter for image files with valid extensions and parse labels from filenames
            self.image_paths = []
            self.labels = []
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    if f.startswith('1_'):
                        label = 1
                        self.labels.append(label)
                    elif f.startswith('0_'):
                        label = 0
                        self.labels.append(label)
                    else:
                        pass
                    path = os.path.join(directory, f)
                    self.image_paths.append(path)
            print(f"Total Training Images Loaded: {len(self.image_paths)}")

            if self.image_paths:
                self.unlabeled_indices = list(range(len(self.image_paths)))  # Initialize unlabeled indices
                self.features = [None] * len(self.image_paths)  # Initialize features list
                self.label_count = 0  # Reset label count
                self.count_label.config(text=f"Images Labelled: {self.label_count}")  # Update label count in GUI
                self.current_image_index = random.choice(self.unlabeled_indices)  # Select random image
                self.display_image()  # Display selected image
                print("Training images loaded successfully.")
            else:
                messagebox.showerror("Error", "No images found in the selected folder.")
                print("Load Images Failed: No valid images found.")

    def check_prelabelled(self):
        """Checks if all images are pre-labeled based on filename prefixes."""
        return all(os.path.basename(p).startswith(('1_', '0_')) for p in self.image_paths)

    # ------------------ Benchmarking Functions ------------------

    def run_benchmark(self):
        """Runs benchmarking for approaches F1-F3, repeats each 3 times, and averages the results per label count."""
        print("Starting Benchmark Run...")
        directory = filedialog.askdirectory(title="Select Benchmark Training Image Folder")
        if not directory:
            print("Benchmark Run Cancelled: No directory selected.")
            return

        print(f"Selected Benchmark Directory: {directory}")

        image_paths = []
        labels = []

        filenames = sorted(os.listdir(directory))
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                if f.startswith('1_'):
                    label = 1
                elif f.startswith('0_'):
                    label = 0
                else:
                    print(f"Skipping file with invalid prefix: {f}")
                    continue
                path = os.path.join(directory, f)
                image_paths.append(path)
                labels.append(label)

        print(f"Total Images Found: {len(image_paths)}")

        test_directory = '../sample_labelled_onychrom_100_test'
        if not os.path.exists(test_directory):
            messagebox.showerror("Error", f"Test directory '{test_directory}' does not exist.")
            print(f"Benchmark Run Terminated: Test directory '{test_directory}' not found.")
            return

        test_image_paths = []
        test_labels = []
        test_filenames = sorted(os.listdir(test_directory))
        for f in test_filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                if f.startswith('1_'):
                    label = 1
                elif f.startswith('0_'):
                    label = 0
                else:
                    print(f"Skipping test file with invalid prefix: {f}")
                    continue
                path = os.path.join(test_directory, f)
                test_image_paths.append(path)
                test_labels.append(label)

        print(f"Total Test Images Found: {len(test_image_paths)}")

        approach_results = {}
        for approach_num in range(1, 4):
            print(f"\n--- Benchmarking Approach F{approach_num} ---")
            all_runs_results = []
            for run in range(1, 4):
                print(f"\nRun {run} for Approach F{approach_num}")
                self.approach.set(approach_num)
                self.setup_model()
                self.reset_state()
                self.image_paths = image_paths.copy()
                true_labels = labels.copy()
                self.unlabeled_indices = list(range(len(self.image_paths)))
                random.shuffle(self.unlabeled_indices)

                # Initialize features and labels lists with the correct length
                self.features = [None] * len(self.image_paths)
                self.labels = [None] * len(self.image_paths)

                self.label_count = 0
                self.count_label.config(text=f"Images Labelled: {self.label_count}")
                self.current_iteration = run

                iteration_accuracies = []
                training_times = []
                testing_times = []
                label_counts = []

                while self.unlabeled_indices:
                    idx = self.unlabeled_indices.pop()
                    self.current_image_index = idx

                    img_path = self.image_paths[self.current_image_index]
                    label = true_labels[self.current_image_index]
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Failed to read image: {img_path}")
                        continue
                    self.process_image(img, label)
                    self.label_count += 1
                    print(f"Labeled {self.label_count}/{len(self.image_paths)}: {'Relevant' if label == 1 else 'Irrelevant'}")

                    if (self.label_count >= 10 and self.label_count % 2 == 0) or self.label_count == len(self.image_paths):
                        print(f"Updating model after {self.label_count} labels...")
                        start_time = time.time()
                        self.update_model()
                        training_time = time.time() - start_time

                        start_time = time.time()
                        test_accuracy = self.evaluate_on_test_set(test_image_paths, test_labels)
                        testing_time = time.time() - start_time

                        if test_accuracy is not None:
                            print(f"Run {run}, Labels {self.label_count}, Test Accuracy: {test_accuracy:.4f}, "
                                  f"Training Time: {training_time:.2f} seconds, Testing Time: {testing_time:.2f} seconds")
                            iteration_accuracies.append(test_accuracy)
                            training_times.append(training_time)
                            testing_times.append(testing_time)
                            label_counts.append(self.label_count)
                        else:
                            print("Test Accuracy could not be calculated.")

                # Store the accuracies and times for this run
                run_results = pd.DataFrame({
                    'Number of Labelled Images': label_counts,
                    f'Accuracy_F{approach_num}': iteration_accuracies,
                    f'Training_Time_F{approach_num}': training_times,
                    f'Testing_Time_F{approach_num}': testing_times
                })
                all_runs_results.append(run_results)

                print(f"Completed Run {run} for Approach F{approach_num}")

            # Average the accuracies and times over all runs
            combined_df = pd.concat(all_runs_results)
            averaged_results = combined_df.groupby('Number of Labelled Images').mean().reset_index()
            approach_results[f'Approach_F{approach_num}'] = averaged_results

            print(f"Averaged Results for Approach F{approach_num}:\n{averaged_results}")

        print("\nCombining benchmark results...")
        # Combine results from different approaches
        final_df = None
        for approach_num in range(1, 4):
            approach_df = approach_results[f'Approach_F{approach_num}']
            if final_df is None:
                final_df = approach_df
            else:
                final_df = pd.merge(final_df, approach_df, on='Number of Labelled Images', how='outer')

        output_file = 'benchmark_results_average.xlsx'
        final_df.to_excel(output_file, sheet_name='Benchmark Average Results', index=False)
        print(f"Benchmark average results saved to '{output_file}'.")

        messagebox.showinfo("Benchmark Complete", f"Benchmark average results saved to {output_file}")
        self.change_approach()

    def evaluate_on_test_set(self, test_image_paths, test_labels):
        """Evaluates the current model on a random subset of 10 test images and returns accuracy."""
        # Ensure there are enough samples
        if len(test_labels) < 10:
            sampled_indices = list(range(len(test_labels)))
        else:
            sampled_indices = random.sample(range(len(test_labels)), 10)

        preds = []
        true_labels_eval = []
        for idx in sampled_indices:
            img_path = test_image_paths[idx]
            true_label = test_labels[idx]
            img = cv2.imread(img_path)
            if img is None:
                continue
            if self.approach.get() in [1, 2, 3]:
                features = self.feature_extractor.extract_features(img)
                if self.approach.get() == 1:
                    prob = self.calculate_threshold_score(features)
                else:
                    features_scaled = self.scaler.transform([features])
                    prob = self.model.predict_proba(features_scaled)[0][1]
            else:
                continue  # Approaches F4 and F5 are removed

            prob = max(0, min(prob, 1))
            pred_label = 1 if prob >= 0.5 else 0
            preds.append(pred_label)
            true_labels_eval.append(true_label)

        if not true_labels_eval:
            return None
        accuracy = accuracy_score(true_labels_eval, preds)
        return accuracy

    def simulate_max(self):
        """Simulates training on the full pre-labeled dataset and evaluates on test set."""
        print("Starting Simulate Max...")
        # Prompt the user to select a directory for pre-labeled training images
        directory = filedialog.askdirectory(title="Select Pre-labeled Training Image Folder")
        if not directory:
            print("Simulate Max Cancelled: No directory selected.")
            return

        print(f"Selected Simulation Directory: {directory}")

        # Prepare image paths and labels from filenames in the selected directory
        image_paths = []
        labels = []

        filenames = sorted(os.listdir(directory))
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                if f.startswith('1_'):
                    label = 1
                elif f.startswith('0_'):
                    label = 0
                else:
                    print(f"Skipping file with invalid prefix: {f}")
                    continue  # Skip files without proper prefix
                path = os.path.join(directory, f)
                image_paths.append(path)
                labels.append(label)
                print(f"Added Image: {f} with Label: {'Relevant' if label == 1 else 'Irrelevant'}")

        print(f"Total Images Found for Simulation: {len(image_paths)}")

        if not image_paths:
            messagebox.showerror("Error", "No images found in the selected folder.")
            print("Simulate Max Terminated: No images to process.")
            return

        # Load the test set images and labels from the specified folder
        test_directory = '../sample_labelled_onychrom_100_test'
        if not os.path.exists(test_directory):
            messagebox.showerror("Error", f"Test directory '{test_directory}' does not exist.")
            print(f"Simulate Max Terminated: Test directory '{test_directory}' not found.")
            return

        test_image_paths = []
        test_labels = []
        test_filenames = sorted(os.listdir(test_directory))
        for f in test_filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                if f.startswith('1_'):
                    label = 1
                elif f.startswith('0_'):
                    label = 0
                else:
                    print(f"Skipping test file with invalid prefix: {f}")
                    continue  # Skip files without proper prefix
                path = os.path.join(test_directory, f)
                test_image_paths.append(path)
                test_labels.append(label)
                print(f"Added Test Image: {f} with Label: {'Relevant' if label == 1 else 'Irrelevant'}")

        print(f"Total Test Images Found: {len(test_image_paths)}")

        # Set image paths and labels
        self.image_paths = image_paths
        self.labels = labels
        self.features = [None] * len(self.image_paths)

        # Process all images and extract features
        print("Extracting features from all training images...")
        start_train_time = time.time()
        for idx, img_path in enumerate(self.image_paths, 1):
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
            self.current_image_index = idx - 1  # Zero-based index
            self.process_image(img, self.labels[idx - 1])
            print(f"Processed Image {idx}/{len(self.image_paths)}: {os.path.basename(img_path)}")
        total_training_time = time.time() - start_train_time
        print(f"Total Training Time: {total_training_time:.2f} seconds")

        # Update the model with the full dataset
        print("Training the model with the full dataset...")
        self.update_model()

        # Evaluate the model on the test set
        print("Evaluating the model on the test set...")
        start_test_time = time.time()

        preds = []
        true_labels_eval = []
        probs = []

        for img_path, true_label in zip(test_image_paths, test_labels):
            img = cv2.imread(img_path)
            if img is None:
                continue
            if self.approach.get() in [1, 2, 3]:
                features = self.feature_extractor.extract_features(img)
                if self.approach.get() == 1:
                    prob = self.calculate_threshold_score(features)
                else:
                    features_scaled = self.scaler.transform([features])
                    prob = self.model.predict_proba(features_scaled)[0][1]
            else:
                continue  # Approaches F4 and F5 are removed

            prob = max(0, min(prob, 1))
            pred_label = 1 if prob >= 0.5 else 0
            preds.append(pred_label)
            true_labels_eval.append(true_label)
            probs.append(prob)

        total_testing_time = time.time() - start_test_time
        print(f"Total Testing Time: {total_testing_time:.2f} seconds")

        if not true_labels_eval:
            print("No data to evaluate.")
            messagebox.showinfo("Simulation Complete", "No data to evaluate.")
            return

        # Calculate metrics
        accuracy = accuracy_score(true_labels_eval, preds)
        recall = recall_score(true_labels_eval, preds)
        f1 = f1_score(true_labels_eval, preds)
        conf_matrix = confusion_matrix(true_labels_eval, preds)
        fpr, tpr, thresholds = roc_curve(true_labels_eval, probs)
        roc_auc = roc_auc_score(true_labels_eval, probs)

        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"ROC AUC: {roc_auc:.4f}")

        # Save metrics and data to Excel
        output_file = 'simulate_max_results.xlsx'
        with pd.ExcelWriter(output_file) as writer:
            # Save metrics
            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Recall', 'F1 Score', 'ROC AUC', 'Total Training Time', 'Total Testing Time'],
                'Value': [accuracy, recall, f1, roc_auc, total_training_time, total_testing_time]
            })
            metrics_df.to_excel(writer, sheet_name='Metrics', index=False)

            # Save confusion matrix
            conf_matrix_df = pd.DataFrame(conf_matrix,
                                          index=['Actual Negative', 'Actual Positive'],
                                          columns=['Predicted Negative', 'Predicted Positive'])
            conf_matrix_df.to_excel(writer, sheet_name='Confusion Matrix')

            # Save ROC curve data
            roc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr, 'Thresholds': thresholds})
            roc_df.to_excel(writer, sheet_name='ROC Curve Data', index=False)

            # Save predictions and true labels
            results_df = pd.DataFrame({
                'Image Path': test_image_paths,
                'True Label': true_labels_eval,
                'Predicted Label': preds,
                'Predicted Probability': probs
            })
            results_df.to_excel(writer, sheet_name='Predictions', index=False)

        print(f"Simulation results saved to '{output_file}'.")
        messagebox.showinfo("Simulation Complete", f"Test Accuracy: {accuracy:.4f}\nResults saved to '{output_file}'")

    # ------------------ Feedback Processing Functions ------------------

    def process_feedback(self, label_value):
        """Processes user feedback when an image is marked as relevant or irrelevant."""
        if self.current_image_index is not None:
            img_path = self.image_paths[self.current_image_index]
            img = cv2.imread(img_path)  # Read the image
            self.process_image(img, label_value)  # Process the image with provided label
            self.label_count += 1  # Increment label count
            self.count_label.config(text=f"Images Labelled: {self.label_count}")  # Update label count in GUI
            self.unlabeled_indices.remove(self.current_image_index)  # Remove from unlabeled list
            print(f"Image {self.label_count}: {'Relevant' if label_value == 1 else 'Irrelevant'} labeled for {img_path}")
            if self.label_count >= 5:
                print("Label count reached threshold. Updating model...")
                self.update_model()  # Update the model if enough labels
            if self.unlabeled_indices:
                self.current_image_index = random.choice(self.unlabeled_indices)  # Select next image
                self.display_image()  # Display selected image
            else:
                self.current_image_index = None  # No more images
                self.display_image()  # Update display
                print("All images have been labeled.")
        else:
            messagebox.showinfo("Info", "No more images to label.")
            print("Process Feedback: No images left to label.")

    def process_image(self, img, label):
        """Extracts features and assigns labels based on the selected approach."""
        if self.approach.get() in [1, 2, 3]:
            features = self.feature_extractor.extract_features(img)  # Extract features for ML models
            self.features[self.current_image_index] = features  # Store extracted features
            self.labels[self.current_image_index] = label  # Assign label

    # ------------------ Model Training and Updating Functions ------------------

    def update_model(self):
        """Trains or updates the model based on the extracted features and labels."""
        print("Updating Model...")
        if self.approach.get() == 1:
            # Approach F1: Threshold-based classification
            print("Approach F1: Threshold-based Classification")
            self.trained = True
            # Pair labels with their corresponding features
            paired = [(label, feat) for label, feat in zip(self.labels, self.features) if feat is not None]
            if not paired:
                print("No labeled features available for training.")
                self.current_accuracy = None
                return
            y_true, X = zip(*paired)
            y_true = np.array(y_true)
            X = np.array(X)
            y_pred = [1 if self.calculate_threshold_score(feat) >= 0.5 else 0 for feat in X]
            # Calculate accuracy
            accuracy = accuracy_score(y_true, y_pred)
            print(f"F1 Accuracy: {accuracy:.4f}")
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}")
            self.current_accuracy = accuracy
        elif self.approach.get() in [2, 3]:
            # Approaches F2 and F3: Machine Learning models
            approach_name = "F2: Interpretable Filters with Logistic Regression" if self.approach.get() == 2 else "F3: Detailed Features with Random Forest"
            print(f"Approach {approach_name}")
            # Ensure X and y have the same length by selecting only indices with non-None features
            valid_indices = [i for i, feat in enumerate(self.features) if feat is not None]
            X = np.array([self.features[i] for i in valid_indices])
            y = np.array([self.labels[i] for i in valid_indices])
            if len(np.unique(y)) < 2:
                print("Not enough classes to train the model.")
                self.current_accuracy = None
                return
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.model.fit(X_scaled, y)

            if self.approach.get() == 2:
                # Calculate optimal_feature_values as the mean of features for the positive class
                positive_indices = np.where(y == 1)[0]
                if len(positive_indices) == 0:
                    print("No positive samples to calculate optimal feature values.")
                    self.optimal_feature_values = None
                else:
                    # Mean of scaled features
                    self.optimal_feature_values = X_scaled[positive_indices].mean(axis=0)
                    # Inverse transform to get raw optimal feature values
                    self.optimal_feature_values_raw = self.scaler.inverse_transform([self.optimal_feature_values])[0]
                    print(f"Optimal Feature Values (Scaled): {self.optimal_feature_values}")
                    print(f"Optimal Feature Values (Raw): {self.optimal_feature_values_raw}")

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=kf)
            accuracy = np.mean(cv_scores)
            print(f"Cross-Validation Accuracy: {accuracy:.4f}")
            self.accuracy_label.config(text=f"Accuracy: {accuracy:.2f}")
            self.current_accuracy = accuracy
            self.trained = True
        else:
            # Approaches F4 and F5 are removed
            pass
        print("Model Update Complete.")


    def calculate_threshold_score(self, features):
        """Calculates a combined score based on feature deviations from optimal feature values using a parabolic drop-off."""
        if self.optimal_feature_values is None:
            print("Optimal feature values not set.")
            return 0.0
        deviations = []
        for i, (f_val, optimal_val) in enumerate(zip(features, self.optimal_feature_values)):
            acceptable_range = self.acceptable_ranges[i]
            if acceptable_range == 0:
                deviation = 1.0 if f_val == optimal_val else 0.0
                reason = "Exact match required due to zero acceptable range."
            else:
                squared_difference = (f_val - optimal_val) ** 2
                normalized_squared_difference = squared_difference / (acceptable_range ** 2)
                deviation = max(0.0, 1.0 - normalized_squared_difference)
                reason = f"Computed deviation using parabolic drop-off."
            deviations.append(deviation)
        overall_score = sum(deviations) / len(deviations)
        return overall_score

    # ------------------ Classification and Saving Functions ------------------

    def classify_images(self):
        """Classifies all loaded images and saves the results with score prefixes."""
        if not self.image_paths:
            messagebox.showerror("Error", "No images loaded.")
            return
        if not self.trained:
            messagebox.showerror("Error", "Model is not ready yet. Please label more images.")
            return

        # Define output directory for classified images
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)
        # Clear existing files in the output directory
        for f in os.listdir(output_dir):
            file_path = os.path.join(output_dir, f)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # Duplicate detection setup
        print("Checking for duplicate images...")
        hashes = {}
        duplicates = 0
        unique_image_paths = []

        for img_path in self.image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue  # Skip if image failed to load

            # Compute hash of the image
            img_hash = self.dhash(img)
            if img_hash in hashes:
                # Duplicate found
                duplicates += 1
            else:
                hashes[img_hash] = img_path
                unique_image_paths.append(img_path)

        print(f"Duplicate detection complete. Found {duplicates} duplicate images. Ignoring duplicates for classification.")

        # Now proceed with classification on unique images only
        for img_path in unique_image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
            if self.approach.get() in [1, 2, 3]:
                if self.approach.get() == 1:
                    features = self.feature_extractor.extract_features(img)
                    prob = self.calculate_threshold_score(features)
                else:
                    features = self.feature_extractor.extract_features(img)
                    features_scaled = self.scaler.transform([features])
                    prob = self.model.predict_proba(features_scaled)[0][1]
            else:
                continue  # Approaches F4 and F5 are removed

            prob = max(0, min(prob, 1))
            score_percent = int(prob * 100)
            filename = os.path.basename(img_path)
            new_filename = f"{score_percent:03d}_{filename}"
            output_path = os.path.join(output_dir, new_filename)
            shutil.copy(img_path, output_path)
        messagebox.showinfo("Done", "Classification complete. Results saved in 'output' directory.")

    def save_labeled_data(self):
        """Saves labeled images to a separate directory with label prefixes."""
        if not self.labels:
            messagebox.showerror("Error", "No labels to save.")  # Error if no labels available
            return
        # Define output directory for labeled images
        output_dir = os.path.join(os.path.dirname(__file__), "output_labelled")
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
        # Clear existing files in the output directory
        for f in os.listdir(output_dir):
            file_path = os.path.join(output_dir, f)
            if os.path.isfile(file_path):
                os.unlink(file_path)  # Delete file
        # Iterate through all labels and save appropriately
        for idx, label in enumerate(self.labels):
            if label is not None:
                img_path = self.image_paths[idx]
                filename = os.path.basename(img_path)
                new_filename = f"{label}_{filename}"  # Prefix filename with label
                output_path = os.path.join(output_dir, new_filename)  # Define new file path
                shutil.copy(img_path, output_path)  # Copy image to labeled directory
        messagebox.showinfo("Done", "Labeled images saved in 'output_labelled' directory.")

    def dhash(self, image, hash_size=8):
        """Computes the difference hash for an image."""
        resized = cv2.resize(image, (hash_size + 1, hash_size))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        diff = gray[:, 1:] > gray[:, :-1]
        hash_value = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        return hash_value


# ------------------ Running the Application ------------------

if __name__ == "__main__":
    root = tk.Tk()  # Create main window
    app = ImageClassifierApp(root)  # Initialize application
    root.mainloop()  # Run the GUI event loop
