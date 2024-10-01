import os
import time
import numpy as np
from glob import glob
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import shutil
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# Set up logging to console with a level of INFO
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- Helper Classes ----------------

# Class to handle data loading and preprocessing
class DatasetLoader:
    def __init__(self, dataset_path, img_size=(128, 128), output_standardised_path=None, output_augmented_path=None, color_mode='L'):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.images = []
        self.labels = []
        self.filenames = []  # Store filenames
        self.color_mode = color_mode  # 'L' for grayscale, 'RGB' for color

        if self.color_mode == 'RGB':
            self.channels = 3
        else:
            self.channels = 1

        self.output_standardised_path = output_standardised_path or os.path.join(os.getcwd(), 'output_standardised')
        self.output_augmented_path = output_augmented_path or os.path.join(os.getcwd(), 'output_augmented')

    def load_processed_data(self):
        logging.info("Loading processed data from %s", self.output_augmented_path)
        image_files = glob(os.path.join(self.output_augmented_path, '*.jpg'))
        total_images = len(image_files)
        logging.info("Found %d processed images.", total_images)
        self.images = []
        self.labels = []
        self.filenames = []
        for img_file in image_files:
            image = Image.open(img_file)
            if self.color_mode == 'RGB':
                image = image.convert('RGB')
                image = image.resize(self.img_size)
                image_array = np.array(image)
            else:
                image = image.convert('L')
                image = image.resize(self.img_size)
                image_array = np.array(image).reshape(self.img_size[0], self.img_size[1], 1)
            label_str = os.path.basename(img_file).split('_')[0]  # Extract label from filename
            label = int(label_str)
            self.images.append(image_array)
            self.labels.append(label)
            self.filenames.append(os.path.basename(img_file))
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        logging.info("Processed data loaded.")

    # Load images, extract labels, and save standardized images
    def load_data(self, num_images=None):
        logging.info("Starting to load data from %s", self.dataset_path)
        image_files = glob(os.path.join(self.dataset_path, '*.jpg'))
        if num_images is not None:
            image_files = image_files[:num_images]
        total_images = len(image_files)
        logging.info("Found %d images to process.", total_images)
        if os.path.exists(self.output_standardised_path):
            shutil.rmtree(self.output_standardised_path)
        os.makedirs(self.output_standardised_path, exist_ok=True)

        for idx, img_file in enumerate(image_files):
            if idx % 100 == 0 and idx != 0:
                logging.info("Processed %d/%d images.", idx, total_images)
            label = int(os.path.basename(img_file).split('_')[0])  # Extract label from filename
            image = Image.open(img_file).convert(self.color_mode)
            image = image.resize(self.img_size)

            standardized_filename = os.path.basename(img_file).replace('.jpg', '_0.jpg')
            standardized_filepath = os.path.join(self.output_standardised_path, standardized_filename)
            image.save(standardized_filepath)

            self.images.append(np.array(image))
            self.labels.append(label)
            self.filenames.append(standardized_filename)  # Store standardized filename
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        logging.info("Completed loading data.")

    # Augment data, save augmented images, and enhance augmentation techniques
    def augment_data(self):
        logging.info("Starting data augmentation.")
        augmented_images = []
        augmented_labels = []
        augmented_filenames = []
        if os.path.exists(self.output_augmented_path):
            shutil.rmtree(self.output_augmented_path)
        os.makedirs(self.output_augmented_path, exist_ok=True)

        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.7, 1.3],
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )

        total_images = len(self.images)
        for idx, (image, label, filename) in enumerate(zip(self.images, self.labels, self.filenames)):
            if idx % 100 == 0 and idx != 0:
                logging.info("Augmented %d/%d images.", idx, total_images)

            # Ensure image has consistent shape
            if self.color_mode == 'RGB':
                if image.ndim == 2:
                    image = np.stack((image,) * 3, axis=-1)  # Convert grayscale to RGB
            else:
                if image.ndim == 3 and image.shape[2] == 3:
                    image = image[:, :, 0]  # Convert RGB to grayscale
                image = image.reshape(self.img_size[0], self.img_size[1], 1)

            image_reshaped = image.reshape((1,) + image.shape)

            # Save the original standardized image to the augmented folder
            standardized_filepath = os.path.join(self.output_augmented_path, filename)
            Image.fromarray(image.squeeze().astype('uint8')).save(standardized_filepath)
            augmented_images.append(image)
            augmented_labels.append(label)
            augmented_filenames.append(filename)

            base_name, ext = os.path.splitext(filename)
            # Extract current suffix number and increment for augmented images
            parts = base_name.rsplit('_', 1)
            if len(parts) == 2 and parts[1].isdigit():
                base_filename = parts[0]
                start_num = int(parts[1])
            else:
                base_filename = base_name
                start_num = 0

            aug_iter = datagen.flow(image_reshaped, batch_size=1)
            for aug_num in range(1, 1):
                aug_image = next(aug_iter)[0].astype('uint8')
                augmented_images.append(aug_image)
                augmented_labels.append(label)
                new_num = start_num + aug_num
                aug_filename = f"{base_filename}_{new_num}{ext}"
                aug_filepath = os.path.join(self.output_augmented_path, aug_filename)
                Image.fromarray(aug_image.squeeze()).save(aug_filepath)
                augmented_filenames.append(aug_filename)

        self.images = np.array(augmented_images)
        self.labels = np.array(augmented_labels)
        self.filenames = augmented_filenames
        logging.info("Data augmentation completed.")

    # Prepare training and validation data
    def prepare_train_val_data(self, val_size=0.3):
        logging.info("Preparing training and validation data.")
        self.images = np.array(self.images) / 255.0  # Normalize
        X_train, X_val, y_train, y_val = train_test_split(
            self.images, self.labels, test_size=val_size, random_state=42
        )
        X_train = X_train.reshape((-1, self.img_size[0], self.img_size[1], self.channels))
        X_val = X_val.reshape((-1, self.img_size[0], self.img_size[1], self.channels))
        logging.info("Data preparation for training and validation completed.")
        return X_train, X_val, y_train, y_val

    # Prepare test data without splitting
    def prepare_test_data(self):
        logging.info("Preparing test data.")
        self.images = self.images / 255.0  # Normalize pixel values
        X_test = self.images.reshape((-1, self.img_size[0], self.img_size[1], self.channels))
        logging.info("Test data preparation completed.")
        return X_test, self.labels

# Base class for models
class BaseModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = None

    # Compile the model
    def compile_model(self):
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    # Train the model
    def train(self, X_train, y_train, X_val, y_val):
        logging.info("Starting training.")
        start_time = time.time()
        self.model.fit(
            X_train, y_train,
            epochs=10,
            validation_data=(X_val, y_val),
            batch_size=32
        )
        training_time = time.time() - start_time
        logging.info(f"Training completed in {training_time:.2f} seconds.")
        return training_time

    # Evaluate the model
    def evaluate(self, X_test, y_test):
        logging.info("Evaluating the model.")
        start_time = time.time()
        y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        testing_time = time.time() - start_time
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        logging.info(f"Evaluation completed. Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Testing time: {testing_time:.2f} seconds.")
        return accuracy, f1, testing_time


# Simple CNN Model
class SimpleCNNModel(BaseModel):
    def build_model(self):
        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        self.compile_model()


# Tuned CNN Model
class TunedCNNModel(BaseModel):
    def build_model(self):
        self.model = models.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        self.compile_model()


# Transfer Learning Model using MobileNetV2
class TransferLearningModel(BaseModel):
    def build_model(self):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(1, activation='sigmoid')
        ])
        self.compile_model()

# ---------------- Test Class ----------------

class Tests:
    # Test 1: Standardize and augment a single image
    def test_standardise_single_image(self):
        logging.info("Test 1: Starting standardization and augmentation of a single image.")
        dataset_path = '../P2_filtered_labelled_100'
        loader = DatasetLoader(dataset_path)
        loader.load_data(num_images=1)
        loader.augment_data()
        logging.info("Test 1: Standardization and augmentation on a single image completed.")

    # Test 2: Standardize and augment all images in the first folder
    def test_standardise_all_images(self):
        logging.info("Test 2: Starting standardization and augmentation of all images.")
        dataset_path = '../P2_filtered_labelled_100'
        loader = DatasetLoader(dataset_path)
        loader.load_data()
        loader.augment_data()
        logging.info("Test 2: Standardization and augmentation on all images completed.")

    # Test 3: Train a single CNN method on a single folder and view results
    def test_single_cnn(self):
        logging.info("Test 3: Starting training of a single CNN model.")
        dataset_path_train = '../P2_filtered_labelled_100'
        dataset_path_test = '../labelled_test_set_100'
        model_choice = 'transfer_learning'  # Choose the model to test

        model_types = {
            'simple_cnn': SimpleCNNModel,
            'tuned_cnn': TunedCNNModel,
            'transfer_learning': TransferLearningModel
        }

        # Set color mode based on model choice
        if model_choice == 'transfer_learning':
            color_mode = 'RGB'
        else:
            color_mode = 'L'

        # Load and prepare training data
        loader_train = DatasetLoader(
            dataset_path_train,
            output_standardised_path='output_standardised_train',
            output_augmented_path='output_augmented_train',
            color_mode=color_mode
        )
        if os.path.exists(loader_train.output_augmented_path) and len(glob(os.path.join(loader_train.output_augmented_path, '*.jpg'))) > 0:
            loader_train.load_processed_data()
        else:
            loader_train.load_data()
            loader_train.augment_data()
        X_train, X_val, y_train, y_val = loader_train.prepare_train_val_data()

        # Load and prepare test data from external dataset
        loader_test = DatasetLoader(
            dataset_path_test,
            output_standardised_path='output_standardised_test',
            output_augmented_path='output_augmented_test',
            color_mode=color_mode
        )
        if os.path.exists(loader_test.output_augmented_path) and len(glob(os.path.join(loader_test.output_augmented_path, '*.jpg'))) > 0:
            loader_test.load_processed_data()
        else:
            loader_test.load_data()
            loader_test.augment_data()
        X_test, y_test = loader_test.prepare_test_data()

        input_shape = (loader_train.img_size[0], loader_train.img_size[1], loader_train.channels)
        model_class = model_types[model_choice]
        model = model_class(input_shape)
        model.build_model()

        model.train(X_train, y_train, X_val, y_val)
        model.evaluate(X_test, y_test)
        logging.info("Test 3: Single CNN model training and evaluation completed.")

    # Test 4: Run full benchmark across all methods and folders
    def run_full_benchmark(self):
        logging.info("Test 4: Starting full benchmark across all models and datasets.")
        dataset_paths_train = [
            '../P1_scraped_labelled_12408',
            '../P2_filtered_labelled_5396',
            '../P3_manual_labelled_3021'
        ]
        dataset_path_test = '../labelled_test_set_100'  # External test dataset
        model_choices = ['simple_cnn', 'tuned_cnn', 'transfer_learning']

        model_types = {
            'simple_cnn': SimpleCNNModel,
            'tuned_cnn': TunedCNNModel,
            'transfer_learning': TransferLearningModel
        }

        results = []  # List to store the results

        for dataset_path_train in dataset_paths_train:
            dataset_name = os.path.basename(dataset_path_train)
            # Define output directories
            output_standardised_train = 'output_standardised_train'
            output_augmented_train = 'output_augmented_train'
            output_standardised_test = 'output_standardised_test'
            output_augmented_test = 'output_augmented_test'

            # Wipe output directories for training data
            for folder in [output_standardised_train, output_augmented_train]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    logging.info(f"Wiped folder '{folder}'.")

            # Wipe output directories for test data
            for folder in [output_standardised_test, output_augmented_test]:
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                    logging.info(f"Wiped folder '{folder}'.")

            for model_choice in model_choices:
                logging.info(f"Running model '{model_choice}' on dataset '{dataset_name}'")

                # Set color mode based on model choice
                if model_choice == 'transfer_learning':
                    color_mode = 'RGB'
                else:
                    color_mode = 'L'

                # Initialize DatasetLoader for training data
                loader_train = DatasetLoader(
                    dataset_path_train,
                    output_standardised_path=output_standardised_train,
                    output_augmented_path=output_augmented_train,
                    color_mode=color_mode
                )

                # Perform standardization and augmentation
                loader_train.load_data()
                loader_train.augment_data()
                X_train, X_val, y_train, y_val = loader_train.prepare_train_val_data()

                # Initialize DatasetLoader for test data
                loader_test = DatasetLoader(
                    dataset_path_test,
                    output_standardised_path=output_standardised_test,
                    output_augmented_path=output_augmented_test,
                    color_mode=color_mode
                )

                # Perform standardization and augmentation on test data
                loader_test.load_data()
                loader_test.augment_data()
                X_test, y_test = loader_test.prepare_test_data()

                input_shape = (loader_train.img_size[0], loader_train.img_size[1], loader_train.channels)

                model_class = model_types[model_choice]
                model = model_class(input_shape)
                model.build_model()

                training_time = model.train(X_train, y_train, X_val, y_val)
                accuracy, f1, testing_time = model.evaluate(X_test, y_test)
                logging.info(f"Completed model '{model_choice}' on dataset '{dataset_name}'")

                results.append({
                    'dataset': dataset_name,
                    'model': model_choice,
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'training_time': training_time,
                    'testing_time': testing_time
                })

        # Create a DataFrame and save to Excel
        df_results = pd.DataFrame(results)
        df_results.to_excel('benchmark_results.xlsx', index=False)
        logging.info("Test 4: Full benchmark completed. Results saved to 'benchmark_results.xlsx'.")

# ---------------- Main Function ----------------
def main():
    tests = Tests()

    # Test 1: Standardize and augment a single image
    tests.test_standardise_single_image()

    # Test 2: Standardize and augment all images in the first folder
    # tests.test_standardise_all_images()

    # Test 3: Train a single CNN method on a single folder and view results
    # tests.test_single_cnn()

    # Test 4: Run full benchmark across all methods and folders
    # tests.run_full_benchmark()


if __name__ == "__main__":
    main()
