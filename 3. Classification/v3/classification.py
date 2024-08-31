import os
import cv2
import numpy as np
from skimage import exposure
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
import shutil

# Constants
# Standardisation
IMG_SIZE = (128, 128)
# Augmentation
AUGMENTATION_FACTOR = 5
# Classification
BATCH_SIZE = 8
EPOCHS = 10
NUM_CLASSES = 2
TEST_SIZE = 0.2  # 20% of data will be used for testing
# Cross-Validation
NUM_FOLDS = 5
# Hyperparameter Tuning
HYPERPARAMETER_TUNING = True
LEARNING_RATES = [0.001, 0.0001]
FILTERS = [32, 64]
DROPOUT_RATES = [0.5, 0.3]

def prepare_output_directory(output_dir):
    """Clears the contents of the output directory if it exists, but does not remove the directory itself."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

class ImageStandardiser:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def standardise_image(self, image):
        """standardises the image by applying resizing, normalization, mean subtraction, grayscale conversion, histogram equalization, and noise reduction."""
        resized_image = cv2.resize(image, self.target_size)
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        normalised_image = gray_image / 255.0
        mean_value = np.mean(normalised_image)
        mean_subtracted_image = normalised_image - mean_value
        equalised_image = exposure.equalize_hist(mean_subtracted_image)
        standardised_image = cv2.GaussianBlur(equalised_image, (5, 5), 0)
        return standardised_image

    def standardise_images_in_directory(self, input_dir, output_dir):
        """standardises all images in the input directory and saves them to the output directory."""
        prepare_output_directory(output_dir)
        for filename in os.listdir(input_dir):
            file_path = os.path.join(input_dir, filename)
            if os.path.isdir(file_path):
                continue
            image = cv2.imread(file_path)
            if image is None:
                print(f"Failed to read {filename}. Skipping...")
                continue
            standardised_image = self.standardise_image(image)
            standardised_filename = os.path.join(output_dir, filename)
            cv2.imwrite(standardised_filename, (standardised_image * 255).astype(np.uint8))
            print(f"standardised and saved {filename} to {standardised_filename}")

class ImageAugmentor:
    def __init__(self):
        self.datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest'
        )

    def augment_images(self, input_dir, output_dir, num_augmented=5):
        """Applies augmentation to images in the input directory and saves them to the output directory."""
        prepare_output_directory(output_dir)
        total_augmented_images = 0
        for filename in os.listdir(input_dir):
            if filename.endswith('.jpg'):
                img_path = os.path.join(input_dir, filename)
                img = load_img(img_path, color_mode='grayscale')
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)

                # Save the original image with a suffix '_0'
                base_filename = os.path.splitext(filename)[0]
                original_save_path = os.path.join(output_dir, f"{base_filename}_0.jpg")
                img.save(original_save_path)
                total_augmented_images += 1

                # Generate and save augmented images with suffixes '_1', '_2', etc.
                i = 1
                for batch in self.datagen.flow(x, batch_size=1):
                    augmented_save_path = os.path.join(output_dir, f"{base_filename}_{i}.jpg")
                    img_to_save = batch[0].astype('uint8')
                    img_to_save = img_to_save.squeeze()  # Remove the channel dimension for saving
                    cv2.imwrite(augmented_save_path, img_to_save)
                    i += 1
                    total_augmented_images += 1
                    if i > num_augmented:
                        break
        print(f"Augmentation complete. Total augmented images: {total_augmented_images}")

class CNNClassifier:
    def __init__(self, input_shape, num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def _build_model(self, input_shape, num_classes, filters, dropout_rate, learning_rate):
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(filters, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 2, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(dropout_rate),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_and_evaluate(self, X, y, num_folds=5):
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        best_accuracy = 0
        best_config = None

        for lr in LEARNING_RATES:
            for filters in FILTERS:
                for dropout_rate in DROPOUT_RATES:
                    fold_accuracies = []
                    for train_index, test_index in kf.split(X):
                        X_train, X_test = X[train_index], X[test_index]
                        y_train, y_test = y[train_index], y[test_index]

                        y_train_cat = to_categorical(y_train, NUM_CLASSES)
                        y_test_cat = to_categorical(y_test, NUM_CLASSES)

                        model = self._build_model(self.input_shape, self.num_classes, filters, dropout_rate, lr)
                        model.fit(X_train, y_train_cat, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)
                        y_pred = np.argmax(model.predict(X_test), axis=1)
                        accuracy = accuracy_score(y_test, y_pred)
                        fold_accuracies.append(accuracy)

                    avg_accuracy = np.mean(fold_accuracies)
                    print(f"Config: Filters={filters}, Dropout={dropout_rate}, Learning Rate={lr}, Accuracy={avg_accuracy:.4f}")

                    if avg_accuracy > best_accuracy:
                        best_accuracy = avg_accuracy
                        best_config = (filters, dropout_rate, lr)

        print(f"Best Config: Filters={best_config[0]}, Dropout={best_config[1]}, Learning Rate={best_config[2]}, Accuracy={best_accuracy:.4f}")

if __name__ == "__main__":
    # Paths
    input_dir = '../sample_labelled_20'
    standardised_dir = 'output_standardised'
    augmented_dir = 'output_augmented'

    # Step 1: Standardization
    standardiser = ImageStandardiser(target_size=IMG_SIZE)
    standardiser.standardise_images_in_directory(input_dir, standardised_dir)
    print("Standardization complete.")

    # Step 2: Augmentation
    augmentor = ImageAugmentor()
    augmentor.augment_images(standardised_dir, augmented_dir, num_augmented=AUGMENTATION_FACTOR)

    # Step 3: Classification
    X, y = [], []
    for filename in os.listdir(augmented_dir):
        if filename.endswith('.jpg'):
            label = 0 if 'healthy' in filename else 1
            img_path = os.path.join(augmented_dir, filename)
            img = load_img(img_path, color_mode='grayscale')
            img_array = img_to_array(img) / 255.0
            X.append(img_array)
            y.append(label)
    X = np.array(X)
    y = np.array(y)

    if HYPERPARAMETER_TUNING:
        classifier = CNNClassifier(input_shape=IMG_SIZE + (1,))
        classifier.train_and_evaluate(X, y, num_folds=NUM_FOLDS)
    else:
        # Default training without hyperparameter tuning
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
        classifier = CNNClassifier(input_shape=IMG_SIZE + (1,))
        classifier.train_and_evaluate(X_train, y_train, X_test, y_test)
    print("Training and evaluation complete.")
