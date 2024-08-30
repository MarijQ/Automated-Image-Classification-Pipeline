import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf

# Constants
IMG_SIZE = (128, 128)
BATCH_SIZE = 8
EPOCHS = 10
NUM_CLASSES = 2


# Load and preprocess the dataset
def load_data(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpg'):
            label = 0 if 'healthy' in filename else 1
            img_path = os.path.join(folder, filename)
            img = load_img(img_path, color_mode='grayscale')  # Load image as is
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)


# Build a simple CNN model
def build_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Main function to run cross-validation
def run_cross_validation(X, y, num_folds=5):
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_no = 1
    for train_idx, val_idx in kfold.split(X, y):
        print(f'Training fold {fold_no}...')

        model = build_model(X.shape[1:], NUM_CLASSES)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        y_train_cat = to_categorical(y_train, NUM_CLASSES)
        y_val_cat = to_categorical(y_val, NUM_CLASSES)

        model.fit(X_train, y_train_cat, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=0)

        y_pred = np.argmax(model.predict(X_val), axis=1)
        print(f'Accuracy for fold {fold_no}: {accuracy_score(y_val, y_pred)}')
        print(classification_report(y_val, y_pred, target_names=['Healthy', 'Beau']))

        fold_no += 1


if __name__ == "__main__":
    # Load data
    X, y = load_data('output')

    # Run cross-validation
    run_cross_validation(X, y)
