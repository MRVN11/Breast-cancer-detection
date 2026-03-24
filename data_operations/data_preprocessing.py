import os

import cv2
import numpy as np
import pandas as pd
from imutils import paths
from sklearn.utils import  class_weight
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

"""
can change whether to train and test with calc or mass.
"""
data = "calc"

def preprocess_images(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0.5)
    image = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    image = image.astype("float32") / 255.0
    return image

def encode_labels(labels_list: np.ndarray, label_encoder) -> np.ndarray:
    labels = label_encoder.fit_transform(labels_list)
    return to_categorical(labels)
    # if label_encoder.classes_.size == 2:
    #     return labels
    # else:
    #     return to_categorical(labels)

def dataset_stratified_split(split: float, dataset: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Split dataset into training and testing sets with stratification."""
    train_x, test_x, train_y, test_y = train_test_split(dataset,
                                                        labels,
                                                        test_size=split,
                                                        stratify=labels,
                                                        random_state=42,
                                                        shuffle=True)
    return train_x, test_x, train_y, test_y

def import_dataset(data_dir: str, label_encoder) -> (np.ndarray, np.ndarray):
    images = list()
    labels = list()

    for image_path in list(paths.list_images(data_dir)):
        images.append(preprocess_images(image_path))
        labels.append(image_path.split(os.path.sep)[-2])
    images = np.array(images, dtype="float32")
    labels = np.array(labels)

    labels = encode_labels(labels, label_encoder)

    images, labels = shuffle(images, labels, random_state=42)
    return images, labels

def calculate_weights(y_train, label_encoder):
    # If one-hot → convert to class indices
    if len(y_train.shape) > 1:
        y_train = np.argmax(y_train, axis=1)

    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(weights))
    return class_weights