import os
import ssl

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import  Dense, Dropout, Flatten
from tensorflow.keras import Sequential

ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle, class_weight


def main() -> None:

    l_e =  LabelEncoder()

    images, labels = import_MIAS_dataset(data_dir ="data/MIAS_data/Processed-Images", label_encoder = l_e)
    num_classes = len(l_e.classes_)
    X_train, X_test, y_train, y_test = dataset_stratified_split(split= 0.20, dataset= images, labels= labels)

    X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=X_train, labels=y_train
    )
    model = create_desnet121(num_classes)
    class_weights = caculate_weights(y_train, l_e)

# training the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val), class_weight=class_weights)

    for layer in model.layers[0].layers[-30:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy" if num_classes > 2 else "binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=10, batch_size=16)


    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)


def create_desnet121(num_classes: int):

    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze pretrained layers
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax" if num_classes > 2 else "sigmoid")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy" if num_classes > 2 else "binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def caculate_weights(y_train, label_encoder):
    if label_encoder.classes_.size != 2:
        y_train = label_encoder.inverse_transform(np.argmax (y_train, axis=1))
    weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

    class_weights = dict(enumerate(weights))
    return class_weights


def dataset_stratified_split(split: float, dataset: np.ndarray, labels: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """Split dataset into training and testing sets with stratification."""
    train_x, test_x, train_y, test_y = train_test_split(dataset, labels, test_size=split, stratify=labels, random_state=42, shuffle=True)
    return train_x, test_x, train_y, test_y

def import_MIAS_dataset(data_dir: str, label_encoder) -> (np.ndarray, np.ndarray):
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

def preprocess_images(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    image = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float32") / 255.0
    return image

def encode_labels(labels_list: np.ndarray, label_encoder) -> np.ndarray:
    labels = label_encoder.fit_transform(labels_list)
    if label_encoder.classes_.size == 2:
        return labels
    else:
        return to_categorical(labels)

if __name__ == "__main__":
    main()