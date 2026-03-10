import os
import ssl

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import  Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras import Sequential

from data_operations.data_preprocessing import dataset_stratified_split, import_MIAS_dataset

ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import  class_weight


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
    history = model.fit(X_train, y_train, epochs=10, batch_size=8, validation_data=(X_val, y_val), class_weight=class_weights)

    for layer in model.layers[0].layers[-30:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy" if num_classes > 2 else "binary_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=30, batch_size=8)


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
        # Flatten(),
        GlobalAveragePooling2D(),
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

    weights = class_weight.compute_class_weight('balanced',
                                                classes =  np.unique(y_train),
                                                y = y_train)

    class_weights = dict(enumerate(weights))
    return class_weights



if __name__ == "__main__":
    main()