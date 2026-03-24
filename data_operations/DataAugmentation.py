import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def DataAugmentation(X, y):
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        width_shift_range=0.05,
        height_shift_range=0.05,
        fill_mode="nearest"
    )

    y_label = np.argmax(y, axis=1)
    X_aug = []
    y_aug = []

    for i in range(len(X)):
        if y_label[i] == 1:
            img = X[i]
            img = img.reshape((1,) + img.shape)

            aug_iter = datagen.flow(img, batch_size=32)
            for _ in range(2):
                X_aug.append(next(aug_iter)[0])
                y_aug.append(y[i])
    X_aug = np.array(X_aug)
    y_aug = np.array(y_aug)

    return np.concatenate([X, X_aug]), np.concatenate([y, y_aug])