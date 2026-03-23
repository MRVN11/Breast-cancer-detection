from keras.applications import ResNet50
from tensorflow.keras.layers import  Dense, Dropout, Input, Flatten, Concatenate
from tensorflow.keras import Sequential
from keras import Model
import ssl

from tensorflow.python.layers.core import fully_connected

ssl._create_default_https_context = ssl._create_unverified_context
def create_ResNet50(num_classes: int):
    img_input = Input(shape=(512, 512, 3))

    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=img_input,
    )

    model = Sequential()
    model.add(base_model)

    model.add(Flatten())

    fully_connected = Sequential(name="fully_connected")

    fully_connected.add(Dropout(rate=0.2, name="1"))
    fully_connected.add(Dense(units=512, activation="relu", name="dense1"))
    fully_connected.add(Dropout(rate=0.2, name="2"))
    fully_connected.add(Dense(units=32, activation="relu", name="dense2"))

    if num_classes == 2:
       fully_connected.add(Dense(units=1, activation="sigmoid", kernel_initializer="random_uniform", name="output"))
    else:
        fully_connected.add(Dense(num_classes, activation="softmax", kernel_initializer="random_uniform", name="output"))


    model.add(fully_connected)
    return model