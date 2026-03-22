from keras.applications import DenseNet121
from tensorflow.keras.layers import  Dense, Dropout, Input, GlobalAveragePooling2D, Concatenate, Flatten
from tensorflow.keras import Sequential
from keras import Model
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

def create_densenet201(num_classes: int):

    img_input = Input(shape=(512, 512, 3))
    img_conc = Concatenate()([img_input, img_input, img_input])

    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_tensor=img_input,
        # pooling="non",
    )
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(224, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation="softmax" if num_classes > 2 else "sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs)
    # base_model = DenseNet121(
    #     weights="imagenet",
    #     include_top=False,
    #     input_tensor=img_input,
    # )
    #
    # model = Sequential()
    # model.add(base_model)
    #
    # model.add(GlobalAveragePooling2D())
    #
    # fully_connected = Sequential(name="fully_connected")
    #
    # fully_connected.add(Dropout(rate=0.2, name="1"))
    # fully_connected.add(Dense(units=512, activation="relu", name="dense1"))
    # fully_connected.add(Dropout(rate=0.2, name="2"))
    # fully_connected.add(Dense(units=32, activation="relu", name="dense2"))
    #
    # if num_classes == 2:
    #     fully_connected.add(Dense(units=1, activation="sigmoid", kernel_initializer="random_uniform", name="output"))
    # else:
    #     fully_connected.add(
    #         Dense(num_classes, activation="softmax", kernel_initializer="random_uniform", name="output"))
    #
    # model.add(fully_connected)

    return model