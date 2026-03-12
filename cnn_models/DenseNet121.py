from keras.applications import DenseNet121
from tensorflow.keras.layers import  Dense, Dropout, Input, GlobalAveragePooling2D, Concatenate
from tensorflow.keras import Sequential
from keras import Model
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
def create_densenet121(num_classes: int):

    img_input = Input(shape=(224, 224, 3))
    # img_conc = Concatenate()([img_input, img_input, img_input])

    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_tensor=img_input,
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation="softmax" if num_classes > 2 else "sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    return model