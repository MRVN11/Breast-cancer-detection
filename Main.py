from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import set_random_seed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

import tensorflow_io as tfio

from cnn_models.DenseNet121 import create_densenet121
from cnn_models.ResNet50 import create_ResNet50
import tensorflow as tf
from data_operations.data_preprocessing import dataset_stratified_split, import_MIAS_dataset, calculate_weights, import_CBIS_dataset
from sklearn.preprocessing import LabelEncoder



def print_num_gpus_available():
    print("Number of GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


print_num_gpus_available()

"""
model that will be made are densenet121 and VGG 
model can be swaped out quickly my changing Model_in_use
dataset can be swapped as well between MIAS and CBIS
"""
epochs = 20
epochs2 = 15
batch_size = 32
Model_in_use = "densenet"
history = None
dataset = "MIAS"
# Learning_rate = 1e-6

def main() -> None:

    l_e =  LabelEncoder()

    try:
        if dataset == "MIAS":
            images, labels = import_MIAS_dataset(data_dir ="data/MIAS_data/Processed-Images", label_encoder = l_e)
            num_classes = len(l_e.classes_)
            X_train, X_test, y_train, y_test = dataset_stratified_split(split=0.20, dataset=images, labels=labels)
            class_weights = calculate_weights(y_train, l_e)
        elif dataset == "CBIS":
            images, labels = import_CBIS_dataset(l_e)
            X_train, X_test, y_train, y_test = dataset_stratified_split(split=0.25, dataset=images, labels=labels)
            class_weights = calculate_weights(y_train, l_e)
    except FileNotFoundError:
        print("Dataset not found")



    if Model_in_use == "densenet":
        model = create_densenet121(num_classes)
    elif Model_in_use == "VGG":
        pass
    elif Model_in_use == "resnet50":
        model = create_ResNet50(num_classes)
    else:
        raise ValueError(f"Model_in_use {Model_in_use} not recognized")



    # for layer in model.layers:
    #     layer.trainable = False

    model.compile(
        optimizer=Adam(1e-5),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True,
        width_shift_range=0.05,
        height_shift_range=0.05,
        fill_mode="nearest"
    )

    X_train, X_val, y_train, y_val = dataset_stratified_split(split=0.25, dataset=X_train, labels=y_train
    )
    model.summary()
# training the model
    history = model.fit(
        datagen.flow(X_train,y_train,batch_size=batch_size),
        # steps_per_epoch=len(X_train)//batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weights)

    for layer in model.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=Adam(1e-6),
        loss = CategoricalCrossentropy(),
        metrics = [CategoricalAccuracy()]
    )

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)


if __name__ == "__main__":
    main()