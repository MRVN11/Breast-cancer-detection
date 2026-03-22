import tensorflow as tf
from Tools.scripts.mailerdaemon import emparse_list
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.preprocessing import LabelEncoder

from cnn_models.DenseNet121 import create_densenet201
from cnn_models.ResNet50 import create_ResNet50
from data_operations.data_preprocessing import (
    dataset_stratified_split,
    import_dataset,
    calculate_weights,
)

# ======================
# CONFIG
# ======================
EPOCHS = 100
FINE_TUNE_EPOCHS = 50
# BATCH_SIZE = 32
MODEL_NAME = "densenet" #densenet, ResNet50 & EfficientNetB7
DATASET = "CBIS"
PATIENCE = EPOCHS // 10
if DATASET == "MIAS":
    BATCH_SIZE = 16
elif DATASET == "CBIS":
    BATCH_SIZE = 32
# ======================
# MODEL CLASS
# ======================
class CNNModel:
    def __init__(self, model_name: str, num_classes: int):
        self.model_name = model_name
        self.model = self._build_model(num_classes)

    def _build_model(self, num_classes):
        if self.model_name == "densenet":
            return create_densenet201(num_classes)
        elif self.model_name == "ResNet50":
            return create_ResNet50(num_classes)
        else:
            raise ValueError(f"{self.model_name} not supported")

    def compile(self, lr=1e-5):
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()]
        )

    def train(self, X_train, X_val, y_train, y_val, class_weights):
        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            horizontal_flip=True,
            width_shift_range=0.05,
            height_shift_range=0.05,
            fill_mode="nearest"
        )

        # callbacks = [
        #     EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        #     ReduceLROnPlateau(patience=PATIENCE // 2)
        # ]
        if DATASET == "MIAS":
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                steps_per_epoch=len(X_train) // BATCH_SIZE,
                validation_data=(X_val, y_val),
                validation_steps=len(X_val) // BATCH_SIZE,
                epochs=EPOCHS,
                class_weight=class_weights,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                    ReduceLROnPlateau(patience=PATIENCE // 2)
                ]
            )
        elif DATASET == "CBIS":
            history = self.model.fit(
                #datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                x=X_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                class_weight=class_weights,
                batch_size=BATCH_SIZE,
                callbacks=[
                    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                    ReduceLROnPlateau(patience=PATIENCE // 2)
                ]
            )
        return history
# ======================
# FineTuning PIPELINE
# ======================
    def fine_tune(self, X_train, X_val, y_train, y_val, class_weights, unfreeze_layer=30):
        for layer in self.model.layers[-unfreeze_layer:]:
            layer.trainable = True

        self.compile(lr=1e-6)
        # callbacks = [
        #     EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
        #     ReduceLROnPlateau(patience=PATIENCE // 2)
        # ]

        history = self.model.fit(
            X_train,
            y_train,
            batch_size=BATCH_SIZE,
            epochs=FINE_TUNE_EPOCHS,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                ReduceLROnPlateau(patience=PATIENCE // 2)
            ]
        )

        return history

# ======================
# Evaluation
# ======================
    def evaluate(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        return loss, acc

# ======================
# MAIN PIPELINE
# ======================
def main():
    label_encoder = LabelEncoder()
    try:
        if DATASET == "MIAS":
            images, labels = import_dataset(
                data_dir="data/CBIS_data/CBIS_images", label_encoder=label_encoder)
            num_classes = len(label_encoder.classes_)
            # Train/Test split
            X_train, X_test, y_train, y_test = dataset_stratified_split(
                split=0.20, dataset=images, labels=labels
            )
            model = CNNModel(MODEL_NAME, num_classes)
            # Train/Validation split
            X_train, X_val, y_train, y_val = dataset_stratified_split(
                split=0.25, dataset=X_train, labels=y_train
            )
            class_weights = calculate_weights(y_train, label_encoder)
            # Build model
            model.compile()
            # Train
            model.train(X_train, X_val, y_train, y_val, class_weights)
            model.fine_tune(X_train, X_val, y_train, y_val, class_weights)
            model.evaluate(X_test, y_test)

        elif DATASET == "CBIS":
            images, labels = import_dataset(data_dir="data/CBIS_data/CBIS_images", label_encoder=label_encoder)
            num_classes = len(label_encoder.classes_)
            # Train/Validation split
            X_train, X_val, y_train, y_val = dataset_stratified_split(
                split=0.25, dataset=images, labels=labels
            )
            # Build model
            model = CNNModel(MODEL_NAME, num_classes)
            class_weights = calculate_weights(y_train, label_encoder)
            model.compile()
            # Train
            model.train(X_train, X_val, y_train, y_val, class_weights)
            model.fine_tune(X_train, X_val, y_train, y_val, class_weights)
            # model.evaluate(X_test, y_test)

    except FileNotFoundError as e:
        print("Dataset not found:", e)

if __name__ == "__main__":
    main()