from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from cnn_models.DenseNet import create_densenet169
from cnn_models.ResNet50 import create_ResNet50
from cnn_models.EfficientNet import create_EfficientNet
from data_operations.data_preprocessing import (
    dataset_stratified_split,
    import_dataset,
    calculate_weights,
)

# ======================
# Config
# ======================
TF_ENABLE_ONEDNN_OPTS=0
EPOCHS = 100
FINE_TUNE_EPOCHS = 50
MODEL_NAME = "densenet"  # densenet, ResNet50 & EfficientNet
DATASET = "CBIS"
PATIENCE = EPOCHS // 10
BATCH_SIZE = 32

# ======================
# Model Class
# ======================
class CNNModel:
    def __init__(self, model_name: str, num_classes: int):
        self.model_name = model_name
        self.model = self._build_model(num_classes)

    def _build_model(self, num_classes):
        if self.model_name == "densenet":
            return create_densenet169(num_classes)
        elif self.model_name == "ResNet50":
            return create_ResNet50(num_classes)
        elif self.model_name == "EfficientNet":
            return create_EfficientNet(num_classes)
        else:
            raise ValueError(f"{self.model_name} not supported")

    def compile(self, lr=1e-4):
        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=CategoricalCrossentropy(),
            metrics=[CategoricalAccuracy()]
        )

    # ======================
    # Model Training
    # ======================
    def train(self, X_train, X_val, y_train, y_val, class_weights):
        datagen = ImageDataGenerator(
            rotation_range=10,
            zoom_range=0.1,
            horizontal_flip=True,
            width_shift_range=0.05,
            height_shift_range=0.05,
            fill_mode="nearest"
        )

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
                x=X_train,
                y=y_train,
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

    def fine_tune(self, X_train, X_val, y_train, y_val, class_weights, unfreeze_layer=30):
        for layer in self.model.layers[-unfreeze_layer:]:
            layer.trainable = True

        self.compile(lr=1e-6)

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

    def Make_prediction(self, x):
        if DATASET == "MIAS":
            self.prediction = self.model.predict(x=x.astype("float32"), batch_size=BATCH_SIZE)
        elif DATASET == "CBIS":
            self.prediction = self.model.predict(x=x)

    def plot_confusion_matrix(self, y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    # ======================
    # Evaluation
    # ======================
    def evaluate(self, y_true, label_encoder, classification_type: str):
        y_pred_probs = self.prediction

        # Convert probabilities → class labels
        if y_pred_probs.shape[1] > 1:
            y_pred = np.argmax(y_pred_probs, axis=1)
            y_true_labels = np.argmax(y_true, axis=1)
        else:
            y_pred = (y_pred_probs > 0.5).astype(int).flatten()
            y_true_labels = y_true.flatten()

        acc = accuracy_score(y_true_labels, y_pred)

        print("\n====================")
        print("📊 MODEL EVALUATION")
        print("====================")
        print(f"Accuracy: {acc:.4f}\n")

        print("Classification Report:")
        print(classification_report(
            y_true_labels,
            y_pred,
            target_names=label_encoder.classes_
        ))

        cm = confusion_matrix(y_true_labels, y_pred)

        self.plot_confusion_matrix(y_true_labels, y_pred, label_encoder.classes_)

        print("Confusion Matrix:")
        print(cm)

        # Sensitivity & Specificity (binary only)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)

            print(f"Sensitivity (Recall): {sensitivity:.4f}")
            print(f"Specificity: {specificity:.4f}")
        else:
            print("Sensitivity/Specificity only for binary classification.")

        # ROC-AUC
        try:
            if len(label_encoder.classes_) == 2:
                auc = roc_auc_score(y_true_labels, y_pred_probs[:, 1])
            else:
                auc = roc_auc_score(y_true, y_pred_probs, multi_class='ovr')

            print(f"\nROC-AUC: {auc:.4f}")
        except Exception as e:
            print("ROC-AUC could not be computed:", e)

        # Save report
        with open("Classification_report.txt", "a") as f:
            f.write(f"Model used: {MODEL_NAME}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(classification_report(
                y_true_labels,
                y_pred,
                target_names=label_encoder.classes_
            ))
# ======================
# Main
# ======================
def main():
    label_encoder = LabelEncoder()
    try:
        if DATASET == "MIAS":
            images, labels = import_dataset(
                data_dir="D:\Data\MIAS\Processed-Images", label_encoder=label_encoder)
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
            model.compile()
            model.train(X_train, X_val, y_train, y_val, class_weights)
            model.fine_tune(X_train, X_val, y_train, y_val, class_weights)
            model.Make_prediction(X_val)
            model.evaluate(y_val, label_encoder, 'B_M')

        elif DATASET == "CBIS":
            images, labels = import_dataset(
                data_dir="data/Combined_Images", label_encoder=label_encoder)
            num_classes = len(label_encoder.classes_)
            # Train/Validation split
            X_train, X_val, y_train, y_val = dataset_stratified_split(
                split=0.25, dataset=images, labels=labels
            )
            model = CNNModel(MODEL_NAME, num_classes)
            class_weights = calculate_weights(y_train, label_encoder)
            model.compile()
            model.train(X_train, X_val, y_train, y_val, class_weights)
            model.fine_tune(X_train, X_val, y_train, y_val, class_weights)
            model.Make_prediction(X_val)
            model.evaluate(y_val, label_encoder, 'B_M')

    except FileNotFoundError as e:
        print("Dataset not found:", e)

if __name__ == "__main__":
    main()