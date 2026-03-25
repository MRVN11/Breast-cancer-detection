from sklearn.utils import  class_weight
import cv2
import numpy as np

def preprocess_images(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0.5)
    image = cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (512, 512))
    image = image.astype("float32") / 255.0
    return image

# def calculate_weights(y_train, label_encoder):
#     if len(y_train.shape) > 1:
#         y_train = np.argmax(y_train, axis=1)
#     weights = class_weight.compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(y_train),
#         y=y_train
#     )
#     class_weights = dict(enumerate(weights))
#     return class_weights