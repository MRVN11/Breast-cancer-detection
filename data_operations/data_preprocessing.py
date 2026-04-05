import cv2
import numpy as np

def preprocess_images(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)

    # --- Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Step 2: CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- Step 3: Breast segmentation (threshold)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop BOTH image and gray
    cropped = image[y:y+h, x:x+w]
    cropped_gray = gray[y:y+h, x:x+w]

    # --- Step 4: Blur (on grayscale)
    blur = cv2.GaussianBlur(cropped_gray, (5, 5), 0.5)

    # --- Step 5: Edge detection
    edges = cv2.Canny(blur, 50, 150).astype("float32") / 255.0

    # --- Step 6: Blend edges
    enhanced = blur.astype("float32") / 255.0
    enhanced = enhanced + 0.3 * edges
    enhanced = np.clip(enhanced, 0, 1)

    # --- Step 7: Convert to 3-channel
    image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    # --- Step 8: Resize
    image = cv2.resize(image, (512, 512))
    return image