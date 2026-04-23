import cv2
import numpy as np

def preprocess_images(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

        # --- Step 1: Resize
    resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)

    # --- Step 2: Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # --- Step 3: Breast segmentation FIRST (on raw gray, before CLAHE)
    # Otsu's thresholding is more robust than a fixed value of 10
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill small holes in the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No breast region found in image.")
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Add a small padding around the crop
    pad = 10
    x, y = max(0, x - pad), max(0, y - pad)
    w = min(image.shape[1] - x, w + 2 * pad)
    h = min(image.shape[0] - y, h + 2 * pad)

    cropped_gray = gray[y:y + h, x:x + w]

    # --- Step 4: CLAHE (now applied AFTER cropping — much more effective)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(cropped_gray)

    # --- Step 5: Denoise before edge detection
    denoised = cv2.GaussianBlur(clahe_gray, (5, 5), 0.5)

    # --- Step 6: Edge detection
    edges = cv2.Canny(denoised, 30, 100).astype("float32") / 255.0

    # --- Step 7: Subtle edge blending (reduced weight to avoid over-sharpening)
    enhanced = denoised.astype("float32") / 255.0
    enhanced = enhanced + 0.15 * edges  # was 0.3 — too aggressive
    enhanced = np.clip(enhanced, 0, 1)

    # --- Step 8: Convert back to uint8 for resize
    enhanced_uint8 = (enhanced * 255).astype(np.uint8)
    resized_2 = cv2.resize(enhanced_uint8, (224, 224), interpolation=cv2.INTER_LINEAR)

    # --- Step 9: Stack to 3-channel and normalise to float32 [0, 1]
    image_3ch = np.stack([resized_2, resized_2, resized_2], axis=-1).astype("float32") / 255.0

    return image_3ch