import os
import pandas as pd
import cv2
from tqdm import tqdm

def organize_cbis_images(csv_path: str, output_root: str) -> None:
    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"Reading CSV with {len(df)} entries")

    # Define folder mapping
    label_to_folder = {
        "MALIGNANT": "Malignant_cases",
        "BENIGN": "Benign_cases"
    }

    # Create folders
    for folder in label_to_folder.values():
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)

    # Process images
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["img_path"]
        label = row["label"].strip().upper()  # Normalize label
        img_folder = row.get("img_folder", "Unknown")  # optional folder info

        # Skip unknown labels
        if label not in label_to_folder:
            print(f"⚠️  Skipping unknown label: {label} for image {img_path}")
            continue

        # Destination folder
        dst_folder = os.path.join(output_root, label_to_folder[label])

        # Create unique filename
        filename = f"{img_folder}_{os.path.basename(img_path)}".replace(".jpg", ".png").replace(".jpeg", ".png")
        dst_path = os.path.join(dst_folder, filename)

        # Read and save as PNG
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️  Failed to read image: {img_path}")
            continue

        cv2.imwrite(dst_path, img)

    print("✅ Done organizing CBIS images!")

if __name__ == "__main__":
    organize_cbis_images(
        csv_path="../data/CBIS_data/CBIS_DDSM/CBIS_dataset.csv",
        output_root="../data/CBIS_data/CBIS_images"
    )