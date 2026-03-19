import os
import shutil
import pandas as pd
import cv2
from tqdm import tqdm

def organize_cbis_images(csv_path, output_root):
    df = pd.read_csv(csv_path)
    print("reading csv")

    # Create output folders
    benign_dir = os.path.join(output_root, "Benign")
    malignant_dir = os.path.join(output_root, "Malignant")
    print("creating folders")

    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(malignant_dir, exist_ok=True)

    copied = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["img_path"]
        label = row["label"]
        print(f"[green]processing image", {img_path})
        # Decide destination
        if label == "MALIGNANT":
            dst_dir = malignant_dir
        else:
            dst_dir = benign_dir

        # Safer filename (prevents duplicates)
        filename = f"{row['img_folder']}_{os.path.basename(img_path)}"
        dst_path = os.path.join(dst_dir, filename)

        if not os.path.exists(dst_path):
            # shutil.copy2(img_path, dst_path)
            # copied += 1
            img = cv2.imread(img_path)
            dst_path = dst_path.replace(".jpg", ".png").replace(".jpeg", ".png")
            cv2.imwrite(dst_path, img)

    print(f"✅ Done. {copied} images copied.")


if __name__ == "__main__":
    organize_cbis_images(
        csv_path="../data/CBIS_data/CBIS_DDSM/CBIS_dataset.csv",
        output_root="../data/CBIS_data/CBIS_images"
    )