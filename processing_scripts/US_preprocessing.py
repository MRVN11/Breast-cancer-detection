import os
import pandas as pd
import cv2

def main(CSV_PATH, IMG_DIR) -> None:


    df = pd.read_csv(CSV_PATH, header=0, index_col=1)


    print("Columns:", df.columns.tolist())
    print(df.head())

    CLASSIFICATION_COL = "Classification"

    for img_png in os.listdir(IMG_DIR):
        if not img_png.endswith(".png"):
            continue

        img_name = img_png.split("\n")[0]
        # img_name = img_png
        if img_name not in df.index:
            print(f"Skipping {img_name}: not found in CSV")
            continue

        label = df.loc[img_name, CLASSIFICATION_COL]
        if isinstance(label, pd.Series):
            label = label.values[0]
        label = str(label).strip().lower()

        img_path = os.path.join(IMG_DIR, img_png)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: could not read image {img_path}")
            continue

        if label == "benign":
            new_path = f"../data/Combined_Images/Benign_cases/{img_name}"
        elif label == "malignant":
            new_path = f"../data/Combined_Images/Malignant_cases/{img_name}"
        else:
            print(f"Skipping {img_name}: unknown label '{label}'")
            continue

        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        cv2.imwrite(new_path, img)
        print(f"Copied {img_png} -> {new_path}")

    print("Done!")

if __name__ == "__main__":
    CSV_PATH = r"D:\BrEaST-Lesions_USG-images_and_masks-Dec-15-2023\BrEaST-Lesions-USG-clinical-data-Dec-15-2023 - BrEaST-Lesions-USG clinical dat.csv"
    IMG_DIR = r"D:\BrEaST-Lesions_USG-images_and_masks-Dec-15-2023\BrEaST-Lesions_USG-images_and_masks"
    main(CSV_PATH, IMG_DIR)