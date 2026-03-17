import os
import pandas as pd

def main():
    csv_root = "../data/CBIS_data/Ori_csv"
    img_root = r"D:\CBIS-DDSM\jpeg"
    csv_output_path = "../data/CBIS_data/CBIS_DDSM"

    type_dict = {
        'Calc-Test': 'calc_case_description_test_set.csv',
        'Calc-Training': 'calc_case_description_train_set.csv',
        'Mass-Test': 'mass_case_description_test_set.csv',
        'Mass-Training': 'mass_case_description_train_set.csv'
    }

    # Build a mapping of all JPEG folders to their images
    rows = []
    for root, dirs, files in os.walk(img_root):
        for file in files:
            if file.lower().endswith(".jpeg") or file.lower().endswith(".jpg"):
                img_folder = os.path.basename(root)  # UID folder
                img_path = os.path.join(root, file)
                rows.append([img_folder, img_path])

    df_imgs = pd.DataFrame(rows, columns=["img_folder", "img_path"])

    for t, csv_file in type_dict.items():
        df_csv = pd.read_csv(os.path.join(csv_root, csv_file), usecols=["pathology", "image file path"])

        # Extract the **last UID folder** from CSV path (matches JPEG folder)
        def extract_uid(path):
            parts = path.split("/")
            # last numeric-looking folder
            for part in reversed(parts):
                if part.startswith("1.3.6.1.4.1"):
                    return part
            return None

        df_csv["img_folder"] = df_csv["image file path"].apply(extract_uid)

        # Fix pathology labels
        df_csv["label"] = df_csv["pathology"].apply(lambda x: "BENIGN" if x == "BENIGN_WITHOUT_CALLBACK" else x)

        # Drop duplicates
        df_csv = df_csv[["img_folder", "label"]].drop_duplicates()

        # Merge with actual JPEG paths
        df_merge = pd.merge(df_imgs, df_csv, on="img_folder", how="inner")

        # Save CSV
        df_merge.to_csv(os.path.join(csv_output_path, f"{t.lower()}.csv"), index=False)
        print(f"{t} done, {len(df_merge)} images matched.")

    print("Finished pre-processing CSVs.")

if __name__ == "__main__":
    main()