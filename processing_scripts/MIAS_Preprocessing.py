import os
import pandas as pd
import cv2

def main() ->  None:
    KEY_LABEL = 3
    df = pd.read_csv('../data/MIAS_data/MIAS_Description.csv', header=None, index_col=0)
    df[KEY_LABEL] = df[KEY_LABEL].fillna("N")
    df[KEY_LABEL] = df[KEY_LABEL].str.strip()

    for img_pgm in os.listdir("../data/MIAS_data/MIAS_images_ori"):
        if img_pgm.endswith(".pgm"):
            img = cv2.imread("../data/MIAS_data/MIAS_images_ori/{}".format(img_pgm))
            img_name = img_pgm.split(".")[0]
            label = df.loc[img_name, KEY_LABEL]
            if isinstance(label, pd.Series):
                label = label.values[0]

            if label == "N":
                new_path = f"../data/MIAS_data/Normal_cases/{img_name}.png"
            elif label == "B":
                new_path = f"../data/MIAS_data/Benign_cases/{img_name}.png"
            elif label == "M":
                new_path = f"../data/MIAS_data/Malignant_cases/{img_name}.png"
            cv2.imwrite(new_path, img)
            print("Converted {} to png" .format(img_pgm))
    print("Done!")
if __name__ =="__main__":
    main()