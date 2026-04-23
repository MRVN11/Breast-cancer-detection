# data_operations/preprocess_cache.py
import os
import numpy as np
from data_operations.data_preprocessing import preprocess_images

def cache_preprocessed(input_dir, output_dir):
    """
    Run once before training. Saves preprocessed images as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)

    for class_name in os.listdir(input_dir):
        class_in  = os.path.join(input_dir,  class_name)
        class_out = os.path.join(output_dir, class_name)
        os.makedirs(class_out, exist_ok=True)

        files = [f for f in os.listdir(class_in) if not f.startswith('.')]
        print(f"Processing class '{class_name}': {len(files)} images...")

        for file in files:
            in_path  = os.path.join(class_in,  file)
            out_path = os.path.join(class_out, file.replace(".png", ".npy")
                                                   .replace(".jpg", ".npy")
                                                   .replace(".pgm", ".npy"))
            if os.path.exists(out_path):
                continue  # skip already cached

            try:
                processed = preprocess_images(in_path)
                np.save(out_path, processed)
                print("processed image saved", file)
            except Exception as e:
                print(f"[Warning] Failed on {in_path}: {e}")

    print("Caching complete.")

if __name__ == "__main__":
    cache_preprocessed(
        input_dir  = "../data/Combined_Images",
        output_dir = "../data/Combined_Images_Cached"
    )