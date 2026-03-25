import os
import torch
from torch.utils.data import Dataset
from data_operations.data_preprocessing import preprocess_images

class BreastCancerDataset(Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)

            for file in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, file))
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = preprocess_images(self.image_paths[idx])
        label = self.labels[idx]

        image = torch.tensor(image).permute(2, 0, 1)
        label = torch.tensor(label).float()

        return image, label