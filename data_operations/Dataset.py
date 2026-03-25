import os
import random
import torch
from torch.utils.data import Dataset
from collections import Counter
from data_operations.data_preprocessing import preprocess_images

class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None, oversample=False):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)

            for file in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, file))
                self.labels.append(label)

        if oversample:
            self._oversample()

    def _oversample(self):
        counts = Counter(self.labels)
        max_count = max(counts.values())

        new_paths = []
        new_labels = []

        for label in counts:
            class_items = [
                (p, l) for p, l in zip(self.image_paths, self.labels) if l == label
            ]
            # Oversample minority
            while len(class_items) < max_count:
                class_items.append(random.choice(class_items))
            for p, l in class_items:
                new_paths.append(p)
                new_labels.append(l)

        self.image_paths = new_paths
        self.labels = new_labels

        print("Class distribution after oversampling:", Counter(self.labels))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = preprocess_images(self.image_paths[idx])
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(label).float()
        return image, label

