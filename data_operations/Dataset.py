# data_operations/Dataset.py
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

class BreastCancerDataset(Dataset):
    def __init__(self, root_dir, transform=None, oversample=False):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for file in sorted(os.listdir(class_path)):
                if file.startswith('.'):
                    continue
                self.image_paths.append(os.path.join(class_path, file))
                self.labels.append(label)

        if oversample:
            self._oversample()

    def _oversample(self):
        counts = Counter(self.labels)
        max_count = max(counts.values())
        new_paths, new_labels = [], []

        for label in counts:
            class_items = [(p, l) for p, l in zip(self.image_paths, self.labels) if l == label]
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
        image = np.load(self.image_paths[idx]).astype(np.float32)

        image = np.ascontiguousarray(image)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx]).float()
        return image, label

    @classmethod
    def from_subset(cls, subset, transform=None, oversample=False):
        instance = cls.__new__(cls)
        source = subset.dataset
        indices = subset.indices
        instance.image_paths = [source.image_paths[i] for i in indices]
        instance.labels = [source.labels[i]       for i in indices]
        instance.classes = source.classes
        instance.transform = transform

        if oversample:
            instance._oversample()

        return instance