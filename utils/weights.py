import numpy as np
import torch

def calculate_class_weights(labels):
    labels = np.array(labels)
    class_counts = np.bincount(labels)

    weights = 1. / class_counts
    weights = weights / weights.sum()  # normalize

    return torch.tensor(weights, dtype=torch.float32)