import numpy as np
import torch
import datetime

def calculate_class_weights(labels):
    """
    Calculates pos_weight for BCEWithLogitsLoss.
    Returns the ratio of negatives to positives (with smoothing),
    which is what BCEWithLogitsLoss pos_weight expects.
    """
    labels = np.array(labels)
    class_counts = np.bincount(labels)

    if len(class_counts) < 2:
        raise ValueError(f"Expected 2 classes, found {len(class_counts)}")

    if np.any(class_counts == 0):
        raise ValueError(f"One or more classes have zero samples: {class_counts}")

    neg, pos = class_counts[0], class_counts[1]

    print(f"Class distribution — Negative: {neg} | Positive: {pos} | Ratio: {neg/pos:.2f}| Time {datetime.datetime.now()}")

    smoothing = 0.1
    pos_weight = (neg + smoothing) / (pos + smoothing)

    return torch.tensor(pos_weight, dtype=torch.float32)