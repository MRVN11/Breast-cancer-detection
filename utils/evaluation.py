import torch
import numpy as np
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(model, loader, device, class_names):
    model.eval()

    preds   = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            with autocast(device_type=device.type):
                outputs = model(images)

            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    preds   = np.array(preds).flatten()
    targets = np.array(targets).flatten()

    # --- Optimal threshold via Youden Index ---
    try:
        fpr, tpr, thresholds = roc_curve(targets, preds)
        youden_idx        = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[youden_idx]
        roc_auc           = auc(fpr, tpr)
        print(f"Optimal threshold: {optimal_threshold:.4f}")
    except ValueError as e:
        print(f"[Warning] ROC curve failed: {e}")
        optimal_threshold = 0.5
        fpr = tpr = roc_auc = None

    y_pred = (preds > optimal_threshold).astype(int)
    y_true = targets.astype(int)

    # --- Metrics ---
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC:     {roc_auc:.4f}")
    print(report)

    # --- Save report ---
    with open("classification_report.txt", "w") as f:
        f.write(f"Accuracy:         {accuracy:.4f}\n")
        f.write(f"Sensitivity:      {sensitivity:.4f}\n")
        f.write(f"Specificity:      {specificity:.4f}\n")
        if roc_auc is not None:
            f.write(f"ROC-AUC:          {roc_auc:.4f}\n")
        f.write(f"Optimal Threshold:{optimal_threshold:.4f}\n\n")
        f.write(report)

    # --- Confusion matrix (counts + normalised) ---
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, matrix, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion Matrix (Counts)", "Confusion Matrix (Normalised)"],
        ["d", ".2f"]
    ):
        sns.heatmap(matrix, annot=True, fmt=fmt, ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()

    # --- ROC Curve ---
    if roc_auc is not None:
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.scatter(fpr[youden_idx], tpr[youden_idx],
                    marker='o', color='red', label=f"Optimal threshold ({optimal_threshold:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig("roc_curve.png", dpi=150, bbox_inches="tight")
        plt.show()

    return accuracy, sensitivity, specificity, roc_auc