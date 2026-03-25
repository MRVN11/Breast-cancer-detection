import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, loader, device, class_names):
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy())
            targets.extend(labels.numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    y_pred = (preds > 0.5).astype(int).flatten()
    y_true = targets.flatten()

    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.show()

    try:
        print("ROC-AUC:", roc_auc_score(y_true, preds))
    except:
        pass