import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
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
            targets.extend(labels.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    y_pred = (preds > 0.4).astype(int).flatten()
    y_true = targets.flatten()

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)

    sns.heatmap(cm, annot=True, fmt='d',
               xticklabels=class_names,
               yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    with open("Classification_report.txt", "w") as f:
        f.write("Accuracy: {:.4f}\n".format(accuracy_score(y_true, y_pred)))
        f.write(classification_report(
            y_true,
            y_pred,
            target_names=class_names)
        )
    try:
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, preds.flatten())
        # Compute AUC
        roc_auc = auc(fpr, tpr)
        print("ROC-AUC:", roc_auc)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
        plt.plot([0, 1], [0, 1], 'k--')  # diagonal line (random classifier)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.show()
    except:
        pass
    return accuracy