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
        print("ROC-AUC:", roc_auc_score(y_true, preds.flatten()))
    except:
        pass
    return accuracy