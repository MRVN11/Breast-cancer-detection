
from torch.amp import GradScaler

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import datetime

from data_operations.Dataset import BreastCancerDataset
from utils.Training import train_one_epoch, validate
from utils.evaluation import evaluate
from utils.weights import calculate_class_weights

from cnn_models.DenseNet import DenseNet121Model
from cnn_models.ResNet import ResNet
from cnn_models.EfficentNet import EfficientNet

from torchvision import transforms

# ======================
# CONFIG
# ======================
MODEL_NAME = "DenseNet121"
FINE_TUNE_EPOCHS = 50
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = int(EPOCHS / 10)
K_FOLDS = 5

DATA_PATH = "data/Combined_Images"

# ======================
# MODEL UTILS
# ======================
def get_model(device):
    if MODEL_NAME == "DenseNet121":
        model = DenseNet121Model()
    elif MODEL_NAME == "ResNet50":
        model = ResNet()
    elif MODEL_NAME == "EfficientNet":
        model = EfficientNet()
    else:
        model = DenseNet121Model()
    model = model.to(device)
    # model = torch.compile(model)

    return model

def freeze_backbone(model):
    """Freezes the layers of the model."""
    for param in model.base_model.parameters():
        param.requires_grad = False


def unfreeze_backbone(model, num_layers=10):
    """Unfreezes the layers of the model. for fine-tuning"""
    if MODEL_NAME == "ResNet50":
        layers = list(model.base_model.children())
    else:
        layers = list(model.base_model.features.children())
    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


# ======================
# TRAINING LOOP
# ======================
def train_model(model, train_loader, val_loader, device, fold, weights):
    # weights = calculate_class_weights(full_dataset.labels).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights[1])
    scaler = GradScaler('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")
    patience_counter = 0

    # ---------- Initial Training ----------
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"[Fold {fold+1}] Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {datetime.datetime.now()}")

        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

    # ---------- Fine-tuning ----------
    print(f"[Fold {fold+1}] Starting fine-tuning...")

    torch.cuda.empty_cache()

    unfreeze_backbone(model)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-6
    )
    best_loss = float("inf")
    patience_counter = 0
    for epoch in range(FINE_TUNE_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"[Fold {fold+1}][FT] Epoch {epoch+1} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        scheduler.step()

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered (FT)")
            break
    return best_loss


# ======================
# MAIN
# ======================
def main():
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # ---------- Transforms ----------
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])

    # ---------- Dataset ----------
    global full_dataset
    full_dataset = BreastCancerDataset(DATA_PATH)
    class_names = full_dataset.classes
    # ---------- K-Fold ----------
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_results = []
    fold_accuracy = []
    weights = calculate_class_weights(full_dataset.labels).to(device)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_dataset)):
        print(f"\n========== FOLD {fold+1} ==========")

        # Create datasets with transforms
        train_dataset = BreastCancerDataset(DATA_PATH, transform=train_transform, oversample=True)
        val_dataset = BreastCancerDataset(DATA_PATH, transform=val_transform)

        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(val_dataset, val_idx)

        # train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        train_loader = DataLoader(
            train_subset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,  # or 8 if CPU allows
            pin_memory=True,
            persistent_workers=True
        )

        # val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)
        val_loader = DataLoader(
            val_subset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )

        # Model
        model = get_model(device)
        freeze_backbone(model)

        # Train
        best_loss = train_model(model, train_loader, val_loader, device, fold, weights)
        fold_results.append(best_loss)

        # Evaluate best model
        model.load_state_dict(torch.load(f"best_model_fold{fold}.pth"))
        accuracy = evaluate(model, val_loader, device, class_names)
        fold_accuracy.append(accuracy)

    print("\n========== FINAL RESULTS ==========")
    for i, loss in enumerate(fold_results):
        print(f"Fold {i+1}: Fold Loss: {loss:.4f}")

    for i, acc in enumerate(fold_accuracy):
        print(f"Fold {i}: Fold accuracy: {acc:.4f}")

    print(f"Average Loss: {sum(fold_results)/len(fold_results):.4f}")
    print(f"Average Accuracy: {sum(fold_accuracy)/len(fold_accuracy):.4f}")

if __name__ == "__main__":
    main()