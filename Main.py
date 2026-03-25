import torch
from torch.utils.data import DataLoader, random_split
from data_operations.Dataset import BreastCancerDataset
from utils.Training import train_one_epoch, validate
from utils.evaluation import evaluate
from utils.weights import calculate_class_weights
from cnn_models.DenseNet import DenseNet121Model
from torchvision import transforms

EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 10

def freeze_backbone(model):
    for param in model.base_model.parameters():
        param.requires_grad = False
def unfreeze_backbone(model, num_layers = 30):
    layers = list(model.base_model.features.children())

    for layer in layers[-num_layers:]:
        for param in layer.parameters():
            param.requires_grad = True

def main():
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======================
    # Transforms
    # ======================
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

    # ======================
    # Full dataset (NO transform yet)
    # ======================
    full_dataset = BreastCancerDataset("data/Combined_Images")
    class_names = full_dataset.classes

    # ======================
    # Split indices FIRST
    # ======================
    indices = torch.randperm(len(full_dataset))
    train_size = int(0.80 * len(full_dataset))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # ======================
    # Recreate datasets with transforms
    # ======================
    train_dataset = BreastCancerDataset(
        "data/Combined_Images",
        transform=train_transform,
        oversample=True
    )

    val_dataset = BreastCancerDataset(
        "data/Combined_Images",
        transform=val_transform,
        oversample=False
    )

    # Apply split safely
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # ======================
    # DataLoaders
    # ======================
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # ======================
    # Model
    # ======================
    model = DenseNet121Model().to(device)
    freeze_backbone(model)

    # ======================
    # Loss & Optimizer
    # ======================
    weights = calculate_class_weights(full_dataset.labels).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    best_loss = float("inf")
    patience_counter = 0

    # ======================
    # Initial Training
    # ======================
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"[Train] Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break
    # ======================
    # Fine-tuning
    # ======================
    print("Starting fine-tuning...")
    unfreeze_backbone(model, num_layers=30)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-6
    )

    for epoch in range(50):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"[FT] Epoch {epoch+1}: {train_loss:.4f} | Val: {val_loss:.4f}")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

    model.load_state_dict(torch.load("best_model.pth"))

    # ======================
    # Evaluation
    # ======================
    evaluate(model, val_loader, device, class_names)

if __name__ == "__main__":
    main()