import torch
from torch.utils.data import DataLoader, random_split
from data_operations.Dataset import BreastCancerDataset
from utils.Training import train_one_epoch, validate
from utils.evaluation import evaluate
from utils.weights import calculate_class_weights
from cnn_models.DenseNet import DenseNet121Model

EPOCHS = 50
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
    if torch.cuda.is_available():
       print("Using GPU")
    else:
        print("No GPU available, using CPU instead")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    dataset = BreastCancerDataset("data/Combined_Images")
    class_names = dataset.classes

    # Split
    train_size = int(0.75 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


    model = DenseNet121Model().to(device)
    freeze_backbone(model)

    # Class weights
    weights = calculate_class_weights(dataset.labels).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights[1])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    best_loss = float("inf")
    patience_counter = 0

 #Training
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

    # Fine-tuning
    unfreeze_last_layer(model, num_layers = 30)
    optmizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
                                lr=1e-6)
    for epoch in range(50):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print("Early stopping")
            break
    model.load_state_dict(torch.load("best_model.pth"))
    evaluate(model, val_loader, device, class_names)

if __name__ == "__main__":
    main()