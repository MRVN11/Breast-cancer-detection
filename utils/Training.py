import torch
from torch.amp import autocast, GradScaler

# scaler = GradScaler()


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0

    for inputs, labels in loader:
        # inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)
        optimizer.zero_grad()

        # ✅ Correct autocast usage
        with autocast(device_type="cuda"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # ✅ Scaled backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            # ✅ Correct autocast usage
            with autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(loader)
