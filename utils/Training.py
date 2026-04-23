import torch
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score


def _forward(model, inputs):
    """Handles models that return auxiliary outputs (e.g. InceptionV3)."""
    outputs = model(inputs)
    if isinstance(outputs, tuple):
        return outputs[0], outputs[1]  # logits, aux_logits
    return outputs, None


def _compute_loss(criterion, logits, aux_logits, labels):
    loss = criterion(logits, labels)
    if aux_logits is not None:
        loss += 0.4 * criterion(aux_logits, labels)
    return loss


def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss  = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type):
            logits, aux = _forward(model, inputs)
            loss = _compute_loss(criterion, logits, aux, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        all_preds.append(torch.sigmoid(logits).detach().cpu())
        all_labels.append(labels.detach().cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    return total_loss / len(loader), roc_auc_score(labels, preds)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float().unsqueeze(1)

            with autocast(device_type=device.type):
                logits, _ = _forward(model, inputs)  # no aux in eval mode
                loss = criterion(logits, labels)

            total_loss += loss.item()
            all_preds.append(torch.sigmoid(logits).cpu())
            all_labels.append(labels.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    return total_loss / len(loader), roc_auc_score(labels, preds)