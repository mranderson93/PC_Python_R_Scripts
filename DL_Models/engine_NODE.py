from tqdm.auto import tqdm
from torchmetrics.classification import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    AUROC,
)
import torch
import torch.nn as nn


# === Train Step for NODE ===
def train_step_NODE(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0

    accuracy = Accuracy(task="multiclass", num_classes=model.leaf_weights.shape[-1]).to(
        device
    )
    precision = Precision(
        task="multiclass", num_classes=model.leaf_weights.shape[-1]
    ).to(device)
    recall = Recall(task="multiclass", num_classes=model.leaf_weights.shape[-1]).to(
        device
    )
    f1 = F1Score(task="multiclass", num_classes=model.leaf_weights.shape[-1]).to(device)
    auroc = AUROC(task="multiclass", num_classes=model.leaf_weights.shape[-1]).to(
        device
    )

    all_probs, all_labels = [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)  # shape: [batch, num_classes]
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
        probs = torch.softmax(logits, dim=1)
        all_probs.append(probs.detach())
        all_labels.append(y.detach())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    preds = torch.argmax(all_probs, dim=1)

    acc = accuracy(preds, all_labels)
    prec = precision(preds, all_labels)
    rec = recall(preds, all_labels)
    f1_score = f1(preds, all_labels)
    roc_auc = auroc(all_probs, all_labels)

    avg_loss = train_loss / len(dataloader.dataset)
    return (
        avg_loss,
        acc.item(),
        prec.item(),
        rec.item(),
        f1_score.item(),
        roc_auc.item(),
    )


# === Test Step for NODE ===
def test_step_NODE(model, dataloader, loss_fn, device):
    model.eval()
    test_loss = 0

    accuracy = Accuracy(task="multiclass", num_classes=model.leaf_weights.shape[-1]).to(
        device
    )
    precision = Precision(
        task="multiclass", num_classes=model.leaf_weights.shape[-1]
    ).to(device)
    recall = Recall(task="multiclass", num_classes=model.leaf_weights.shape[-1]).to(
        device
    )
    f1 = F1Score(task="multiclass", num_classes=model.leaf_weights.shape[-1]).to(device)
    auroc = AUROC(task="multiclass", num_classes=model.leaf_weights.shape[-1]).to(
        device
    )

    all_probs, all_labels = [], []

    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            test_loss += loss.item() * x.size(0)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs)
            all_labels.append(y)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    preds = torch.argmax(all_probs, dim=1)

    acc = accuracy(preds, all_labels)
    prec = precision(preds, all_labels)
    rec = recall(preds, all_labels)
    f1_score = f1(preds, all_labels)
    roc_auc = auroc(all_probs, all_labels)

    avg_loss = test_loss / len(dataloader.dataset)
    return (
        avg_loss,
        acc.item(),
        prec.item(),
        rec.item(),
        f1_score.item(),
        roc_auc.item(),
    )


# === Training Loop for NODE ===
def train_NODE(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int = 5,
):
    results = {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "train_roc_auc": [],
        "test_loss": [],
        "test_acc": [],
        "test_precision": [],
        "test_recall": [],
        "test_f1": [],
        "test_roc_auc": [],
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = (
            train_step_NODE(model, train_dataloader, loss_fn, optimizer, device)
        )
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = test_step_NODE(
            model, test_dataloader, loss_fn, device
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["train_precision"].append(train_prec)
        results["train_recall"].append(train_rec)
        results["train_f1"].append(train_f1)
        results["train_roc_auc"].append(train_auc)

        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["test_precision"].append(test_prec)
        results["test_recall"].append(test_rec)
        results["test_f1"].append(test_f1)
        results["test_roc_auc"].append(test_auc)

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, ROC-AUC: {train_auc:.4f} || "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, ROC-AUC: {test_auc:.4f}"
        )

    return results
