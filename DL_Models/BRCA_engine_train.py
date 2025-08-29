from tqdm.auto import tqdm
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
)
import torch


# === Train Step ===
def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0

    # Metrics
    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    auroc = BinaryAUROC().to(device)

    all_probs, all_labels = [], []

    for X, y in dataloader:
        X, y = X.to(device), y.to(device).float()

        optimizer.zero_grad()
        logits = model(X).squeeze()
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * X.size(0)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.detach())
        all_labels.append(y.detach())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    preds = (all_probs > 0.5).float()

    return (
        train_loss / len(dataloader.dataset),
        accuracy(preds, all_labels).item(),
        precision(preds, all_labels).item(),
        recall(preds, all_labels).item(),
        f1(preds, all_labels).item(),
        auroc(all_probs, all_labels).item(),
    )


# === Test Step ===
def test_step(model, dataloader, loss_fn, device):
    model.eval()
    test_loss = 0

    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    auroc = BinaryAUROC().to(device)

    all_probs, all_labels = [], []

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).float()
            logits = model(X).squeeze()
            loss = loss_fn(logits, y)
            test_loss += loss.item() * X.size(0)
            probs = torch.sigmoid(logits)
            all_probs.append(probs)
            all_labels.append(y)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    preds = (all_probs > 0.5).float()

    return (
        test_loss / len(dataloader.dataset),
        accuracy(preds, all_labels).item(),
        precision(preds, all_labels).item(),
        recall(preds, all_labels).item(),
        f1(preds, all_labels).item(),
        auroc(all_probs, all_labels).item(),
    )


# === Training Loop ===
def train_TabTransformer(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    loss_fn,
    device,
    epochs=50,
    patience=10,
    scheduler=None,
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

    best_loss = float("inf")
    best_epoch = 0
    best_state = None

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = train_step(
            model, train_dataloader, loss_fn, optimizer, device
        )
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = test_step(
            model, test_dataloader, loss_fn, device
        )

        # Scheduler step if provided
        if scheduler is not None:
            scheduler.step(test_loss)

        # Save results
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

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch
            best_state = model.state_dict()
        elif epoch - best_epoch >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

        print(
            f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, ROC-AUC: {train_auc:.4f} || "
            f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, ROC-AUC: {test_auc:.4f}"
        )

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return results
