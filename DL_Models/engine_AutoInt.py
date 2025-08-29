from tqdm.auto import tqdm
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAUROC,
)
import torch
import torch.nn as nn


# === Train Step ===
def train_step_autoint(model, dataloader, loss_fn, optimizer, device):
    """
    Single training step for AutoInt model.
    Supports numeric + optional categorical features.
    """
    model.train()
    train_loss = 0

    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    auroc = BinaryAUROC().to(device)

    all_probs, all_labels = [], []

    for batch in dataloader:
        # unpack batch: (x_numeric, [x_categ], y)
        if len(batch) == 3:
            x_num, x_cat, y = batch
            x_num, y = x_num.to(device), y.to(device).float()
            x_cat = [c.to(device) for c in x_cat]
            logits = model(x_num, x_cat).squeeze()
        else:
            x_num, y = batch
            x_num, y = x_num.to(device), y.to(device).float()
            logits = model(x_num).squeeze()

        probs = torch.sigmoid(logits)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        all_probs.append(probs.detach())
        all_labels.append(y.detach())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    preds = (all_probs > 0.5).long()
    all_labels = all_labels.long()

    acc = accuracy(preds, all_labels)
    prec = precision(preds, all_labels)
    rec = recall(preds, all_labels)
    f1_score = f1(preds, all_labels)
    roc_auc = auroc(all_probs, all_labels)

    avg_loss = train_loss / len(dataloader)
    return (
        avg_loss,
        acc.item(),
        prec.item(),
        rec.item(),
        f1_score.item(),
        roc_auc.item(),
    )


# === Test Step ===
def test_step_autoint(model, dataloader, loss_fn, device):
    """
    Single evaluation step for AutoInt model.
    Supports numeric + optional categorical features.
    """
    model.eval()
    test_loss = 0

    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    auroc = BinaryAUROC().to(device)

    all_probs, all_labels = [], []

    with torch.inference_mode():
        for batch in dataloader:
            if len(batch) == 3:
                x_num, x_cat, y = batch
                x_num, y = x_num.to(device), y.to(device).float()
                x_cat = [c.to(device) for c in x_cat]
                logits = model(x_num, x_cat).squeeze()
            else:
                x_num, y = batch
                x_num, y = x_num.to(device), y.to(device).float()
                logits = model(x_num).squeeze()

            probs = torch.sigmoid(logits)
            loss = loss_fn(logits, y)

            test_loss += loss.item()
            all_probs.append(probs)
            all_labels.append(y)

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    preds = (all_probs > 0.5).long()
    all_labels = all_labels.long()

    acc = accuracy(preds, all_labels)
    prec = precision(preds, all_labels)
    rec = recall(preds, all_labels)
    f1_score = f1(preds, all_labels)
    roc_auc = auroc(all_probs, all_labels)

    avg_loss = test_loss / len(dataloader)
    return (
        avg_loss,
        acc.item(),
        prec.item(),
        rec.item(),
        f1_score.item(),
        roc_auc.item(),
    )


# === Training Loop ===
# === Training Loop with Early Stopping and Final Results ===
def train_AutoInt(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int = 5,
    patience: int = 20,   # early stopping
):
    """
    Complete training loop for AutoInt with metrics tracking + early stopping.
    """
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
    patience_counter = 0

    for epoch in tqdm(range(epochs)):
        # ---- Train step ----
        train_loss, train_acc, train_prec, train_rec, train_f1, train_auc = (
            train_step_autoint(model, train_dataloader, loss_fn, optimizer, device)
        )

        # ---- Eval step ----
        test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = test_step_autoint(
            model, test_dataloader, loss_fn, device
        )

        # ---- Store results ----
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
            f"Epoch: {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | "
            f"Recall: {train_rec:.4f} | F1: {train_f1:.4f} | ROC-AUC: {train_auc:.4f} || "
            f"Test Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Prec: {test_prec:.4f} | "
            f"Recall: {test_rec:.4f} | F1: {test_f1:.4f} | ROC-AUC: {test_auc:.4f}"
        )

        # ---- Early Stopping Check ----
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1} (patience={patience})\n")
            break

    # === Final Summary ===
    print("\n===== Final Training Results =====")
    print(f"Train Loss: {results['train_loss'][-1]:.4f} | "
          f"Acc: {results['train_acc'][-1]:.4f} | "
          f"Prec: {results['train_precision'][-1]:.4f} | "
          f"Recall: {results['train_recall'][-1]:.4f} | "
          f"F1: {results['train_f1'][-1]:.4f} | "
          f"ROC-AUC: {results['train_roc_auc'][-1]:.4f}")

    print(f"Test Loss: {results['test_loss'][-1]:.4f} | "
          f"Acc: {results['test_acc'][-1]:.4f} | "
          f"Prec: {results['test_precision'][-1]:.4f} | "
          f"Recall: {results['test_recall'][-1]:.4f} | "
          f"F1: {results['test_f1'][-1]:.4f} | "
          f"ROC-AUC: {results['test_roc_auc'][-1]:.4f}")
    print("==================================\n")

    return results

