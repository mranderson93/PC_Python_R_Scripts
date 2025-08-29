import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinaryAUROC
from tqdm.auto import tqdm
def train_step_NODE(model, dataloader, loss_fn, optimizer, device):
    model.train()
    train_loss = 0

    # Metrics
    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    auroc = BinaryAUROC().to(device)

    all_probs, all_labels = [], []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * x.size(0)
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
        auroc(all_probs, all_labels).item()
    )

def test_step_NODE(model, dataloader, loss_fn, device):
    model.eval()
    test_loss = 0

    accuracy = BinaryAccuracy().to(device)
    precision = BinaryPrecision().to(device)
    recall = BinaryRecall().to(device)
    f1 = BinaryF1Score().to(device)
    auroc = BinaryAUROC().to(device)

    all_probs, all_labels = [], []

    with torch.inference_mode():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            logits = model(x)
            loss = loss_fn(logits, y)

            test_loss += loss.item() * x.size(0)
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
        auroc(all_probs, all_labels).item()
    )

# ==============================
# Training Loop
# ==============================
# ==============================
# Training Loop with Early Stopping
# ==============================
def train_NODE(
    model,
    train_dataloader,
    test_dataloader,
    optimizer,
    loss_fn,
    device,
    epochs=100,
    patience=20  # early stopping patience
):
    results = {
        "train_loss": [], "train_acc": [], "train_precision": [], "train_recall": [], "train_f1": [], "train_roc_auc": [],
        "test_loss": [], "test_acc": [], "test_precision": [], "test_recall": [], "test_f1": [], "test_roc_auc": [],
    }

    best_loss = float("inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in tqdm(range(epochs)):
        train_metrics = train_step_NODE(model, train_dataloader, loss_fn, optimizer, device)
        test_metrics = test_step_NODE(model, test_dataloader, loss_fn, device)

        results["train_loss"].append(train_metrics[0])
        results["train_acc"].append(train_metrics[1])
        results["train_precision"].append(train_metrics[2])
        results["train_recall"].append(train_metrics[3])
        results["train_f1"].append(train_metrics[4])
        results["train_roc_auc"].append(train_metrics[5])

        results["test_loss"].append(test_metrics[0])
        results["test_acc"].append(test_metrics[1])
        results["test_precision"].append(test_metrics[2])
        results["test_recall"].append(test_metrics[3])
        results["test_f1"].append(test_metrics[4])
        results["test_roc_auc"].append(test_metrics[5])

        # Early stopping check
        if test_metrics[0] < best_loss:
            best_loss = test_metrics[0]
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_metrics[0]:.4f}, Acc: {train_metrics[1]:.4f}, "
        f"Precision: {train_metrics[2]:.4f}, Recall: {train_metrics[3]:.4f}, "
        f"F1: {train_metrics[4]:.4f}, ROC-AUC: {train_metrics[5]:.4f} || "
        f"Test Loss: {test_metrics[0]:.4f}, Acc: {test_metrics[1]:.4f}, "
        f"Precision: {test_metrics[2]:.4f}, Recall: {test_metrics[3]:.4f}, "
        f"F1: {test_metrics[4]:.4f}, ROC-AUC: {test_metrics[5]:.4f}"
    )

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return results
