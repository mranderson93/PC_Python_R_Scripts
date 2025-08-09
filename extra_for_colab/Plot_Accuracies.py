import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

# Create folder if not exists
os.makedirs("images", exist_ok=True)

def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training & testing curves (loss, accuracy, precision, recall, F1, ROC-AUC)."""

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(20, 18))

    # 1. Loss Plot
    plt.subplot(3, 2, 1)
    plt.plot(epochs, results['train_loss'], label='Train Loss')
    plt.plot(epochs, results['test_loss'], label='Test Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # 2. Accuracy Plot
    plt.subplot(3, 2, 2)
    plt.plot(epochs, results['train_acc'], label='Train Accuracy')
    plt.plot(epochs, results['test_acc'], label='Test Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    # 3. Precision Plot
    plt.subplot(3, 2, 3)
    plt.plot(epochs, results['train_precision'], label='Train Precision')
    plt.plot(epochs, results['test_precision'], label='Test Precision')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.legend()

    # 4. Recall Plot
    plt.subplot(3, 2, 4)
    plt.plot(epochs, results['train_recall'], label='Train Recall')
    plt.plot(epochs, results['test_recall'], label='Test Recall')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.legend()

    # 5. F1 Score Plot
    plt.subplot(3, 2, 5)
    plt.plot(epochs, results['train_f1'], label='Train F1 Score')
    plt.plot(epochs, results['test_f1'], label='Test F1 Score')
    plt.title('F1 Score')
    plt.xlabel('Epochs')
    plt.legend()

    # 6. ROC-AUC Plot
    plt.subplot(3, 2, 6)
    plt.plot(epochs, results['train_roc_auc'], label='Train ROC-AUC')
    plt.plot(epochs, results['test_roc_auc'], label='Test ROC-AUC')
    plt.title('ROC-AUC')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig("images/Transformer_Training_Testing_Metrics.png", dpi=600)
    plt.show()
