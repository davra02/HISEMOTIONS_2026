import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from dataset import EMOTION_COLS


def get_predictions(model, dataloader: DataLoader, device: torch.device) -> tuple:
    """Run inference and return (logits, labels) as numpy arrays."""
    model.eval()
    all_logits, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids')
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            all_logits.append(logits.cpu().numpy())
            if 'labels' in batch:
                all_labels.append(batch['labels'].numpy())

    logits = np.vstack(all_logits)
    labels = np.vstack(all_labels) if all_labels else None
    return logits, labels


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return f1_score(y_true, y_pred, average='micro', zero_division=0)


def find_best_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    search_range: np.ndarray = None,
) -> tuple:
    """
    Search for the best per-label threshold maximising micro F1.

    Returns
    -------
    best_thresholds : np.ndarray  shape (num_labels,)
    best_f1        : float
    """
    if search_range is None:
        search_range = np.arange(0.10, 0.91, 0.05)

    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    num_labels = logits.shape[1]
    best_thresholds = np.full(num_labels, 0.5)
    best_f1 = 0.0

    # First pass: per-label greedy optimisation
    for label_idx in range(num_labels):
        best_t = 0.5
        best_label_f1 = -1.0
        for t in search_range:
            thresholds = best_thresholds.copy()
            thresholds[label_idx] = t
            preds = (probs >= thresholds).astype(int)
            f1 = micro_f1(labels, preds)
            if f1 > best_label_f1:
                best_label_f1 = f1
                best_t = t
        best_thresholds[label_idx] = best_t

    preds = (probs >= best_thresholds).astype(int)
    best_f1 = micro_f1(labels, preds)

    print("Thresholds per label:")
    for i, (col, t) in enumerate(zip(EMOTION_COLS, best_thresholds)):
        print(f"  {col:10s}: {t:.2f}")
    print(f"Micro F1 on dev: {best_f1:.4f}")

    return best_thresholds, best_f1


def evaluate_with_thresholds(
    logits: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= thresholds).astype(int)
    return micro_f1(labels, preds)
