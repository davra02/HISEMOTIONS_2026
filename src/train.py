"""
Training script for multi-label emotion detection (HISEMOTIONS @ IberLEF 2026).

Usage (local M4 or Colab):
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --merge_dev   # train on train+dev
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Allow running from project root or src/
sys.path.insert(0, os.path.dirname(__file__))
from dataset import EmotionDataset, EMOTION_COLS, load_split
from model import MultiLabelEmotionClassifier
from evaluate import get_predictions, find_best_thresholds, micro_f1


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def compute_pos_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """pos_weight = (# negative) / (# positive) per label."""
    counts = df[EMOTION_COLS].sum()
    n = len(df)
    weights = (n - counts) / counts.clip(lower=1)
    print("Positive weights per label:")
    for col, w in weights.items():
        print(f"  {col:10s}: {w:.1f}")
    return torch.tensor(weights.values, dtype=torch.float32).to(device)


def train_epoch(model, loader, optimizer, scheduler, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main(cfg: dict, merge_dev: bool = False):
    device = get_device()
    print(f"Using device: {device}")

    # --- Data ---
    train_df = load_split(cfg['train_path'])
    dev_df = load_split(cfg['dev_path'])

    if merge_dev:
        print("Merging train + dev for final training run.")
        train_df = pd.concat([train_df, dev_df], ignore_index=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

    train_ds = EmotionDataset(train_df, tokenizer, cfg['max_length'])
    dev_ds = EmotionDataset(dev_df, tokenizer, cfg['max_length'])

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=cfg.get('num_workers', 0), pin_memory=(device.type == 'cuda'),
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=cfg['batch_size'], shuffle=False,
        num_workers=cfg.get('num_workers', 0),
    )

    # --- Model ---
    model = MultiLabelEmotionClassifier(cfg['model_name'], dropout=cfg['dropout'])
    model.to(device)

    # --- Loss with class imbalance correction ---
    pos_weights = compute_pos_weights(train_df, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # --- Optimiser & Scheduler ---
    num_training_steps = len(train_loader) * cfg['num_epochs']
    num_warmup_steps = int(num_training_steps * cfg.get('warmup_ratio', 0.1))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['learning_rate'],
        weight_decay=cfg.get('weight_decay', 0.01),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # --- Training loop ---
    best_f1 = 0.0
    best_thresholds = np.full(len(EMOTION_COLS), 0.5)
    os.makedirs(cfg['output_dir'], exist_ok=True)
    model_path = os.path.join(cfg['output_dir'], 'best_model.pt')

    for epoch in range(1, cfg['num_epochs'] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)

        if not merge_dev:
            dev_logits, dev_labels = get_predictions(model, dev_loader, device)
            thresholds, f1 = find_best_thresholds(dev_logits, dev_labels)
            print(f"Epoch {epoch}/{cfg['num_epochs']} | loss={train_loss:.4f} | dev micro F1={f1:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_thresholds = thresholds
                torch.save({'model': model.state_dict(), 'thresholds': thresholds}, model_path)
                print(f"  -> Saved best model (F1={best_f1:.4f})")
        else:
            print(f"Epoch {epoch}/{cfg['num_epochs']} | loss={train_loss:.4f}")

    if merge_dev:
        torch.save({'model': model.state_dict(), 'thresholds': best_thresholds}, model_path)
        print(f"Final model saved to {model_path}")
    else:
        print(f"\nBest dev micro F1: {best_f1:.4f}")

    # Save thresholds separately for easy inspection
    thresh_path = os.path.join(cfg['output_dir'], 'thresholds.npy')
    np.save(thresh_path, best_thresholds)
    print(f"Thresholds saved to {thresh_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--merge_dev', action='store_true',
                        help='Train on train+dev (for final submission)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, merge_dev=args.merge_dev)
