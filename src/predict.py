"""
Generate predictions for the test set.

Usage:
    python src/predict.py --config configs/config.yaml --test_path test/test.csv
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from dataset import EmotionDataset, EMOTION_COLS, load_split
from model import MultiLabelEmotionClassifier
from evaluate import get_predictions


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main(cfg: dict, test_path: str, output_path: str):
    device = get_device()
    print(f"Using device: {device}")

    test_df = load_split(test_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])
    test_ds = EmotionDataset(test_df, tokenizer, cfg['max_length'])
    test_loader = DataLoader(test_ds, batch_size=cfg['batch_size'], shuffle=False)

    # Load model + thresholds
    model_path = os.path.join(cfg['output_dir'], 'best_model.pt')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = MultiLabelEmotionClassifier(cfg['model_name'], dropout=cfg['dropout'])
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    thresholds = checkpoint.get('thresholds', np.full(len(EMOTION_COLS), 0.5))
    print("Thresholds used:")
    for col, t in zip(EMOTION_COLS, thresholds):
        print(f"  {col:10s}: {t:.2f}")

    logits, _ = get_predictions(model, test_loader, device)
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= thresholds).astype(int)

    result_df = test_df[['text']].copy() if 'text' in test_df.columns else test_df.copy()
    for i, col in enumerate(EMOTION_COLS):
        result_df[col] = preds[:, i]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    print(f"Label distribution in predictions:")
    print(result_df[EMOTION_COLS].sum())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--test_path', default='test/test.csv')
    parser.add_argument('--output', default='submissions/run1.csv')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.test_path, args.output)
