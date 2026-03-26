import torch
from torch.utils.data import Dataset
import pandas as pd

EMOTION_COLS = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'hope']


def load_split(path: str) -> pd.DataFrame:
    """Load a CSV split, drop rows with missing text, fill missing labels with 0."""
    df = pd.read_csv(path)
    df = df.dropna(subset=['text']).reset_index(drop=True)
    for col in EMOTION_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)
    return df


class EmotionDataset(Dataset):
    """PyTorch Dataset for multi-label emotion classification."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 256):
        self.texts = df['text'].tolist()
        self.labels = (
            df[EMOTION_COLS].values.astype('float32')
            if all(c in df.columns for c in EMOTION_COLS)
            else None
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
        if 'token_type_ids' in encoding:
            item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item
