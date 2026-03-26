import torch.nn as nn
from transformers import AutoModel


class MultiLabelEmotionClassifier(nn.Module):
    """BERT/RoBERTa encoder + linear head for multi-label emotion detection."""

    def __init__(self, model_name: str, num_labels: int = 6, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits  # raw logits — use BCEWithLogitsLoss during training
