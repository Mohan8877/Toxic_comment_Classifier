from __future__ import annotations

from transformers import DistilBertConfig, DistilBertForSequenceClassification


def build_model(vocab_size: int, embedding_dim: int = 768, max_len: int = 200, num_labels: int = 6):
    config = DistilBertConfig(
        vocab_size=vocab_size,
        max_position_embeddings=max_len,
        num_labels=num_labels,
        dim=embedding_dim,
        hidden_dim=embedding_dim * 2,
        n_layers=6,
        n_heads=12,
        dropout=0.1,
        attention_dropout=0.1,
        seq_classif_dropout=0.2,
    )
    model = DistilBertForSequenceClassification(config)
    return model
