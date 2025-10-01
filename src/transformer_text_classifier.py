#!/usr/bin/env python3
"""
Custom Transformer Encoder for Spam Classification (Pure DL, no pretrained models)
- Implements a Transformer encoder with learned positional embeddings
- Masked mean pooling for sequence representation
- Binary classification head (Spam vs Ham) using BCEWithLogitsLoss
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    max_len: int = 512
    dropout: float = 0.1
    pad_idx: int = 0


class TransformerTextClassifier(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx,
        )
        # Learned positional embeddings (simpler and effective for classification)
        self.pos_embedding = nn.Embedding(
            num_embeddings=config.max_len,
            embedding_dim=config.d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Binary classification head (single logit)
        self.classifier = nn.Linear(config.d_model, 1)

        # Initialize parameters
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: LongTensor of shape (batch, seq_len)
            attention_mask: Bool/BYTE/BFloat mask (batch, seq_len), 1 for tokens to attend, 0 for padding
        Returns:
            logits: FloatTensor of shape (batch,) — raw logits for BCEWithLogitsLoss
        """
        bsz, seq_len = input_ids.size()
        device = input_ids.device

        # Embedding + positional encoding
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # Build key padding mask for nn.Transformer (True indicates padding)
        if attention_mask is None:
            # Assume pad_idx is 0
            key_padding_mask = input_ids.eq(self.config.pad_idx)
        else:
            # attention_mask: 1 for real tokens, 0 for pad -> need bool pad mask
            key_padding_mask = ~attention_mask.bool()

        # Encode
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)

        # Masked mean pooling (exclude padding)
        if attention_mask is None:
            mask = (~key_padding_mask).float().unsqueeze(-1)  # (b, s, 1)
        else:
            mask = attention_mask.float().unsqueeze(-1)

        x = x * mask
        sum_x = x.sum(dim=1)  # (b, d)
        len_x = mask.sum(dim=1).clamp(min=1e-6)  # (b, 1)
        pooled = sum_x / len_x

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        return logits


def build_model(config: TransformerConfig) -> TransformerTextClassifier:
    return TransformerTextClassifier(config)
