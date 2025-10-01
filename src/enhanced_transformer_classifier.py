#!/usr/bin/env python3
"""
Enhanced TransformerTextClassifier for Maximum Spam Recall
- [CLS] token classification (better than mean pooling)
- Optimized for all spam types (phishing, financial, romance, tech, crypto)
- Focal loss support for hard examples
- Advanced attention mechanisms
- Special token awareness
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EnhancedTransformerConfig:
    vocab_size: int
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    max_len: int = 512
    dropout: float = 0.1
    pad_idx: int = 0
    cls_idx: int = 1  # [CLS] token index
    # Special tokens for spam detection
    special_token_ids: Dict[str, int] = None


class MultiHeadAttentionWithBias(nn.Module):
    """Enhanced multi-head attention with learned positional bias"""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Learned attention bias for spam-specific patterns
        self.attention_bias = nn.Parameter(torch.zeros(1, nhead, 1, 1))
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # Compute Q, K, V
        q = self.q_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Attention scores with bias
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.attention_bias
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights and output
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection with residual connection
        out = self.out_linear(out)
        return self.layer_norm(x + self.dropout(out))


class SpamAwareTransformerLayer(nn.Module):
    """Transformer layer optimized for spam detection"""
    
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Enhanced attention
        self.self_attn = MultiHeadAttentionWithBias(d_model, nhead, dropout)
        
        # Feed-forward with spam-specific activation
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Self-attention
        x = self.self_attn(x, mask)
        
        # Feed-forward with residual
        ff_out = F.gelu(self.ff1(x))  # GELU works better for text
        ff_out = self.dropout1(ff_out)
        ff_out = self.ff2(ff_out)
        x = self.layer_norm2(x + self.dropout2(ff_out))
        
        return x


class EnhancedTransformerTextClassifier(nn.Module):
    """Enhanced Transformer for comprehensive spam detection"""
    
    def __init__(self, config: EnhancedTransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_idx,
        )
        
        # Learned positional embeddings
        self.pos_embedding = nn.Embedding(
            num_embeddings=config.max_len,
            embedding_dim=config.d_model,
        )
        
        # Special token type embeddings (for spam-specific tokens)
        self.token_type_embedding = nn.Embedding(10, config.d_model)  # 10 different token types
        
        # Transformer layers
        self.layers = nn.ModuleList([
            SpamAwareTransformerLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Classification head using [CLS] token
        self.cls_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.d_model // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 4, 1)  # Binary classification
        )
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    
    def get_token_type_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate token type IDs for special tokens"""
        token_type_ids = torch.zeros_like(input_ids)
        
        if self.config.special_token_ids:
            for token_name, token_id in self.config.special_token_ids.items():
                if token_name in ['URL', 'EMAIL', 'DOMAIN', 'PHONE']:
                    token_type_ids[input_ids == token_id] = 1
                elif token_name in ['MONEY', 'NUMBER']:
                    token_type_ids[input_ids == token_id] = 2
                elif token_name == 'URGENT':
                    token_type_ids[input_ids == token_id] = 3
        
        return token_type_ids
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced forward pass with CLS classification
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] - 1 for real tokens, 0 for padding
        
        Returns:
            logits: [batch_size] - raw logits for binary classification
        """
        batch_size, seq_len = input_ids.size()
        device = input_ids.device
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Positional embeddings
        positions = torch.arange(0, seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        pos_embeds = self.pos_embedding(positions)
        
        # Token type embeddings (for special tokens)
        token_type_ids = self.get_token_type_ids(input_ids)
        type_embeds = self.token_type_embedding(token_type_ids)
        
        # Combine embeddings
        x = token_embeds + pos_embeds + type_embeds
        x = self.dropout(x)
        
        # Build attention mask (1 for attend, 0 for ignore)
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_idx)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        x = self.layer_norm(x)
        
        # Use [CLS] token for classification (first token)
        cls_output = x[:, 0, :]  # [batch_size, d_model]
        
        # Classification
        logits = self.cls_classifier(cls_output).squeeze(-1)  # [batch_size]
        
        return logits


class FocalLoss(nn.Module):
    """Focal Loss for handling hard examples and class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Convert to probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def build_enhanced_model(config: EnhancedTransformerConfig) -> EnhancedTransformerTextClassifier:
    """Build enhanced transformer model"""
    return EnhancedTransformerTextClassifier(config)


def test_model():
    """Test the enhanced model"""
    config = EnhancedTransformerConfig(
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_layers=6,
        max_len=128
    )
    
    model = build_enhanced_model(config)
    
    # Test input
    batch_size = 2
    seq_len = 50
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        print(f"Model output shape: {logits.shape}")
        print(f"Sample logits: {logits}")
        
        # Convert to probabilities
        probs = torch.sigmoid(logits)
        print(f"Sample probabilities: {probs}")


if __name__ == "__main__":
    test_model()