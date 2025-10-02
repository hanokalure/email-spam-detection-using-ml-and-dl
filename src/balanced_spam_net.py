#!/usr/bin/env python3
"""
BalancedSpamNet - Dual-Head Deep Learning Architecture for Balanced Spam Detection

Key Innovations:
1. Dual-Head Architecture: Separate specialized heads for SPAM and HAM detection
2. Hybrid Feature Extraction: CNN + BiLSTM + Attention mechanisms
3. Business Context Awareness: Special handling for legitimate business emails
4. Balanced Training: Equal focus on spam and ham during training
5. Advanced Loss Functions: Focal loss + class balancing

Architecture Overview:
Input → Enhanced Preprocessing → Embedding → CNN Features → BiLSTM Context → 
Dual Attention → [SPAM Head | HAM Head] → Balanced Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import math


class BusinessContextEmbedding(nn.Module):
    """Enhanced embedding layer with business context awareness"""
    
    def __init__(self, vocab_size: int, embed_dim: int, business_vocab_size: int = 1000):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Main embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Business context embedding (for banking, finance, commerce terms)
        self.business_embedding = nn.Embedding(business_vocab_size, embed_dim // 4)
        
        # Learnable combination weights
        self.context_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, input_ids: torch.Tensor, business_mask: Optional[torch.Tensor] = None):
        # Main embeddings
        embeddings = self.embedding(input_ids)
        
        if business_mask is not None:
            # Enhanced embeddings for business terms
            business_embeddings = self.business_embedding(business_mask)
            # Pad business embeddings to match main embedding dimension
            business_embeddings = F.pad(business_embeddings, (0, self.embed_dim - self.embed_dim // 4))
            embeddings = embeddings + self.context_weight * business_embeddings
            
        return embeddings


class MultiScaleCNN(nn.Module):
    """Multi-scale CNN for pattern detection at different granularities"""
    
    def __init__(self, embed_dim: int, num_filters: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_filters = num_filters
        
        # Multiple kernel sizes for different pattern scales
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=3, padding=1),  # Local patterns
            nn.Conv1d(embed_dim, num_filters, kernel_size=5, padding=2),  # Medium patterns  
            nn.Conv1d(embed_dim, num_filters, kernel_size=7, padding=3),  # Long patterns
            nn.Conv1d(embed_dim, num_filters, kernel_size=1, padding=0),  # Point-wise
        ])
        
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(num_filters * 4)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, embed_dim]
        x = x.transpose(1, 2)  # [batch_size, embed_dim, seq_len]
        
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x))  # [batch_size, num_filters, seq_len]
            conv_outputs.append(conv_out)
            
        # Concatenate all conv outputs
        combined = torch.cat(conv_outputs, dim=1)  # [batch_size, num_filters*4, seq_len]
        combined = combined.transpose(1, 2)  # [batch_size, seq_len, num_filters*4]
        
        combined = self.dropout(combined)
        combined = self.layer_norm(combined)
        
        return combined


class ContextualBiLSTM(nn.Module):
    """Bidirectional LSTM with residual connections"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, bidirectional=True, dropout=0.2
        )
        
        # Residual connection projection
        self.residual_proj = nn.Linear(input_dim, hidden_dim * 2)
        self.dropout = nn.Dropout(0.3)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.bilstm(x)  # [batch_size, seq_len, hidden_dim*2]
        
        # Residual connection
        residual = self.residual_proj(x)
        combined = lstm_out + residual
        
        combined = self.dropout(combined)
        combined = self.layer_norm(combined)
        
        return combined


class DualHeadAttention(nn.Module):
    """Dual attention mechanism for SPAM and HAM specific pattern focus"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # Separate attention for SPAM and HAM detection
        self.spam_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        self.ham_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # Learnable query vectors
        self.spam_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.ham_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        self.dropout = nn.Dropout(0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        
        # Expand query vectors for batch
        spam_query = self.spam_query.expand(batch_size, -1, -1)
        ham_query = self.ham_query.expand(batch_size, -1, -1)
        
        # SPAM-focused attention
        spam_attended, spam_weights = self.spam_attention(
            spam_query, x, x, key_padding_mask=attention_mask
        )
        spam_attended = self.dropout(spam_attended)
        spam_attended = self.layer_norm(spam_attended)
        
        # HAM-focused attention  
        ham_attended, ham_weights = self.ham_attention(
            ham_query, x, x, key_padding_mask=attention_mask
        )
        ham_attended = self.dropout(ham_attended)
        ham_attended = self.layer_norm(ham_attended)
        
        return spam_attended.squeeze(1), ham_attended.squeeze(1)


class BalancedSpamNet(nn.Module):
    """
    Dual-Head Architecture for Balanced Spam Detection
    
    Solves the critical HAM detection problem through specialized architecture
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        cnn_filters: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # 1. Enhanced Embedding with Business Context
        self.embedding = BusinessContextEmbedding(vocab_size, embed_dim)
        
        # 2. Multi-Scale CNN Feature Extraction
        self.cnn = MultiScaleCNN(embed_dim, cnn_filters)
        cnn_output_dim = cnn_filters * 4
        
        # 3. Contextual BiLSTM
        self.bilstm = ContextualBiLSTM(cnn_output_dim, lstm_hidden, lstm_layers)
        lstm_output_dim = lstm_hidden * 2
        
        # 4. Dual-Head Attention Mechanism
        self.dual_attention = DualHeadAttention(lstm_output_dim, attention_heads)
        
        # 5. Specialized Classification Heads
        self.spam_head = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 4, 1)
        )
        
        self.ham_head = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, lstm_output_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(lstm_output_dim // 4, 1)
        )
        
        # 6. Final Combination Layer
        self.final_classifier = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create attention mask (True for padding tokens)"""
        return (input_ids == 0)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        business_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        # Create attention mask
        attention_mask = self.create_attention_mask(input_ids)
        
        # 1. Enhanced Embedding
        embedded = self.embedding(input_ids, business_mask)
        
        # 2. CNN Feature Extraction
        cnn_features = self.cnn(embedded)
        
        # 3. BiLSTM Context Modeling
        lstm_output = self.bilstm(cnn_features)
        
        # 4. Dual-Head Attention
        spam_features, ham_features = self.dual_attention(lstm_output, attention_mask)
        
        # 5. Specialized Head Predictions
        spam_logits = self.spam_head(spam_features)  # [batch_size, 1]
        ham_logits = self.ham_head(ham_features)     # [batch_size, 1]
        
        # 6. Final Combined Prediction
        combined_features = torch.cat([spam_logits, ham_logits], dim=1)  # [batch_size, 2]
        final_logits = self.final_classifier(combined_features)  # [batch_size, 1]
        
        outputs = {
            'logits': final_logits.squeeze(1),  # [batch_size]
            'spam_logits': spam_logits.squeeze(1),
            'ham_logits': ham_logits.squeeze(1),
            'spam_features': spam_features,
            'ham_features': ham_features
        }
        
        return outputs


class BalancedFocalLoss(nn.Module):
    """
    Balanced Focal Loss for addressing class imbalance and hard examples
    Combines focal loss with dual-head balancing
    """
    
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, spam_weight: float = 1.0, ham_weight: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.spam_weight = spam_weight
        self.ham_weight = ham_weight
        
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Main classification loss
        logits = outputs['logits']
        spam_logits = outputs['spam_logits']
        ham_logits = outputs['ham_logits']
        
        # Focal loss for main prediction
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        main_loss = -alpha_weight * focal_weight * torch.log(pt + 1e-8)
        main_loss = main_loss.mean()
        
        # Specialized head losses
        spam_targets = targets.float()
        ham_targets = 1.0 - targets.float()
        
        spam_loss = F.binary_cross_entropy_with_logits(spam_logits, spam_targets, reduction='mean')
        ham_loss = F.binary_cross_entropy_with_logits(ham_logits, ham_targets, reduction='mean')
        
        # Weighted combination
        total_loss = (
            main_loss + 
            self.spam_weight * spam_loss + 
            self.ham_weight * ham_loss
        )
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'spam_loss': spam_loss,
            'ham_loss': ham_loss
        }


def create_balanced_spam_net(vocab_size: int, **kwargs) -> BalancedSpamNet:
    """Factory function to create BalancedSpamNet model"""
    return BalancedSpamNet(vocab_size=vocab_size, **kwargs)


if __name__ == "__main__":
    # Test model creation
    vocab_size = 50000
    model = create_balanced_spam_net(vocab_size)
    
    # Test forward pass
    batch_size = 4
    seq_len = 128
    
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, 2, (batch_size,)).float()
    
    outputs = model(input_ids)
    
    print("BalancedSpamNet Test:")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"SPAM head shape: {outputs['spam_logits'].shape}")
    print(f"HAM head shape: {outputs['ham_logits'].shape}")
    
    # Test loss
    criterion = BalancedFocalLoss()
    loss_dict = criterion(outputs, targets)
    print(f"Total loss: {loss_dict['total_loss']:.4f}")
    print("✅ BalancedSpamNet architecture test passed!")