#!/usr/bin/env python3
"""
Advanced Training for Enhanced Transformer with Subword Tokenization
- Implements SentencePiece BPE tokenization for better rare word handling
- Focal loss for hard examples and class imbalance
- Advanced learning rate scheduling and regularization
- Extensive validation on diverse spam types
"""

import os
import json
import math
import argparse
import time
import pickle
from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Install if not available: pip install sentencepiece
try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    print("⚠️ SentencePiece not available. Install with: pip install sentencepiece")
    SENTENCEPIECE_AVAILABLE = False

# Local imports
from enhanced_spam_preprocessor import EnhancedSpamPreprocessor
from enhanced_transformer_classifier import (
    EnhancedTransformerConfig, 
    EnhancedTransformerTextClassifier,
    FocalLoss,
    build_enhanced_model
)


class SubwordTokenizer:
    """Subword tokenizer using SentencePiece BPE"""
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.model_file = None
        self.sp_model = None
        
    def train_tokenizer(self, texts: List[str], model_prefix: str = "spam_tokenizer"):
        """Train SentencePiece tokenizer on corpus"""
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("SentencePiece not available")
            
        # Write training data to temp file
        train_file = f"{model_prefix}_train.txt"
        with open(train_file, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(f"{text}\n")
        
        # Train SentencePiece model
        model_file = f"{model_prefix}.model"
        spm.SentencePieceTrainer.Train(
            input=train_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',  # Byte-Pair Encoding
            character_coverage=0.9995,
            normalization_rule_name='nmt_nfkc_cf',
            # Spam-specific settings
            user_defined_symbols=['[URL]', '[EMAIL]', '[DOMAIN]', '[PHONE]', '[MONEY]', '[URGENT]'],
            pad_id=0,
            unk_id=1,
            bos_id=2,  # beginning of sentence
            eos_id=3,  # end of sentence
            control_symbols=['[PAD]', '[UNK]', '[BOS]', '[EOS]']
        )
        
        # Load trained model
        self.model_file = model_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_file)
        
        # Clean up temp file
        os.remove(train_file)
        
        print(f"✅ Trained subword tokenizer with vocab size: {self.sp_model.GetPieceSize()}")
        
    def load_tokenizer(self, model_file: str):
        """Load existing tokenizer"""
        if not SENTENCEPIECE_AVAILABLE:
            raise ImportError("SentencePiece not available")
            
        self.model_file = model_file
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.Load(model_file)
        print(f"✅ Loaded subword tokenizer with vocab size: {self.sp_model.GetPieceSize()}")
        
    def encode(self, text: str) -> List[int]:
        """Encode text to subword token IDs"""
        if self.sp_model is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.sp_model.EncodeAsIds(text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text"""
        if self.sp_model is None:
            raise ValueError("Tokenizer not trained or loaded")
        return self.sp_model.DecodeIds(ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        if self.sp_model is None:
            return self.vocab_size
        return self.sp_model.GetPieceSize()


class AdvancedSpamDataset(Dataset):
    """Advanced dataset with subword tokenization support"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: SubwordTokenizer, 
                 preprocessor: EnhancedSpamPreprocessor, max_len: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # Preprocess text
        processed_text = self.preprocessor.preprocess(text)
        
        # Tokenize with subwords
        token_ids = self.tokenizer.encode(processed_text)
        
        # Add [CLS] token at beginning
        cls_id = 2  # [BOS] serves as [CLS]
        token_ids = [cls_id] + token_ids
        
        # Truncate if too long
        if len(token_ids) > self.max_len:
            token_ids = token_ids[:self.max_len]
        
        return token_ids, label

    def collate_fn(self, batch):
        seqs, labels = zip(*batch)
        
        # Dynamic padding
        max_len_batch = min(max(len(s) for s in seqs), self.max_len)
        batch_size = len(seqs)

        input_ids = torch.zeros((batch_size, max_len_batch), dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len_batch), dtype=torch.long)

        for i, seq in enumerate(seqs):
            seq_len = min(len(seq), max_len_batch)
            input_ids[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)
            attention_mask[i, :seq_len] = 1

        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return input_ids, attention_mask, labels_tensor


class FocalLossAdvanced(nn.Module):
    """Advanced Focal Loss with label smoothing and class balancing"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, label_smoothing: float = 0.0,
                 pos_weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # Convert to probabilities
        probs = torch.sigmoid(inputs)
        
        # Compute BCE loss
        import torch.nn.functional as F
        if self.pos_weight is not None:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth, 
                                                         pos_weight=self.pos_weight, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets_smooth, reduction='none')
        
        # Compute focal weight
        p_t = probs * targets_smooth + (1 - probs) * (1 - targets_smooth)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def load_comprehensive_dataset(csv_path: str) -> pd.DataFrame:
    """Load comprehensive spam dataset"""
    df = pd.read_csv(csv_path)
    
    # Normalize columns to text/label
    if {"text", "label"}.issubset(df.columns):
        df = df[["text", "label"]].copy()
    elif {"v1", "v2"}.issubset(df.columns):
        df = df[["v2", "v1"]].copy()
        df.columns = ["text", "label"]
    elif len(df.columns) >= 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["text", "label"]
    else:
        raise ValueError("Unsupported dataset format")

    # Normalize labels to 0/1
    if df["label"].dtype == "O":
        df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})
    
    # Drop missing
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    return df


def compute_advanced_metrics(y_true: List[int], y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute comprehensive metrics for spam detection"""
    y_pred = (y_scores >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Class-specific metrics
    spam_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    spam_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spam_f1 = 2 * (spam_precision * spam_recall) / (spam_precision + spam_recall) if (spam_precision + spam_recall) > 0 else 0.0
    
    ham_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    ham_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ham_f1 = 2 * (ham_precision * ham_recall) / (ham_precision + ham_recall) if (ham_precision + ham_recall) > 0 else 0.0
    
    return {
        "accuracy": acc,
        "precision": prec, 
        "recall": rec,
        "f1": f1,
        "spam_precision": spam_precision,
        "spam_recall": spam_recall,
        "spam_f1": spam_f1,
        "ham_precision": ham_precision,
        "ham_recall": ham_recall,
        "ham_f1": ham_f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }


def cosine_annealing_with_restarts(optimizer, T_0: int, T_mult: int = 2, eta_min: float = 1e-7):
    """Cosine annealing with warm restarts"""
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
    )


def main():
    parser = argparse.ArgumentParser(description="Advanced Training for Enhanced Transformer")
    parser.add_argument("--data", type=str, default="data/comprehensive_spam_dataset.csv", 
                       help="Path to comprehensive dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--vocab-size", type=int, default=32000, help="Subword vocab size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--focal-loss", action="store_true", help="Use advanced focal loss")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Label smoothing")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--save-dir", type=str, default="models", help="Save directory")
    parser.add_argument("--tokenizer-path", type=str, help="Path to existing tokenizer")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("ADVANCED TRANSFORMER TRAINING WITH SUBWORD TOKENIZATION")
    print("=" * 80)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load comprehensive dataset
    print(f"\nLoading dataset: {args.data}")
    df = load_comprehensive_dataset(args.data)
    print(f"Dataset size: {len(df)}")
    spam_count = int(df["label"].sum())
    ham_count = len(df) - spam_count
    print(f"Ham: {ham_count} | Spam: {spam_count} | Spam ratio: {spam_count/len(df):.3f}")

    # Initialize preprocessor
    print("\nInitializing enhanced preprocessor...")
    preprocessor = EnhancedSpamPreprocessor()
    
    # Train/Val/Test split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Initialize or load subword tokenizer
    tokenizer = SubwordTokenizer(vocab_size=args.vocab_size)
    
    if args.tokenizer_path and os.path.exists(args.tokenizer_path):
        print(f"Loading existing tokenizer: {args.tokenizer_path}")
        tokenizer.load_tokenizer(args.tokenizer_path)
    else:
        if not SENTENCEPIECE_AVAILABLE:
            print("❌ SentencePiece not available. Using fallback word-level tokenization.")
            # Fallback to existing word-level approach
            from train_enhanced_transformer import build_enhanced_vocab, texts_to_ids_enhanced, EnhancedSpamDataset
            vocab = build_enhanced_vocab(train_df["text"].tolist(), preprocessor)
            special_tokens = preprocessor.get_special_tokens()
            
            # Create datasets with word-level tokenization
            train_ids = texts_to_ids_enhanced(train_df["text"].tolist(), vocab, preprocessor)
            val_ids = texts_to_ids_enhanced(val_df["text"].tolist(), vocab, preprocessor)
            test_ids = texts_to_ids_enhanced(test_df["text"].tolist(), vocab, preprocessor)

            train_ds = EnhancedSpamDataset(train_ids, train_df["label"].tolist(), args.max_len, vocab, special_tokens)
            val_ds = EnhancedSpamDataset(val_ids, val_df["label"].tolist(), args.max_len, vocab, special_tokens)
            test_ds = EnhancedSpamDataset(test_ids, test_df["label"].tolist(), args.max_len, vocab, special_tokens)
            
            vocab_size = len(vocab)
            
        else:
            print("Training subword tokenizer...")
            # Preprocess training texts for tokenizer training
            train_texts_processed = [preprocessor.preprocess(text) for text in train_df["text"].tolist()]
            tokenizer_path = os.path.join(args.save_dir, "spam_tokenizer")
            tokenizer.train_tokenizer(train_texts_processed, tokenizer_path)
            
            # Create datasets with subword tokenization
            train_ds = AdvancedSpamDataset(train_df["text"].tolist(), train_df["label"].tolist(), 
                                         tokenizer, preprocessor, args.max_len)
            val_ds = AdvancedSpamDataset(val_df["text"].tolist(), val_df["label"].tolist(),
                                       tokenizer, preprocessor, args.max_len)  
            test_ds = AdvancedSpamDataset(test_df["text"].tolist(), test_df["label"].tolist(),
                                        tokenizer, preprocessor, args.max_len)
            
            vocab_size = tokenizer.get_vocab_size()

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=train_ds.collate_fn, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=val_ds.collate_fn, pin_memory=(device.type=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=test_ds.collate_fn, pin_memory=(device.type=="cuda"))

    # Build model with enhanced config
    print(f"\nBuilding model with vocab size: {vocab_size}")
    config = EnhancedTransformerConfig(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        max_len=args.max_len,
        dropout=args.dropout,
        pad_idx=0,  # SentencePiece pad token
        cls_idx=2,  # SentencePiece BOS token serves as CLS
    )
    
    model = build_enhanced_model(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Advanced loss function
    pos_weight = torch.tensor([max(1.0, ham_count / max(1, spam_count))], device=device)
    
    if args.focal_loss:
        criterion = FocalLossAdvanced(
            alpha=1.0, gamma=2.0, label_smoothing=args.label_smoothing,
            pos_weight=pos_weight, reduction='mean'
        )
        print(f"Using Advanced Focal Loss (gamma=2.0, label_smoothing={args.label_smoothing})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using weighted BCE Loss (pos_weight={pos_weight.item():.2f})")

    # Advanced optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    scheduler = cosine_annealing_with_restarts(optimizer, T_0=len(train_loader)*5, eta_min=1e-7)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    best_val_recall = 0.0
    best_epoch = -1
    epochs_no_improve = 0

    def run_eval(data_loader, split_name=""):
        model.eval()
        all_scores = []
        all_labels = []
        
        with torch.no_grad():
            for input_ids, attn_mask, labels in data_loader:
                input_ids = input_ids.to(device, non_blocking=True)
                attn_mask = attn_mask.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                logits = model(input_ids, attn_mask)
                probs = torch.sigmoid(logits)
                
                all_scores.append(probs.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())
        
        y_scores = np.concatenate(all_scores)
        y_true = np.concatenate(all_labels).astype(int)
        
        return compute_advanced_metrics(y_true, y_scores, threshold=0.5)

    print(f"\nStarting advanced training (optimizing for spam recall)...")
    training_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        
        epoch_start = time.time()
        
        for batch_idx, (input_ids, attn_mask, labels) in enumerate(train_loader, 1):
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=(device.type == "cuda")):
                logits = model(input_ids, attn_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch:02d} | Batch {batch_idx:04d}/{len(train_loader)} | "
                      f"Loss: {running_loss/batch_idx:.4f} | "
                      f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        
        # Validation
        val_metrics = run_eval(val_loader, "VAL")
        
        print(f"\nEpoch {epoch:02d} ({epoch_time:.1f}s) - Loss: {avg_loss:.4f}")
        print(f"VAL => Acc: {val_metrics['accuracy']:.4f} | "
              f"Spam Rec: {val_metrics['spam_recall']:.4f} | "
              f"Ham Acc: {val_metrics['ham_recall']:.4f} | "
              f"F1: {val_metrics['f1']:.4f}")

        # Early stopping based on spam recall
        current_recall = val_metrics['spam_recall']
        
        if current_recall > best_val_recall:
            best_val_recall = current_recall
            best_epoch = epoch
            epochs_no_improve = 0
            
            # Save best model
            save_path = os.path.join(args.save_dir, "advanced_transformer_best.pt")
            torch.save({
                'model_state': model.state_dict(),
                'config': config.__dict__,
                'tokenizer_path': getattr(tokenizer, 'model_file', None),
                'metrics': val_metrics,
                'epoch': epoch
            }, save_path)
            
            print(f"🎯 New best spam recall: {best_val_recall:.4f} (saved)")
            
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{args.patience} epochs")
            
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    training_time = time.time() - training_start
    print(f"\nTraining completed in {training_time/60:.1f} minutes")
    print(f"Best spam recall: {best_val_recall:.4f} at epoch {best_epoch}")

    # Final test evaluation
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    
    test_metrics = run_eval(test_loader, "TEST")
    
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Spam Precision: {test_metrics['spam_precision']:.4f}")
    print(f"Spam Recall: {test_metrics['spam_recall']:.4f}")
    print(f"Ham Recall: {test_metrics['ham_recall']:.4f}")
    print(f"Overall F1: {test_metrics['f1']:.4f}")
    print(f"False Negatives: {test_metrics['false_negatives']}")
    print(f"False Positives: {test_metrics['false_positives']}")


if __name__ == "__main__":
    main()