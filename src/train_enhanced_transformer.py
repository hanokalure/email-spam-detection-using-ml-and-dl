#!/usr/bin/env python3
"""
Train Enhanced Transformer for Maximum Spam Recall
- Uses enhanced preprocessor with URL/domain preservation
- Enhanced transformer architecture with CLS classification
- Comprehensive dataset with all spam types
- Focal loss for hard examples and maximum recall
- GPU-optimized training

Usage:
  python src/train_enhanced_transformer.py --data data/comprehensive_spam_dataset.csv --epochs 25 --batch-size 16 --max-recall
"""

import os
import json
import math
import argparse
import time
from collections import Counter
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Local imports
from enhanced_spam_preprocessor import EnhancedSpamPreprocessor
from enhanced_transformer_classifier import (
    EnhancedTransformerConfig, 
    EnhancedTransformerTextClassifier,
    FocalLoss,
    build_enhanced_model
)


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


def build_enhanced_vocab(texts: List[str], preprocessor: EnhancedSpamPreprocessor, 
                        min_freq: int = 2, max_size: int = 50000) -> Dict[str, int]:
    """Build vocabulary including special tokens for spam detection"""
    
    # Start with special tokens
    special_tokens = preprocessor.get_special_tokens()
    vocab = {token: idx for idx, token in enumerate(special_tokens.values())}
    
    # Count words
    counter = Counter()
    for text in texts:
        processed = preprocessor.preprocess(text)
        for word in processed.split():
            # Skip special tokens (already in vocab)
            if word not in vocab:
                counter[word] += 1
    
    # Add frequent words
    frequent_words = [(w, c) for w, c in counter.items() if c >= min_freq]
    frequent_words.sort(key=lambda x: (-x[1], x[0]))
    
    for word, _ in frequent_words[:max_size - len(vocab)]:
        vocab[word] = len(vocab)
    
    return vocab


def texts_to_ids_enhanced(texts: List[str], vocab: Dict[str, int], preprocessor: EnhancedSpamPreprocessor) -> List[List[int]]:
    """Convert texts to token IDs using enhanced preprocessor"""
    special_tokens = preprocessor.get_special_tokens()
    unk_idx = vocab.get(special_tokens['UNK'], 1)
    
    ids_list = []
    for text in texts:
        processed = preprocessor.preprocess(text)
        tokens = processed.split()
        ids = [vocab.get(token, unk_idx) for token in tokens]
        ids_list.append(ids)
    
    return ids_list


class EnhancedSpamDataset(Dataset):
    """Enhanced dataset with special token support"""
    
    def __init__(self, ids_list: List[List[int]], labels: List[int], max_len: int, 
                 vocab: Dict[str, int], special_tokens: Dict[str, str]):
        self.ids_list = ids_list
        self.labels = labels
        self.max_len = max_len
        
        # Get special token IDs
        self.pad_idx = vocab[special_tokens['PAD']]
        self.cls_idx = vocab[special_tokens['CLS']]
        
    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        return self.ids_list[idx], int(self.labels[idx])

    def collate_fn(self, batch):
        seqs, labels = zip(*batch)
        
        # Dynamic padding to max_len or max in batch
        max_in_batch = min(max(len(s) for s in seqs), self.max_len)
        batch_size = len(seqs)

        input_ids = torch.full((batch_size, max_in_batch), self.pad_idx, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_in_batch), dtype=torch.long)

        for i, seq in enumerate(seqs):
            # Truncate if needed
            trunc_seq = seq[:max_in_batch]
            input_ids[i, :len(trunc_seq)] = torch.tensor(trunc_seq, dtype=torch.long)
            attention_mask[i, :len(trunc_seq)] = 1

        labels_tensor = torch.tensor(labels, dtype=torch.float32)
        return input_ids, attention_mask, labels_tensor


def warmup_cosine_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    """Learning rate scheduler with warmup and cosine decay"""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_metrics_with_recall_focus(y_true: List[int], y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute metrics with focus on recall (catching all spam)"""
    y_pred = (y_scores >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    
    # Confusion matrix for detailed analysis
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Additional metrics for spam detection
    spam_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    spam_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spam_f1 = 2 * (spam_precision * spam_recall) / (spam_precision + spam_recall) if (spam_precision + spam_recall) > 0 else 0.0
    
    return {
        "accuracy": acc,
        "precision": prec, 
        "recall": rec,
        "f1": f1,
        "spam_precision": spam_precision,
        "spam_recall": spam_recall,
        "spam_f1": spam_f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced Transformer for Maximum Spam Recall")
    parser.add_argument("--data", type=str, default="data/comprehensive_spam_dataset.csv", help="Path to comprehensive dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--min-freq", type=int, default=2, help="Min token frequency")
    parser.add_argument("--vocab-size", type=int, default=50000, help="Max vocab size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--max-recall", action="store_true", help="Optimize for maximum recall (catch all spam)")
    parser.add_argument("--focal-loss", action="store_true", help="Use focal loss for hard examples")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 80)
    print("ENHANCED TRANSFORMER FOR MAXIMUM SPAM RECALL")
    print("=" * 80)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load comprehensive dataset
    print(f"\nLoading comprehensive dataset: {args.data}")
    df = load_comprehensive_dataset(args.data)
    print(f"Dataset size: {len(df)}")
    spam_count = int(df["label"].sum())
    ham_count = len(df) - spam_count
    print(f"Ham: {ham_count} | Spam: {spam_count} | Spam ratio: {spam_count/len(df):.3f}")

    # Enhanced preprocessing
    print("\nInitializing enhanced preprocessor...")
    preprocessor = EnhancedSpamPreprocessor()
    df["processed_text"] = df["text"].astype(str).apply(preprocessor.preprocess)

    # Train/Val/Test split with stratification
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Build enhanced vocabulary
    print("Building enhanced vocabulary with special tokens...")
    vocab = build_enhanced_vocab(
        train_df["processed_text"].tolist(), 
        preprocessor, 
        min_freq=args.min_freq, 
        max_size=args.vocab_size
    )
    
    special_tokens = preprocessor.get_special_tokens()
    print(f"Vocabulary size: {len(vocab)} (includes {len(special_tokens)} special tokens)")

    # Convert texts to IDs
    train_ids = texts_to_ids_enhanced(train_df["processed_text"].tolist(), vocab, preprocessor)
    val_ids = texts_to_ids_enhanced(val_df["processed_text"].tolist(), vocab, preprocessor)
    test_ids = texts_to_ids_enhanced(test_df["processed_text"].tolist(), vocab, preprocessor)

    train_labels = train_df["label"].astype(int).tolist()
    val_labels = val_df["label"].astype(int).tolist()
    test_labels = test_df["label"].astype(int).tolist()

    # Create datasets
    train_ds = EnhancedSpamDataset(train_ids, train_labels, args.max_len, vocab, special_tokens)
    val_ds = EnhancedSpamDataset(val_ids, val_labels, args.max_len, vocab, special_tokens)
    test_ds = EnhancedSpamDataset(test_ids, test_labels, args.max_len, vocab, special_tokens)

    # Data loaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                             collate_fn=train_ds.collate_fn, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=val_ds.collate_fn, pin_memory=(device.type=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=test_ds.collate_fn, pin_memory=(device.type=="cuda"))

    # Build enhanced model
    print("\nBuilding enhanced transformer model...")
    config = EnhancedTransformerConfig(
        vocab_size=len(vocab),
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        max_len=args.max_len,
        dropout=0.1,
        pad_idx=vocab[special_tokens['PAD']],
        cls_idx=vocab[special_tokens['CLS']],
        special_token_ids={name: vocab[token] for name, token in special_tokens.items()}
    )
    
    model = build_enhanced_model(config).to(device)
    
    # Model size info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} ({trainable_params:,} trainable)")

    # Loss function - optimize for maximum recall
    if args.focal_loss:
        # Focal loss for hard examples
        criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
        print("Using Focal Loss for hard examples")
    elif args.max_recall:
        # Class weighting to favor spam detection (reduce false negatives)
        pos_weight = torch.tensor([3.0], device=device)  # Higher weight for spam
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Using weighted BCE Loss optimized for maximum recall")
    else:
        # Balanced class weighting
        pos_count = sum(train_labels)
        neg_count = len(train_labels) - pos_count
        pos_weight = torch.tensor([max(1.0, neg_count / max(1, pos_count))], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using balanced BCE Loss (pos_weight={pos_weight.item():.2f})")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    best_val_metric = 0.0
    best_epoch = -1
    epochs_no_improve = 0
    
    # For maximum recall, use recall as the primary metric
    metric_key = "spam_recall" if args.max_recall else "spam_f1"

    def run_eval(data_loader, split_name="") -> Dict[str, float]:
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
        
        return compute_metrics_with_recall_focus(y_true, y_scores, threshold=0.5)

    print(f"\nStarting training (optimizing for {metric_key})...")
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
              f"Spam Prec: {val_metrics['spam_precision']:.4f} | "
              f"Spam Rec: {val_metrics['spam_recall']:.4f} | "
              f"Spam F1: {val_metrics['spam_f1']:.4f}")

        # Early stopping based on chosen metric
        current_metric = val_metrics[metric_key]
        
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_epoch = epoch
            epochs_no_improve = 0

            # Save best checkpoint
            ckpt_path = os.path.join(args.save_dir, "enhanced_transformer_best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config": config.__dict__,
                "vocab": vocab,
                "special_tokens": special_tokens,
                "preprocessor_class": "EnhancedSpamPreprocessor",
                "val_metrics": val_metrics,
                "epoch": epoch,
                "args": vars(args)
            }, ckpt_path)
            
            print(f"✅ New best {metric_key}: {best_val_metric:.4f} - Saved to {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s) "
                  f"(best {metric_key}={best_val_metric:.4f} @ epoch {best_epoch})")

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered!")
            break
        print()

    training_time = time.time() - training_start
    
    # Load best model for final evaluation
    ckpt_path = os.path.join(args.save_dir, "enhanced_transformer_best.pt")
    if os.path.exists(ckpt_path):
        print(f"\nLoading best checkpoint for final evaluation...")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])

    # Final test evaluation
    test_metrics = run_eval(test_loader, "TEST")
    
    print("\n" + "=" * 80)
    print("FINAL TEST RESULTS - ENHANCED TRANSFORMER")
    print("=" * 80)
    print(f"Accuracy     : {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Spam Precision: {test_metrics['spam_precision']:.4f} ({test_metrics['spam_precision']*100:.2f}%)")
    print(f"Spam Recall   : {test_metrics['spam_recall']:.4f} ({test_metrics['spam_recall']*100:.2f}%)")
    print(f"Spam F1-Score : {test_metrics['spam_f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (Ham correctly classified): {test_metrics['true_negatives']}")
    print(f"  False Positives (Ham classified as Spam) : {test_metrics['false_positives']}")
    print(f"  False Negatives (Spam classified as Ham) : {test_metrics['false_negatives']}")
    print(f"  True Positives  (Spam correctly classified): {test_metrics['true_positives']}")

    # Save final report
    report = {
        "model_type": "EnhancedTransformerTextClassifier",
        "config": config.__dict__,
        "training_args": vars(args),
        "test_metrics": test_metrics,
        "best_val_metric": float(best_val_metric),
        "best_epoch": int(best_epoch),
        "training_time_seconds": training_time,
        "dataset_info": {
            "total_samples": len(df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "vocab_size": len(vocab),
            "spam_ratio": spam_count / len(df)
        }
    }
    
    report_path = os.path.join(args.save_dir, "enhanced_transformer_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Training report saved: {report_path}")
    print(f"🚀 Enhanced model saved: {ckpt_path}")
    print(f"⏱️  Total training time: {training_time/60:.1f} minutes")
    
    print("\n" + "=" * 80)
    print(f"ENHANCED TRANSFORMER TRAINING COMPLETE!")
    print(f"🎯 Final Spam Recall: {test_metrics['spam_recall']*100:.2f}% (Goal: Maximum recall)")
    print(f"📊 Final Accuracy: {test_metrics['accuracy']*100:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()