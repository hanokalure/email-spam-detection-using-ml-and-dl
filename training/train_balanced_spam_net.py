#!/usr/bin/env python3
"""
Training Script for BalancedSpamNet - GPU Optimized

Key Features:
1. Balanced sampling during training
2. GPU optimization with mixed precision
3. Advanced data preprocessing
4. Real-time monitoring of HAM vs SPAM performance
5. Early stopping based on balanced metrics

Usage:
python training/train_balanced_spam_net.py --epochs 50 --batch-size 32 --gpu
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from balanced_spam_net import BalancedSpamNet, BalancedFocalLoss, create_balanced_spam_net
from simple_svm_classifier import SimplePreprocessor


class BusinessAwarePreprocessor:
    """Enhanced preprocessor with business context awareness"""
    
    def __init__(self):
        self.base_preprocessor = SimplePreprocessor()
        
        # Business/Financial vocabulary
        self.business_terms = {
            'account', 'banking', 'financial', 'investment', 'credit', 'debit',
            'transaction', 'payment', 'invoice', 'receipt', 'statement', 'balance',
            'loan', 'mortgage', 'insurance', 'policy', 'premium', 'claim',
            'business', 'corporate', 'company', 'enterprise', 'commercial',
            'professional', 'service', 'customer', 'client', 'contract',
            'agreement', 'terms', 'conditions', 'legal', 'compliance',
            'security', 'secure', 'encrypted', 'verification', 'authenticate'
        }
        
        # Create business term to ID mapping
        self.business_vocab = {term: i for i, term in enumerate(self.business_terms)}
        
    def preprocess(self, text: str) -> str:
        """Enhanced preprocessing preserving business context"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        # Base preprocessing
        cleaned = self.base_preprocessor.preprocess(text)
        
        # Preserve important business indicators
        cleaned = cleaned.replace('$', ' dollar ')
        cleaned = cleaned.replace('%', ' percent ')
        cleaned = cleaned.replace('@', ' at ')
        
        return cleaned
    
    def get_business_mask(self, text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
        """Create business context mask for enhanced embeddings"""
        words = text.split()[:max_len]
        mask = []
        
        for word in words:
            if word.lower() in self.business_vocab:
                mask.append(self.business_vocab[word.lower()])
            else:
                mask.append(0)  # No business context
                
        # Pad to max_len
        while len(mask) < max_len:
            mask.append(0)
            
        return mask[:max_len]


def load_comprehensive_dataset(csv_path: str) -> pd.DataFrame:
    """Load and normalize the comprehensive spam dataset"""
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Normalize columns
    if 'text' in df.columns and 'label' in df.columns:
        df = df[['text', 'label']].copy()
    elif 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v2', 'v1']].copy()
        df.columns = ['text', 'label']
    else:
        df = df.iloc[:, :2].copy()
        df.columns = ['text', 'label']
    
    # Normalize labels to 0/1
    if df['label'].dtype == 'object':
        df['label'] = df['label'].str.lower().map({'ham': 0, 'spam': 1})
    
    # Clean data
    df = df.dropna(subset=['text', 'label']).copy()
    df['label'] = df['label'].astype(int)
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"HAM: {sum(df['label'] == 0)} ({sum(df['label'] == 0)/len(df)*100:.1f}%)")
    print(f"SPAM: {sum(df['label'] == 1)} ({sum(df['label'] == 1)/len(df)*100:.1f}%)")
    
    return df


def build_balanced_vocab(texts: List[str], min_freq: int = 3, max_size: int = 50000) -> Dict[str, int]:
    """Build vocabulary with special tokens"""
    counter = Counter()
    for text in texts:
        for word in text.split():
            counter[word] += 1
    
    # Special tokens
    vocab = {'<PAD>': 0, '<UNK>': 1}
    
    # Add frequent words
    for word, count in counter.most_common(max_size - 2):
        if count >= min_freq:
            vocab[word] = len(vocab)
    
    return vocab


def texts_to_ids(texts: List[str], vocab: Dict[str, int], max_len: int = 512) -> List[List[int]]:
    """Convert texts to token IDs with padding"""
    unk_id = vocab.get('<UNK>', 1)
    pad_id = vocab.get('<PAD>', 0)
    
    ids_list = []
    for text in texts:
        words = text.split()[:max_len]
        ids = [vocab.get(word, unk_id) for word in words]
        
        # Pad to max_len
        while len(ids) < max_len:
            ids.append(pad_id)
            
        ids_list.append(ids[:max_len])
    
    return ids_list


class BalancedSpamDataset(Dataset):
    """Dataset with business context awareness"""
    
    def __init__(
        self,
        texts: List[str], 
        labels: List[int], 
        vocab: Dict[str, int],
        preprocessor: BusinessAwarePreprocessor,
        max_len: int = 512
    ):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.preprocessor = preprocessor
        self.max_len = max_len
        
        # Convert texts to IDs
        print("Converting texts to token IDs...")
        self.input_ids = texts_to_ids(texts, vocab, max_len)
        
        # Create business context masks
        print("Creating business context masks...")
        self.business_masks = []
        for text in texts:
            mask = self.preprocessor.get_business_mask(text, vocab, max_len)
            self.business_masks.append(mask)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'business_mask': torch.tensor(self.business_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float)
        }


def create_balanced_dataloaders(
    train_dataset: BalancedSpamDataset,
    val_dataset: BalancedSpamDataset,
    test_dataset: BalancedSpamDataset,
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create balanced data loaders with weighted sampling"""
    
    # Calculate class weights for balanced sampling
    train_labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
    class_counts = Counter(train_labels)
    
    # Weight inversely to frequency
    weights = {0: 1.0/class_counts[0], 1: 1.0/class_counts[1]}
    sample_weights = [weights[int(label)] for label in train_labels]
    
    # Weighted sampler for balanced batches
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def compute_balanced_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute balanced metrics focused on HAM/SPAM performance"""
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Class-specific metrics
    ham_mask = (y_true == 0)
    spam_mask = (y_true == 1)
    
    ham_accuracy = accuracy_score(y_true[ham_mask], y_pred[ham_mask]) if ham_mask.sum() > 0 else 0.0
    spam_accuracy = accuracy_score(y_true[spam_mask], y_pred[spam_mask]) if spam_mask.sum() > 0 else 0.0
    
    # Balanced accuracy (mean of class accuracies)
    balanced_accuracy = (ham_accuracy + spam_accuracy) / 2.0
    
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'ham_accuracy': ham_accuracy,
        'spam_accuracy': spam_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train_epoch(
    model: BalancedSpamNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    criterion: BalancedFocalLoss,
    scaler: GradScaler,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train one epoch with balanced focus"""
    
    model.train()
    total_loss = 0.0
    total_main_loss = 0.0
    total_spam_loss = 0.0
    total_ham_loss = 0.0
    
    predictions = []
    targets = []
    
    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        business_mask = batch['business_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast(enabled=(device.type == 'cuda')):
            outputs = model(input_ids, business_mask)
            loss_dict = criterion(outputs, labels)
            total_loss_batch = loss_dict['total_loss']
        
        # Backward pass
        scaler.scale(total_loss_batch).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:
            scheduler.step()
        
        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_main_loss += loss_dict['main_loss'].item()
        total_spam_loss += loss_dict['spam_loss'].item()
        total_ham_loss += loss_dict['ham_loss'].item()
        
        # Collect predictions for metrics
        with torch.no_grad():
            preds = (torch.sigmoid(outputs['logits']) > 0.5).float()
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
        
        # Progress logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                  f"Loss: {total_loss_batch.item():.4f}")
    
    # Calculate epoch metrics
    predictions = np.array(predictions)
    targets = np.array(targets)
    metrics = compute_balanced_metrics(targets, predictions)
    
    avg_losses = {
        'total_loss': total_loss / len(dataloader),
        'main_loss': total_main_loss / len(dataloader),
        'spam_loss': total_spam_loss / len(dataloader),
        'ham_loss': total_ham_loss / len(dataloader)
    }
    
    return {**metrics, **avg_losses}


def evaluate_model(
    model: BalancedSpamNet,
    dataloader: DataLoader,
    criterion: BalancedFocalLoss,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model with detailed metrics"""
    
    model.eval()
    total_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            business_mask = batch['business_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            outputs = model(input_ids, business_mask)
            loss_dict = criterion(outputs, labels)
            total_loss += loss_dict['total_loss'].item()
            
            preds = (torch.sigmoid(outputs['logits']) > 0.5).float()
            predictions.extend(preds.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    metrics = compute_balanced_metrics(targets, predictions)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train BalancedSpamNet")
    parser.add_argument('--data', type=str, default='data/comprehensive_spam_dataset.csv',
                       help='Path to dataset CSV')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--max-len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--save-dir', type=str, default='models', help='Model save directory')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu')
    print(f"🔥 Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("🚀 Starting BalancedSpamNet Training")
    print("=" * 70)
    
    # Load and preprocess data
    df = load_comprehensive_dataset(args.data)
    
    # Initialize preprocessor
    preprocessor = BusinessAwarePreprocessor()
    print("Preprocessing texts...")
    df['processed_text'] = df['text'].astype(str).apply(preprocessor.preprocess)
    
    # Split data: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df['label']
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df['label']
    )
    
    print(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Build vocabulary on training data
    print("Building vocabulary...")
    vocab = build_balanced_vocab(
        train_df['processed_text'].tolist(),
        min_freq=3,
        max_size=args.vocab_size
    )
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = BalancedSpamDataset(
        train_df['processed_text'].tolist(),
        train_df['label'].tolist(),
        vocab, preprocessor, args.max_len
    )
    val_dataset = BalancedSpamDataset(
        val_df['processed_text'].tolist(),
        val_df['label'].tolist(),
        vocab, preprocessor, args.max_len
    )
    test_dataset = BalancedSpamDataset(
        test_df['processed_text'].tolist(),
        test_df['label'].tolist(),
        vocab, preprocessor, args.max_len
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_balanced_dataloaders(
        train_dataset, val_dataset, test_dataset, args.batch_size
    )
    
    # Create model
    print("Initializing BalancedSpamNet...")
    model = create_balanced_spam_net(
        vocab_size=len(vocab),
        embed_dim=256,
        cnn_filters=128,
        lstm_hidden=256,
        lstm_layers=2,
        attention_heads=8,
        dropout=0.3
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss and optimizer
    criterion = BalancedFocalLoss(
        alpha=0.5, gamma=2.0, 
        spam_weight=1.2, ham_weight=1.5  # Higher weight for HAM detection
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Mixed precision scaler
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # Training loop
    best_balanced_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    print("🎯 TARGET: HAM Accuracy >85%, SPAM Accuracy >92%")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n📅 EPOCH {epoch}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, 
            criterion, scaler, device, epoch
        )
        
        # Validate
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Logging
        print(f"TRAIN => Acc: {train_metrics['accuracy']:.4f} | "
              f"Bal-Acc: {train_metrics['balanced_accuracy']:.4f} | "
              f"HAM: {train_metrics['ham_accuracy']:.4f} | "
              f"SPAM: {train_metrics['spam_accuracy']:.4f}")
        
        print(f"VAL   => Acc: {val_metrics['accuracy']:.4f} | "
              f"Bal-Acc: {val_metrics['balanced_accuracy']:.4f} | "
              f"HAM: {val_metrics['ham_accuracy']:.4f} | "
              f"SPAM: {val_metrics['spam_accuracy']:.4f}")
        
        # Check if this is the best model (based on balanced accuracy)
        if val_metrics['balanced_accuracy'] > best_balanced_acc:
            best_balanced_acc = val_metrics['balanced_accuracy']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(args.save_dir, 'balanced_spam_net_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'vocab': vocab,
                'val_metrics': val_metrics,
                'train_metrics': train_metrics,
                'args': vars(args)
            }, model_path)
            
            print(f"✅ NEW BEST! Balanced Accuracy: {best_balanced_acc:.4f} (Saved)")
            
            # Check if we reached target performance
            if val_metrics['ham_accuracy'] >= 0.85 and val_metrics['spam_accuracy'] >= 0.92:
                print("🎉 TARGET ACHIEVED! HAM >85% and SPAM >92%")
                
        else:
            patience_counter += 1
            print(f"⏳ No improvement for {patience_counter} epochs (best: {best_balanced_acc:.4f})")
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"⏹️ Early stopping after {patience_counter} epochs without improvement")
            break
    
    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("🔍 FINAL TEST EVALUATION")
    print("=" * 70)
    
    # Load best model
    best_model_path = os.path.join(args.save_dir, 'balanced_spam_net_best.pt')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print(f"🎯 FINAL RESULTS:")
    print(f"Overall Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f} ({test_metrics['balanced_accuracy']*100:.2f}%)")
    print(f"HAM Accuracy: {test_metrics['ham_accuracy']:.4f} ({test_metrics['ham_accuracy']*100:.2f}%)")
    print(f"SPAM Accuracy: {test_metrics['spam_accuracy']:.4f} ({test_metrics['spam_accuracy']*100:.2f}%)")
    print(f"F1-Score: {test_metrics['f1']:.4f}")
    
    # Compare with existing models
    print(f"\n📊 COMPARISON WITH EXISTING MODELS:")
    print(f"Enhanced Transformer HAM:  56.0% ❌ → BalancedSpamNet HAM:  {test_metrics['ham_accuracy']*100:.1f}% ✅")
    print(f"Enhanced Transformer SPAM: 96.0% ✅ → BalancedSpamNet SPAM: {test_metrics['spam_accuracy']*100:.1f}% ✅")
    
    improvement_ham = test_metrics['ham_accuracy'] - 0.56
    print(f"\n🚀 HAM Detection Improvement: +{improvement_ham*100:.1f} percentage points!")
    
    # Save final results
    results = {
        'test_metrics': test_metrics,
        'best_epoch': best_epoch,
        'model_path': best_model_path,
        'vocab_size': len(vocab),
        'total_parameters': total_params
    }
    
    results_path = os.path.join(args.save_dir, 'balanced_spam_net_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Training completed! Results saved to {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()