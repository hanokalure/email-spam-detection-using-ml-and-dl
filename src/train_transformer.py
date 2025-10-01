#!/usr/bin/env python3
"""
Train a Custom Transformer Spam Classifier (Pure DL, GPU-ready)
- Uses SimplePreprocessor for consistent cleaning
- Word-level vocabulary with min frequency threshold
- Stratified split (train/val/test = 80/10/10)
- Mixed precision (AMP), AdamW, linear warmup + cosine decay
- Early stopping on validation F1
- Saves best checkpoint and tokenizer/config artifacts

Usage (PowerShell):
  python src/train_transformer.py --data data/mega_spam_dataset.csv --epochs 30 --max-len 512 --batch-size 16 --deeper

Notes:
- For maximum accuracy, use --deeper (6 layers, d_model=512). For faster runs omit it (4 layers, d_model=384).
- Automatically uses GPU if available (CUDA). Falls back to CPU.
"""

import os
import json
import math
import argparse
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
from simple_svm_classifier import SimplePreprocessor
from transformer_text_classifier import TransformerConfig, build_model


SPECIAL_TOKENS = {
    "PAD": "[PAD]",
    "UNK": "[UNK]",
}


def load_email_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize columns to text/label (0=ham,1=spam)
    if {"text", "label"}.issubset(df.columns):
        df = df[["text", "label"]].copy()
        if df["label"].dtype == "O":
            df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})
    elif {"v1", "v2"}.issubset(df.columns):
        df = df[["v2", "v1"]].copy()
        df.columns = ["text", "label"]
        df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})
    elif len(df.columns) >= 2:
        df = df.iloc[:, :2].copy()
        df.columns = ["text", "label"]
        if df["label"].dtype == "O":
            df["label"] = df["label"].str.lower().map({"ham": 0, "spam": 1})
    else:
        raise ValueError("Unsupported dataset format. Need columns like ('text','label') or ('v1','v2').")

    # Drop missing
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(int)
    return df


def build_vocab(texts: List[str], min_freq: int = 2, max_size: int = 50000) -> Dict[str, int]:
    counter = Counter()
    for t in texts:
        for w in t.split():
            counter[w] += 1

    # Reserve indices 0,1 for PAD, UNK
    vocab = {SPECIAL_TOKENS["PAD"]: 0, SPECIAL_TOKENS["UNK"]: 1}
    # Most common tokens over min_freq
    items = [(w, c) for w, c in counter.items() if c >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))

    for w, _ in items[: max_size - len(vocab)]:
        vocab[w] = len(vocab)
    return vocab


def texts_to_ids(texts: List[str], vocab: Dict[str, int], unk_idx: int = 1) -> List[List[int]]:
    ids = []
    for t in texts:
        tokens = t.split()
        ids.append([vocab.get(tok, unk_idx) for tok in tokens])
    return ids


class SpamDataset(Dataset):
    def __init__(self, ids_list: List[List[int]], labels: List[int], max_len: int, pad_idx: int = 0):
        self.ids_list = ids_list
        self.labels = labels
        self.max_len = max_len
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.ids_list)

    def __getitem__(self, idx):
        return self.ids_list[idx], int(self.labels[idx])

    def collate_fn(self, batch):
        # Dynamic pad to max_len (or max in batch, capped by max_len)
        seqs, labels = zip(*batch)
        max_in_batch = min(max(len(s) for s in seqs), self.max_len)
        batch_size = len(seqs)

        input_ids = torch.full((batch_size, max_in_batch), self.pad_idx, dtype=torch.long)
        attn_mask = torch.zeros((batch_size, max_in_batch), dtype=torch.long)

        for i, seq in enumerate(seqs):
            trunc = seq[:max_in_batch]
            input_ids[i, : len(trunc)] = torch.tensor(trunc, dtype=torch.long)
            attn_mask[i, : len(trunc)] = 1

        labels_t = torch.tensor(labels, dtype=torch.float32)
        return input_ids, attn_mask, labels_t


def warmup_cosine_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_metrics(y_true: List[int], y_scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_scores >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="Train Custom Transformer Spam Classifier (GPU-ready)")
    parser.add_argument("--data", type=str, default="data/mega_spam_dataset.csv", help="Path to CSV dataset")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length (tokens)")
    parser.add_argument("--min-freq", type=int, default=2, help="Min token frequency for vocab")
    parser.add_argument("--vocab-size", type=int, default=50000, help="Max vocab size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--deeper", action="store_true", help="Use deeper/larger model for maximum accuracy")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (0 recommended on Windows)")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    print("=" * 70)
    print("CUSTOM TRANSFORMER SPAM CLASSIFIER (PURE DL)")
    print("=" * 70)

    # Device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA Device: {torch.cuda.get_device_name(0)} | Capability: {torch.cuda.get_device_capability(0)}")

    # Load data
    print(f"Loading dataset: {args.data}")
    df = load_email_dataset(args.data)
    print(f"Dataset size: {len(df)}")
    spam = int(df["label"].sum())
    ham = len(df) - spam
    print(f"Ham: {ham} | Spam: {spam} | Spam ratio: {spam/len(df):.3f}")

    # Preprocess text
    pre = SimplePreprocessor()
    df["processed_text"] = df["text"].astype(str).apply(pre.preprocess)

    # Split
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
    train_df, val_df = train_test_split(train_df, test_size=0.1111, random_state=42, stratify=train_df["label"])  # ~0.1

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # Build vocab on train only
    vocab = build_vocab(train_df["processed_text"].tolist(), min_freq=args.min_freq, max_size=args.vocab_size)
    pad_idx = vocab[SPECIAL_TOKENS["PAD"]]
    unk_idx = vocab[SPECIAL_TOKENS["UNK"]]
    print(f"Vocab size: {len(vocab)} (min_freq={args.min_freq}, max={args.vocab_size})")

    # Convert texts -> ids
    train_ids = texts_to_ids(train_df["processed_text"].tolist(), vocab, unk_idx)
    val_ids = texts_to_ids(val_df["processed_text"].tolist(), vocab, unk_idx)
    test_ids = texts_to_ids(test_df["processed_text"].tolist(), vocab, unk_idx)

    train_labels = train_df["label"].astype(int).tolist()
    val_labels = val_df["label"].astype(int).tolist()
    test_labels = test_df["label"].astype(int).tolist()

    # Datasets & loaders
    train_ds = SpamDataset(train_ids, train_labels, max_len=args.max_len, pad_idx=pad_idx)
    val_ds = SpamDataset(val_ids, val_labels, max_len=args.max_len, pad_idx=pad_idx)
    test_ds = SpamDataset(test_ids, test_labels, max_len=args.max_len, pad_idx=pad_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=train_ds.collate_fn, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=val_ds.collate_fn, pin_memory=(device.type=="cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=test_ds.collate_fn, pin_memory=(device.type=="cuda"))

    # Model config (deeper = maximum accuracy)
    if args.deeper:
        cfg = TransformerConfig(
            vocab_size=len(vocab),
            d_model=512,
            nhead=8,
            num_layers=6,
            dim_feedforward=2048,
            max_len=args.max_len,
            dropout=0.1,
            pad_idx=pad_idx,
        )
    else:
        cfg = TransformerConfig(
            vocab_size=len(vocab),
            d_model=384,
            nhead=6,
            num_layers=4,
            dim_feedforward=1536,
            max_len=args.max_len,
            dropout=0.1,
            pad_idx=pad_idx,
        )

    model = build_model(cfg).to(device)

    # Loss: BCE with logits + class weighting (pos_weight)
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    pos_weight = torch.tensor([max(1.0, neg_count / max(1, pos_count))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs // max(1, args.grad_accum)
    warmup_steps = int(0.1 * total_steps)
    scheduler = warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Training loop
    best_val_f1 = 0.0
    best_epoch = -1
    epochs_no_improve = 0

    def run_eval(data_loader) -> Dict[str, float]:
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
        return compute_metrics(y_true, y_scores, threshold=0.5)

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        step = 0

        for it, (input_ids, attn_mask, labels) in enumerate(train_loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(enabled=(device.type == "cuda")):
                logits = model(input_ids, attn_mask)
                loss = criterion(logits, labels)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if it % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running_loss += loss.item() * args.grad_accum
            step += 1

            if it % 100 == 0 or it == len(train_loader):
                print(f"Epoch {epoch:02d} | Step {it:04d}/{len(train_loader)} | Loss {running_loss/step:.4f}")

        # Validation
        val_metrics = run_eval(val_loader)
        print(f"\nEpoch {epoch:02d} VAL => Acc: {val_metrics['accuracy']:.4f} | Prec: {val_metrics['precision']:.4f} | Rec: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")

        # Early stopping on F1
        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch
            epochs_no_improve = 0

            # Save checkpoint + artifacts
            ckpt_path = os.path.join(args.save_dir, "transformer_best.pt")
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "vocab": vocab,
                "val_metrics": val_metrics,
                "epoch": epoch,
            }, ckpt_path)
            print(f"✅ Saved best model to {ckpt_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s) (best F1={best_val_f1:.4f} @ epoch {best_epoch})")

        if epochs_no_improve >= args.patience:
            print("Early stopping triggered.")
            break

    # Final test evaluation (load best)
    ckpt_path = os.path.join(args.save_dir, "transformer_best.pt")
    if os.path.exists(ckpt_path):
        print(f"\nLoading best checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    else:
        print("\n⚠️ Best checkpoint not found, using last epoch model for test.")

    test_metrics = run_eval(test_loader)
    print("\n" + "=" * 70)
    print("FINAL TEST METRICS (CUSTOM TRANSFORMER)")
    print("=" * 70)
    print(f"Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall   : {test_metrics['recall']:.4f}")
    print(f"F1-score : {test_metrics['f1']:.4f}")

    # Save final report
    report = {
        "config": cfg.__dict__,
        "test_metrics": test_metrics,
        "best_val_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "dataset": os.path.basename(args.data),
        "samples": int(len(df)),
    }
    report_path = os.path.join(args.save_dir, "transformer_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 Report saved: {report_path}")


if __name__ == "__main__":
    main()
