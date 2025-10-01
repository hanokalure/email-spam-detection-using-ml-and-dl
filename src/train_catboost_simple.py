#!/usr/bin/env python3
"""
Simple CatBoost Improvement
- Use the SAME feature space as the original CatBoost (15,000 char n-grams)
- Just improve the CatBoost parameters for better accuracy
- Should be faster and still improve accuracy
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import time

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CatBoost
from catboost import CatBoostClassifier

# Reuse your existing preprocessor for consistency
import sys
sys.path.append(os.path.dirname(__file__))
from simple_svm_classifier import SimplePreprocessor


def load_email_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset and normalize to columns: text, label (0=ham,1=spam)"""
    df = pd.read_csv(csv_path)

    if 'text' in df.columns and 'label' in df.columns:
        pass
    elif 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v2', 'v1']].copy()
        df.columns = ['text', 'label']
    else:
        df = df.iloc[:, :2].copy()
        df.columns = ['text', 'label']

    if df['label'].dtype == 'O':
        df['label'] = df['label'].str.lower().map({'ham': 0, 'spam': 1})

    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int)
    return df


def train_catboost_simple(data_path: str, model_path: str) -> float:
    """Train CatBoost with better parameters but same features as original"""
    
    print("=" * 70)
    print("SIMPLE CATBOOST IMPROVEMENT (SAME FEATURES, BETTER PARAMETERS)")
    print("=" * 70)

    print(f"Loading data from {data_path}...")
    df = load_email_dataset(data_path)
    print(f"Dataset loaded: {len(df)} messages")
    spam = int(df['label'].sum())
    ham = len(df) - spam
    print(f"Spam messages: {spam} ({spam/len(df)*100:.1f}%)")
    print(f"Ham messages: {ham} ({ham/len(df)*100:.1f}%)")

    # Preprocess
    preprocessor = SimplePreprocessor()
    print("Preprocessing text data...")
    df['processed_text'] = df['text'].apply(preprocessor.preprocess)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    # Split training set for validation
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train
    )
    
    print(f"Training set: {len(X_train_fit)} messages")
    print(f"Validation set: {len(X_val)} messages")
    print(f"Test set: {len(X_test)} messages")

    # Use SAME vectorizer as original CatBoost (from train_catboost.py)
    print("Using same vectorizer as original (char n-grams only)...")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=15000,   # SAME as original
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents='ascii'
    )
    
    print("Vectorizing text...")
    start_vec = time.time()
    X_train_vec = vectorizer.fit_transform(X_train_fit)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    vec_time = time.time() - start_vec

    print(f"Feature space: {X_train_vec.shape[1]} features (same as original)")
    print(f"Vectorization time: {vec_time:.1f} seconds")

    # IMPROVED CatBoost parameters (this is the key improvement)
    print("\nTraining CatBoost with improved parameters...")
    start_time = time.time()
    
    model = CatBoostClassifier(
        # Better parameters than original
        iterations=600,           # More iterations than original (200)
        depth=6,                  # Deeper than original (4)  
        learning_rate=0.06,       # Lower learning rate for stability
        l2_leaf_reg=8,           # More regularization
        
        # Better class handling
        class_weights=[1.0, 1.3], # Better class balance than original
        
        # Training improvements
        loss_function='Logloss',
        eval_metric='Accuracy',
        use_best_model=True,      # Use best model from validation
        early_stopping_rounds=50, # Stop if no improvement
        
        # Reproducibility
        random_seed=42,
        verbose=50,  # Show progress every 50 iterations
    )

    # Train with validation (key improvement over original)
    model.fit(
        X_train_vec, y_train_fit,
        eval_set=(X_val_vec, y_val),
        verbose=True  # Show training progress
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.1f} seconds")

    # Evaluate on test set
    print(f"\n📊 EVALUATION:")
    y_pred = model.predict(X_test_vec)
    y_pred = np.array(y_pred).astype(int)
    
    test_accuracy = accuracy_score(y_test, y_pred)

    print("=" * 70)
    print("IMPROVED CATBOOST RESULTS")
    print("=" * 70)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Compare with original CatBoost
    original_accuracy = 0.9798  # 97.98%
    improvement = (test_accuracy - original_accuracy) * 100
    print(f"Improvement over original CatBoost: {improvement:+.2f} percentage points")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Ham correctly classified): {cm[0, 0]}")
    print(f"False Positives (Ham classified as Spam): {cm[0, 1]}")
    print(f"False Negatives (Spam classified as Ham): {cm[1, 0]}")
    print(f"True Positives (Spam correctly classified): {cm[1, 1]}")

    # Calculate metrics
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if cm[1, 1] + cm[0, 1] > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm[1, 1] + cm[1, 0] > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    print(f"\nSpam Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Spam Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"Spam F1-Score: {f1:.4f}")

    # Save improved model
    model_artifact = {
        'classifier': model,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor,
        'accuracy': float(test_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'training_time_seconds': training_time,
        'improvement_over_original': float(improvement),
        'timestamp': datetime.now().isoformat()
    }
    
    Path(os.path.dirname(model_path) or '.').mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, model_path)
    print(f"\n✅ Improved model saved to {model_path}")

    print("\n" + "=" * 70)
    print(f"FINAL ACCURACY: {test_accuracy*100:.2f}%")
    print(f"Training time: {training_time:.1f} seconds")
    print("=" * 70)

    return test_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train improved CatBoost spam classifier")
    parser.add_argument('--data', type=str, default='data/mega_spam_dataset.csv', 
                       help='Path to CSV dataset')
    parser.add_argument('--model', type=str, default='models/catboost_tuned.pkl', 
                       help='Output model path')
    args = parser.parse_args()

    print(f"🎯 Goal: Improve CatBoost accuracy beyond 97.98%")
    print(f"⚡ Strategy: Better parameters, same features, validation training")
    print(f"📊 Dataset: {args.data}")
    print(f"💾 Output: {args.model}")
    print()

    accuracy = train_catboost_simple(args.data, args.model)
    
    print(f"\n🎉 Training completed!")
    
    # Show comparison
    print(f"\n📈 ACCURACY COMPARISON:")
    print(f"  SVM (5K):             99.66% 🏆")
    print(f"  SVM (14K):            98.29%")
    print(f"  CatBoost (original):  97.98%")
    print(f"  CatBoost (improved):  {accuracy*100:.2f}% ⭐")
    
    if accuracy > 0.9798:
        improvement = (accuracy - 0.9798) * 100
        print(f"\n✅ SUCCESS! Improved by {improvement:.2f} percentage points!")
    else:
        print(f"\n⚠️ No improvement this time, but the model has better training setup")


if __name__ == '__main__':
    main()