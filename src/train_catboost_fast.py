#!/usr/bin/env python3
"""
Fast CatBoost Improvement - Key optimizations for better accuracy
- Better features: combined char + word n-grams (smaller scale)
- Optimized hyperparameters based on best practices
- No expensive hyperparameter search - just use proven settings
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
from sklearn.pipeline import FeatureUnion

# CatBoost
from catboost import CatBoostClassifier

# Reuse your existing preprocessor for consistency
import sys
sys.path.append(os.path.dirname(__file__))
from simple_svm_classifier import SimplePreprocessor


def load_email_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset and normalize to columns: text, label (0=ham,1=spam)"""
    df = pd.read_csv(csv_path)

    # Try to normalize to 'text' and 'label'
    if 'text' in df.columns and 'label' in df.columns:
        pass
    elif 'v1' in df.columns and 'v2' in df.columns:
        # UCI format
        df = df[['v2', 'v1']].copy()
        df.columns = ['text', 'label']
    else:
        # Try infer first two columns
        df = df.iloc[:, :2].copy()
        df.columns = ['text', 'label']

    # Normalize labels to 0/1
    if df['label'].dtype == 'O':
        df['label'] = df['label'].str.lower().map({'ham': 0, 'spam': 1})

    # Drop rows with missing values
    df = df.dropna(subset=['text', 'label'])

    # Ensure numeric labels
    df['label'] = df['label'].astype(int)
    return df


def create_optimized_vectorizer():
    """Create optimized combined vectorizer (smaller, faster)"""
    
    # Character n-grams (good for obfuscated spam) - reduced size
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=8000,  # Reduced from 12000
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents='ascii'
    )
    
    # Word n-grams (good for phrases) - reduced size
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=5000,  # Reduced from 8000
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents='ascii',
        stop_words=None  # Keep stop words for spam detection
    )
    
    # Combine both vectorizers
    combined_vectorizer = FeatureUnion([
        ('char_ngrams', char_vectorizer),
        ('word_ngrams', word_vectorizer)
    ])
    
    return combined_vectorizer


def train_catboost_optimized(data_path: str, model_path: str) -> float:
    """Train CatBoost with optimized settings (no hyperparameter search)"""
    
    print("=" * 70)
    print("FAST CATBOOST OPTIMIZATION")
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

    # Split data - use more data for training, less for validation
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    # Split training set for validation
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,  # Smaller validation set
        random_state=42,
        stratify=y_train
    )
    
    print(f"Training set: {len(X_train_fit)} messages")
    print(f"Validation set: {len(X_val)} messages")
    print(f"Test set: {len(X_test)} messages")

    # Create optimized vectorizer
    print("Creating optimized combined vectorizer...")
    vectorizer = create_optimized_vectorizer()
    
    print("Vectorizing text...")
    start_vec = time.time()
    X_train_vec = vectorizer.fit_transform(X_train_fit)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)
    vec_time = time.time() - start_vec

    print(f"Combined feature space: {X_train_vec.shape[1]} features")
    print(f"Vectorization time: {vec_time:.1f} seconds")

    # Optimized CatBoost model (based on research and best practices)
    print("\nTraining optimized CatBoost model...")
    start_time = time.time()
    
    model = CatBoostClassifier(
        # Core parameters optimized for text classification
        iterations=500,           # Good balance of performance vs time
        depth=6,                  # Optimal for most text tasks
        learning_rate=0.08,       # Conservative learning rate
        l2_leaf_reg=5,           # Regularization to prevent overfitting
        
        # Class balancing 
        class_weights=[1.0, 1.4], # Slightly favor spam detection
        
        # Training optimization
        loss_function='Logloss',
        eval_metric='Accuracy',
        use_best_model=True,
        early_stopping_rounds=40,
        
        # Speed and reproducibility
        random_seed=42,
        verbose=25,  # Show progress every 25 iterations
        thread_count=4  # Limit CPU usage
    )

    # Train with validation
    model.fit(
        X_train_vec, y_train_fit,
        eval_set=(X_val_vec, y_val),
        verbose=False
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")

    # Evaluate on test set
    print(f"\n📊 EVALUATION ON TEST SET:")
    y_pred = model.predict(X_test_vec)
    y_pred = np.array(y_pred).astype(int)
    
    test_accuracy = accuracy_score(y_test, y_pred)

    print("=" * 70)
    print("OPTIMIZED CATBOOST RESULTS")
    print("=" * 70)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Compare with baseline
    baseline_accuracy = 0.9798
    improvement = (test_accuracy - baseline_accuracy) * 100
    print(f"Improvement over baseline: {improvement:+.2f} percentage points")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Ham correctly classified): {cm[0, 0]}")
    print(f"False Positives (Ham classified as Spam): {cm[0, 1]}")
    print(f"False Negatives (Spam classified as Ham): {cm[1, 0]}")
    print(f"True Positives (Spam correctly classified): {cm[1, 1]}")

    # Calculate key metrics
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if cm[1, 1] + cm[0, 1] > 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm[1, 1] + cm[1, 0] > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    print(f"\nKey Metrics:")
    print(f"Spam Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Spam Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"Spam F1-Score: {f1:.4f}")

    # Save model
    model_artifact = {
        'classifier': model,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor,
        'accuracy': float(test_accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'training_time_seconds': training_time,
        'improvement_over_baseline': float(improvement),
        'timestamp': datetime.now().isoformat()
    }
    
    Path(os.path.dirname(model_path) or '.').mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, model_path)
    print(f"\n✅ Optimized model saved to {model_path}")

    # Show top features
    try:
        feature_names = vectorizer.get_feature_names_out()
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        
        print(f"\n🔝 TOP 10 IMPORTANT FEATURES:")
        for i, idx in enumerate(top_indices, 1):
            feature_name = feature_names[idx]
            # Truncate long feature names
            if len(feature_name) > 20:
                feature_name = feature_name[:17] + "..."
            print(f"  {i:2d}. {feature_name:<20} ({importances[idx]:.4f})")
    except Exception as e:
        print(f"⚠️ Could not extract feature importance: {e}")

    print("\n" + "=" * 70)
    print(f"FINAL ACCURACY: {test_accuracy*100:.2f}%")
    print(f"Total training time: {training_time:.1f} seconds")
    print("=" * 70)

    return test_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train fast optimized CatBoost spam classifier")
    parser.add_argument('--data', type=str, default='data/mega_spam_dataset.csv', 
                       help='Path to CSV dataset')
    parser.add_argument('--model', type=str, default='models/catboost_tuned.pkl', 
                       help='Output model path')
    args = parser.parse_args()

    print(f"🎯 Target: Improve CatBoost accuracy beyond 97.98%")
    print(f"⚡ Strategy: Fast optimization with proven settings")
    print(f"📊 Dataset: {args.data}")
    print(f"💾 Output: {args.model}")
    print()

    accuracy = train_catboost_optimized(args.data, args.model)
    
    print(f"\n🎉 Training completed!")
    print(f"Final accuracy: {accuracy*100:.2f}%")
    
    # Compare with your existing models
    print(f"\n📈 ACCURACY COMPARISON:")
    print(f"  SVM (5K):           99.66% 🏆")
    print(f"  SVM (14K):          98.29%")
    print(f"  CatBoost (original): 97.98%")
    print(f"  CatBoost (optimized): {accuracy*100:.2f}% ⭐")


if __name__ == '__main__':
    main()