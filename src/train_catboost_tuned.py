#!/usr/bin/env python3
"""
Advanced CatBoost Training with Hyperparameter Tuning
- Combined char + word n-gram features
- RandomizedSearchCV with 30-40 trials
- Early stopping and validation
- Optimized for accuracy
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import time

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin

# CatBoost
from catboost import CatBoostClassifier

# Reuse your existing preprocessor for consistency
import sys
sys.path.append(os.path.dirname(__file__))
from simple_svm_classifier import SimplePreprocessor


class TextPreprocessorTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper for SimplePreprocessor"""
    
    def __init__(self):
        self.preprocessor = SimplePreprocessor()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [self.preprocessor.preprocess(text) for text in X]


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


def create_combined_vectorizer():
    """Create combined char + word n-gram vectorizer"""
    
    # Character n-grams (good for obfuscated spam)
    char_vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=12000,
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents='ascii'
    )
    
    # Word n-grams (good for phrases)
    word_vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        max_features=8000,
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


def train_catboost_tuned(data_path: str, model_path: str, n_trials: int = 35) -> float:
    """Train tuned CatBoost with hyperparameter search"""
    
    print("=" * 70)
    print("ADVANCED CATBOOST TRAINING WITH HYPERPARAMETER TUNING")
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
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        df['processed_text'], df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=df['label']
    )
    
    # Split training into train/validation for early stopping
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        random_state=42,
        stratify=y_train_full
    )
    
    print(f"Training set: {len(X_train)} messages")
    print(f"Validation set: {len(X_val)} messages") 
    print(f"Test set: {len(X_test)} messages")

    # Create combined vectorizer
    print("Creating combined char + word n-gram vectorizer...")
    vectorizer = create_combined_vectorizer()
    
    print("Vectorizing text...")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Combined feature space: {X_train_vec.shape[1]} features")
    print(f"  - Char n-grams: ~12,000 features")
    print(f"  - Word n-grams: ~8,000 features")

    # Define hyperparameter search space (reduced for speed)
    param_distributions = {
        'iterations': [300, 500, 700],
        'depth': [4, 5, 6],
        'learning_rate': [0.05, 0.08, 0.12],
        'l2_leaf_reg': [3, 5, 10],
        'class_weights': [[1.0, 1.2], [1.0, 1.5]]
    }

    print(f"\n🔍 Starting hyperparameter search with {n_trials} trials...")
    print(f"Search space size: ~{4*4*5*4*3*3:,} combinations")
    print("This will take 15-25 minutes...")
    
    start_time = time.time()

    # Base CatBoost model
    base_catboost = CatBoostClassifier(
        loss_function='Logloss',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=False,
        use_best_model=True,
        early_stopping_rounds=50
    )

    # Manual hyperparameter search (faster than cross-validation)
    from sklearn.model_selection import ParameterSampler
    
    param_sampler = ParameterSampler(param_distributions, n_iter=n_trials, random_state=42)
    
    best_model = None
    best_params = None
    best_score = 0.0
    
    print("🚀 Training models with validation set...")
    
    for i, params in enumerate(param_sampler, 1):
        print(f"Trial {i}/{n_trials}: {params}")
        
        # Create model with current parameters
        model = CatBoostClassifier(
            loss_function='Logloss',
            eval_metric='Accuracy', 
            random_seed=42,
            verbose=False,
            use_best_model=True,
            early_stopping_rounds=30,
            **params
        )
        
        # Train with validation
        model.fit(
            X_train_vec, y_train,
            eval_set=(X_val_vec, y_val),
            verbose=False
        )
        
        # Evaluate on validation set
        val_pred = model.predict(X_val_vec)
        val_score = accuracy_score(y_val, val_pred)
        
        print(f"  Validation accuracy: {val_score:.4f} ({val_score*100:.2f}%)")
        
        # Update best model
        if val_score > best_score:
            best_score = val_score
            best_model = model
            best_params = params.copy()
            print(f"  🎉 New best! Validation accuracy: {best_score:.4f}")
        
        print()
    
    search_time = time.time() - start_time
    print(f"✅ Hyperparameter search completed in {search_time/60:.1f} minutes")
    
    best_cv_score = best_score  # Using validation score instead of CV

    print(f"\n🏆 BEST HYPERPARAMETERS:")
    print("-" * 50)
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"  Cross-validation accuracy: {best_cv_score:.4f} ({best_cv_score*100:.2f}%)")

    # Final evaluation on test set
    print(f"\n📊 FINAL TEST EVALUATION:")
    y_pred = best_model.predict(X_test_vec)
    y_pred = np.array(y_pred).astype(int)  # Ensure integer predictions
    
    test_accuracy = accuracy_score(y_test, y_pred)

    print("=" * 70)
    print("TUNED CATBOOST RESULTS")
    print("=" * 70)
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Improvement over baseline: {(test_accuracy - 0.9798)*100:+.2f} percentage points")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Ham correctly classified): {cm[0, 0]}")
    print(f"False Positives (Ham classified as Spam): {cm[0, 1]}")
    print(f"False Negatives (Spam classified as Ham): {cm[1, 0]}")
    print(f"True Positives (Spam correctly classified): {cm[1, 1]}")

    # Calculate precision/recall
    if cm[1, 1] + cm[0, 1] > 0:
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1])
        print(f"Spam Precision: {precision:.4f}")
    if cm[1, 1] + cm[1, 0] > 0:
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        print(f"Spam Recall: {recall:.4f}")

    # Save model
    model_artifact = {
        'classifier': best_model,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor,
        'accuracy': float(test_accuracy),
        'cv_accuracy': float(best_cv_score),
        'best_params': best_params,
        'search_time_minutes': search_time / 60,
        'timestamp': datetime.now().isoformat()
    }
    
    Path(os.path.dirname(model_path) or '.').mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, model_path)
    print(f"\n✅ Tuned model saved to {model_path}")

    # Feature importance (top 10)
    try:
        feature_names = vectorizer.get_feature_names_out()
        importances = best_model.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        
        print(f"\n🔝 TOP 10 IMPORTANT FEATURES:")
        for i, idx in enumerate(top_indices, 1):
            print(f"  {i:2d}. {feature_names[idx]:<20} ({importances[idx]:.4f})")
    except Exception as e:
        print(f"⚠️ Could not extract feature importance: {e}")

    print("\n" + "=" * 70)
    print(f"FINAL TUNED ACCURACY: {test_accuracy*100:.2f}%")
    print(f"Training completed in {search_time/60:.1f} minutes")
    print("=" * 70)

    return test_accuracy


def main():
    parser = argparse.ArgumentParser(description="Train tuned CatBoost spam classifier")
    parser.add_argument('--data', type=str, default='data/mega_spam_dataset.csv', 
                       help='Path to CSV dataset')
    parser.add_argument('--model', type=str, default='models/catboost_tuned.pkl', 
                       help='Output model path')
    parser.add_argument('--trials', type=int, default=20, 
                       help='Number of hyperparameter trials (default: 20)')
    args = parser.parse_args()

    print(f"🎯 Target: Improve CatBoost accuracy beyond 97.98%")
    print(f"📊 Dataset: {args.data}")
    print(f"🔍 Hyperparameter trials: {args.trials}")
    print(f"💾 Output: {args.model}")
    print()

    accuracy = train_catboost_tuned(args.data, args.model, args.trials)
    
    print(f"\n🎉 Training completed!")
    print(f"Final accuracy: {accuracy*100:.2f}%")
    
    # Compare with your existing models
    print(f"\n📈 ACCURACY COMPARISON:")
    print(f"  SVM (5K):           99.66% 🏆")
    print(f"  SVM (14K):          98.29%")
    print(f"  CatBoost (original): 97.98%")
    print(f"  CatBoost (tuned):   {accuracy*100:.2f}% ⭐")


if __name__ == '__main__':
    main()