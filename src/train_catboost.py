#!/usr/bin/env python3
"""
Train CatBoost model for Email Spam Detection using the mega dataset.
- Uses the same SimplePreprocessor as your SVM for consistency
- Vectorizes text with TF-IDF (character n-grams)
- Saves model as models/catboost.pkl by default
"""

import os
import argparse
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CatBoost
from catboost import CatBoostClassifier

# Reuse your existing preprocessor for consistency
import sys
import os
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


def train_catboost(data_path: str, model_path: str) -> float:
    print("=" * 60)
    print("EMAIL SPAM DETECTION USING CATBOOST")
    print("=" * 60)

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

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    print(f"Training set: {len(X_train)} messages")
    print(f"Test set: {len(X_test)} messages")

    # Vectorize (character n-grams like SVM, but a bit larger feature space)
    print("Vectorizing text using TF-IDF (char n-grams)...")
    vectorizer = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        max_features=15000,   # Reduced for faster training
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        lowercase=True,
        strip_accents='ascii'
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print(f"Feature matrix shape: {X_train_vec.shape}")

    # CatBoost model (simplified for faster training)
    model = CatBoostClassifier(
        iterations=200,  # Reduced from 600
        depth=4,         # Reduced from 6
        learning_rate=0.15,
        loss_function='Logloss',
        random_seed=42,
        verbose=50,      # Show progress every 50 iterations
        class_weights=[1.0, 1.2]  # Slightly upweight spam
    )

    print("Training CatBoost classifier...")
    # CatBoost supports sparse matrices via pool, but can accept directly too
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    # y_pred returns strings sometimes; coerce to int
    y_pred = pd.Series(y_pred).astype(int)

    acc = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 60)
    print("CATBOOST CLASSIFIER RESULTS")
    print("=" * 60)
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives (Ham correctly classified): {cm[0, 0]}")
    print(f"False Positives (Ham classified as Spam): {cm[0, 1]}")
    print(f"False Negatives (Spam classified as Ham): {cm[1, 0]}")
    print(f"True Positives (Spam correctly classified): {cm[1, 1]}")

    # Save
    model_artifact = {
        'classifier': model,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor,
        'accuracy': float(acc),
        'timestamp': datetime.now().isoformat()
    }
    Path(os.path.dirname(model_path) or '.').mkdir(parents=True, exist_ok=True)
    joblib.dump(model_artifact, model_path)
    print(f"\nModel saved to {model_path}")

    print("\n" + "=" * 60)
    print(f"FINAL ACCURACY: {acc*100:.2f}%")
    print("=" * 60)

    return acc


def main():
    parser = argparse.ArgumentParser(description="Train a CatBoost spam classifier")
    parser.add_argument('--data', type=str, default='data/mega_spam_dataset.csv', help='Path to CSV dataset')
    parser.add_argument('--model', type=str, default='models/catboost.pkl', help='Output model path')
    args = parser.parse_args()

    train_catboost(args.data, args.model)


if __name__ == '__main__':
    main()
