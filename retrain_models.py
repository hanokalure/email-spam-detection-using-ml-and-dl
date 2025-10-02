#!/usr/bin/env python3
"""
Quick script to retrain SVM and CatBoost models with proper serialization
This fixes the pickle loading issues in the interactive predictor
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from simple_svm_classifier import SimplePreprocessor
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    print("⚠️ SimplePreprocessor not available, using basic preprocessing")

try:
    import catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available, skipping CatBoost training")

def load_dataset():
    """Load a basic dataset for retraining"""
    # Try to load comprehensive dataset, fallback to any available CSV
    data_files = [
        "data/comprehensive_spam_dataset.csv",
        "data/mega_spam_dataset.csv", 
        "data/spam_assassin_large.csv"
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"Loading dataset: {data_file}")
            df = pd.read_csv(data_file)
            
            # Normalize columns
            if 'text' in df.columns and 'label' in df.columns:
                df = df[['text', 'label']].copy()
            elif 'v1' in df.columns and 'v2' in df.columns:
                df = df[['v2', 'v1']].copy()
                df.columns = ['text', 'label']
            else:
                print(f"⚠️ Unknown format for {data_file}")
                continue
                
            # Normalize labels
            if df['label'].dtype == 'object':
                df['label'] = df['label'].str.lower().map({'ham': 0, 'spam': 1})
            
            df = df.dropna().reset_index(drop=True)
            print(f"Dataset loaded: {len(df)} samples")
            return df
    
    print("❌ No suitable dataset found!")
    return None

def retrain_svm():
    """Retrain SVM model with proper serialization"""
    print("\n=== Retraining SVM Model ===")
    
    df = load_dataset()
    if df is None:
        return False
    
    # Initialize preprocessor
    if PREPROCESSOR_AVAILABLE:
        preprocessor = SimplePreprocessor()
        df['processed_text'] = df['text'].apply(preprocessor.preprocess)
    else:
        # Basic preprocessing
        df['processed_text'] = df['text'].str.lower().str.strip()
        preprocessor = None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train SVM
    classifier = LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=10000)
    classifier.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"SVM Accuracy: {accuracy:.4f}")
    
    # Save with proper structure
    model_data = {
        'classifier': classifier,
        'vectorizer': vectorizer,
        'preprocessor': preprocessor,
        'accuracy': accuracy,
        'model_type': 'LinearSVM'
    }
    
    model_path = "models/svm_full.pkl"
    joblib.dump(model_data, model_path)
    print(f"✅ SVM model saved to {model_path}")
    
    return True

def retrain_catboost():
    """Retrain CatBoost model with proper serialization"""
    if not CATBOOST_AVAILABLE:
        print("\n❌ CatBoost not available, skipping")
        return False
        
    print("\n=== Retraining CatBoost Model ===")
    
    df = load_dataset()
    if df is None:
        return False
    
    # Take a smaller sample for quick training
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)
        print(f"Using sample of {len(df)} for quick training")
    
    # Initialize preprocessor
    if PREPROCESSOR_AVAILABLE:
        preprocessor = SimplePreprocessor()
        df['processed_text'] = df['text'].apply(preprocessor.preprocess)
    else:
        df['processed_text'] = df['text'].str.lower().str.strip()
        preprocessor = None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    
    # Vectorize
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train CatBoost
    classifier = catboost.CatBoostClassifier(
        iterations=100,  # Quick training
        depth=6,
        learning_rate=0.1,
        random_seed=42,
        verbose=False
    )
    
    classifier.fit(X_train_vec.toarray(), y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test_vec.toarray())
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"CatBoost Accuracy: {accuracy:.4f}")
    
    # Save with proper structure
    model_data = {
        'model': classifier,
        'classifier': classifier,  # Both keys for compatibility
        'vectorizer': vectorizer,
        'preprocessor': preprocessor,
        'accuracy': accuracy,
        'test_accuracy': accuracy,
        'model_type': 'CatBoostClassifier'
    }
    
    model_path = "models/catboost_tuned.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ CatBoost model saved to {model_path}")
    
    return True

def main():
    print("🔄 Retraining Models for Compatibility")
    print("=" * 50)
    
    os.makedirs("models", exist_ok=True)
    
    # Retrain both models
    svm_success = retrain_svm()
    catboost_success = retrain_catboost()
    
    print(f"\n📊 Results:")
    print(f"SVM: {'✅ Success' if svm_success else '❌ Failed'}")
    print(f"CatBoost: {'✅ Success' if catboost_success else '❌ Failed'}")
    
    if svm_success or catboost_success:
        print(f"\n🎉 Model retraining completed!")
        print(f"You can now use predict_main.py with the fixed models")
    else:
        print(f"\n⚠️ No models were successfully retrained")
        print(f"Please check that you have the required datasets and dependencies")

if __name__ == "__main__":
    main()