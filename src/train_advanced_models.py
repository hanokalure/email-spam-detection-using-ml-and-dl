#!/usr/bin/env python3
"""
Train advanced ML models for email spam detection
Supports: XGBoost, LightGBM, Logistic Regression, Calibrated SVM, Complement Naive Bayes
"""

import os
import sys
import pickle
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Optional imports (install if available)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True  
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not installed. Install with: pip install lightgbm")

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

class SimplePreprocessor:
    """Text preprocessor for spam detection"""
    
    def __init__(self):
        # Common English stopwords
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
            'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once', 'to', 'from'
        }
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        import re
        import pandas as pd
        
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Keep digits and important punctuation for spam detection
        text = re.sub(r'[^a-z0-9\s!?$£%+\-.,:]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """Remove common stopwords"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 1]
        return ' '.join(filtered_words)
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

class AdvancedSpamClassifier:
    """Advanced spam classifier with multiple ML algorithms"""
    
    def __init__(self, model_type='xgboost'):
        """
        Initialize classifier
        
        Args:
            model_type: 'xgboost', 'lightgbm', 'logreg', 'calibrated_svm', 'complement_nb'
        """
        self.model_type = model_type
        self.vectorizer = None
        self.classifier = None
        self.preprocessor = SimplePreprocessor()
        
    def load_combined_data(self, datasets=None):
        """Load and combine multiple datasets"""
        
        if datasets is None:
            datasets = [
                "data/mega_spam_dataset.csv",  # Combined mega dataset
                "data/spam.csv",  # Original SpamAssassin (fallback)
            ]
        
        all_data = []
        total_emails = 0
        
        for dataset_path in datasets:
            if not os.path.exists(dataset_path):
                print(f"⚠️ Dataset not found: {dataset_path}")
                continue
                
            print(f"📥 Loading {dataset_path}...")
            
            try:
                df = pd.read_csv(dataset_path)
                
                # Standardize column names
                if 'v1' in df.columns and 'v2' in df.columns:
                    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
                elif 'Category' in df.columns and 'Message' in df.columns:
                    df = df.rename(columns={'Category': 'label', 'Message': 'text'})
                
                # Ensure we have the right columns
                if 'label' not in df.columns or 'text' not in df.columns:
                    print(f"❌ Invalid format in {dataset_path}")
                    continue
                
                # Standardize labels
                df['label'] = df['label'].str.lower()
                df['label'] = df['label'].map({'ham': 'ham', 'spam': 'spam'})
                
                # Remove invalid rows
                df = df.dropna(subset=['label', 'text'])
                df = df[df['label'].isin(['ham', 'spam'])]
                
                all_data.append(df)
                total_emails += len(df)
                
                print(f"✅ Loaded {len(df)} emails from {dataset_path}")
                print(f"   Ham: {len(df[df['label'] == 'ham'])}, Spam: {len(df[df['label'] == 'spam'])}")
                
            except Exception as e:
                print(f"❌ Error loading {dataset_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid datasets found!")
        
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        print(f"🔄 Removing duplicates...")
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
        after_dedup = len(combined_df)
        
        print(f"📊 Combined Dataset Summary:")
        print(f"Total emails: {after_dedup} (removed {before_dedup - after_dedup} duplicates)")
        print(f"Ham emails: {len(combined_df[combined_df['label'] == 'ham'])}")
        print(f"Spam emails: {len(combined_df[combined_df['label'] == 'spam'])}")
        
        return combined_df['text'].tolist(), combined_df['label'].tolist()
    
    def train(self, texts, labels, test_size=0.2, random_state=42):
        """Train the classifier"""
        
        print(f"\n🚀 Training {self.model_type.upper()} model...")
        
        # Preprocess texts
        print("🔄 Preprocessing texts...")
        processed_texts = [self.preprocessor.preprocess(text) for text in texts]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        print(f"Training set: {len(X_train)} emails")
        print(f"Test set: {len(X_test)} emails")
        
        # Vectorize text
        print("🔄 Creating TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            analyzer='char_wb',   # Character n-grams
            min_df=2,            # Ignore rare features
            max_df=0.95,         # Ignore very common features
            max_features=50000,  # Limit feature space
            sublinear_tf=True,   # Apply sublinear scaling
            binary=False         # Use term frequency
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Feature space: {X_train_vec.shape[1]} features")
        
        # Train model based on type
        start_time = time.time()
        
        if self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.classifier = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            )
            
        elif self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.classifier = lgb.LGBMClassifier(
                objective='binary',
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=random_state,
                n_jobs=-1,
                verbosity=-1
            )
            
        elif self.model_type == 'logreg':
            self.classifier = LogisticRegression(
                C=1.0,
                solver='liblinear',
                random_state=random_state,
                max_iter=1000,
                n_jobs=-1
            )
            
        elif self.model_type == 'calibrated_svm':
            base_svm = LinearSVC(
                C=1.0,
                random_state=random_state,
                max_iter=2000
            )
            self.classifier = CalibratedClassifierCV(
                base_svm,
                method='sigmoid',
                cv=3
            )
            
        elif self.model_type == 'complement_nb':
            self.classifier = ComplementNB(
                alpha=1.0,
                norm=False
            )
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train the model
        print(f"🔄 Training {self.model_type} model...")
        
        # Convert labels to binary
        y_train_binary = [1 if label == 'spam' else 0 for label in y_train]
        y_test_binary = [1 if label == 'spam' else 0 for label in y_test]
        
        self.classifier.fit(X_train_vec, y_train_binary)
        
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Evaluate model
        print("\n📊 Model Evaluation:")
        
        # Predictions
        y_pred = self.classifier.predict(X_test_vec)
        
        # Accuracy
        accuracy = accuracy_score(y_test_binary, y_pred)
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Detailed classification report
        target_names = ['Ham', 'Spam']
        print("\nDetailed Classification Report:")
        print(classification_report(y_test_binary, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test_binary, y_pred)
        print("\nConfusion Matrix:")
        print(f"             Predicted")
        print(f"Actual    Ham    Spam")
        print(f"Ham      {cm[0,0]:4d}   {cm[0,1]:4d}")
        print(f"Spam     {cm[1,0]:4d}   {cm[1,1]:4d}")
        
        # Test prediction speed
        start_time = time.time()
        sample_predictions = self.classifier.predict(X_test_vec[:100])
        prediction_time = (time.time() - start_time) / 100
        
        print(f"\nPrediction Speed: {prediction_time*1000:.2f}ms per email")
        
        return accuracy
    
    def predict(self, text):
        """Predict if text is spam or ham"""
        if self.classifier is None or self.vectorizer is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess
        processed_text = self.preprocessor.preprocess(text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.classifier.predict(text_vec)[0]
        
        # Get probability if available
        try:
            proba = self.classifier.predict_proba(text_vec)[0]
            confidence = max(proba)
        except:
            confidence = None
        
        result = 'spam' if prediction == 1 else 'ham'
        
        return result, confidence
    
    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            'model_type': self.model_type,
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'preprocessor': self.preprocessor
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model_type = model_data['model_type']
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.preprocessor = model_data['preprocessor']
        
        print(f"✅ Model loaded from {filepath}")

def train_all_models():
    """Train all available models and compare performance"""
    
    # Define models to train
    models_to_train = []
    
    if XGBOOST_AVAILABLE:
        models_to_train.append('xgboost')
    if LIGHTGBM_AVAILABLE:
        models_to_train.append('lightgbm')
    
    models_to_train.extend(['logreg', 'calibrated_svm', 'complement_nb'])
    
    results = {}
    
    print("🚀 Training all available models...")
    print(f"Models to train: {', '.join(models_to_train)}")
    
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training: {model_type.upper()}")
        print('='*60)
        
        try:
            # Initialize classifier
            classifier = AdvancedSpamClassifier(model_type=model_type)
            
            # Load data
            texts, labels = classifier.load_combined_data()
            
            # Train model
            accuracy = classifier.train(texts, labels)
            
            # Save model with meaningful name
            model_filename = f"models/{model_type}_mega14k.pkl"
            classifier.save_model(model_filename)
            
            results[model_type] = {
                'accuracy': accuracy,
                'model_file': model_filename
            }
            
        except Exception as e:
            print(f"❌ Error training {model_type}: {e}")
            results[model_type] = {
                'accuracy': 0.0,
                'error': str(e)
            }
    
    # Print final comparison
    print(f"\n{'='*60}")
    print("FINAL MODEL COMPARISON")
    print('='*60)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('accuracy', 0), reverse=True)
    
    print("| Model | Accuracy | Status |")
    print("|-------|----------|--------|")
    
    for model_type, result in sorted_results:
        if 'error' in result:
            print(f"| {model_type:15} | {'ERROR':8} | {result['error'][:30]} |")
        else:
            accuracy = result['accuracy']
            print(f"| {model_type:15} | {accuracy:7.4f} | ✅ Saved to {result['model_file']} |")
    
    # Recommend best model
    if sorted_results and 'error' not in sorted_results[0][1]:
        best_model = sorted_results[0][0]
        best_accuracy = sorted_results[0][1]['accuracy']
        print(f"\n🏆 BEST MODEL: {best_model.upper()} with {best_accuracy:.4f} accuracy")
        print(f"📁 Model file: {sorted_results[0][1]['model_file']}")
    
    return results

if __name__ == "__main__":
    print("🚀 Advanced ML Spam Detection Training")
    print("="*50)
    
    # Check if required packages are installed
    missing_packages = []
    if not XGBOOST_AVAILABLE:
        missing_packages.append("xgboost")
    if not LIGHTGBM_AVAILABLE:
        missing_packages.append("lightgbm")
    
    if missing_packages:
        print(f"\n💡 For best results, install: pip install {' '.join(missing_packages)}")
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train all models
    results = train_all_models()
    
    print(f"\n✅ Training complete!")
    print("Use the best performing model for your spam detection needs.")