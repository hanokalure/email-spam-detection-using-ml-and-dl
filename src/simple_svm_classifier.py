"""
SMS Spam Detection using Support Vector Machine (SVM) - Simplified Version
This script trains an SVM classifier on the SMS spam dataset without NLTK dependencies.
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from datetime import datetime
import argparse

class SimplePreprocessor:
    """Simple text preprocessing without NLTK"""
    
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
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Keep digits and important punctuation for spam detection
        # Remove only problematic characters, keep letters, digits, and key punctuation
        text = re.sub(r'[^a-z0-9\s!?$£%+\-.,:]', ' ', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """Remove common stopwords (less aggressive for short messages)"""
        words = text.split()
        # Keep more words in short messages, only remove very common stopwords
        # and don't enforce minimum length for SMS-style texts
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 1]
        return ' '.join(filtered_words)
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

class SimpleSVMClassifier:
    """Simple SVM-based SMS Spam Classifier"""
    
    def __init__(self):
        self.preprocessor = SimplePreprocessor()
        # Use character n-grams which work much better for short SMS texts
        self.vectorizer = TfidfVectorizer(
            analyzer='char_wb',  # Character n-grams within word boundaries
            ngram_range=(3, 5),  # 3-5 character sequences
            max_features=10000,
            min_df=2,
            max_df=0.95,
            lowercase=True,
            sublinear_tf=True,  # Better for character n-grams
            strip_accents='ascii'
        )
        self.classifier = LinearSVC(
            C=1.0, 
            class_weight='balanced',  # Handle class imbalance
            random_state=42, 
            max_iter=10000
        )
        self.is_trained = False
    
    def load_data(self, file_path):
        """Load SMS data from CSV file with support for multiple formats"""
        print(f"Loading data from {file_path}...")
        
        # Read CSV with specific columns
        df = pd.read_csv(file_path, encoding='latin-1')
        
        # Detect dataset format and standardize
        if 'text' in df.columns and 'target' in df.columns:
            # Format: text, target (0=ham, 1=spam)
            df = df[['text', 'target']].copy()
            df.columns = ['text', 'label']
            print(f"Detected format: text/target (0=ham, 1=spam)")
        elif 'text' in df.columns and 'label' in df.columns:
            # Format: text, label (spam/ham or 0/1)
            df = df[['text', 'label']].copy()
            # Check if labels are already numeric (0/1) or string (spam/ham)
            if df['label'].dtype in ['int64', 'float64']:
                print(f"Detected format: text/label (numeric 0=ham, 1=spam)")
            else:
                df['label'] = df['label'].map({'spam': 1, 'ham': 0})
                print(f"Detected format: text/label (spam/ham)")
        elif 'v1' in df.columns and 'v2' in df.columns:
            # UCI SMS format: v1=label (ham/spam), v2=text
            df = df[['v2', 'v1']].copy()
            df.columns = ['text', 'label']
            df['label'] = df['label'].map({'spam': 1, 'ham': 0})
            print(f"Detected format: UCI SMS (v1=label, v2=text)")
        elif len(df.columns) >= 2:
            # Legacy format: assume first two columns
            df = df.iloc[:, :2]
            df.columns = ['text', 'label']
            # Check if first column contains ham/spam
            if df['text'].dtype == 'object' and any(df['text'].str.contains('ham|spam', na=False)):
                # Swap columns if first column is actually labels
                df = df[['label', 'text']].copy()
                df.columns = ['text', 'label']
            df['label'] = df['label'].map({'spam': 1, 'ham': 0})
            print(f"Detected format: legacy (first 2 columns)")
        else:
            raise ValueError("Dataset format not recognized")
        
        # Remove any rows with missing data
        df = df.dropna()
        
        # Remove rows where label mapping failed (only for string labels)
        if not df['label'].dtype in ['int64', 'float64']:
            df = df.dropna(subset=['label'])
        
        print(f"Dataset loaded: {len(df)} messages")
        print(f"Spam messages: {sum(df['label'])} ({sum(df['label'])/len(df)*100:.1f}%)")
        print(f"Ham messages: {len(df) - sum(df['label'])} ({(len(df) - sum(df['label']))/len(df)*100:.1f}%)")
        
        return df
    
    def preprocess_data(self, df):
        """Preprocess the text data"""
        print("Preprocessing text data...")
        df['processed_text'] = df['text'].apply(self.preprocessor.preprocess)
        return df
    
    def train(self, file_path, test_size=0.2):
        """Train the SVM classifier"""
        # Load and preprocess data
        df = self.load_data(file_path)
        df = self.preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['label'], 
            test_size=test_size, 
            random_state=42, 
            stratify=df['label']
        )
        
        print(f"Training set: {len(X_train)} messages")
        print(f"Test set: {len(X_test)} messages")
        
        # Vectorize text
        print("Vectorizing text using TF-IDF...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_vec.shape}")
        
        # Train classifier
        print("Training SVM classifier...")
        self.classifier.fit(X_train_vec, y_train)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*60)
        print("SVM CLASSIFIER RESULTS")
        print("="*60)
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives (Ham correctly classified): {cm[0, 0]}")
        print(f"False Positives (Ham classified as Spam): {cm[0, 1]}")
        print(f"False Negatives (Spam classified as Ham): {cm[1, 0]}")
        print(f"True Positives (Spam correctly classified): {cm[1, 1]}")
        
        # Calculate additional metrics
        precision_spam = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        recall_spam = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        f1_spam = 2 * (precision_spam * recall_spam) / (precision_spam + recall_spam) if (precision_spam + recall_spam) > 0 else 0
        
        print(f"\nSpam Detection Metrics:")
        print(f"Precision: {precision_spam:.4f}")
        print(f"Recall: {recall_spam:.4f}")
        print(f"F1-Score: {f1_spam:.4f}")
        
        self.is_trained = True
        
        # Store test data for later use
        self.X_test = X_test
        self.y_test = y_test
        self.accuracy = accuracy
        
        return accuracy
    
    def predict(self, text):
        """Predict if a single message is spam or not"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Preprocess the text
        processed_text = self.preprocessor.preprocess(text)
        
        # Vectorize
        text_vec = self.vectorizer.transform([processed_text])
        
        # Predict
        prediction = self.classifier.predict(text_vec)[0]
        confidence = self.classifier.decision_function(text_vec)[0]
        
        return prediction, confidence
    
    def save_model(self, model_path):
        """Save trained model and vectorizer"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'classifier': self.classifier,
            'vectorizer': self.vectorizer,
            'preprocessor': self.preprocessor,
            'accuracy': self.accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load trained model and vectorizer"""
        model_data = joblib.load(model_path)
        
        self.classifier = model_data['classifier']
        self.vectorizer = model_data['vectorizer']
        self.preprocessor = model_data['preprocessor']
        self.accuracy = model_data['accuracy']
        self.is_trained = True
        
        print(f"Model loaded from {model_path}")
        print(f"Model accuracy: {self.accuracy:.4f}")

def main():
    """Main function to train and evaluate SVM classifier"""
    parser = argparse.ArgumentParser(description="Train an SVM spam classifier")
    parser.add_argument("--data", type=str, default="data/spam.csv", help="Path to CSV with SMS data")
    parser.add_argument("--model", type=str, default="models/svm_model.pkl", help="Output path for the trained model")
    args = parser.parse_args()

    print("="*60)
    print("SMS SPAM DETECTION USING SVM")
    print("="*60)
    
    # Initialize classifier
    svm_classifier = SimpleSVMClassifier()
    
    # Train the model
    data_path = args.data
    accuracy = svm_classifier.train(data_path)
    
    # Save the trained model
    model_path = args.model
    svm_classifier.save_model(model_path)
    
    # Test with sample messages
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    
    test_messages = [
        "Hi how are you doing today?",
        "FREE! Win a £1000 cash prize! Text WIN to 85233 now!",
        "Can you pick up milk on your way home?",
        "URGENT! You have won a lottery prize of $50000! Call now!",
        "Meeting postponed to 3pm tomorrow",
        "Congratulations! You're eligible for a free iPhone! Click here!",
        "Call me when you get this message",
        "WINNER!! You have been selected to receive a £900 prize reward!"
    ]
    
    for i, msg in enumerate(test_messages, 1):
        pred, confidence = svm_classifier.predict(msg)
        label = "SPAM" if pred == 1 else "HAM"
        print(f"\n{i}. Message: '{msg[:50]}{'...' if len(msg) > 50 else ''}'")
        print(f"   Prediction: {label} (confidence: {abs(confidence):.3f})")
    
    print("\n" + "="*60)
    print(f"FINAL ACCURACY: {accuracy*100:.2f}%")
    print("="*60)
    
    return accuracy

if __name__ == "__main__":
    accuracy = main()
