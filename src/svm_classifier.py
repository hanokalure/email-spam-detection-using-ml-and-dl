"""
SMS Spam Detection using Support Vector Machine (SVM)
This script trains an SVM classifier on the SMS spam dataset and evaluates its performance.
"""

import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class SMSPreprocessor:
    """Handles text preprocessing for SMS data"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
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
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\b\d{10,11}\b', '', text)
        
        # Remove extra whitespace and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """Remove common stopwords"""
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        return ' '.join(filtered_words)
    
    def preprocess(self, text):
        """Full preprocessing pipeline"""
        text = self.clean_text(text)
        text = self.remove_stopwords(text)
        return text

class SVMSpamClassifier:
    """SVM-based SMS Spam Classifier"""
    
    def __init__(self):
        self.preprocessor = SMSPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,
            max_df=0.95
        )
        self.classifier = LinearSVC(C=1.0, random_state=42, max_iter=10000)
        self.is_trained = False
    
    def load_data(self, file_path):
        """Load SMS data from CSV file"""
        print(f"Loading data from {file_path}...")
        
        # Read CSV with specific columns
        df = pd.read_csv(file_path, encoding='latin-1')
        
        # Use first two columns (assuming v1=label, v2=text)
        df = df.iloc[:, :2]
        df.columns = ['label', 'text']
        
        # Remove any rows with missing data
        df = df.dropna()
        
        # Convert labels to binary (spam=1, ham=0)
        df['label'] = df['label'].map({'spam': 1, 'ham': 0})
        
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
        
        # Train classifier
        print("Training SVM classifier...")
        self.classifier.fit(X_train_vec, y_train)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test_vec)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n" + "="*50)
        print("SVM CLASSIFIER RESULTS")
        print("="*50)
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"True Negatives (Ham correctly classified): {cm[0, 0]}")
        print(f"False Positives (Ham classified as Spam): {cm[0, 1]}")
        print(f"False Negatives (Spam classified as Ham): {cm[1, 0]}")
        print(f"True Positives (Spam correctly classified): {cm[1, 1]}")
        
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
        probability = self.classifier.decision_function(text_vec)[0]
        
        return prediction, probability
    
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
    # Initialize classifier
    svm_classifier = SVMSpamClassifier()
    
    # Train the model
    data_path = "data/spam.csv"
    accuracy = svm_classifier.train(data_path)
    
    # Save the trained model
    model_path = "models/svm_model.pkl"
    svm_classifier.save_model(model_path)
    
    # Test with sample messages
    print("\n" + "="*50)
    print("SAMPLE PREDICTIONS")
    print("="*50)
    
    test_messages = [
        "Hi how are you doing today?",
        "FREE! Win a £1000 cash prize! Text WIN to 85233 now!",
        "Can you pick up milk on your way home?",
        "URGENT! You have won a lottery prize of $50000! Call now!",
        "Meeting postponed to 3pm tomorrow",
        "Congratulations! You're eligible for a free iPhone! Click here!"
    ]
    
    for msg in test_messages:
        pred, prob = svm_classifier.predict(msg)
        label = "SPAM" if pred == 1 else "HAM"
        confidence = abs(prob)
        print(f"\nMessage: '{msg[:50]}{'...' if len(msg) > 50 else ''}'")
        print(f"Prediction: {label} (confidence: {confidence:.3f})")
    
    return accuracy

if __name__ == "__main__":
    accuracy = main()