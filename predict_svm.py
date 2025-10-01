#!/usr/bin/env python3
"""
SVM Spam Predictor
Easy interface for using trained SVM models
"""

import os
import sys
import joblib
import numpy as np
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from simple_svm_classifier import SimplePreprocessor
    SIMPLE_SVM_AVAILABLE = True
except ImportError:
    SIMPLE_SVM_AVAILABLE = False

class SVMPredictor:
    """Easy-to-use predictor for SVM models"""
    
    def __init__(self, model_path: str):
        """Initialize predictor with saved model
        
        Args:
            model_path: Path to saved .pkl model file
        """
        self.model_path = model_path
        
        # Load model and components
        self._load_model()
        
        print(f"✅ SVM model loaded")
        print(f"📊 Model: LinearSVM with TF-IDF vectorization")
        print(f"🎯 Optimized for email spam detection")
    
    def _load_model(self):
        """Load model, vectorizer, and preprocessor"""
        # Load the saved model
        model_data = joblib.load(self.model_path)
        
        if isinstance(model_data, dict):
            # New format with all components
            self.classifier = model_data['classifier']
            self.vectorizer = model_data['vectorizer']
            self.preprocessor = model_data.get('preprocessor')
            self.accuracy = model_data.get('accuracy', 'Unknown')
        else:
            # Legacy format - just the classifier
            self.classifier = model_data
            self.vectorizer = None
            self.preprocessor = None
            self.accuracy = 'Unknown'
        
        # Initialize preprocessor if not available
        if self.preprocessor is None and SIMPLE_SVM_AVAILABLE:
            self.preprocessor = SimplePreprocessor()
        
        # Model info
        self.model_info = {
            'name': 'Support Vector Machine (SVM)',
            'accuracy': f'{self.accuracy:.2%}' if isinstance(self.accuracy, float) else str(self.accuracy),
            'type': 'LinearSVM',
            'features': 'TF-IDF vectorization',
            'dataset_size': 'Unknown',
            'speed': 'Fast (~1-5ms)'
        }
    
    def predict(self, text: str, return_details: bool = False) -> Dict:
        """Predict if text is spam or ham
        
        Args:
            text: Input text to classify
            return_details: If True, return detailed prediction info
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess text if preprocessor available
        if self.preprocessor:
            processed_text = self.preprocessor.preprocess(text)
        else:
            processed_text = text.lower().strip()
        
        # Vectorize text
        if self.vectorizer:
            text_vec = self.vectorizer.transform([processed_text])
        else:
            raise ValueError("Model vectorizer not available")
        
        # Predict
        prediction = self.classifier.predict(text_vec)[0]
        
        # Get probability/confidence
        try:
            if hasattr(self.classifier, 'decision_function'):
                confidence_score = self.classifier.decision_function(text_vec)[0]
                # Convert decision function score to probability-like value
                probability = 1 / (1 + np.exp(-confidence_score))  # Sigmoid
            elif hasattr(self.classifier, 'predict_proba'):
                proba = self.classifier.predict_proba(text_vec)[0]
                probability = proba[1] if len(proba) > 1 else proba[0]
            else:
                probability = 0.5  # Default if no probability available
        except:
            probability = 0.5
        
        # Convert prediction to string
        prediction_label = 'SPAM' if prediction == 1 else 'HAM'
        confidence = probability if prediction == 1 else (1 - probability)
        
        # Basic result
        result = {
            'prediction': prediction_label,
            'probability': probability,
            'confidence': confidence,
            'is_spam': prediction == 1
        }
        
        # Add details if requested
        if return_details:
            result.update({
                'original_text': text,
                'processed_text': processed_text,
                'text_length': len(text),
                'model_info': self.model_info,
                'features_extracted': text_vec.shape[1] if hasattr(text_vec, 'shape') else 'Unknown'
            })
        
        return result


def main():
    """Test the SVM predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SVM Spam Predictor")
    parser.add_argument("--model", default="models/svm_full.pkl", help="Path to SVM model")
    parser.add_argument("--text", default="Free money! Click here now!", help="Text to predict")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    try:
        predictor = SVMPredictor(args.model)
        result = predictor.predict(args.text, return_details=True)
        
        print(f"\n📧 Text: {args.text}")
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"📈 Probability: {result['probability']:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()