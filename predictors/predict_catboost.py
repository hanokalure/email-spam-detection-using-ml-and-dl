#!/usr/bin/env python3
"""
CatBoost Spam Predictor
Easy interface for using trained CatBoost models
"""

import os
import sys
import pickle
import numpy as np
from typing import Dict, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    import catboost
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from simple_svm_classifier import SimplePreprocessor
    SIMPLE_PREPROCESSOR_AVAILABLE = True
except ImportError:
    SIMPLE_PREPROCESSOR_AVAILABLE = False

class CatBoostPredictor:
    """Easy-to-use predictor for CatBoost models"""
    
    def __init__(self, model_path: str):
        """Initialize predictor with saved model
        
        Args:
            model_path: Path to saved .pkl model file
        """
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost not installed. Install with: pip install catboost")
        
        self.model_path = model_path
        
        # Load model and components
        self._load_model()
        
        print(f"✅ CatBoost model loaded")
        print(f"📊 Model: CatBoost Gradient Boosting")
        print(f"🎯 Optimized for high accuracy spam detection")
    
    def _load_model(self):
        """Load model, vectorizer, and preprocessor"""
        # Load the saved model with joblib first (most compatible) then fallback to pickle
        import sys
        import types
        import joblib
        
        # Add src to path for class imports and handle __main__ mappings
        src_path = os.path.join(os.path.dirname(__file__), 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Handle potential __main__.SimplePreprocessor references
        try:
            from simple_svm_classifier import SimplePreprocessor as SSP
            if '__main__' not in sys.modules:
                sys.modules['__main__'] = types.ModuleType('__main__')
            setattr(sys.modules['__main__'], 'SimplePreprocessor', SSP)
        except Exception:
            pass
        
        # Try joblib first (more reliable for sklearn/catboost models)
        try:
            model_data = joblib.load(self.model_path)
            print("✅ Model loaded successfully with joblib")
        except Exception as e:
            print(f"⚠️ Joblib loading failed: {e}")
            print("Trying pickle loading...")
            
            # Fallback to pickle
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                print("✅ Model loaded successfully with pickle")
            except Exception as e2:
                print(f"❌ Pickle loading also failed: {e2}")
                # Create minimal structure
                model_data = {'model': None, 'vectorizer': None, 'preprocessor': None}
                print("Creating minimal model structure...")
        
        if isinstance(model_data, dict):
            # New format with all components
            self.classifier = model_data.get('model') or model_data.get('classifier')
            self.vectorizer = model_data.get('vectorizer')
            self.preprocessor = model_data.get('preprocessor')
            self.accuracy = model_data.get('test_accuracy', model_data.get('accuracy', 'Unknown'))
            self.feature_names = model_data.get('feature_names', [])
        else:
            # Legacy format - just the classifier
            self.classifier = model_data
            self.vectorizer = None
            self.preprocessor = None
            self.accuracy = 'Unknown'
            self.feature_names = []
        
        # Initialize preprocessor if not available
        if self.preprocessor is None and SIMPLE_PREPROCESSOR_AVAILABLE:
            self.preprocessor = SimplePreprocessor()
        
        # Model info
        self.model_info = {
            'name': 'CatBoost Gradient Boosting',
            'accuracy': f'{self.accuracy:.2%}' if isinstance(self.accuracy, float) else str(self.accuracy),
            'type': 'CatBoostClassifier',
            'features': f'{len(self.feature_names)} features' if self.feature_names else 'TF-IDF + engineered features',
            'dataset_size': 'Comprehensive spam dataset',
            'speed': 'Medium (~5-15ms)'
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
        
        # Convert to format expected by CatBoost
        if hasattr(text_vec, 'toarray'):
            text_features = text_vec.toarray()[0]
        else:
            text_features = text_vec[0]
        
        # Predict
        try:
            prediction = self.classifier.predict([text_features])[0]
            
            # Get probability
            try:
                probabilities = self.classifier.predict_proba([text_features])[0]
                probability = probabilities[1] if len(probabilities) > 1 else probabilities[0]
            except:
                # Fallback if predict_proba not available
                probability = 0.5
                
        except Exception as e:
            print(f"Prediction error: {e}")
            prediction = 0
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
                'features_extracted': len(text_features) if isinstance(text_features, (list, np.ndarray)) else 'Unknown'
            })
        
        return result


def main():
    """Test the CatBoost predictor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test CatBoost Spam Predictor")
    parser.add_argument("--model", default="models/catboost_tuned.pkl", help="Path to CatBoost model")
    parser.add_argument("--text", default="Congratulations! You've won $1000! Click here to claim.", help="Text to predict")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return
    
    try:
        predictor = CatBoostPredictor(args.model)
        result = predictor.predict(args.text, return_details=True)
        
        print(f"\n📧 Text: {args.text}")
        print(f"🎯 Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.3f}")
        print(f"📈 Probability: {result['probability']:.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()