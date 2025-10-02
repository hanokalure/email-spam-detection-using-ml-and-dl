#!/usr/bin/env python3
"""
BalancedSpamNet Predictor - Production-ready inference for the dual-head spam detection model

Key Features:
1. Fast inference with GPU support
2. Business-aware preprocessing
3. Dual-head interpretation (SPAM + HAM confidence)
4. Consistent interface with other predictors
5. Robust error handling
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import time

# Add src to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))

try:
    from balanced_spam_net import BalancedSpamNet, create_balanced_spam_net
    from train_balanced_spam_net import BusinessAwarePreprocessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure BalancedSpamNet model files are available")


class BalancedSpamNetPredictor:
    """Production predictor for BalancedSpamNet model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.vocab = None
        self.preprocessor = None
        self.device = None
        self.model_info = {}
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the BalancedSpamNet model and supporting components"""
        try:
            print(f"🔄 Loading BalancedSpamNet from {self.model_path}...")
            
            # Determine device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract components
            self.vocab = checkpoint['vocab']
            vocab_size = len(self.vocab)
            
            # Get model configuration from checkpoint
            if 'args' in checkpoint:
                args = checkpoint['args']
                embed_dim = 256  # Default from training
                cnn_filters = 128
                lstm_hidden = 256
                lstm_layers = 2
                attention_heads = 8
            else:
                # Use defaults if args not available
                embed_dim = 256
                cnn_filters = 128
                lstm_hidden = 256
                lstm_layers = 2
                attention_heads = 8
            
            # Create model
            self.model = create_balanced_spam_net(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                cnn_filters=cnn_filters,
                lstm_hidden=lstm_hidden,
                lstm_layers=lstm_layers,
                attention_heads=attention_heads,
                dropout=0.3
            ).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Initialize preprocessor
            self.preprocessor = BusinessAwarePreprocessor()
            
            # Store model info
            if 'val_metrics' in checkpoint:
                val_metrics = checkpoint['val_metrics']
                self.model_info = {
                    'accuracy': val_metrics.get('accuracy', 0.0),
                    'ham_accuracy': val_metrics.get('ham_accuracy', 0.0),
                    'spam_accuracy': val_metrics.get('spam_accuracy', 0.0),
                    'balanced_accuracy': val_metrics.get('balanced_accuracy', 0.0),
                    'f1_score': val_metrics.get('f1', 0.0),
                    'epoch': checkpoint.get('epoch', 0),
                    'vocab_size': vocab_size,
                    'device': str(self.device)
                }
            
            print(f"✅ BalancedSpamNet loaded successfully!")
            print(f"   Device: {self.device}")
            print(f"   Vocabulary size: {vocab_size:,}")
            if self.model_info:
                print(f"   Model accuracy: {self.model_info['accuracy']*100:.2f}%")
                print(f"   HAM accuracy: {self.model_info['ham_accuracy']*100:.2f}%")
                print(f"   SPAM accuracy: {self.model_info['spam_accuracy']*100:.2f}%")
            
        except Exception as e:
            print(f"❌ Error loading BalancedSpamNet: {e}")
            raise e
    
    def preprocess_text(self, text: str, max_len: int = 512) -> Dict[str, torch.Tensor]:
        """Preprocess text for BalancedSpamNet prediction"""
        try:
            # Clean text
            processed_text = self.preprocessor.preprocess(text)
            
            # Convert to token IDs
            words = processed_text.split()[:max_len]
            unk_id = self.vocab.get('<UNK>', 1)
            pad_id = self.vocab.get('<PAD>', 0)
            
            # Tokenize
            input_ids = [self.vocab.get(word, unk_id) for word in words]
            
            # Pad to max_len
            while len(input_ids) < max_len:
                input_ids.append(pad_id)
            input_ids = input_ids[:max_len]
            
            # Create business context mask
            business_mask = self.preprocessor.get_business_mask(processed_text, self.vocab, max_len)
            
            # Convert to tensors
            input_ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
            business_mask_tensor = torch.tensor([business_mask], dtype=torch.long, device=self.device)
            
            return {
                'input_ids': input_ids_tensor,
                'business_mask': business_mask_tensor,
                'processed_text': processed_text
            }
            
        except Exception as e:
            print(f"❌ Preprocessing error: {e}")
            raise e
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Make prediction on input text
        
        Returns:
            Dict with prediction, probability, confidence scores, and detailed analysis
        """
        if not self.model or not self.vocab:
            return {
                'error': 'Model not loaded properly',
                'prediction': 'HAM',
                'probability': 0.5
            }
        
        try:
            start_time = time.time()
            
            # Preprocess text
            preprocessed = self.preprocess_text(text)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(
                    preprocessed['input_ids'], 
                    preprocessed['business_mask']
                )
                
                # Get probabilities
                main_prob = torch.sigmoid(outputs['logits']).item()
                spam_prob = torch.sigmoid(outputs['spam_logits']).item()
                ham_prob = torch.sigmoid(outputs['ham_logits']).item()
                
                # Final prediction (main logit)
                is_spam = main_prob > 0.5
                prediction = "SPAM" if is_spam else "HAM"
                
                # Confidence calculation
                confidence = max(main_prob, 1 - main_prob)
                
            prediction_time = time.time() - start_time
            
            # Prepare detailed result
            result = {
                'prediction': prediction,
                'probability': main_prob,
                'confidence': confidence,
                'spam_probability': spam_prob,
                'ham_probability': ham_prob,
                'prediction_time': prediction_time,
                'processed_text': preprocessed['processed_text'],
                'model_type': 'BalancedSpamNet',
                'device': str(self.device)
            }
            
            # Add business context analysis
            business_terms = self.get_business_terms(text)
            if business_terms:
                result['business_terms_detected'] = business_terms
                result['business_context'] = True
            else:
                result['business_context'] = False
            
            return result
            
        except Exception as e:
            return {
                'error': f'Prediction error: {e}',
                'prediction': 'HAM',
                'probability': 0.5,
                'prediction_time': 0.0
            }
    
    def get_business_terms(self, text: str) -> List[str]:
        """Extract business terms found in the text"""
        try:
            text_lower = text.lower()
            found_terms = []
            
            for term in self.preprocessor.business_terms:
                if term in text_lower:
                    found_terms.append(term)
            
            return found_terms
            
        except Exception:
            return []
    
    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """Predict on a batch of texts efficiently"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                result = self.predict(text)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed model information"""
        info = {
            'model_name': 'BalancedSpamNet',
            'model_type': 'Dual-Head Deep Learning',
            'architecture': 'CNN + BiLSTM + Dual Attention',
            'device': str(self.device),
            'vocab_size': len(self.vocab) if self.vocab else 0,
            'specializations': [
                'Balanced HAM/SPAM detection',
                'Business context awareness',
                'Dual-head architecture',
                'Advanced attention mechanisms'
            ]
        }
        
        if self.model_info:
            info.update(self.model_info)
        
        return info
    
    def explain_prediction(self, text: str) -> Dict[str, Any]:
        """Provide explanation for prediction"""
        result = self.predict(text)
        
        explanation = {
            'prediction': result['prediction'],
            'confidence': result.get('confidence', 0.5),
            'reasoning': []
        }
        
        # Business context reasoning
        if result.get('business_context', False):
            business_terms = result.get('business_terms_detected', [])
            explanation['reasoning'].append(
                f"Business terms detected: {', '.join(business_terms[:5])}"
            )
            explanation['reasoning'].append(
                "Enhanced business context analysis applied"
            )
        
        # Dual head analysis
        spam_prob = result.get('spam_probability', 0.5)
        ham_prob = result.get('ham_probability', 0.5)
        
        explanation['dual_head_analysis'] = {
            'spam_head_confidence': spam_prob,
            'ham_head_confidence': ham_prob,
            'heads_agreement': abs(spam_prob - (1 - ham_prob)) < 0.1
        }
        
        # Confidence reasoning
        confidence = result.get('confidence', 0.5)
        if confidence > 0.9:
            explanation['reasoning'].append("Very high confidence prediction")
        elif confidence > 0.7:
            explanation['reasoning'].append("High confidence prediction")  
        elif confidence > 0.6:
            explanation['reasoning'].append("Moderate confidence prediction")
        else:
            explanation['reasoning'].append("Low confidence - uncertain classification")
        
        return explanation


def test_predictor():
    """Test the BalancedSpamNet predictor"""
    model_path = "models/balanced_spam_net_best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Make sure you've trained BalancedSpamNet first")
        return False
    
    try:
        # Create predictor
        predictor = BalancedSpamNetPredictor(model_path)
        
        # Test predictions
        test_cases = [
            "FREE! Win a $1000 cash prize! Text WIN to 85233 now!",
            "Hi, can you pick up milk on your way home?",
            "Your account has been compromised. Click here to verify your banking details.",
            "Meeting rescheduled to 3pm tomorrow. Please confirm.",
            "Congratulations! You're eligible for a free iPhone! Click here!"
        ]
        
        print("\n🧪 Testing BalancedSpamNet Predictions:")
        print("=" * 60)
        
        for i, text in enumerate(test_cases, 1):
            result = predictor.predict(text)
            print(f"\n{i}. Text: '{text[:50]}...'")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   SPAM Head: {result['spam_probability']:.3f}")
            print(f"   HAM Head: {result['ham_probability']:.3f}")
            if result.get('business_context'):
                terms = result.get('business_terms_detected', [])
                print(f"   Business Terms: {', '.join(terms[:3])}")
        
        print("\n✅ BalancedSpamNet predictor test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    test_predictor()