#!/usr/bin/env python3
"""
Test just the three problematic messages
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predict_enhanced_transformer import EnhancedTransformerPredictor

def test_problematic():
    """Test the three messages that should be HAM but are classified as SPAM"""
    
    model_path = "models/enhanced_transformer_best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    predictor = EnhancedTransformerPredictor(model_path)
    
    # The three problematic messages
    messages = [
        "Info: ₹2500 cash withdrawn using ATM card ending 7890. Transaction Date: 15-Jan-24, Time: 14:45. Current Balance: ₹12500.",
        "Payment of $75 has been processed successfully. Debited from account ending 5678. Transaction ID: TXN123456789. Balance: $425.",
        "Dear Customer, your mini statement: 1. 14-Jan: ₹2000 CR, 2. 15-Jan: ₹500 DR. Current Balance: ₹47500. Account statement available."
    ]
    
    print("🔍 Testing the 3 problematic messages:")
    print("=" * 80)
    
    for i, text in enumerate(messages, 1):
        result = predictor.predict(text, return_details=True)
        
        print(f"\n{i}. Message: {text}")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Raw Probability: {result['probability']:.6f}")
        print(f"   Confidence: {result['confidence']:.6f}")
        
        # Let's also manually test the _classify_with_context method
        manual_prediction = predictor._classify_with_context(text, result['probability'])
        print(f"   Manual classification: {manual_prediction}")

if __name__ == "__main__":
    test_problematic()