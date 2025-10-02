#!/usr/bin/env python3
"""
Test BalancedSpamNet integration with the prediction system
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'predictors'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from predict_balanced_spam_net import BalancedSpamNetPredictor
    print("✅ BalancedSpamNet import successful")
    
    # Test model loading and prediction
    model_path = "models/balanced_spam_net_best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        exit(1)
    
    # Create predictor
    print("🔄 Loading BalancedSpamNet...")
    predictor = BalancedSpamNetPredictor(model_path)
    
    # Test predictions
    test_cases = [
        "FREE! Win $1000 cash prize! Click here now!",
        "Hi, can you pick up milk on your way home?",
        "Your banking account has been compromised. Verify now.",
        "Meeting rescheduled to 3pm tomorrow"
    ]
    
    print("\n🧪 Testing BalancedSpamNet Predictions:")
    print("=" * 60)
    
    for i, text in enumerate(test_cases, 1):
        result = predictor.predict(text)
        
        prediction = result.get('prediction', 'HAM')
        confidence = result.get('confidence', 0.5)
        spam_prob = result.get('spam_probability', 0.5)
        ham_prob = result.get('ham_probability', 0.5)
        pred_time = result.get('prediction_time', 0.0)
        
        print(f"\n{i}. Text: '{text[:50]}...'")
        print(f"   Prediction: {prediction}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   SPAM Head: {spam_prob:.3f}")
        print(f"   HAM Head: {ham_prob:.3f}")
        print(f"   Time: {pred_time*1000:.1f}ms")
        
        # Check for business context
        if result.get('business_context'):
            terms = result.get('business_terms_detected', [])
            print(f"   Business Terms: {', '.join(terms[:3])}")
    
    print("\n✅ BalancedSpamNet integration test completed successfully!")
    print("🎯 Ready to add to main prediction interface!")

except Exception as e:
    print(f"❌ Error during test: {e}")
    import traceback
    traceback.print_exc()