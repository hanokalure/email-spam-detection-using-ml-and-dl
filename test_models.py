#!/usr/bin/env python3
"""
Test script to validate models work with various messages and thresholds
"""
import sys
import os
sys.path.append('src')
from simple_svm_classifier import SimpleSVMClassifier

def test_model(model_path, messages, threshold=0.0):
    """Test a model with given messages and threshold"""
    print(f"\n🔍 Testing model: {os.path.basename(model_path)}")
    print(f"📏 Threshold: {threshold}")
    print("=" * 70)
    
    try:
        classifier = SimpleSVMClassifier()
        classifier.load_model(model_path)
        
        for i, message in enumerate(messages, 1):
            prediction, confidence = classifier.predict(message)
            
            # Apply threshold
            if confidence > threshold:
                final_pred = 1  # Spam
                final_label = "SPAM"
            else:
                final_pred = 0  # Ham
                final_label = "HAM"
            
            # Color coding
            if final_pred == 1:
                icon = "🚨" if abs(confidence) > 1.0 else "⚠️"
            else:
                icon = "✅" if abs(confidence) > 1.0 else "🤔"
            
            print(f"{i:2}. {icon} {final_label:<4} (conf: {confidence:6.3f}, thresh: {threshold}) | {message}")
        
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def main():
    # Test messages (email-focused)
    test_messages = [
        "Dear colleague, please find the attached quarterly report for review.",
        "Meeting scheduled for tomorrow at 10 AM in conference room B.",  
        "Thank you for your email. I will respond by end of business today.",
        "Please review the attached contract and provide your feedback.",
        "Important system maintenance scheduled for this weekend.",
        "Congratulations! You have won a $1000 gift card. Click here to claim!",
        "URGENT! Your account will be suspended unless you verify immediately!",
        "Free iPhone! Limited time offer - act now to claim your prize!",
        "WINNER!! You have been selected for our exclusive lottery prize!",
        "Click here to claim your inheritance of $2,000,000 now!",
        "Exclusive mortgage rates - call 555-123-4567 for instant approval",
        "Your loan application approved for $50000 - call now to proceed"
    ]
    
    # Available models
    models = [
        "models/svm_best.pkl",  # Primary email model (99.66%)
        "models/svm.pkl"        # Fallback mixed model (98.90%)
    ]
    
    # Test with different thresholds
    thresholds = [0.0, 0.5, 1.0]
    
    for model in models:
        if os.path.exists(model):
            print(f"\n\n{'='*80}")
            print(f"TESTING MODEL: {os.path.basename(model)}")
            print('='*80)
            
            for threshold in thresholds:
                test_model(model, test_messages, threshold)
                
        else:
            print(f"❌ Model not found: {model}")

if __name__ == "__main__":
    main()