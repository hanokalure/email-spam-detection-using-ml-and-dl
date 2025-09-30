#!/usr/bin/env python3
"""
Smart SMS Spam Prediction with Confidence Thresholds
Handles domain mismatch better by using confidence levels
"""
import sys
import os
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from simple_svm_classifier import SimpleSVMClassifier, SimplePreprocessor

# Backward-compat: map pickled __main__.SimplePreprocessor to real class
try:
    import __main__ as _m
    if not hasattr(_m, 'SimplePreprocessor'):
        _m.SimplePreprocessor = SimplePreprocessor
except Exception:
    pass


def classify_with_confidence(prediction, confidence):
    """Classify with confidence-based interpretation"""
    abs_conf = abs(confidence)
    
    if prediction == 1:  # Spam
        if abs_conf > 1.0:
            return "SPAM", "HIGH", "🚨"
        elif abs_conf > 0.5:
            return "SPAM", "MEDIUM", "⚠️"
        else:
            return "UNCERTAIN (likely spam)", "LOW", "🤔"
    else:  # Ham
        if abs_conf > 1.0:
            return "HAM", "HIGH", "✅"
        elif abs_conf > 0.5:
            return "HAM", "MEDIUM", "✅"
        else:
            return "UNCERTAIN (likely ham)", "LOW", "🤔"


def main():
    parser = argparse.ArgumentParser(description="Smart spam prediction with confidence analysis")
    parser.add_argument("message", nargs="?", help="Message to classify")
    parser.add_argument("--model", "-m", default="models/svm_best.pkl", help="Path to model file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--threshold", "-t", type=float, default=0.0, help="Spam threshold (default: 0.0). Higher = less sensitive to spam")
    
    args = parser.parse_args()

    # Load classifier
    classifier = SimpleSVMClassifier()
    try:
        classifier.load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    def predict_smart(text, threshold):
        if not text.strip():
            return
        
        try:
            prediction, confidence = classifier.predict(text)
            
            # Apply threshold-based classification
            if confidence > threshold:
                final_prediction = 1  # Spam
            else:
                final_prediction = 0  # Ham
            
            label, conf_level, icon = classify_with_confidence(final_prediction, confidence)
            
            print(f"{icon} {label}")
            print(f"   Confidence: {conf_level} ({abs(confidence):.3f})")
            print(f"   Threshold: {threshold}")
            print(f"   Message: {text}")
            
            # Add interpretation
            if "UNCERTAIN" in label:
                print(f"   💡 Tip: Low confidence may indicate domain mismatch or edge case")
            print()
        except Exception as e:
            print(f"Error: {e}")

    # Single message mode
    if args.message:
        predict_smart(args.message, args.threshold)
        return 0

    # Interactive mode
    if args.interactive or len(sys.argv) == 1:
        print("🧠 Smart SMS Spam Classifier with Confidence Analysis")
        print("Type your messages (or 'quit' to exit):")
        print("=" * 60)
        
        while True:
            try:
                message = input("Message: ").strip()
                if message.lower() in ['quit', 'exit', 'q']:
                    break
                if message:
                    predict_smart(message, args.threshold)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())