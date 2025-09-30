"""
SMS Spam Detection - Interactive Testing Script
Load the trained SVM model and test individual messages from command line.
"""

import sys
import os
from simple_svm_classifier import SimpleSVMClassifier, SimplePreprocessor

# Fix pickle compatibility issue
sys.modules['__main__'].SimplePreprocessor = SimplePreprocessor
sys.modules['__main__'].SimpleSVMClassifier = SimpleSVMClassifier
import simple_svm_classifier as svm_mod
import sys
sys.modules['__main__'].SimplePreprocessor = svm_mod.SimplePreprocessor

def test_message_interactive():
    """Interactive testing of SMS messages"""
    
    # Check if model exists
    model_path = "models/svm_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Error: Model not found!")
        print("Please run 'python src/simple_svm_classifier.py' first to train the model.")
        return
    
    # Load the trained model
    print("🔄 Loading trained SVM model...")
    classifier = SimpleSVMClassifier()
    classifier.load_model(model_path)
    
    print("\n" + "="*60)
    print("SMS SPAM DETECTION - INTERACTIVE TESTING")
    print("="*60)
    print(f"Model Accuracy: {classifier.accuracy*100:.2f}%")
    print("Type your SMS message to test (or 'quit' to exit)")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            message = input("\n📱 Enter SMS message: ").strip()
            
            if message.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not message:
                print("⚠️  Please enter a valid message.")
                continue
            
            # Make prediction
            prediction, confidence = classifier.predict(message)
            
            # Format results
            label = "🚫 SPAM" if prediction == 1 else "✅ HAM (Not Spam)"
            confidence_score = abs(confidence)
            
            # Color coding for confidence levels
            if confidence_score > 1.0:
                confidence_level = "HIGH"
            elif confidence_score > 0.5:
                confidence_level = "MEDIUM"
            else:
                confidence_level = "LOW"
            
            print("\n" + "-" * 40)
            print(f"📊 RESULT: {label}")
            print(f"🎯 Confidence: {confidence_score:.3f} ({confidence_level})")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

def test_single_message(message):
    """Test a single message from command line argument"""
    
    model_path = "models/svm_model.pkl"
    if not os.path.exists(model_path):
        print("❌ Error: Model not found!")
        print("Please run 'python src/simple_svm_classifier.py' first to train the model.")
        return
    
    # Load model
    classifier = SimpleSVMClassifier()
    classifier.load_model(model_path)
    
    # Make prediction
    prediction, confidence = classifier.predict(message)
    
    label = "SPAM" if prediction == 1 else "HAM"
    print(f"Message: '{message}'")
    print(f"Prediction: {label} (confidence: {abs(confidence):.3f})")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        # Test single message from command line
        message = " ".join(sys.argv[1:])
        test_single_message(message)
    else:
        # Interactive mode
        test_message_interactive()

if __name__ == "__main__":
    main()