#!/usr/bin/env python3
"""
Simple SMS Spam Prediction Script
Usage: python predict.py "Your message here"
"""
import sys
import argparse
from src.simple_svm_classifier import SimpleSVMClassifier


def main():
    parser = argparse.ArgumentParser(description="Predict if a message is spam or ham")
    parser.add_argument("message", nargs="?", help="Message to classify")
    parser.add_argument("--model", "-m", default="models/svm_large_corpus.pkl", help="Path to model file")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()

    # Load the classifier
    classifier = SimpleSVMClassifier()
    try:
        classifier.load_model(args.model)
    except FileNotFoundError:
        print(f"Error: Model file '{args.model}' not found!")
        print("Please train the model first by running:")
        print("python src/simple_svm_classifier.py --data data/large_spamassassin_corpus.csv --model models/svm_large_corpus.pkl")
        return 1
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    def predict_message(text):
        if not text.strip():
            return
        
        try:
            prediction, confidence = classifier.predict(text)
            label = "SPAM" if prediction == 1 else "HAM "
            print(f"{label} | confidence: {abs(confidence):.3f} | {text}")
        except Exception as e:
            print(f"Error predicting: {e}")

    # Single message mode
    if args.message:
        predict_message(args.message)
        return 0

    # Interactive mode
    if args.interactive or len(sys.argv) == 1:
        print("Interactive SMS Spam Classifier")
        print("Type your messages (or 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                message = input("Message: ").strip()
                if message.lower() in ['quit', 'exit', 'q']:
                    break
                if message:
                    predict_message(message)
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break
        return 0

    # If no arguments provided, show help
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())