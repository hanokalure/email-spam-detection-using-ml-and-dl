#!/usr/bin/env python3
"""
Enhanced Email Spam Prediction with Model Selection
Supports multiple models with friendly names and aliases
"""

import sys
import os
import argparse
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from simple_svm_classifier import SimpleSVMClassifier, SimplePreprocessor
from train_advanced_models import AdvancedSpamClassifier
from model_manager import ModelRegistry

# Backward-compat: map pickled __main__.SimplePreprocessor to real class
try:
    import __main__ as _m
    if not hasattr(_m, 'SimplePreprocessor'):
        _m.SimplePreprocessor = SimplePreprocessor
except Exception:
    pass

def load_classifier(model_path):
    """Load classifier based on model type"""
    
    # Try advanced classifier first (for new models)
    try:
        # Check if it's an advanced model by looking at the pickle content
        import pickle
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        if isinstance(model_data, dict) and 'model_type' in model_data:
            # Advanced model
            classifier = AdvancedSpamClassifier(model_data['model_type'])
            classifier.load_model(model_path)
            return classifier, 'advanced'
        else:
            # Legacy SVM model
            classifier = SimpleSVMClassifier()
            classifier.load_model(model_path)
            return classifier, 'legacy'
    
    except Exception as e:
        # Fallback to legacy SVM
        try:
            classifier = SimpleSVMClassifier()
            classifier.load_model(model_path)
            return classifier, 'legacy'
        except Exception as e2:
            raise Exception(f"Could not load model: {e}")

def classify_with_confidence_analysis(prediction, confidence, threshold=0.0):
    """Enhanced classification with confidence analysis"""
    
    abs_conf = abs(confidence) if confidence is not None else 0.0
    
    # Apply threshold
    if abs_conf > threshold:
        final_prediction = 1 if prediction == 1 or prediction == 'spam' else 0
    else:
        final_prediction = 0  # Default to HAM if below threshold
    
    # Confidence levels
    if abs_conf > 1.5:
        conf_level = "VERY HIGH"
        conf_icon = "🔥"
    elif abs_conf > 1.0:
        conf_level = "HIGH"
        conf_icon = "💪"
    elif abs_conf > 0.5:
        conf_level = "MEDIUM"
        conf_icon = "👍"
    elif abs_conf > 0.2:
        conf_level = "LOW"
        conf_icon = "🤔"
    else:
        conf_level = "VERY LOW"
        conf_icon = "❓"
    
    # Final classification
    if final_prediction == 1:
        label = "SPAM"
        result_icon = "🚨"
    else:
        label = "HAM"
        result_icon = "✅"
    
    return label, conf_level, conf_icon, result_icon, abs_conf

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Email Spam Prediction with Model Selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Selection Examples:
  --model best              Use best available model (alias)
  --model xgboost          Use XGBoost mega model (alias)
  --model fastest          Use fastest model (alias) 
  --model svm_best.pkl     Use specific model file
  --models                 List all available models

Prediction Examples:
  python predict_enhanced.py "Check this email content"
  python predict_enhanced.py "Email text" --model xgboost --threshold 0.5
  python predict_enhanced.py --interactive --model fastest
        """
    )
    
    parser.add_argument("message", nargs="?", help="Message to classify")
    parser.add_argument("--model", "-m", default="best", help="Model name, alias, or file path (default: best)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--threshold", "-t", type=float, default=0.0, 
                      help="Spam threshold (default: 0.0). Higher = less sensitive to spam")
    parser.add_argument("--models", action="store_true", help="List available models and exit")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed prediction info")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Show prediction speed")
    
    args = parser.parse_args()
    
    # Initialize model registry
    registry = ModelRegistry()
    
    # List models if requested
    if args.models:
        registry.print_model_list()
        return 0
    
    # Resolve model path
    model_path = registry.resolve_model_path(args.model)
    if not model_path:
        print(f"❌ Model not found: {args.model}")
        print("\n💡 Available models:")
        models = registry.list_models()
        for model in models[:5]:  # Show first 5
            if model['exists']:
                print(f"   • {model['filename']} - {model['name']}")
        print(f"\n   Use --models to see all available models")
        return 1
    
    # Get model info
    model_info = registry.get_model_info(args.model)
    if args.verbose and model_info:
        print(f"📊 Using: {model_info['name']} ({model_info['type']})")
        print(f"   Dataset: {model_info['dataset']}")
        print(f"   Accuracy: {model_info['accuracy']}")
        print(f"   Speed: {model_info['speed']}")
        print()
    
    # Load classifier
    try:
        print(f"🔄 Loading model: {os.path.basename(model_path)}...")
        classifier, model_type = load_classifier(model_path)
        print(f"✅ Model loaded ({model_type} type)")
        if args.verbose:
            print(f"   Path: {model_path}")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    def predict_message(text):
        """Predict with enhanced output"""
        if not text.strip():
            return
        
        start_time = time.time() if args.benchmark else None
        
        try:
            if model_type == 'advanced':
                prediction, confidence = classifier.predict(text)
                # Convert string predictions to numeric
                pred_numeric = 1 if prediction == 'spam' else 0
            else:
                pred_numeric, confidence = classifier.predict(text)
                prediction = 'spam' if pred_numeric == 1 else 'ham'
            
            prediction_time = time.time() - start_time if start_time else None
            
            label, conf_level, conf_icon, result_icon, abs_conf = classify_with_confidence_analysis(
                pred_numeric, confidence, args.threshold
            )
            
            # Output format
            print(f"{result_icon} {label}")
            
            if args.verbose:
                print(f"   Original prediction: {prediction}")
                print(f"   Confidence: {conf_level} ({abs_conf:.3f}) {conf_icon}")
                print(f"   Threshold applied: {args.threshold}")
                if prediction_time:
                    print(f"   Prediction time: {prediction_time*1000:.1f}ms")
                print(f"   Message: {text}")
            else:
                print(f"   Confidence: {abs_conf:.3f} | {text}")
            
            # Interpretation tips
            if abs_conf < 0.3:
                print(f"   💡 Low confidence - consider trying different model or threshold")
            elif args.threshold > 0 and abs_conf <= args.threshold:
                print(f"   💡 Below threshold ({args.threshold}) - classified as HAM")
            
            print()
            
        except Exception as e:
            print(f"❌ Error predicting: {e}")
    
    # Single message mode
    if args.message:
        predict_message(args.message)
        return 0
    
    # Interactive mode
    if args.interactive or len(sys.argv) == 1:
        model_name = model_info['name'] if model_info else os.path.basename(model_path)
        print(f"🧠 Interactive Email Spam Detection")
        print(f"📊 Model: {model_name}")
        print(f"⚙️  Threshold: {args.threshold}")
        print("=" * 60)
        print("Type your messages (or 'quit' to exit, 'models' to list models):")
        print()
        
        while True:
            try:
                message = input("📧 Message: ").strip()
                
                if message.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif message.lower() == 'models':
                    print("\n" + "="*60)
                    registry.print_model_list()
                    print("="*60 + "\n")
                    continue
                elif message.lower().startswith('model '):
                    # Switch model
                    new_model = message[6:].strip()
                    new_path = registry.resolve_model_path(new_model)
                    if new_path:
                        try:
                            classifier, model_type = load_classifier(new_path)
                            model_info = registry.get_model_info(new_model)
                            model_name = model_info['name'] if model_info else os.path.basename(new_path)
                            print(f"✅ Switched to: {model_name}")
                        except Exception as e:
                            print(f"❌ Error loading {new_model}: {e}")
                    else:
                        print(f"❌ Model not found: {new_model}")
                    continue
                
                if message:
                    predict_message(message)
                    
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                break
        
        return 0
    
    # Show help if no arguments
    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())