#!/usr/bin/env python3
"""
Email Spam Detection - Model Selection Interface
Main entry point for choosing and using different spam detection models
"""

import os
import sys
import joblib
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from simple_svm_classifier import SimpleSVMClassifier, SimplePreprocessor

class ModelManager:
    """Manages multiple spam detection models"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.available_models = self.scan_models()
        self.current_model = None
        self.current_model_info = None
    
    def scan_models(self):
        """Scan for available model files"""
        models = {}
        
        # Define model information
        model_info = {
            "svm_full.pkl": {
                "name": "Support Vector Machine (SVM)", 
                "accuracy": "98.29%",
                "type": "SVM"
            },
            "catboost_tuned.pkl": {
                "name": "CatBoost",
                "accuracy": "97.60%",
                "type": "CatBoost"
            }
        }
        
        # Check which models exist
        for filename, info in model_info.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                models[filename] = {**info, "path": str(model_path), "exists": True}
        
        return models
    
    def display_models(self):
        """Display available models in a nice format"""
        print("🎯 Email Spam Detection - Model Selection")
        print("=" * 50)
        print()
        
        if not self.available_models:
            print("❌ No models found in models/ directory!")
            return False
        
        print("Available Models:")
        print("-" * 30)
        
        for i, (filename, info) in enumerate(self.available_models.items(), 1):
            print(f"{i}. {info['name']} - {info['accuracy']}")
        print()
        
        return True
    
    def select_model(self):
        """Interactive model selection"""
        while True:
            if not self.display_models():
                return False
            
            try:
                choice = input("Select model number (or 'q' to quit): ").strip().lower()
                
                if choice == 'q':
                    return False
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.available_models):
                    model_items = list(self.available_models.items())
                    filename, info = model_items[choice_num - 1]
                    
                    print(f"\n🔄 Loading {info['name']}...")
                    
                    if self.load_model(filename, info):
                        print(f"✅ Model loaded successfully!")
                        print(f"📊 Ready to predict with {info['name']} ({info['accuracy']} accuracy)")
                        return True
                    else:
                        print("❌ Failed to load model. Try another one.")
                        print()
                        continue
                else:
                    print("❌ Invalid selection. Please choose a valid number.")
                    print()
                    
            except ValueError:
                print("❌ Please enter a valid number or 'q' to quit.")
                print()
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return False
    
    def load_model(self, filename, info):
        """Load the selected model"""
        try:
            model_path = info["path"]
            
            if info["type"] == "SVM":
                # Load SVM model
                classifier = SimpleSVMClassifier()
                classifier.load_model(model_path)
                self.current_model = classifier
                self.predict_method = "svm"
                
            elif info["type"] == "CatBoost":
                # Load CatBoost model
                model_data = joblib.load(model_path)
                self.current_model = model_data
                self.predict_method = "catboost"
            
            self.current_model_info = info
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_text(self, text, threshold=0.0):
        """Make prediction with current model"""
        if not self.current_model:
            return None, None, "No model loaded"
        
        start_time = time.time()
        
        try:
            if self.predict_method == "svm":
                # SVM prediction
                prediction, confidence = self.current_model.predict(text)
                label = "SPAM" if prediction == 1 else "HAM"
                
                # Apply threshold
                if abs(confidence) <= threshold:
                    label = "HAM"  # Below threshold = HAM
                
            elif self.predict_method == "catboost":
                # CatBoost prediction
                preprocessor = self.current_model["preprocessor"]
                vectorizer = self.current_model["vectorizer"] 
                classifier = self.current_model["classifier"]
                
                # Preprocess and vectorize
                processed_text = preprocessor.preprocess(text)
                text_vec = vectorizer.transform([processed_text])
                
                # Predict
                prediction = classifier.predict(text_vec)[0]
                try:
                    # Try to get prediction probability
                    proba = classifier.predict_proba(text_vec)[0]
                    confidence = max(proba)
                    if prediction == 0:
                        confidence = proba[0]  # Ham confidence
                    else:
                        confidence = proba[1]  # Spam confidence
                except:
                    confidence = 0.8 if prediction == 1 else 0.8
                
                label = "SPAM" if prediction == 1 else "HAM"
                
                # Apply threshold
                if confidence <= threshold:
                    label = "HAM"
            
            prediction_time = time.time() - start_time
            return label, confidence, prediction_time
            
        except Exception as e:
            return None, None, f"Prediction error: {e}"

def main():
    """Main interactive loop"""
    print("🚀 Email Spam Detection System")
    print("=" * 50)
    
    manager = ModelManager()
    
    # Model selection
    if not manager.select_model():
        print("👋 Goodbye!")
        return
    
    print("\n" + "=" * 70)
    print("📧 Email Prediction Mode")
    print("=" * 70)
    print("Commands:")
    print("  - Enter email text to classify")
    print("  - 'models' to switch models")
    print("  - 'threshold X' to set spam threshold (e.g., 'threshold 0.5')")
    print("  - 'info' to show current model info")
    print("  - 'quit' to exit")
    print("-" * 70)
    
    current_threshold = 0.0
    
    while True:
        try:
            print()
            user_input = input("📧 Enter email text (or command): ").strip()
            
            if not user_input:
                continue
                
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            elif user_input.lower() == 'models':
                print("\n" + "=" * 70)
                if manager.select_model():
                    print("\n" + "=" * 70)
                    print("📧 Email Prediction Mode (Model switched)")
                    print("-" * 70)
                else:
                    break
                continue
                
            elif user_input.lower().startswith('threshold '):
                try:
                    new_threshold = float(user_input.split()[1])
                    current_threshold = new_threshold
                    print(f"✅ Threshold set to {current_threshold}")
                except:
                    print("❌ Invalid threshold. Use: threshold 0.5")
                continue
                
            elif user_input.lower() == 'info':
                if manager.current_model_info:
                    info = manager.current_model_info
                    print(f"\n📊 Current Model: {info['name']}")
                    print(f"   Accuracy: {info['accuracy']}")
                    print(f"   Dataset: {info['dataset']}")
                    print(f"   Speed: {info['speed']}")
                    print(f"   Threshold: {current_threshold}")
                continue
            
            # Make prediction
            print(f"\n🔄 Analyzing with {manager.current_model_info['name']}...")
            
            label, confidence, pred_time = manager.predict_text(user_input, current_threshold)
            
            if label:
                # Format output
                if label == "SPAM":
                    result_icon = "🚨"
                    result_color = "SPAM"
                else:
                    result_icon = "✅" 
                    result_color = "HAM"
                
                print(f"\n{result_icon} Result: {result_color}")
                if isinstance(confidence, (int, float)):
                    print(f"   Confidence: {confidence:.3f}")
                if isinstance(pred_time, (int, float)):
                    print(f"   Prediction time: {pred_time*1000:.1f}ms")
                
                # Threshold info
                if current_threshold > 0:
                    print(f"   Threshold applied: {current_threshold}")
                    if label == "HAM" and confidence and confidence <= current_threshold:
                        print(f"   💡 Classified as HAM due to threshold")
                
            else:
                print(f"❌ {pred_time}")  # Error message
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()