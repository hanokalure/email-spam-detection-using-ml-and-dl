#!/usr/bin/env python3
"""
Email Spam Detection - BalancedSpamNet Main Interface (Windows Compatible)
Clean prediction interface with no Unicode issues
"""

import os
import sys
import time
from pathlib import Path

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'predictors'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import BalancedSpamNet
try:
    from predict_balanced_spam_net import BalancedSpamNetPredictor
    print("[+] BalancedSpamNet imported successfully")
except ImportError as e:
    print(f"[!] BalancedSpamNet import error: {e}")
    sys.exit(1)

# Try to import other predictors (optional)
try:
    from predict_svm import SVMPredictor
    SVM_AVAILABLE = True
except ImportError:
    SVM_AVAILABLE = False

try:
    from predict_catboost import CatBoostPredictor  
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class ModelManager:
    """Model manager for spam detection"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.available_models = self.scan_models()
        self.current_model = None
        self.current_model_info = None
        self.predict_method = None
    
    def scan_models(self):
        """Scan for available model files"""
        models = {}
        
        # Model definitions
        model_definitions = {
            "balanced_spam_net_best.pt": {
                "name": "[1] BalancedSpamNet - Revolutionary Dual-Head Model",
                "accuracy": "98.15% (98.1% HAM + 98.3% SPAM)",
                "type": "BalancedSpamNet",
                "description": "Breakthrough in HAM detection - solves legitimate email classification",
                "specialties": ["Balanced HAM/SPAM detection", "Business context awareness", "Dual-head architecture"]
            },
            "svm_full.pkl": {
                "name": "[2] SVM - Fast Traditional ML",  
                "accuracy": "~95-98%",
                "type": "SVM",
                "description": "Fast and reliable traditional machine learning",
                "specialties": ["Speed", "CPU efficiency", "Proven reliability"]
            } if SVM_AVAILABLE else None,
            "catboost_tuned.pkl": {
                "name": "[3] CatBoost - Gradient Boosting",
                "accuracy": "~96-97%",
                "type": "CatBoost", 
                "description": "Advanced gradient boosting ensemble method",
                "specialties": ["Ensemble learning", "Feature engineering", "Robustness"]
            } if CATBOOST_AVAILABLE else None
        }
        
        # Filter None values and check which models exist
        for filename, info in model_definitions.items():
            if info is not None:
                model_path = self.models_dir / filename
                if model_path.exists():
                    models[filename] = {**info, "path": str(model_path), "exists": True}
        
        return models
    
    def display_models(self):
        """Display available models"""
        print("Email Spam Detection - Model Selection")
        print("=" * 60)
        
        if not self.available_models:
            print("[!] No models found!")
            return False
        
        print("Available Models:")
        print("-" * 40)
        
        for i, (filename, info) in enumerate(self.available_models.items(), 1):
            print(f"{i}. {info['name']}")
            print(f"   Accuracy: {info['accuracy']}")
            print(f"   Description: {info['description']}")
            print(f"   Specialties: {', '.join(info['specialties'])}")
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
                    
                    print(f"\n[Loading] {info['name']}...")
                    
                    if self.load_model(filename, info):
                        print("[+] Model loaded successfully!")
                        return True
                    else:
                        print("[!] Failed to load model.")
                        continue
                else:
                    print("[!] Invalid selection.")
                    
            except ValueError:
                print("[!] Please enter a valid number.")
            except KeyboardInterrupt:
                print("\nGoodbye!")
                return False
    
    def load_model(self, filename, info):
        """Load the selected model"""
        try:
            model_path = info["path"]
            model_type = info["type"]
            
            if model_type == "BalancedSpamNet":
                self.current_model = BalancedSpamNetPredictor(model_path)
                self.predict_method = "balanced_spam_net"
            
            elif model_type == "SVM" and SVM_AVAILABLE:
                self.current_model = SVMPredictor(model_path)
                self.predict_method = "svm"
            
            elif model_type == "CatBoost" and CATBOOST_AVAILABLE:
                self.current_model = CatBoostPredictor(model_path)
                self.predict_method = "catboost"
            
            else:
                print(f"[!] Model type not supported: {model_type}")
                return False
            
            self.current_model_info = info
            return True
            
        except Exception as e:
            print(f"[!] Error loading model: {e}")
            return False
    
    def predict_text(self, text):
        """Make prediction with current model"""
        if not self.current_model:
            return None, None, "No model loaded"
        
        try:
            start_time = time.time()
            
            if self.predict_method == "balanced_spam_net":
                # BalancedSpamNet returns detailed dict
                result = self.current_model.predict(text)
                prediction = result.get('prediction', 'HAM')
                
                prediction_time = time.time() - start_time
                return prediction, result, prediction_time
            
            else:
                # Other models 
                result = self.current_model.predict(text)
                if isinstance(result, dict):
                    prediction = result.get('prediction', 'HAM')
                    confidence = result.get('probability', 0.5)
                else:
                    prediction, confidence = result
                
                prediction_time = time.time() - start_time
                return prediction, confidence, prediction_time
                
        except Exception as e:
            return None, None, f"Prediction error: {e}"


def main():
    """Main interactive loop"""
    print("Email Spam Detection System - BalancedSpamNet Edition")
    print("=" * 60)
    
    manager = ModelManager()
    
    # Model selection
    if not manager.select_model():
        print("Goodbye!")
        return
    
    print("\n" + "=" * 60)
    print("Email Prediction Mode")
    print("=" * 60)
    print("Commands:")
    print("  - Enter email text to classify")
    print("  - 'models' to switch models")
    print("  - 'info' to show current model info")
    print("  - 'quit' to exit")
    print("-" * 60)
    
    while True:
        try:
            print()
            user_input = input("Enter email text (or command): ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            elif user_input.lower() == 'models':
                print("\n" + "=" * 60)
                if manager.select_model():
                    print("\n" + "=" * 60)
                    print("Email Prediction Mode (Model switched)")
                    print("-" * 60)
                else:
                    break
                continue
            
            elif user_input.lower() == 'info':
                if manager.current_model_info:
                    info = manager.current_model_info
                    print(f"\nCurrent Model: {info['name']}")
                    print(f"   Accuracy: {info['accuracy']}")
                    print(f"   Type: {info['type']}")
                    print(f"   Description: {info['description']}")
                    print(f"   Specialties: {', '.join(info['specialties'])}")
                continue
            
            # Make prediction
            print(f"\n[Analyzing] {manager.current_model_info['name']}...")
            
            prediction, result_data, pred_time = manager.predict_text(user_input)
            
            if prediction:
                # Format output
                if prediction == "SPAM":
                    result_icon = "[SPAM]"
                    result_color = "SPAM"
                else:
                    result_icon = "[HAM]"
                    result_color = "HAM"
                
                print(f"\n{result_icon} Result: {result_color}")
                
                # Show timing
                if isinstance(pred_time, (int, float)):
                    print(f"   Prediction time: {pred_time*1000:.1f}ms")
                
                # Show BalancedSpamNet specific details
                if manager.predict_method == "balanced_spam_net" and isinstance(result_data, dict):
                    confidence = result_data.get('confidence', 0.5)
                    spam_prob = result_data.get('spam_probability', 0.5)
                    ham_prob = result_data.get('ham_probability', 0.5)
                    
                    print(f"   Overall Confidence: {confidence:.3f}")
                    print(f"   SPAM Head: {spam_prob:.3f}")
                    print(f"   HAM Head: {ham_prob:.3f}")
                    
                    # Business context
                    if result_data.get('business_context'):
                        business_terms = result_data.get('business_terms_detected', [])
                        if business_terms:
                            print(f"   Business Context: {', '.join(business_terms[:3])}")
                            print(f"   [+] Business-aware analysis applied")
                
                else:
                    # Other models
                    if isinstance(result_data, (int, float)):
                        print(f"   Confidence: {result_data:.3f}")
            
            else:
                print(f"[!] {pred_time}")  # Error message
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"[!] Error: {e}")


if __name__ == "__main__":
    main()