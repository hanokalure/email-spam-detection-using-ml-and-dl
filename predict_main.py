#!/usr/bin/env python3
"""
Email Spam Detection - Model Selection Interface
Main entry point for choosing and using different spam detection models
"""

import os
import sys
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all available predictors
from predict_enhanced_transformer import EnhancedTransformerPredictor
from predict_transformer import TransformerPredictor
from predict_svm import SVMPredictor
from predict_catboost import CatBoostPredictor

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
        
        # All available ML models
        model_info = {
            # Enhanced Transformer - Latest model with 99.48% recall
            "enhanced_transformer_99recall.pt": {
                "name": "🥇 Enhanced Transformer – 99.48% Spam Recall",
                "accuracy": "96.65% (99.48% spam recall!)",
                "type": "Enhanced_Transformer",
                "dataset": "Comprehensive dataset (15.5K)",
                "speed": "~10-20ms (GPU)",
                "details": "Latest trained model with focal loss and max recall optimization"
            },
            # Original Transformer
            "transformer_best.pt": {
                "name": "🥈 Transformer (Original) – Balanced Performance",
                "accuracy": "~95-96% (balanced precision/recall)",
                "type": "Transformer",
                "dataset": "Mixed dataset",
                "speed": "~10-20ms (GPU)",
                "details": "Original transformer model with balanced performance"
            },
            # SVM Model
            "svm_full.pkl": {
                "name": "🥉 SVM (Support Vector Machine) – Fast & Reliable",
                "accuracy": "~95-98% (traditional ML)",
                "type": "SVM",
                "dataset": "Email corpus",
                "speed": "~1-5ms (CPU)",
                "details": "Fast traditional ML model with TF-IDF features"
            },
            # CatBoost Model
            "catboost_tuned.pkl": {
                "name": "🚀 CatBoost – Gradient Boosting Power",
                "accuracy": "~96-97% (ensemble method)",
                "type": "CatBoost",
                "dataset": "Comprehensive dataset",
                "speed": "~5-15ms (CPU)",
                "details": "Gradient boosting with automatic feature engineering"
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
            model_type = info["type"]
            
            if model_type == "Enhanced_Transformer":
                # Load Enhanced Transformer predictor
                predictor = EnhancedTransformerPredictor(model_path)
                self.current_model = predictor
                self.predict_method = "enhanced_transformer"
            
            elif model_type == "Transformer":
                # Load Original Transformer predictor
                predictor = TransformerPredictor(model_path)
                self.current_model = predictor
                self.predict_method = "transformer"
            
            elif model_type == "SVM":
                # Load SVM predictor
                predictor = SVMPredictor(model_path)
                self.current_model = predictor
                self.predict_method = "svm"
            
            elif model_type == "CatBoost":
                # Load CatBoost predictor
                predictor = CatBoostPredictor(model_path)
                self.current_model = predictor
                self.predict_method = "catboost"
                
            else:
                print(f"❌ Unknown model type: {model_type}")
                return False
            
            self.current_model_info = info
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict_text(self, text):
        """Make prediction with current model"""
        if not self.current_model:
            return None, None, "No model loaded"
        
        start_time = time.time()
        
        try:
            # Enhanced Transformer prediction (context-aware, no manual threshold)
            result = self.current_model.predict(text)
            label = result.get("prediction", "HAM")
            confidence = result.get("probability", 0.5)
            
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
    print("  - 'info' to show current model info")
    print("  - 'quit' to exit")
    print("-" * 70)
    
    # No manual threshold; Enhanced Transformer applies context-aware logic by default
    print("💡 Transformer (Enhanced) uses built-in context-aware classification; no threshold needed.")
    
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
                
                
            elif user_input.lower() == 'info':
                if manager.current_model_info:
                    info = manager.current_model_info
                    print(f"\n📊 Current Model: {info['name']}")
                    print(f"   Accuracy: {info['accuracy']}")
                    print(f"   Dataset: {info['dataset']}")
                    print(f"   Speed: {info['speed']}")
                    if 'details' in info:
                        print(f"   Details: {info['details']}")
                    print(f"   Uses context-aware classification (no manual threshold)")
                continue
            
            # Make prediction
            print(f"\n🔄 Analyzing with {manager.current_model_info['name']}...")
            
            label, confidence, pred_time = manager.predict_text(user_input)
            
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
                
                
            else:
                print(f"❌ {pred_time}")  # Error message
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()