#!/usr/bin/env python3
"""
Model Management System for Email Spam Detection
Provides model discovery, selection, and metadata management
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import glob

class ModelRegistry:
    """Registry for managing multiple spam detection models"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / "model_registry.json"
        self.load_registry()
    
    def load_registry(self):
        """Load model registry from file or create new one"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": {},
                "aliases": {
                    "svm": "svm_best.pkl",
                    "best": "svm_best.pkl", 
                    "default": "svm_best.pkl",
                    "email": "svm_best.pkl"
                }
            }
            self.scan_and_register_models()
    
    def save_registry(self):
        """Save registry to file"""
        self.models_dir.mkdir(exist_ok=True)
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def scan_and_register_models(self):
        """Scan models directory and auto-register found models"""
        
        # Define model metadata based on filename patterns
        model_patterns = {
            "svm_best.pkl": {
                "name": "SVM Email (Original)",
                "type": "LinearSVM",
                "dataset": "SpamAssassin Email Corpus",
                "size": "~5.8K emails",
                "accuracy": "99.66%",
                "description": "Original high-accuracy email spam model",
                "tags": ["email", "production", "original"],
                "speed": "fast"
            },
            "svm.pkl": {
                "name": "SVM Mixed (Legacy)",
                "type": "LinearSVM", 
                "dataset": "Mixed Email+SMS",
                "size": "~11K messages",
                "accuracy": "98.90%",
                "description": "Legacy mixed email and SMS model",
                "tags": ["mixed", "legacy"],
                "speed": "fast"
            },
            "xgboost_mega14k.pkl": {
                "name": "XGBoost Mega Dataset",
                "type": "XGBoost",
                "dataset": "Mega Combined (Enron + SpamAssassin)",
                "size": "14.3K emails",
                "accuracy": "98.22%", 
                "description": "XGBoost trained on mega dataset for high accuracy",
                "tags": ["mega", "xgboost", "high-accuracy"],
                "speed": "medium"
            },
            "lightgbm_mega14k.pkl": {
                "name": "LightGBM Mega Dataset", 
                "type": "LightGBM",
                "dataset": "Mega Combined (Enron + SpamAssassin)",
                "size": "14.3K emails",
                "accuracy": "98.22%",
                "description": "LightGBM trained on mega dataset for speed + accuracy",
                "tags": ["mega", "lightgbm", "fast"],
                "speed": "fast"
            },
            "logreg_mega14k.pkl": {
                "name": "Logistic Regression Mega",
                "type": "LogisticRegression",
                "dataset": "Mega Combined (Enron + SpamAssassin)", 
                "size": "14.3K emails",
                "accuracy": "97.94%",
                "description": "Logistic Regression with calibrated probabilities",
                "tags": ["mega", "logistic", "interpretable"],
                "speed": "very-fast"
            },
            "calibrated_svm_mega14k.pkl": {
                "name": "Calibrated SVM Mega - BEST",
                "type": "CalibratedSVM",
                "dataset": "Mega Combined (Enron + SpamAssassin)",
                "size": "14.3K emails", 
                "accuracy": "98.75%",
                "description": "🏆 HIGHEST ACCURACY - SVM with calibrated probability outputs",
                "tags": ["mega", "svm", "calibrated", "best"],
                "speed": "medium"
            },
            "complement_nb_mega14k.pkl": {
                "name": "Complement Naive Bayes Mega",
                "type": "ComplementNB",
                "dataset": "Mega Combined (Enron + SpamAssassin)",
                "size": "14.3K emails",
                "accuracy": "79.69%", 
                "description": "Ultra-fast but lower accuracy - good for speed tests",
                "tags": ["mega", "naive-bayes", "ultra-fast"],
                "speed": "ultra-fast"
            }
        }
        
        # Scan for actual model files
        for model_file in self.models_dir.glob("*.pkl"):
            filename = model_file.name
            
            if filename in model_patterns:
                # Use predefined metadata
                metadata = model_patterns[filename].copy()
            else:
                # Auto-generate metadata for unknown models
                metadata = {
                    "name": filename.replace('.pkl', '').replace('_', ' ').title(),
                    "type": "Unknown",
                    "dataset": "Unknown",
                    "size": "Unknown",
                    "accuracy": "Unknown",
                    "description": f"Auto-discovered model: {filename}",
                    "tags": ["auto-discovered"],
                    "speed": "unknown"
                }
            
            # Add file metadata
            try:
                stat = model_file.stat()
                metadata.update({
                    "file_size": f"{stat.st_size / 1024:.1f} KB",
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime
                })
            except:
                pass
            
            # Try to extract more info from the model file
            try:
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    if isinstance(model_data, dict):
                        if 'model_type' in model_data:
                            metadata['type'] = model_data['model_type']
            except:
                pass
            
            self.registry["models"][filename] = metadata
        
        # Update aliases for new models
        self.registry["aliases"].update({
            "xgboost": "xgboost_mega14k.pkl",
            "lightgbm": "lightgbm_mega14k.pkl", 
            "lgbm": "lightgbm_mega14k.pkl",
            "logreg": "logreg_mega14k.pkl",
            "logistic": "logreg_mega14k.pkl",
            "svm_cal": "calibrated_svm_mega14k.pkl",
            "calibrated": "calibrated_svm_mega14k.pkl",
            "nb": "complement_nb_mega14k.pkl",
            "naive_bayes": "complement_nb_mega14k.pkl",
            "mega": "calibrated_svm_mega14k.pkl",  # NEW BEST MODEL (98.75%)
            "fastest": "complement_nb_mega14k.pkl",
            "accurate": "calibrated_svm_mega14k.pkl",  # MOST ACCURATE
            "new_best": "calibrated_svm_mega14k.pkl",
            "production": "calibrated_svm_mega14k.pkl",
            "recommended": "calibrated_svm_mega14k.pkl"
        })
        
        self.save_registry()
    
    def list_models(self, show_details=False) -> List[Dict]:
        """List all available models"""
        models = []
        
        for filename, metadata in self.registry["models"].items():
            model_path = self.models_dir / filename
            
            model_info = {
                "filename": filename,
                "exists": model_path.exists(),
                **metadata
            }
            
            if show_details:
                # Add aliases that point to this model
                aliases = [alias for alias, target in self.registry["aliases"].items() 
                          if target == filename]
                model_info["aliases"] = aliases
            
            models.append(model_info)
        
        return models
    
    def resolve_model_path(self, model_identifier: str) -> Optional[str]:
        """Resolve model identifier to actual file path"""
        
        # Direct filename
        if model_identifier.endswith('.pkl'):
            model_path = self.models_dir / model_identifier
            if model_path.exists():
                return str(model_path)
        
        # Alias lookup
        if model_identifier in self.registry["aliases"]:
            filename = self.registry["aliases"][model_identifier]
            model_path = self.models_dir / filename
            if model_path.exists():
                return str(model_path)
        
        # Direct path
        if os.path.exists(model_identifier):
            return model_identifier
        
        return None
    
    def get_model_info(self, model_identifier: str) -> Optional[Dict]:
        """Get detailed information about a model"""
        
        resolved_path = self.resolve_model_path(model_identifier)
        if not resolved_path:
            return None
        
        filename = os.path.basename(resolved_path)
        
        if filename in self.registry["models"]:
            return self.registry["models"][filename]
        
        return None
    
    def register_model(self, filename: str, metadata: Dict):
        """Register a new model with metadata"""
        self.registry["models"][filename] = metadata
        self.save_registry()
    
    def add_alias(self, alias: str, target_filename: str):
        """Add an alias for a model"""
        self.registry["aliases"][alias] = target_filename
        self.save_registry()
    
    def print_model_list(self):
        """Print formatted list of available models"""
        models = self.list_models(show_details=True)
        
        print("📊 Available Spam Detection Models:")
        print("=" * 80)
        
        for model in models:
            status = "✅ Available" if model["exists"] else "❌ Missing"
            
            print(f"\n📁 {model['filename']}")
            print(f"   Name: {model['name']}")
            print(f"   Type: {model['type']}")
            print(f"   Dataset: {model['dataset']} ({model['size']})")
            print(f"   Accuracy: {model['accuracy']}")
            print(f"   Speed: {model['speed']}")
            print(f"   Status: {status}")
            
            if model.get('aliases'):
                print(f"   Aliases: {', '.join(model['aliases'])}")
            
            if model.get('file_size'):
                print(f"   Size: {model['file_size']}")
        
        print(f"\n💡 Usage Examples:")
        print(f"   python predict.py \"email text\" --model best")
        print(f"   python predict.py \"email text\" --model xgboost") 
        print(f"   python predict.py \"email text\" --model fastest")
        print(f"   python predict.py \"email text\" --model models/svm_best.pkl")

def main():
    """CLI for model management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Spam Detection Model Manager")
    parser.add_argument("--list", "-l", action="store_true", help="List all models")
    parser.add_argument("--scan", "-s", action="store_true", help="Scan for new models")
    parser.add_argument("--info", "-i", help="Get info about specific model")
    
    args = parser.parse_args()
    
    registry = ModelRegistry()
    
    if args.scan:
        print("🔍 Scanning for models...")
        registry.scan_and_register_models()
        print("✅ Registry updated")
    
    if args.info:
        info = registry.get_model_info(args.info)
        if info:
            print(f"📊 Model Info: {args.info}")
            print(json.dumps(info, indent=2))
        else:
            print(f"❌ Model not found: {args.info}")
    
    if args.list or not any([args.scan, args.info]):
        registry.print_model_list()

if __name__ == "__main__":
    main()