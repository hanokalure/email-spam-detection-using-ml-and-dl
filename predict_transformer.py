#!/usr/bin/env python3
"""
Prediction script for Custom Transformer Spam Classifier
- Loads trained model from models/transformer_best.pt
- Provides single message prediction and interactive mode
- Shows confidence scores and attention insights

Usage:
  python predict_transformer.py "Your email content here"
  python predict_transformer.py --interactive
"""

import os
import sys
import json
import argparse
from typing import Dict, List

import torch
import torch.nn.functional as F
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from simple_svm_classifier import SimplePreprocessor
from transformer_text_classifier import TransformerConfig, build_model


class TransformerPredictor:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = SimplePreprocessor()
        
        # Load checkpoint
        print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore config and vocab
        self.config = TransformerConfig(**checkpoint["config"])
        self.vocab = checkpoint["vocab"]
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Build and load model
        self.model = build_model(self.config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        
        # Get validation metrics if available
        self.val_metrics = checkpoint.get("val_metrics", {})
        print(f"Model loaded! Device: {self.device}")
        if self.val_metrics:
            print(f"Validation F1: {self.val_metrics.get('f1', 'N/A'):.4f}")
    
    def preprocess_text(self, text: str) -> List[int]:
        """Preprocess text and convert to token IDs"""
        processed = self.preprocessor.preprocess(text)
        tokens = processed.split()
        unk_idx = self.vocab.get("[UNK]", 1)
        return [self.vocab.get(token, unk_idx) for token in tokens]
    
    def predict(self, text: str, return_attention: bool = False) -> Dict:
        """
        Predict spam probability for a single text
        
        Returns:
            Dict with keys: prediction, probability, confidence_level, tokens (if return_attention)
        """
        # Preprocess
        token_ids = self.preprocess_text(text)
        
        if not token_ids:
            return {
                "prediction": "HAM", 
                "probability": 0.0, 
                "confidence_level": "VERY LOW",
                "error": "Empty text after preprocessing"
            }
        
        # Convert to tensor
        max_len = min(len(token_ids), self.config.max_len)
        input_ids = torch.zeros(1, max_len, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros(1, max_len, dtype=torch.long, device=self.device)
        
        input_ids[0, :len(token_ids[:max_len])] = torch.tensor(token_ids[:max_len], dtype=torch.long)
        attention_mask[0, :len(token_ids[:max_len])] = 1
        
        # Predict
        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probability = torch.sigmoid(logits).item()
        
        # Determine prediction and confidence
        prediction = "SPAM" if probability > 0.5 else "HAM"
        
        # Confidence levels based on distance from 0.5
        confidence_score = abs(probability - 0.5) * 2  # 0 to 1
        if confidence_score > 0.8:
            confidence_level = "VERY HIGH"
            conf_icon = "🔥"
        elif confidence_score > 0.6:
            confidence_level = "HIGH" 
            conf_icon = "💪"
        elif confidence_score > 0.4:
            confidence_level = "MEDIUM"
            conf_icon = "👍"
        elif confidence_score > 0.2:
            confidence_level = "LOW"
            conf_icon = "🤔"
        else:
            confidence_level = "VERY LOW"
            conf_icon = "❓"
        
        result = {
            "prediction": prediction,
            "probability": probability,
            "confidence_score": confidence_score,
            "confidence_level": confidence_level,
            "confidence_icon": conf_icon,
            "num_tokens": len(token_ids[:max_len])
        }
        
        if return_attention:
            # Get tokens for interpretation
            tokens = [self.reverse_vocab.get(tid, "[UNK]") for tid in token_ids[:max_len]]
            result["tokens"] = tokens
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict multiple texts efficiently"""
        return [self.predict(text) for text in texts]


def format_prediction(result: Dict, text: str) -> str:
    """Format prediction result for display"""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    pred_icon = "🚨" if result["prediction"] == "SPAM" else "✅"
    output = f"{pred_icon} {result['prediction']}\n"
    output += f"   Probability: {result['probability']:.4f}\n"
    output += f"   Confidence: {result['confidence_level']} ({result['confidence_score']:.3f}) {result['confidence_icon']}\n"
    output += f"   Tokens: {result['num_tokens']}\n"
    output += f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}"
    return output


def main():
    parser = argparse.ArgumentParser(description="Custom Transformer Spam Prediction")
    parser.add_argument("message", nargs="?", help="Message to classify")
    parser.add_argument("--model", "-m", default="models/transformer_best.pt", help="Path to trained model")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--batch", "-b", help="File with messages to classify (one per line)")
    parser.add_argument("--attention", "-a", action="store_true", help="Show token attention (experimental)")
    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("Please train the model first with: python src/train_transformer.py --deeper")
        return 1

    # Load predictor
    try:
        predictor = TransformerPredictor(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1

    # Single message mode
    if args.message:
        result = predictor.predict(args.message, return_attention=args.attention)
        print(format_prediction(result, args.message))
        return 0

    # Batch mode
    if args.batch:
        if not os.path.exists(args.batch):
            print(f"❌ Batch file not found: {args.batch}")
            return 1
        
        with open(args.batch, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Processing {len(texts)} messages...")
        results = predictor.predict_batch(texts)
        
        for i, (text, result) in enumerate(zip(texts, results), 1):
            print(f"\n{i}. {format_prediction(result, text)}")
        return 0

    # Interactive mode
    if args.interactive or len(sys.argv) == 1:
        print("🤖 Custom Transformer Spam Classifier - Interactive Mode")
        print("=" * 60)
        print("Commands:")
        print("  - Enter email/SMS text to classify")
        print("  - 'quit' or 'exit' to stop")
        print("  - 'stats' to show model info")
        print("-" * 60)
        
        while True:
            try:
                message = input("\n📧 Enter message: ").strip()
                
                if message.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                elif message.lower() == 'stats':
                    print(f"\n📊 Model Statistics:")
                    print(f"   Architecture: Custom Transformer ({predictor.config.num_layers} layers)")
                    print(f"   Model size: {predictor.config.d_model} dimensions, {predictor.config.nhead} heads")
                    print(f"   Vocabulary: {len(predictor.vocab):,} tokens")
                    print(f"   Max length: {predictor.config.max_len} tokens")
                    print(f"   Device: {predictor.device}")
                    if predictor.val_metrics:
                        print(f"   Validation metrics: {predictor.val_metrics}")
                    continue
                
                if message:
                    result = predictor.predict(message, return_attention=args.attention)
                    print(f"\n{format_prediction(result, message)}")
                    
                    if args.attention and "tokens" in result:
                        print(f"\n   🔍 Tokens: {' | '.join(result['tokens'][:20])}")
                
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        return 0

    # Show help if no arguments
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())