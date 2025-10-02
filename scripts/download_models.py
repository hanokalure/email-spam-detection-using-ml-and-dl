#!/usr/bin/env python3
"""
Model Download Script for Spam Detection Models

Downloads trained models from Google Drive to the models/ directory.
Requires: pip install gdown

Usage:
    python scripts/download_models.py           # Download all models
    python scripts/download_models.py svm       # Download specific model
    python scripts/download_models.py --help    # Show help
"""

import os
import sys
import argparse
from pathlib import Path

try:
    import gdown
except ImportError:
    print("❌ gdown is required but not installed.")
    print("📦 Install it with: pip install gdown")
    print("🔄 Then run this script again.")
    sys.exit(1)

try:
    import toml
except ImportError:
    print("⚠️  toml package recommended but not found.")
    print("📦 Install it with: pip install toml")
    print("🔄 Using fallback configuration...")
    toml = None

# Fallback configuration if toml not available
MODELS_CONFIG = {
    'svm': {
        'filename': 'svm_full.pkl',
        'direct_url': 'https://drive.google.com/uc?id=1Vxjz4QV3FESvm7gMeKNqklA5uARPntLv',
        'description': 'Support Vector Machine classifier',
        'size_mb': '~15MB'
    },
    'enhanced_transformer': {
        'filename': 'enhanced_transformer_99recall.pt',
        'direct_url': 'https://drive.google.com/uc?id=1kGD6Tg5JLIko0XhYPk2-WtAswj1S2Pgs',
        'description': 'Enhanced Transformer model (99% spam recall)',
        'size_mb': '~172MB'
    },
    'catboost': {
        'filename': 'catboost_tuned.pkl',
        'direct_url': 'https://drive.google.com/uc?id=1ofS_IU9QiypgkvFqNGLjUvSdUfEi9hjO',
        'description': 'Tuned CatBoost classifier',
        'size_mb': '~10MB'
    }
}

def load_config():
    """Load models configuration from TOML file or use fallback"""
    config_path = Path(__file__).parent.parent / 'config' / 'models.toml'
    
    if toml and config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = toml.load(f)
            return config['models']
        except Exception as e:
            print(f"⚠️  Could not load config file: {e}")
    
    print("🔄 Using built-in configuration...")
    return MODELS_CONFIG

def download_model(model_name, model_config, models_dir):
    """Download a single model"""
    filename = model_config['filename']
    url = model_config['direct_url']
    description = model_config['description']
    size = model_config.get('size_mb', 'Unknown size')
    
    output_path = models_dir / filename
    
    print(f"\n📥 Downloading {model_name.upper()} model:")
    print(f"   📄 {description}")
    print(f"   📦 Size: {size}")
    print(f"   🎯 Target: {output_path}")
    
    if output_path.exists():
        print(f"   ✅ File already exists: {output_path}")
        return True
    
    try:
        print(f"   🔄 Downloading from Google Drive...")
        gdown.download(url, str(output_path), quiet=False)
        
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✅ Download successful! ({file_size:.1f} MB)")
            return True
        else:
            print(f"   ❌ Download failed - file not created")
            return False
            
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Download spam detection models from Google Drive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available models:
  svm                Support Vector Machine (~15MB)
  enhanced_transformer  Enhanced Transformer (~172MB)  
  catboost           CatBoost classifier (~10MB)
  
Examples:
  python scripts/download_models.py                    # Download all
  python scripts/download_models.py svm catboost       # Download specific models
  python scripts/download_models.py --list             # List available models
        """
    )
    
    parser.add_argument('models', nargs='*', 
                       help='Specific models to download (default: all)')
    parser.add_argument('--list', action='store_true',
                       help='List available models and exit')
    parser.add_argument('--models-dir', default='models',
                       help='Directory to download models to (default: models)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    if args.list:
        print("📋 Available models:")
        for name, info in config.items():
            print(f"   • {name:<20} - {info['description']} ({info.get('size_mb', '?')})")
        return
    
    # Setup models directory
    models_dir = Path(args.models_dir)
    models_dir.mkdir(exist_ok=True)
    
    print(f"🎯 Models directory: {models_dir.absolute()}")
    
    # Determine which models to download
    if args.models:
        requested_models = args.models
        # Validate requested models
        invalid = [m for m in requested_models if m not in config]
        if invalid:
            print(f"❌ Invalid model names: {invalid}")
            print(f"✅ Available models: {list(config.keys())}")
            return 1
    else:
        requested_models = list(config.keys())
        print(f"📦 Downloading all models: {requested_models}")
    
    # Download models
    successful = []
    failed = []
    
    for model_name in requested_models:
        if download_model(model_name, config[model_name], models_dir):
            successful.append(model_name)
        else:
            failed.append(model_name)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Download Summary:")
    if successful:
        print(f"   ✅ Successful ({len(successful)}): {', '.join(successful)}")
    if failed:
        print(f"   ❌ Failed ({len(failed)}): {', '.join(failed)}")
    
    print(f"\n🎯 Models location: {models_dir.absolute()}")
    
    if successful:
        print("\n🚀 Ready to use! Try running:")
        print("   python predictors/predict_main.py \"Test spam message\"")
    
    return 0 if not failed else 1

if __name__ == '__main__':
    sys.exit(main())