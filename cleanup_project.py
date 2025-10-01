#!/usr/bin/env python3
"""
Project Cleanup Script - Remove test files, debug files, and legacy components
Keeps only the Essential Enhanced Transformer spam detection system
"""

import os
import shutil
from pathlib import Path

# Files and directories to KEEP (core functionality)
KEEP_FILES = {
    # Main files
    "predict_main.py",  # Main prediction interface (updated for Enhanced Transformer only)
    "predict_enhanced_transformer.py",  # Enhanced Transformer predictor
    "requirements.txt",  # Dependencies
    
    # Documentation
    "README.md",
    "LICENSE",
    ".gitignore",
    
    # Core Enhanced Transformer implementation
    "src/enhanced_spam_preprocessor.py",
    "src/enhanced_transformer_classifier.py", 
    "src/train_enhanced_transformer.py",
    
    # Model files (keep the trained model)
    "models/enhanced_transformer_best.pt",
    "models/README.md",
    
    # Data directory structure (but will clean contents)
    "data/README.md"
}

# Files and directories to REMOVE (test/debug/legacy/unused)
REMOVE_ITEMS = [
    # Test and debug files
    "check_models.py",
    "debug_banking.py", 
    "test_banking_scenarios.py",
    "test_spam_categories.py",
    "test_models.py",
    "evaluate_all_models.py",
    "detailed_analysis.py",
    "message_extremes.py",
    "analyze_dataset.py",
    "analyze_large_dataset.py",
    
    # Legacy prediction interfaces
    "predict.py",
    "predict_enhanced.py",
    "predict_smart.py",
    "predict_transformer.py",  # Old transformer predictor
    
    # Legacy model files and training
    "models/transformer_best.pt",  # Old transformer model
    "models/commands_to_run.txt",
    
    # Legacy src files (non-Enhanced Transformer)
    "src/download_all_datasets.py",
    "src/download_enhanced_dataset.py", 
    "src/download_enron_dataset.py",
    "src/download_large_dataset.py",
    "src/download_spamassassin.py",
    "src/comprehensive_spam_dataset.py",
    "src/train_transformer.py",  # Old transformer training
    "src/transformer_text_classifier.py",  # Old transformer model
    "src/model_manager.py",
    "src/predict.py",
    "src/predict_cli.py",
    "src/process_existing_corpus.py",
    "src/simple_svm_classifier.py",  # SVM removed from main interface
    "src/svm_classifier.py",
    "src/test_svm.py",
    "src/train_advanced_models.py",
    "src/train_catboost.py",
    "src/train_catboost_fast.py",
    "src/train_catboost_simple.py",
    "src/train_catboost_tuned.py",
    
    # Legacy documentation
    "EMAIL_SPAM_GUIDE.md",
    "MODEL_COMPARISON_GUIDE.md", 
    "RUN_MODEL_GUIDE.md",
    "SOLUTION_SUMMARY.md",
    
    # CatBoost artifacts
    "catboost_info/",
    
    # Raw data files (can be re-downloaded if needed)
    "data/additional_ham.tar.bz2",
    "data/enron_raw/",
    "data/spamassassin_additional/"
]

def cleanup_project():
    """Clean up the project by removing test files and legacy components"""
    
    print("🧹 Starting project cleanup...")
    print("=" * 60)
    
    removed_count = 0
    kept_count = 0
    
    # Remove specified files and directories
    for item in REMOVE_ITEMS:
        item_path = Path(item)
        
        if item_path.exists():
            if item_path.is_file():
                print(f"🗑️  Removing file: {item}")
                item_path.unlink()
                removed_count += 1
            elif item_path.is_dir():
                print(f"🗑️  Removing directory: {item}")
                shutil.rmtree(item_path)
                removed_count += 1
        else:
            print(f"⚠️  Not found (skipping): {item}")
    
    # Count kept files
    for item in KEEP_FILES:
        if Path(item).exists():
            kept_count += 1
    
    print("\n" + "=" * 60)
    print("✅ CLEANUP COMPLETE")
    print("=" * 60)
    print(f"🗑️  Items removed: {removed_count}")
    print(f"✅ Core files kept: {kept_count}")
    
    print("\n📁 FINAL PROJECT STRUCTURE:")
    print("=" * 60)
    print("Core Files:")
    for item in sorted(KEEP_FILES):
        if Path(item).exists():
            print(f"  ✅ {item}")
        else:
            print(f"  ❌ {item} (missing)")
    
    print(f"\n🎯 Your project now contains ONLY the Enhanced Transformer spam detection system!")
    print(f"✅ Main interface: predict_main.py")
    print(f"✅ Direct predictor: predict_enhanced_transformer.py")
    print(f"✅ Model: models/enhanced_transformer_best.pt")
    print(f"✅ Training: src/train_enhanced_transformer.py")
    
    return removed_count, kept_count

def confirm_cleanup():
    """Ask for user confirmation before cleanup"""
    print("🚨 PROJECT CLEANUP CONFIRMATION")
    print("=" * 60)
    print("This will remove the following types of files:")
    print("  • Test and debug scripts")
    print("  • Legacy prediction interfaces") 
    print("  • Old Transformer model (non-Enhanced)")
    print("  • SVM and CatBoost components")
    print("  • Raw dataset archives")
    print("  • Analysis and evaluation scripts")
    print()
    print("✅ WILL KEEP:")
    print("  • Enhanced Transformer model and training")
    print("  • Main prediction interface (predict_main.py)")
    print("  • Enhanced preprocessor and classifier")
    print("  • Documentation (README, LICENSE)")
    print("  • Dependencies (requirements.txt)")
    
    response = input(f"\n🤔 Proceed with cleanup? (yes/no): ").strip().lower()
    return response in ['yes', 'y']

if __name__ == "__main__":
    if confirm_cleanup():
        removed, kept = cleanup_project()
        print(f"\n🎉 Cleanup finished! Removed {removed} items, kept {kept} core files.")
    else:
        print("❌ Cleanup cancelled.")