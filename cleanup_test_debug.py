#!/usr/bin/env python3
"""
Simple Project Cleanup - Remove only test and debug files
Keeps all main functionality, models, and prediction interfaces intact
"""

import os
from pathlib import Path

# Only test and debug files to remove
TEST_DEBUG_FILES = [
    # Test files
    "check_models.py",
    "test_banking_scenarios.py", 
    "test_spam_categories.py",
    "test_models.py",
    "src/test_svm.py",
    
    # Debug files
    "debug_banking.py",
    
    # Analysis/evaluation files (not core functionality)
    "evaluate_all_models.py",
    "detailed_analysis.py",
    "message_extremes.py",
    "analyze_dataset.py", 
    "analyze_large_dataset.py"
]

def cleanup_test_debug():
    """Remove only test and debug files"""
    
    print("🧹 Cleaning up test and debug files...")
    print("=" * 50)
    
    removed_count = 0
    
    for file_path in TEST_DEBUG_FILES:
        path = Path(file_path)
        
        if path.exists():
            print(f"🗑️  Removing: {file_path}")
            path.unlink()
            removed_count += 1
        else:
            print(f"⚠️  Not found: {file_path}")
    
    print("\n" + "=" * 50)
    print("✅ CLEANUP COMPLETE")
    print("=" * 50)
    print(f"🗑️  Test/debug files removed: {removed_count}")
    
    print(f"\n🎯 All main functionality preserved:")
    print(f"✅ All prediction interfaces kept")
    print(f"✅ All models kept")
    print(f"✅ All training scripts kept")
    print(f"✅ All documentation kept")
    
    return removed_count

if __name__ == "__main__":
    print("🚨 SIMPLE CLEANUP - TEST & DEBUG FILES ONLY")
    print("=" * 50)
    print("This will remove ONLY:")
    for file in TEST_DEBUG_FILES:
        print(f"  • {file}")
    
    print(f"\n✅ EVERYTHING ELSE WILL BE KEPT")
    
    response = input(f"\n🤔 Proceed? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        removed = cleanup_test_debug()
        print(f"\n🎉 Done! Removed {removed} test/debug files.")
    else:
        print("❌ Cleanup cancelled.")