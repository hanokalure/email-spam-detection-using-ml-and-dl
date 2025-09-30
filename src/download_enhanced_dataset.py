"""
Enhanced SMS Spam Dataset Downloader
Downloads and combines multiple high-quality datasets for better accuracy
"""

import pandas as pd
import requests
import zipfile
import os
from io import StringIO
import numpy as np

def download_file(url, filename):
    """Download file from URL"""
    print(f"Downloading {filename}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {str(e)}")
        return False

def create_synthetic_spam_data():
    """Create additional spam patterns that are commonly missed"""
    synthetic_data = [
        # Modern referral spam
        ("refer and earn click the below link", "spam"),
        ("refer your friends and earn money", "spam"),
        ("earn money by referring friends click here", "spam"),
        ("get cashback by referring friends", "spam"),
        ("refer 3 friends and get 500 rupees", "spam"),
        
        # Prize/lottery spam
        ("congratulations you have won a prize", "spam"),
        ("you are selected for a cash prize", "spam"),
        ("claim your lottery winnings now", "spam"),
        ("winner announcement click to claim", "spam"),
        ("you have won 10000 rupees", "spam"),
        
        # Urgent action spam
        ("urgent action required click here", "spam"),
        ("immediate action needed verify account", "spam"),
        ("your account will be closed click here", "spam"),
        ("limited time offer expires today", "spam"),
        ("act now or lose this opportunity", "spam"),
        
        # Financial spam
        ("get instant loan click here", "spam"),
        ("pre approved loan available", "spam"),
        ("zero interest loan apply now", "spam"),
        ("instant cash advance click here", "spam"),
        ("get credit card with no verification", "spam"),
        
        # Shopping/offer spam
        ("exclusive offer just for you", "spam"),
        ("flat 50 percent discount click now", "spam"),
        ("buy one get one free offer", "spam"),
        ("limited stock hurry up", "spam"),
        ("mega sale ending tonight", "spam"),
        
        # Click-bait spam
        ("you will not believe what happened next", "spam"),
        ("this trick will shock you", "spam"),
        ("doctors hate this simple trick", "spam"),
        ("see what happens when you click", "spam"),
        ("amazing results guaranteed", "spam"),
        
        # Normal messages (ham)
        ("hey how are you doing", "ham"),
        ("can you pick up groceries", "ham"),
        ("meeting is at 3pm tomorrow", "ham"),
        ("happy birthday hope you have a great day", "ham"),
        ("thanks for helping me today", "ham"),
        ("are you free for dinner tonight", "ham"),
        ("call me when you get this message", "ham"),
        ("good morning have a nice day", "ham"),
        ("see you at the office", "ham"),
        ("let me know when you reach home", "ham"),
        ("congratulations on your new job", "ham"),
        ("the presentation went really well", "ham"),
        ("can you send me the document", "ham"),
        ("traffic is heavy today leave early", "ham"),
        ("the movie was really good", "ham"),
    ]
    
    return pd.DataFrame(synthetic_data, columns=['text', 'label'])

def download_kaggle_datasets():
    """Download additional datasets from public sources"""
    datasets = []
    
    # Dataset 1: Additional SMS data from GitHub
    try:
        url1 = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
        response = requests.get(url1, timeout=30)
        if response.status_code == 200:
            # Try to parse the CSV
            try:
                df1 = pd.read_csv(StringIO(response.text), encoding='latin-1')
                if len(df1.columns) >= 2:
                    df1 = df1.iloc[:, :2]
                    df1.columns = ['label', 'text']
                    df1 = df1.dropna()
                    datasets.append(df1)
                    print(f"✅ Downloaded additional dataset: {len(df1)} messages")
            except:
                print("❌ Could not parse additional dataset")
    except:
        print("❌ Could not download additional dataset")
    
    return datasets

def create_enhanced_dataset():
    """Create enhanced dataset by combining multiple sources"""
    print("="*60)
    print("CREATING ENHANCED SMS SPAM DATASET")
    print("="*60)
    
    datasets = []
    
    # 1. Load original dataset
    original_path = "data/spam.csv"
    if os.path.exists(original_path):
        print("Loading original dataset...")
        original_df = pd.read_csv(original_path, encoding='latin-1')
        original_df = original_df.iloc[:, :2]
        original_df.columns = ['label', 'text']
        original_df = original_df.dropna()
        datasets.append(original_df)
        print(f"✅ Original dataset: {len(original_df)} messages")
    
    # 2. Add synthetic data for modern spam patterns
    print("Adding synthetic spam patterns...")
    synthetic_df = create_synthetic_spam_data()
    datasets.append(synthetic_df)
    print(f"✅ Synthetic dataset: {len(synthetic_df)} messages")
    
    # 3. Try to download additional datasets
    print("Attempting to download additional datasets...")
    additional_datasets = download_kaggle_datasets()
    datasets.extend(additional_datasets)
    
    # 4. Combine all datasets
    if not datasets:
        print("❌ No datasets available!")
        return None
    
    print("Combining all datasets...")
    combined_df = pd.concat(datasets, ignore_index=True)
    
    # 5. Clean and deduplicate
    print("Cleaning and deduplicating...")
    combined_df = combined_df.drop_duplicates(subset=['text'])
    combined_df = combined_df.dropna()
    
    # 6. Standardize labels
    combined_df['label'] = combined_df['label'].str.lower()
    combined_df = combined_df[combined_df['label'].isin(['ham', 'spam'])]
    
    # 7. Balance the dataset if needed
    spam_count = sum(combined_df['label'] == 'spam')
    ham_count = sum(combined_df['label'] == 'ham')
    
    print(f"Dataset composition:")
    print(f"  Spam messages: {spam_count} ({spam_count/len(combined_df)*100:.1f}%)")
    print(f"  Ham messages: {ham_count} ({ham_count/len(combined_df)*100:.1f}%)")
    print(f"  Total messages: {len(combined_df)}")
    
    # 8. Save enhanced dataset
    output_path = "data/enhanced_spam.csv"
    combined_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Enhanced dataset saved to {output_path}")
    
    # 9. Show sample messages
    print("\nSample spam messages:")
    spam_samples = combined_df[combined_df['label'] == 'spam']['text'].head(3)
    for i, msg in enumerate(spam_samples, 1):
        print(f"  {i}. {msg[:60]}...")
    
    print("\nSample ham messages:")
    ham_samples = combined_df[combined_df['label'] == 'ham']['text'].head(3)
    for i, msg in enumerate(ham_samples, 1):
        print(f"  {i}. {msg[:60]}...")
    
    return output_path

def main():
    """Main function"""
    # Create enhanced dataset
    enhanced_path = create_enhanced_dataset()
    
    if enhanced_path:
        print(f"\n✅ Enhanced dataset ready at: {enhanced_path}")
        print("Now run: python src/enhanced_svm_classifier.py")
    else:
        print("❌ Failed to create enhanced dataset")

if __name__ == "__main__":
    main()