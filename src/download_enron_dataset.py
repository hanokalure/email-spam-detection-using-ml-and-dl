#!/usr/bin/env python3
"""
Download and prepare Enron-Spam dataset for training
"""

import os
import requests
import tarfile
import pandas as pd
from pathlib import Path
import shutil
import glob

def download_enron_spam():
    """Download Enron-Spam dataset"""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    enron_dir = data_dir / "enron_spam"
    enron_dir.mkdir(exist_ok=True)
    
    # Enron dataset URLs (multiple sources)
    urls = [
        "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz",
        "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron2.tar.gz", 
        "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron3.tar.gz",
        "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron4.tar.gz",
        "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron5.tar.gz",
        "http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron6.tar.gz"
    ]
    
    print("🔄 Downloading Enron-Spam dataset...")
    
    all_emails = []
    
    for i, url in enumerate(urls, 1):
        print(f"📥 Downloading Enron part {i}/6...")
        
        try:
            # Download tar.gz file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            tar_path = enron_dir / f"enron{i}.tar.gz"
            
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"✅ Downloaded enron{i}.tar.gz")
            
            # Extract tar.gz
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(enron_dir)
            
            # Process extracted emails
            extract_dir = enron_dir / f"enron{i}"
            if extract_dir.exists():
                # Look for ham and spam directories
                ham_dir = extract_dir / "ham"
                spam_dir = extract_dir / "spam"
                
                if ham_dir.exists():
                    for email_file in ham_dir.glob("*.txt"):
                        try:
                            content = email_file.read_text(encoding='utf-8', errors='ignore')
                            all_emails.append({"label": "ham", "text": content, "source": f"enron{i}"})
                        except Exception as e:
                            print(f"⚠️ Error reading {email_file}: {e}")
                
                if spam_dir.exists():
                    for email_file in spam_dir.glob("*.txt"):
                        try:
                            content = email_file.read_text(encoding='utf-8', errors='ignore')
                            all_emails.append({"label": "spam", "text": content, "source": f"enron{i}"})
                        except Exception as e:
                            print(f"⚠️ Error reading {email_file}: {e}")
            
            # Clean up tar file
            tar_path.unlink()
            
        except Exception as e:
            print(f"❌ Error downloading enron{i}: {e}")
            continue
    
    if all_emails:
        # Save as CSV
        df = pd.DataFrame(all_emails)
        csv_path = data_dir / "enron_spam_combined.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n📊 Enron-Spam Dataset Summary:")
        print(f"Total emails: {len(df)}")
        print(f"Ham emails: {len(df[df['label'] == 'ham'])}")
        print(f"Spam emails: {len(df[df['label'] == 'spam'])}")
        print(f"Saved to: {csv_path}")
        
        # Clean up extracted directories
        for i in range(1, 7):
            extract_dir = enron_dir / f"enron{i}"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
        
        return csv_path
    else:
        print("❌ No emails were successfully processed")
        return None

def download_alternative_datasets():
    """Download alternative spam datasets"""
    data_dir = Path("data")
    
    print("\n🔄 Trying alternative dataset sources...")
    
    # Try Kaggle-style Enron dataset
    try:
        # This is a simplified version - in practice you might need Kaggle API
        alt_url = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.csv"
        
        response = requests.get(alt_url)
        if response.status_code == 200:
            csv_path = data_dir / "enron_alternative.csv"
            with open(csv_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print(f"✅ Downloaded alternative Enron dataset to {csv_path}")
            return csv_path
    except Exception as e:
        print(f"⚠️ Alternative dataset not available: {e}")
    
    # Try SpamAssassin public corpus expansion
    try:
        # Additional SpamAssassin corpus
        corpus_url = "https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2"
        print(f"📥 Trying SpamAssassin additional corpus...")
        
        response = requests.get(corpus_url, stream=True)
        if response.status_code == 200:
            bz2_path = data_dir / "additional_ham.tar.bz2"
            
            with open(bz2_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print("✅ Downloaded additional SpamAssassin corpus")
            return bz2_path
    except Exception as e:
        print(f"⚠️ Additional corpus not available: {e}")
    
    return None

if __name__ == "__main__":
    print("🚀 Starting Enron-Spam dataset download...")
    
    # Try main Enron download
    result = download_enron_spam()
    
    if result is None:
        print("\n🔄 Main download failed, trying alternatives...")
        alt_result = download_alternative_datasets()
        
        if alt_result is None:
            print("❌ All download attempts failed")
            print("💡 Manual download instructions:")
            print("1. Visit: http://www2.aueb.gr/users/ion/data/enron-spam/")
            print("2. Download preprocessed datasets")
            print("3. Extract to data/enron_spam/ directory")
        else:
            print(f"✅ Alternative dataset downloaded: {alt_result}")
    else:
        print(f"✅ Main Enron dataset ready: {result}")