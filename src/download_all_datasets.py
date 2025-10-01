#!/usr/bin/env python3
"""
Download comprehensive spam datasets for maximum training data
- Enron-Spam Dataset (~33K emails)
- TREC Spam Dataset (~75K emails)  
- SpamAssassin Additional Corpora
- Combined total: ~120K+ emails
"""

import os
import requests
import tarfile
import bz2
import zipfile
import pandas as pd
from pathlib import Path
import shutil
import urllib3
from urllib.request import urlretrieve
import ssl
import warnings

# Disable SSL warnings and verification issues
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

def download_file_robust(url, filepath, description=""):
    """Download file with multiple fallback methods"""
    
    methods = [
        lambda: download_with_requests(url, filepath),
        lambda: download_with_urllib(url, filepath),
        lambda: download_with_requests_no_ssl(url, filepath)
    ]
    
    for i, method in enumerate(methods, 1):
        try:
            print(f"📥 {description} - Method {i}/3...")
            method()
            print(f"✅ Downloaded: {filepath}")
            return True
        except Exception as e:
            print(f"⚠️ Method {i} failed: {str(e)[:100]}...")
            continue
    
    print(f"❌ All download methods failed for {url}")
    return False

def download_with_requests(url, filepath):
    """Download with requests library"""
    response = requests.get(url, stream=True, timeout=300, verify=True)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def download_with_urllib(url, filepath):
    """Download with urllib"""
    urlretrieve(url, filepath)

def download_with_requests_no_ssl(url, filepath):
    """Download with requests, no SSL verification"""
    response = requests.get(url, stream=True, timeout=300, verify=False)
    response.raise_for_status()
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

def download_enron_datasets():
    """Download Enron-Spam datasets"""
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    enron_dir = data_dir / "enron_raw"
    enron_dir.mkdir(exist_ok=True)
    
    print("🚀 Downloading Enron-Spam Datasets...")
    
    # Alternative Enron sources (multiple URLs for reliability)
    enron_sources = [
        {
            'name': 'Enron Preprocessed v1',
            'urls': [
                'http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron1.tar.gz',
                'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/Enron1.tar.gz'
            ]
        },
        {
            'name': 'Enron Preprocessed v2', 
            'urls': [
                'http://www2.aueb.gr/users/ion/data/enron-spam/preprocessed/enron2.tar.gz',
                'https://archive.ics.uci.edu/ml/machine-learning-databases/00228/Enron2.tar.gz'
            ]
        },
        {
            'name': 'Enron from Kaggle Mirror',
            'urls': [
                'https://github.com/MWiechmann/enron_spam_data/raw/master/enron_spam_data.csv'
            ]
        },
        {
            'name': 'Enron from Alternative Source',
            'urls': [
                'https://www.cs.cmu.edu/~./enron/enron_mail_20110402.tar.gz'
            ]
        }
    ]
    
    successful_downloads = []
    all_emails = []
    
    for source in enron_sources:
        print(f"\n📦 Trying {source['name']}...")
        
        for url in source['urls']:
            filename = url.split('/')[-1]
            filepath = enron_dir / filename
            
            if download_file_robust(url, filepath, source['name']):
                successful_downloads.append(filepath)
                
                # Process downloaded file
                try:
                    emails_from_file = process_enron_file(filepath, source['name'])
                    all_emails.extend(emails_from_file)
                    print(f"✅ Processed {len(emails_from_file)} emails from {filename}")
                except Exception as e:
                    print(f"⚠️ Error processing {filename}: {e}")
                
                break  # Success, move to next source
    
    if all_emails:
        # Save combined Enron dataset
        df = pd.DataFrame(all_emails)
        csv_path = data_dir / "enron_combined_large.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n📊 Enron Dataset Summary:")
        print(f"Total emails: {len(df)}")
        print(f"Ham: {len(df[df['label'] == 'ham'])}")
        print(f"Spam: {len(df[df['label'] == 'spam'])}")
        print(f"Saved to: {csv_path}")
        
        return csv_path
    else:
        print("❌ No Enron emails were successfully downloaded")
        return None

def process_enron_file(filepath, source_name):
    """Process different types of Enron files"""
    
    emails = []
    
    try:
        if filepath.suffix == '.csv':
            # CSV file
            df = pd.read_csv(filepath)
            
            # Try different column name combinations
            label_col = None
            text_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'label' in col_lower or 'category' in col_lower or 'spam' in col_lower:
                    label_col = col
                elif 'text' in col_lower or 'message' in col_lower or 'email' in col_lower or 'body' in col_lower:
                    text_col = col
            
            if label_col and text_col:
                for _, row in df.iterrows():
                    label = str(row[label_col]).lower()
                    text = str(row[text_col])
                    
                    # Normalize labels
                    if 'spam' in label or label == '1':
                        label = 'spam'
                    else:
                        label = 'ham'
                    
                    emails.append({
                        'label': label,
                        'text': text,
                        'source': source_name
                    })
        
        elif filepath.suffix in ['.gz', '.tar']:
            # Extract and process tar.gz files
            if tarfile.is_tarfile(filepath):
                with tarfile.open(filepath, 'r:*') as tar:
                    temp_dir = filepath.parent / f"temp_{filepath.stem}"
                    tar.extractall(temp_dir)
                    
                    # Look for ham/spam directories
                    for root, dirs, files in os.walk(temp_dir):
                        if 'ham' in root.lower():
                            label = 'ham'
                        elif 'spam' in root.lower():
                            label = 'spam'
                        else:
                            continue
                        
                        for file in files:
                            if file.endswith('.txt'):
                                file_path = Path(root) / file
                                try:
                                    text = file_path.read_text(encoding='utf-8', errors='ignore')
                                    emails.append({
                                        'label': label,
                                        'text': text,
                                        'source': source_name
                                    })
                                except:
                                    continue
                    
                    # Cleanup temp directory
                    shutil.rmtree(temp_dir, ignore_errors=True)
    
    except Exception as e:
        print(f"⚠️ Error processing {filepath}: {e}")
    
    return emails

def download_trec_dataset():
    """Download TREC Spam Track datasets"""
    
    data_dir = Path("data")
    trec_dir = data_dir / "trec_raw"
    trec_dir.mkdir(exist_ok=True)
    
    print("\n🚀 Downloading TREC Spam Datasets...")
    
    trec_sources = [
        {
            'name': 'TREC 2006 Public Corpus',
            'url': 'https://plg.uwaterloo.ca/~gvcormac/treccorpus06/trec06p.tar.gz',
            'size': '~25K emails'
        },
        {
            'name': 'TREC 2007 Public Corpus',
            'url': 'https://plg.uwaterloo.ca/~gvcormac/treccorpus07/trec07p.tar.gz', 
            'size': '~75K emails'
        }
    ]
    
    all_emails = []
    
    for source in trec_sources:
        print(f"\n📦 Downloading {source['name']} ({source['size']})...")
        
        filename = source['url'].split('/')[-1]
        filepath = trec_dir / filename
        
        if download_file_robust(source['url'], filepath, source['name']):
            try:
                emails_from_file = process_trec_file(filepath, source['name'])
                all_emails.extend(emails_from_file)
                print(f"✅ Processed {len(emails_from_file)} emails from TREC")
            except Exception as e:
                print(f"⚠️ Error processing TREC file: {e}")
    
    if all_emails:
        # Save TREC dataset
        df = pd.DataFrame(all_emails)
        csv_path = data_dir / "trec_combined_large.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n📊 TREC Dataset Summary:")
        print(f"Total emails: {len(df)}")
        print(f"Ham: {len(df[df['label'] == 'ham'])}")
        print(f"Spam: {len(df[df['label'] == 'spam'])}")
        print(f"Saved to: {csv_path}")
        
        return csv_path
    else:
        print("❌ No TREC emails were successfully downloaded")
        return None

def process_trec_file(filepath, source_name):
    """Process TREC dataset files"""
    
    emails = []
    
    try:
        if tarfile.is_tarfile(filepath):
            with tarfile.open(filepath, 'r:*') as tar:
                temp_dir = filepath.parent / f"temp_trec_{filepath.stem}"
                tar.extractall(temp_dir)
                
                # TREC files usually have a specific structure
                # Look for index files and email files
                for root, dirs, files in os.walk(temp_dir):
                    # Look for label files
                    label_files = [f for f in files if 'label' in f.lower() or 'index' in f.lower()]
                    
                    if label_files:
                        label_file = Path(root) / label_files[0]
                        
                        try:
                            with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 2:
                                        label = 'spam' if parts[0].lower() == 'spam' else 'ham'
                                        email_file = Path(root) / parts[1]
                                        
                                        if email_file.exists():
                                            try:
                                                text = email_file.read_text(encoding='utf-8', errors='ignore')
                                                emails.append({
                                                    'label': label,
                                                    'text': text,
                                                    'source': source_name
                                                })
                                            except:
                                                continue
                        except Exception as e:
                            print(f"⚠️ Error reading label file: {e}")
                
                # Cleanup
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    except Exception as e:
        print(f"⚠️ Error processing TREC file: {e}")
    
    return emails

def download_additional_spamassassin():
    """Download additional SpamAssassin corpora"""
    
    data_dir = Path("data")
    spam_dir = data_dir / "spamassassin_additional"
    spam_dir.mkdir(exist_ok=True)
    
    print("\n🚀 Downloading Additional SpamAssassin Corpora...")
    
    sa_sources = [
        {
            'name': 'Easy Ham 2',
            'url': 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham_2.tar.bz2'
        },
        {
            'name': 'Hard Ham',
            'url': 'https://spamassassin.apache.org/old/publiccorpus/20030228_hard_ham.tar.bz2'
        },
        {
            'name': 'Spam 1',
            'url': 'https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2'
        },
        {
            'name': 'Spam 2', 
            'url': 'https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2'
        }
    ]
    
    all_emails = []
    
    for source in sa_sources:
        print(f"\n📦 Downloading {source['name']}...")
        
        filename = source['url'].split('/')[-1]
        filepath = spam_dir / filename
        
        if download_file_robust(source['url'], filepath, source['name']):
            try:
                emails_from_file = process_spamassassin_file(filepath, source['name'])
                all_emails.extend(emails_from_file)
                print(f"✅ Processed {len(emails_from_file)} emails from {source['name']}")
            except Exception as e:
                print(f"⚠️ Error processing {source['name']}: {e}")
    
    if all_emails:
        # Save additional SpamAssassin dataset
        df = pd.DataFrame(all_emails)
        csv_path = data_dir / "spamassassin_additional_large.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"\n📊 Additional SpamAssassin Summary:")
        print(f"Total emails: {len(df)}")
        print(f"Ham: {len(df[df['label'] == 'ham'])}")
        print(f"Spam: {len(df[df['label'] == 'spam'])}")
        print(f"Saved to: {csv_path}")
        
        return csv_path
    else:
        print("❌ No additional SpamAssassin emails downloaded")
        return None

def process_spamassassin_file(filepath, source_name):
    """Process SpamAssassin corpus files"""
    
    emails = []
    
    try:
        # Handle .bz2 files
        if filepath.suffix == '.bz2':
            if filepath.name.endswith('.tar.bz2'):
                with tarfile.open(filepath, 'r:bz2') as tar:
                    temp_dir = filepath.parent / f"temp_sa_{filepath.stem}"
                    tar.extractall(temp_dir)
                    
                    # Determine label from source name
                    if 'spam' in source_name.lower():
                        label = 'spam'
                    else:
                        label = 'ham'
                    
                    # Process all email files
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if not file.startswith('.'):  # Skip hidden files
                                file_path = Path(root) / file
                                try:
                                    text = file_path.read_text(encoding='utf-8', errors='ignore')
                                    emails.append({
                                        'label': label,
                                        'text': text,
                                        'source': source_name
                                    })
                                except:
                                    continue
                    
                    # Cleanup
                    shutil.rmtree(temp_dir, ignore_errors=True)
    
    except Exception as e:
        print(f"⚠️ Error processing SpamAssassin file: {e}")
    
    return emails

def create_mega_dataset():
    """Combine all downloaded datasets into one mega dataset"""
    
    data_dir = Path("data")
    
    print("\n🚀 Creating Mega Combined Dataset...")
    
    dataset_files = [
        "data/spam.csv",  # Original SpamAssassin
        "data/enron_combined_large.csv",
        "data/trec_combined_large.csv", 
        "data/spamassassin_additional_large.csv"
    ]
    
    all_data = []
    total_count = 0
    
    for dataset_file in dataset_files:
        if os.path.exists(dataset_file):
            print(f"📥 Adding {dataset_file}...")
            try:
                df = pd.read_csv(dataset_file)
                
                # Standardize columns
                if 'v1' in df.columns and 'v2' in df.columns:
                    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
                
                # Ensure proper columns exist
                if 'label' in df.columns and 'text' in df.columns:
                    df['label'] = df['label'].str.lower()
                    df = df[df['label'].isin(['ham', 'spam'])]
                    df = df.dropna(subset=['label', 'text'])
                    
                    all_data.append(df)
                    total_count += len(df)
                    print(f"✅ Added {len(df)} emails")
                else:
                    print(f"⚠️ Skipping {dataset_file} - invalid format")
            except Exception as e:
                print(f"⚠️ Error loading {dataset_file}: {e}")
        else:
            print(f"⚠️ {dataset_file} not found")
    
    if all_data:
        # Combine all datasets
        mega_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates based on text content
        print("🔄 Removing duplicates...")
        before_dedup = len(mega_df)
        mega_df = mega_df.drop_duplicates(subset=['text'], keep='first')
        after_dedup = len(mega_df)
        
        # Save mega dataset
        mega_path = data_dir / "mega_spam_dataset.csv"
        mega_df.to_csv(mega_path, index=False)
        
        print(f"\n🎉 MEGA DATASET CREATED!")
        print(f"📊 Final Statistics:")
        print(f"Total emails: {after_dedup:,} (removed {before_dedup - after_dedup:,} duplicates)")
        print(f"Ham emails: {len(mega_df[mega_df['label'] == 'ham']):,}")
        print(f"Spam emails: {len(mega_df[mega_df['label'] == 'spam']):,}")
        print(f"📁 Saved to: {mega_path}")
        
        return mega_path
    else:
        print("❌ No datasets were successfully combined")
        return None

if __name__ == "__main__":
    print("🚀 COMPREHENSIVE SPAM DATASET DOWNLOADER")
    print("=" * 50)
    print("Downloading multiple large-scale spam datasets...")
    print("Expected total: 100K+ emails")
    print("=" * 50)
    
    # Download all datasets
    datasets_downloaded = []
    
    # 1. Download Enron datasets
    enron_result = download_enron_datasets()
    if enron_result:
        datasets_downloaded.append(enron_result)
    
    # 2. Download TREC datasets  
    trec_result = download_trec_dataset()
    if trec_result:
        datasets_downloaded.append(trec_result)
    
    # 3. Download additional SpamAssassin
    sa_result = download_additional_spamassassin()
    if sa_result:
        datasets_downloaded.append(sa_result)
    
    # 4. Create mega combined dataset
    if datasets_downloaded or os.path.exists("data/spam.csv"):
        mega_result = create_mega_dataset()
        
        if mega_result:
            print(f"\n🎉 SUCCESS! Mega dataset ready for training:")
            print(f"📁 {mega_result}")
            print("\nNext step: Run 'python src/train_advanced_models.py' to train all ML models")
        else:
            print("\n❌ Failed to create mega dataset")
    else:
        print("\n❌ No datasets were successfully downloaded")
        print("\n💡 Manual download options:")
        print("1. Visit: https://www.cs.cmu.edu/~./enron/ for Enron dataset")
        print("2. Visit: https://trec.nist.gov/ for TREC spam datasets")
        print("3. Visit: https://spamassassin.apache.org/old/publiccorpus/ for SpamAssassin")