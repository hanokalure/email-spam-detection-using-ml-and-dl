"""
Download Large SMS Spam Dataset
Downloads a high-quality, large SMS spam dataset directly from public sources
"""

import requests
import pandas as pd
import zipfile
import os
from io import BytesIO, StringIO
import urllib.request

def download_dataset_1():
    """Download SMS Spam Collection Dataset - Enhanced version"""
    print("Downloading Enhanced SMS Spam Collection Dataset...")
    
    try:
        # Direct download from UCI repository mirror
        url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        
        print(f"Downloading from: {url}")
        response = requests.get(url, timeout=60, stream=True)
        response.raise_for_status()
        
        # Extract ZIP file
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            zip_file.extractall("data/uci_dataset")
        
        # Find the CSV file
        csv_files = []
        for root, dirs, files in os.walk("data/uci_dataset"):
            for file in files:
                if file.endswith('.txt') or file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
        
        if csv_files:
            # Read the first CSV/TXT file found
            file_path = csv_files[0]
            print(f"Found dataset file: {file_path}")
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    if file_path.endswith('.txt'):
                        # Handle tab-separated file
                        df = pd.read_csv(file_path, sep='\t', encoding=encoding, header=None)
                        df.columns = ['label', 'text']
                    else:
                        df = pd.read_csv(file_path, encoding=encoding)
                        if df.columns[0] != 'label':
                            df = df.iloc[:, :2]
                            df.columns = ['label', 'text']
                    
                    print(f"✅ Successfully loaded with {encoding} encoding")
                    print(f"Dataset shape: {df.shape}")
                    return df
                except Exception as e:
                    continue
        
        print("❌ Could not find or read dataset file")
        return None
        
    except Exception as e:
        print(f"❌ Failed to download dataset 1: {str(e)}")
        return None

def download_dataset_2():
    """Download from GitHub repository with SMS spam data"""
    print("Downloading SMS spam dataset from GitHub...")
    
    try:
        # GitHub repository with SMS spam data
        url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
        
        print(f"Downloading from: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse CSV
        df = pd.read_csv(StringIO(response.text), encoding='latin-1')
        
        # Clean up columns
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ['label', 'text']
            df = df.dropna()
            
            print(f"✅ Successfully downloaded GitHub dataset")
            print(f"Dataset shape: {df.shape}")
            return df
        
        print("❌ Invalid dataset structure")
        return None
        
    except Exception as e:
        print(f"❌ Failed to download dataset 2: {str(e)}")
        return None

def download_dataset_3():
    """Create a larger synthetic dataset with modern spam patterns"""
    print("Creating enhanced synthetic dataset...")
    
    # Modern spam patterns
    spam_patterns = [
        # Referral spam
        "refer and earn click the below link",
        "refer your friends and earn money instantly",
        "earn money by referring friends click here now",
        "get cashback by referring friends today",
        "refer 3 friends and get 500 rupees instantly",
        "share this link and earn commission",
        "invite friends and get bonus points",
        "referral program earn upto 1000 daily",
        "make money by sharing this offer",
        "unlimited earning through referrals",
        
        # Prize/Lottery spam
        "congratulations you have won a cash prize",
        "you are selected for a lottery of 50000",
        "claim your lottery winnings now urgent",
        "winner announcement click to claim prize",
        "you have won 10000 rupees claim now",
        "lucky winner selected for cash reward",
        "jackpot winner click here to claim",
        "you won lottery ticket 123456789",
        "prize money waiting for you claim",
        "congratulations lottery winner selected",
        
        # Financial spam  
        "get instant loan click here apply",
        "pre approved loan available 100000",
        "zero interest loan apply now urgent",
        "instant cash advance click here now",
        "get credit card with no documents",
        "loan approved within 5 minutes apply",
        "emergency cash available click here",
        "personal loan without documents apply",
        "instant money transfer available now",
        "cash credit approved click to get",
        
        # Shopping/Offers spam
        "exclusive offer just for you 90 off",
        "flat 50 percent discount click now",
        "buy one get one free limited offer",
        "limited stock hurry up last chance",
        "mega sale ending tonight shop now",
        "special discount only for you click",
        "clearance sale upto 80 percent off",
        "flash sale ends in 1 hour buy",
        "exclusive deal click before expires",
        "limited time offer grab now fast",
        
        # Urgent action spam
        "urgent action required click here now",
        "immediate action needed verify account today",
        "your account will be closed click",
        "limited time offer expires in hours",
        "act now or lose this opportunity forever",
        "urgent update required click immediately",
        "account suspended click to reactivate now",
        "security alert action required immediately",
        "verify your account before closure urgent",
        "immediate response needed click here",
    ]
    
    # Normal messages (ham)
    ham_patterns = [
        "hey how are you doing today",
        "can you pick up groceries on way home",
        "meeting is at 3pm tomorrow dont forget",
        "happy birthday hope you have great day",
        "thanks for helping me with project today",
        "are you free for dinner tonight",
        "call me when you get this message",
        "good morning have a wonderful day ahead",
        "see you at the office tomorrow",
        "let me know when you reach home safely",
        "congratulations on your new job promotion",
        "the presentation went really well today",
        "can you send me the document please",
        "traffic is heavy today leave early",
        "the movie was really good loved it",
        "thanks for the birthday wishes everyone",
        "reminder about tomorrows meeting at 2pm",
        "can we reschedule our lunch meeting",
        "please confirm if you are coming",
        "hope you feel better soon take care",
        "the weather is nice today perfect",
        "dont forget to bring the documents tomorrow",
        "the project deadline is next friday",
        "can you help me with this task",
        "the train is delayed by 15 minutes",
        "see you at the conference next week",
        "thanks for the wonderful dinner last night",
        "the book you recommended was excellent",
        "please share your contact details with",
        "the meeting has been moved to room",
    ]
    
    # Create variations
    spam_data = []
    ham_data = []
    
    # Add base patterns multiple times with variations
    for pattern in spam_patterns:
        spam_data.append((pattern, "spam"))
        # Add variations
        spam_data.append((pattern.upper(), "spam"))
        spam_data.append((pattern.replace("click", "tap"), "spam"))
        spam_data.append((pattern.replace("now", "today"), "spam"))
        spam_data.append((pattern + " limited time", "spam"))
    
    for pattern in ham_patterns:
        ham_data.append((pattern, "ham"))
        # Add variations
        ham_data.append((pattern.capitalize(), "ham"))
        ham_data.append((pattern.replace("you", "u"), "ham"))
        ham_data.append((pattern + " thanks", "ham"))
    
    # Combine data
    all_data = spam_data + ham_data
    df = pd.DataFrame(all_data, columns=['text', 'label'])
    
    print(f"✅ Created synthetic dataset")
    print(f"Dataset shape: {df.shape}")
    print(f"Spam messages: {sum(df['label'] == 'spam')}")
    print(f"Ham messages: {sum(df['label'] == 'ham')}")
    
    return df

def main():
    """Download and combine multiple datasets"""
    print("="*60)
    print("DOWNLOADING LARGE SMS SPAM DATASET")
    print("="*60)
    
    datasets = []
    
    # Try downloading from multiple sources
    print("\n1. Trying UCI repository...")
    df1 = download_dataset_1()
    if df1 is not None:
        datasets.append(df1)
    
    print("\n2. Trying GitHub repository...")
    df2 = download_dataset_2() 
    if df2 is not None:
        datasets.append(df2)
    
    print("\n3. Creating synthetic dataset...")
    df3 = download_dataset_3()
    if df3 is not None:
        datasets.append(df3)
    
    # Combine all datasets
    if datasets:
        print(f"\n📊 Combining {len(datasets)} datasets...")
        combined_df = pd.concat(datasets, ignore_index=True)
        
        # Clean and deduplicate
        print("🧹 Cleaning data...")
        combined_df = combined_df.dropna()
        combined_df['label'] = combined_df['label'].str.lower()
        combined_df = combined_df[combined_df['label'].isin(['ham', 'spam'])]
        combined_df = combined_df.drop_duplicates(subset=['text'])
        
        # Save dataset
        output_path = "data/large_spam_dataset.csv"
        combined_df.to_csv(output_path, index=False, encoding='utf-8')
        
        # Show statistics
        spam_count = sum(combined_df['label'] == 'spam')
        ham_count = sum(combined_df['label'] == 'ham')
        
        print(f"\n✅ LARGE DATASET READY!")
        print(f"📁 Saved to: {output_path}")
        print(f"📈 Total messages: {len(combined_df)}")
        print(f"🚫 Spam messages: {spam_count} ({spam_count/len(combined_df)*100:.1f}%)")
        print(f"✅ Ham messages: {ham_count} ({ham_count/len(combined_df)*100:.1f}%)")
        
        print(f"\n🚀 Now run:")
        print(f"python src\\simple_svm_classifier.py --data \"{output_path}\" --model \"models\\svm_large.pkl\"")
        
        return output_path
    else:
        print("❌ Failed to download any dataset")
        return None

if __name__ == "__main__":
    main()