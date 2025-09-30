"""
Process Existing SpamAssassin Corpus
Converts the already downloaded SpamAssassin email files to CSV format
"""

import pandas as pd
import os
import email
import email.policy
from pathlib import Path
import re

def clean_text(text):
    """Clean and normalize text content"""
    if not text:
        return ""
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Limit length for SMS-like processing
    if len(text) > 500:
        text = text[:500]
    
    return text

def parse_email_file(file_path):
    """Parse a single email file and extract text content"""
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Try parsing as email
        try:
            msg = email.message_from_bytes(content, policy=email.policy.default)
            
            # Extract text content
            text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        part_content = part.get_content()
                        if part_content:
                            text += part_content + " "
            else:
                if msg.get_content_type() == "text/plain":
                    text = msg.get_content()
            
            # If no body text, try subject
            if not text and msg.get("Subject"):
                text = msg.get("Subject")
            
            # Clean the text
            text = clean_text(text)
            
            # Only return if we have meaningful text
            if len(text) > 10:
                return text
                
        except Exception as e:
            # If email parsing fails, try as plain text
            try:
                text = content.decode('utf-8', errors='ignore')
                text = clean_text(text)
                if len(text) > 10:
                    return text
            except:
                try:
                    text = content.decode('latin-1', errors='ignore')
                    text = clean_text(text)
                    if len(text) > 10:
                        return text
                except:
                    pass
    
    except Exception as e:
        pass
    
    return None

def process_corpus_directory(base_path):
    """Process all email files in the corpus directory structure"""
    print(f"🔍 Scanning directory: {base_path}")
    
    all_messages = []
    
    # Expected folder structure patterns
    spam_folders = ['spam', 'spam_2']
    ham_folders = ['easy_ham', 'hard_ham']
    
    base_path = Path(base_path)
    
    # Process all files recursively
    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        
        # Determine label based on folder structure
        folder_name = root_path.name.lower()
        parent_folder = root_path.parent.name.lower() if root_path.parent != root_path else ""
        
        # Determine if this is spam or ham
        label = None
        if any(spam_folder in folder_name for spam_folder in spam_folders):
            label = "spam"
        elif any(ham_folder in folder_name for ham_folder in ham_folders):
            label = "ham"
        elif any(spam_folder in parent_folder for spam_folder in spam_folders):
            label = "spam" 
        elif any(ham_folder in parent_folder for ham_folder in ham_folders):
            label = "ham"
        
        if not label:
            # Skip folders we can't classify
            continue
            
        print(f"📧 Processing {label} messages in: {root_path.name}")
        folder_count = 0
        
        for file in files:
            # Skip hidden files and directories
            if file.startswith('.') or file.startswith('__'):
                continue
                
            file_path = root_path / file
            
            # Skip if it's actually a directory
            if file_path.is_dir():
                continue
            
            text = parse_email_file(file_path)
            if text:
                all_messages.append((text, label))
                folder_count += 1
        
        if folder_count > 0:
            print(f"✅ Extracted {folder_count} {label} messages from {root_path.name}")
    
    return all_messages

def main():
    """Process existing SpamAssassin corpus and create large dataset"""
    print("="*70)
    print("PROCESSING EXISTING SPAMASSASSIN CORPUS")
    print("="*70)
    
    # Path to your downloaded corpus
    corpus_path = r"C:\Users\hanok\Downloads\archive (1)"
    
    if not os.path.exists(corpus_path):
        print(f"❌ Error: Path not found: {corpus_path}")
        return
    
    print(f"📁 Processing corpus from: {corpus_path}")
    
    # Process all messages
    all_messages = process_corpus_directory(corpus_path)
    
    if not all_messages:
        print("❌ No messages found! Check the directory structure.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_messages, columns=['text', 'label'])
    
    # Basic cleaning and stats
    print(f"\n📊 Initial messages: {len(df)}")
    
    # Remove duplicates and clean
    df = df.dropna()
    df = df.drop_duplicates(subset=['text'])
    df = df[df['text'].str.len() >= 10]  # Remove very short messages
    df = df[df['text'].str.len() <= 1000]  # Remove very long messages
    
    # Count by label
    spam_count = sum(df['label'] == 'spam')
    ham_count = sum(df['label'] == 'ham')
    
    print(f"🧹 After cleaning: {len(df)} messages")
    print(f"🚫 Spam messages: {spam_count:,} ({spam_count/len(df)*100:.1f}%)")
    print(f"✅ Ham messages: {ham_count:,} ({ham_count/len(df)*100:.1f}%)")
    
    # Save the dataset
    os.makedirs("data", exist_ok=True)
    output_path = "data/large_spamassassin_corpus.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n🎉 LARGE SPAMASSASSIN DATASET READY!")
    print(f"📁 Saved to: {output_path}")
    print(f"📈 Total messages: {len(df):,}")
    
    if len(df) > 5000:
        print(f"🚀 EXCELLENT: {len(df):,} messages should give 99%+ accuracy!")
        print(f"🔥 This is a professional-grade dataset!")
    
    # Show sample messages
    print(f"\n📝 Sample spam messages:")
    spam_samples = df[df['label'] == 'spam']['text'].head(3)
    for i, msg in enumerate(spam_samples, 1):
        print(f"{i}. {msg[:100]}...")
    
    print(f"\n📝 Sample ham messages:")
    ham_samples = df[df['label'] == 'ham']['text'].head(3)
    for i, msg in enumerate(ham_samples, 1):
        print(f"{i}. {msg[:100]}...")
    
    print(f"\n🔥 Train the SVM with this large dataset:")
    print(f"python src\\simple_svm_classifier.py --data \"{output_path}\" --model \"models\\svm_large_corpus.pkl\"")
    
    return output_path

if __name__ == "__main__":
    main()