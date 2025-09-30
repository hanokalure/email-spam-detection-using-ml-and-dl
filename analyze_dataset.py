import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('data/large_spamassassin_corpus.csv', encoding='latin-1')

print('DATASET OVERVIEW')
print('=' * 50)
print(f'Total messages: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
print()

# Show label distribution
print('LABEL DISTRIBUTION')
print('=' * 50)
label_counts = df['label'].value_counts()
print(f'Ham (legitimate): {label_counts["ham"]} ({label_counts["ham"]/len(df)*100:.1f}%)')
print(f'Spam: {label_counts["spam"]} ({label_counts["spam"]/len(df)*100:.1f}%)')
print()

# Show sample spam messages
print('SAMPLE SPAM MESSAGES')
print('=' * 50)
spam_messages = df[df['label'] == 'spam']['text'].head(5)
for i, msg in enumerate(spam_messages, 1):
    print(f'{i}. {msg[:300]}...')
    print('-' * 80)

print('\nSAMPLE HAM (LEGITIMATE) MESSAGES')
print('=' * 50)
ham_messages = df[df['label'] == 'ham']['text'].head(5)
for i, msg in enumerate(ham_messages, 1):
    print(f'{i}. {msg[:300]}...')
    print('-' * 80)

# Show message length statistics
print('\nMESSAGE LENGTH STATISTICS')
print('=' * 50)
df['text_length'] = df['text'].str.len()
print(f"Average message length: {df['text_length'].mean():.0f} characters")
print(f"Median message length: {df['text_length'].median():.0f} characters")
print(f"Shortest message: {df['text_length'].min()} characters")
print(f"Longest message: {df['text_length'].max()} characters")
print()

# Show length comparison between spam and ham
spam_lengths = df[df['label'] == 'spam']['text_length']
ham_lengths = df[df['label'] == 'ham']['text_length']

print('LENGTH COMPARISON')
print('=' * 50)
print(f"Average spam length: {spam_lengths.mean():.0f} characters")
print(f"Average ham length: {ham_lengths.mean():.0f} characters")
print()

# Show some random examples from both categories
print('RANDOM SPAM EXAMPLES')
print('=' * 50)
random_spam = df[df['label'] == 'spam'].sample(3, random_state=42)
for i, (_, row) in enumerate(random_spam.iterrows(), 1):
    print(f'{i}. {row["text"][:200]}...')
    print('-' * 80)

print('\nRANDOM HAM EXAMPLES')  
print('=' * 50)
random_ham = df[df['label'] == 'ham'].sample(3, random_state=42)
for i, (_, row) in enumerate(random_ham.iterrows(), 1):
    print(f'{i}. {row["text"][:200]}...')
    print('-' * 80)