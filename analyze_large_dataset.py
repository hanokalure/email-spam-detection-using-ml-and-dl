import pandas as pd

# Load the dataset
df = pd.read_csv('data/spam_assassin_large.csv', encoding='latin-1')

print('LARGE SPAMASSASSIN DATASET ANALYSIS')
print('=' * 50)
print(f'Total messages: {len(df)}')
print(f'Columns: {df.columns.tolist()}')
print(f'File size: 23.4 MB')

print('\nTARGET DISTRIBUTION:')
print('-' * 30)
target_counts = df['target'].value_counts()
for target, count in target_counts.items():
    label_name = "HAM" if target == 0 else "SPAM"
    print(f'{label_name} (target={target}): {count} ({count/len(df)*100:.1f}%)')

print('\nSAMPLE MESSAGES:')
print('-' * 30)
ham_sample = df[df['target']==0]['text'].iloc[0]
spam_sample = df[df['target']==1]['text'].iloc[0]

print('HAM example:')
print(f'  {ham_sample[:300]}...')
print()
print('SPAM example:') 
print(f'  {spam_sample[:300]}...')

print('\nMESSAGE LENGTH ANALYSIS:')
print('-' * 30)
df['text_length'] = df['text'].str.len()
print(f'Average message length: {df["text_length"].mean():.0f} characters')
print(f'Median message length: {df["text_length"].median():.0f} characters')
print(f'Shortest message: {df["text_length"].min()} characters')
print(f'Longest message: {df["text_length"].max()} characters')

# Compare with your current dataset
print('\nCOMPARISON WITH CURRENT DATASET:')
print('-' * 40)
print(f'Current dataset: 3,790 messages')
print(f'Large dataset: {len(df)} messages')
print(f'Improvement: {len(df)/3790:.1f}x larger dataset!')