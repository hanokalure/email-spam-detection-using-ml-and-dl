import pandas as pd

df = pd.read_csv('data/large_spamassassin_corpus.csv', encoding='latin-1')

print('DETAILED SPAM MESSAGE EXAMPLES')
print('=' * 70)

spam_df = df[df['label'] == 'spam']
ham_df = df[df['label'] == 'ham']

# Show different types of spam
print('\nFINANCIAL/MONEY SPAM:')
print('-' * 40)
financial_keywords = ['money', 'cash', 'loan', 'mortgage', 'credit', 'debt', 'income']
financial_spam = spam_df[spam_df['text'].str.contains('|'.join(financial_keywords), case=False, na=False)].head(3)
for i, (_, row) in enumerate(financial_spam.iterrows(), 1):
    print(f'{i}. {row["text"][:350]}...\n')

print('PRODUCT/OFFER SPAM:')
print('-' * 40)
offer_keywords = ['free', 'offer', 'deal', 'buy', 'sale', 'discount', 'win']
offer_spam = spam_df[spam_df['text'].str.contains('|'.join(offer_keywords), case=False, na=False)].head(3)
for i, (_, row) in enumerate(offer_spam.iterrows(), 1):
    print(f'{i}. {row["text"][:350]}...\n')

print('HEALTH/MEDICAL SPAM:')
print('-' * 40)
health_keywords = ['weight', 'fat', 'lose', 'pill', 'health', 'medical', 'drug']
health_spam = spam_df[spam_df['text'].str.contains('|'.join(health_keywords), case=False, na=False)].head(3)
for i, (_, row) in enumerate(health_spam.iterrows(), 1):
    print(f'{i}. {row["text"][:350]}...\n')

print('TYPES OF HAM (LEGITIMATE) MESSAGES:')
print('=' * 70)

print('\nEMAIL DISCUSSIONS/FORUMS:')
print('-' * 40)
forum_ham = ham_df[ham_df['text'].str.contains('wrote:|posted:|From:|Date:', case=False, na=False)].head(2)
for i, (_, row) in enumerate(forum_ham.iterrows(), 1):
    print(f'{i}. {row["text"][:350]}...\n')

print('NEWS/ARTICLES:')
print('-' * 40)
news_ham = ham_df[ham_df['text'].str.contains('news|article|report|press', case=False, na=False)].head(2)
for i, (_, row) in enumerate(news_ham.iterrows(), 1):
    print(f'{i}. {row["text"][:350]}...\n')

print('TECHNICAL/PROGRAMMING DISCUSSIONS:')
print('-' * 40)
tech_ham = ham_df[ham_df['text'].str.contains('code|function|program|error|bug|software', case=False, na=False)].head(2)
for i, (_, row) in enumerate(tech_ham.iterrows(), 1):
    print(f'{i}. {row["text"][:350]}...\n')

print('MESSAGE CHARACTERISTICS SUMMARY:')
print('=' * 70)
print(f'This dataset contains email messages from the SpamAssassin corpus.')
print(f'It includes {len(spam_df)} spam messages and {len(ham_df)} legitimate (ham) messages.')
print(f'The messages appear to be from mailing lists, forums, and personal emails.')
print(f'Spam messages typically contain promotional content, financial offers, and product advertisements.')
print(f'Ham messages include technical discussions, news articles, and normal email conversations.')