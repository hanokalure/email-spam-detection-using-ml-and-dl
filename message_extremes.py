import pandas as pd

df = pd.read_csv('data/large_spamassassin_corpus.csv', encoding='latin-1')
df['text_length'] = df['text'].str.len()

print('SHORTEST AND LONGEST MESSAGES')
print('=' * 60)

shortest_msg = df.loc[df['text_length'].idxmin()]
longest_msg = df.loc[df['text_length'].idxmax()]

print(f'\nSHORTEST MESSAGE ({shortest_msg["text_length"]} characters, {shortest_msg["label"]}):')
print('-' * 50)
print(f'"{shortest_msg["text"]}"')

print(f'\nLONGEST MESSAGE ({longest_msg["text_length"]} characters, {longest_msg["label"]}):')
print('-' * 50)
print(f'{longest_msg["text"][:800]}...')

# Show some very short spam and ham examples
print('\nVERY SHORT MESSAGES:')
print('-' * 30)
short_messages = df[df['text_length'] < 50].head(10)
for _, row in short_messages.iterrows():
    print(f'{row["label"]} ({len(row["text"])} chars): "{row["text"]}"')

print('\nMESSAGE LENGTH DISTRIBUTION:')
print('-' * 30)
bins = [0, 50, 100, 200, 500, 1000, 2000]
labels = ['0-50', '51-100', '101-200', '201-500', '501-1000', '1000+']
df['length_category'] = pd.cut(df['text_length'], bins=bins, labels=labels, include_lowest=True)

print('Character length distribution:')
for cat in labels:
    count = sum(df['length_category'] == cat)
    percent = count / len(df) * 100
    print(f'{cat:>10} chars: {count:>4} messages ({percent:>5.1f}%)')