import os
import re
import pandas as pd

base = r'C:\Users\hp\.gemini\antigravity\scratch\SENTINEL-AI\backend\ml\datasets'
inc_path = os.path.join(base, 'new_sentiment_incoming.tsv')
inc_df = pd.read_csv(inc_path, sep='\t')

print(f'=== INCOMING DATASET AUDIT ===')
print(f'Total incoming rows: {len(inc_df)}')
print('Columns:', inc_df.columns.tolist())

print('\n[1] Sentiment Distribution:')
print(inc_df['sentiment'].value_counts(dropna=False))

print('\n[2] Language Distribution:')
print(inc_df['language'].value_counts(dropna=False))

print('\n[3] Source Distribution:')
print(inc_df['source'].value_counts(dropna=False))

print('\n[4] Domain Distribution:')
print(inc_df['domain'].value_counts(dropna=False))

# Check nulls and short texts
null_texts = inc_df['text'].isnull().sum()
short_texts = inc_df[inc_df['text'].fillna('').str.strip().str.len() < 5]
print(f'\n[5] Null texts: {null_texts}')
print(f'Short texts (<5 chars): {len(short_texts)}')

# Internal duplicates
internal_dup_count = inc_df.duplicated(subset=['text']).sum()
print(f'\n[6] Internal exact text duplicates: {internal_dup_count}')

# Load existing datasets
test_en = pd.read_csv(os.path.join(base, 'test_sentiment.csv'))
test_hi = pd.read_csv(os.path.join(base, 'test_hinglish_sentiment.csv'))
train_en = pd.read_csv(os.path.join(base, 'train_sentiment.csv'))
train_hi = pd.read_csv(os.path.join(base, 'train_hinglish_sentiment.csv'))

def norm(t):
    return re.sub(r'\s+', ' ', str(t).strip().lower())

inc_df['text_norm'] = inc_df['text'].apply(norm)
test_en['text_norm'] = test_en['text'].apply(norm)
test_hi['text_norm'] = test_hi['text'].apply(norm)
train_en['text_norm'] = train_en['text'].apply(norm)
train_hi['text_norm'] = train_hi['text'].apply(norm)

test_en_norm = set(test_en['text_norm'])
test_hi_norm = set(test_hi['text_norm'])
train_en_norm = set(train_en['text_norm'])
train_hi_norm = set(train_hi['text_norm'])

overlap_test_en = inc_df[inc_df['text_norm'].isin(test_en_norm)]
overlap_test_hi = inc_df[inc_df['text_norm'].isin(test_hi_norm)]
overlap_train_en = inc_df[inc_df['text_norm'].isin(train_en_norm)]
overlap_train_hi = inc_df[inc_df['text_norm'].isin(train_hi_norm)]

print('\n=== ZERO-DATA-LEAKAGE AUDIT ===')
print(f'Overlap with test_sentiment.csv (English Holdout Test Set): {len(overlap_test_en)}')
for _, r in overlap_test_en.iterrows():
    print(f'  [LEAKAGE DETECTED & MUST BE PURGED] ID: {r["id"]} | Label: {r["sentiment"]} | Text: {r["text"]}')

print(f'\nOverlap with test_hinglish_sentiment.csv (Hinglish Holdout Test Set): {len(overlap_test_hi)}')
for _, r in overlap_test_hi.iterrows():
    print(f'  [LEAKAGE DETECTED & MUST BE PURGED] ID: {r["id"]} | Label: {r["sentiment"]} | Text: {r["text"]}')

print(f'\nOverlap with train_sentiment.csv (English Train): {len(overlap_train_en)}')
for _, r in overlap_train_en.iterrows():
    print(f'  [EXISTING EN TRAIN] ID: {r["id"]} | Text: {r["text"]}')

print(f'\nOverlap with train_hinglish_sentiment.csv (Hinglish Train): {len(overlap_train_hi)}')
for _, r in overlap_train_hi.iterrows():
    print(f'  [EXISTING HI TRAIN] ID: {r["id"]} | Text: {r["text"]}')
