import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

analyzer = SentimentIntensityAnalyzer()

# Load the free recall text
file_path = '/Users/nachshonkorem/Documents/AntiGrav/Anxiety/data/free_recall.csv'
df = pd.read_csv(file_path)

# Initialize output lists
results = []

for idx, row in df.iterrows():
    sub = row['Sub']
    scores = {'Sub': sub}
    
    column_mapping = [
        ('pre', 'day_1_pre'),
        ('post', 'day_1_post'),
        ('day2_pre', 'day_2_pre'),
        ('day2_post', 'day_2_post')
    ]
    
    for col, prefix in column_mapping:
        val = row.get(col)
        text = str(val) if not pd.isna(val) else ""
        if text.strip():
            pol_scores = analyzer.polarity_scores(text)
            scores[f'{prefix}_pos'] = pol_scores['pos']
            scores[f'{prefix}_neg'] = pol_scores['neg']
            scores[f'{prefix}_compound'] = pol_scores['compound']
        else:
            scores[f'{prefix}_pos'] = pd.NA
            scores[f'{prefix}_neg'] = pd.NA
            scores[f'{prefix}_compound'] = pd.NA
            
    results.append(scores)

# Convert to DataFrame
out_df = pd.DataFrame(results)

# Save to new CSV
out_path = '/Users/nachshonkorem/Documents/AntiGrav/Anxiety/data/free_recall_nlp.csv'
out_df.to_csv(out_path, index=False)

print(f"Saved NLP sentiment results to {out_path}")
print(out_df.head(10))
