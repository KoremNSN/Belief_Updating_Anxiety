import pandas as pd
import os

day1_path = '/Users/nachshonkorem/Documents/AntiGrav/Anxiety/data/allsubs_day1.csv'
survey_path = '/Users/nachshonkorem/Documents/AntiGrav/Anxiety/data/survey.csv'
day2_path = '/Users/nachshonkorem/Documents/AntiGrav/Anxiety/data/allsubs_day2.csv'

df1 = pd.read_csv(day1_path)
df_survey = pd.read_csv(survey_path)
df2 = pd.read_csv(day2_path)

df1['Sub'] = df1['Sub'].astype(str)
df_survey['Sub'] = df_survey['Sub'].astype(str)
df2['sub'] = df2['sub'].astype(str)

df2 = df2.rename(columns={'sub': 'Sub'})

df2_subset = df2[['Sub', 'Win', 'Lose', 'Correct_Win_ID', 'Correct_Lose_ID']].copy()
df2_subset = df2_subset.rename(columns={
    'Win': 'Day2_Win_Success',
    'Lose': 'Day2_Lose_Success',
    'Correct_Win_ID': 'Day2_Correct_Win_ID',
    'Correct_Lose_ID': 'Day2_Correct_Lose_ID'
})

merged_df = pd.merge(df1, df_survey[['Sub', 'GAD7']], on='Sub', how='left')
merged_df = pd.merge(merged_df, df2_subset, on='Sub', how='left')

unique_subs = merged_df[['Sub', 'GAD7', 'Day2_Win_Success']].drop_duplicates()

missing_gad = unique_subs[unique_subs['GAD7'].isna()]['Sub'].tolist()
missing_day2 = unique_subs[unique_subs['Day2_Win_Success'].isna()]['Sub'].tolist()

if missing_gad:
    print(f"Subjects from Day 1 missing GAD7 score: {', '.join(missing_gad)}")
else:
    print("No subjects from Day 1 are missing GAD7 scores.")

if missing_day2:
    print(f"Subjects from Day 1 missing Day 2 success rates: {', '.join(missing_day2)}")
else:
    print("No subjects from Day 1 are missing Day 2 success rates.")

out_path = '/Users/nachshonkorem/Documents/AntiGrav/Anxiety/data/allsubs_combined.csv'
merged_df.to_csv(out_path, index=False)
print(f"\nSaved combined data (Trial-by-Trial) to {out_path}")
