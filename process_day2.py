import glob
import pandas as pd
import os
import re

files = glob.glob('/Volumes/Current_Projects2-CC0931-Comparative_Medicine/Lab_Projects/Belief_Updating_Anxiety/Reversal_Data/AX_*/AX_*_day2.csv')

results = []

for f in files:
    filename = os.path.basename(f)
    match = re.search(r'AX_(\w+)_day2\.csv', filename)
    if match:
        sub_id = match.group(1)
    else:
        sub_id = filename
        
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue
        
    valid_choices = df.dropna(subset=['ChosenID'])
    
    sub_res = {'sub': sub_id, 'Win': None, 'Lose': None}
    
    for t_type in ['W', 'L']:
        type_df = valid_choices[valid_choices['TrialType'] == t_type]
        denominator = len(type_df)
        if denominator > 0:
            numerator = type_df['ChosenID'].astype(str).str.contains('1').sum()
            pct = numerator / denominator
            if t_type == 'W':
                sub_res['Win'] = pct
            else:
                sub_res['Lose'] = pct
                
    results.append(sub_res)

res_df = pd.DataFrame(results)
res_df['sub_num'] = pd.to_numeric(res_df['sub'], errors='coerce')
res_df = res_df.sort_values('sub_num').drop(columns=['sub_num'])

out_path = '/Users/nachshonkorem/Documents/AntiGrav/Anxiety/data/allsubs_day2.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
res_df.to_csv(out_path, index=False)
print(f"Saved results for {len(res_df)} subjects to {out_path}")
print(res_df.head())
