import glob
import pandas as pd
from collections import Counter
import os
import re

files = glob.glob('/Volumes/Current_Projects2-CC0931-Comparative_Medicine/Lab_Projects/Belief_Updating_Anxiety/Reversal_Data/AX_*/AX_*_day1.csv')

print(f"Found {len(files)} files matching the pattern.")

all_data = []
sub_file_counts = Counter()
sub_trial_counts = {}
pwin_violations = []

for f in files:
    filename = os.path.basename(f)
    match = re.search(r'AX_(\w+)_day1\.csv', filename)
    if match:
        sub_id = match.group(1)
    else:
        sub_id = filename
    
    sub_file_counts[sub_id] += 1
    
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"Error reading {f}: {e}")
        continue
    
    num_trials = len(df)
    sub_trial_counts[sub_id] = sub_trial_counts.get(sub_id, 0) + num_trials
    
    if 'W1_Pwin' in df.columns:
        w1_vals = df['W1_Pwin'].dropna().unique()
        for v in w1_vals:
            if v != 0.75:
                pwin_violations.append((sub_id, 'W1_Pwin', v))
    if 'L2_Pwin' in df.columns:
        l2_vals = df['L2_Pwin'].dropna().unique()
        for v in l2_vals:
            if v != 0.75:
                pwin_violations.append((sub_id, 'L2_Pwin', v))
                
    cols_to_keep = ['Sub', 'TrialType', 'IdxWithinType', 'LeftID', 'RightID', 'ChoiceKey', 'ChoiceSide', 'ChosenID', 'ChosenPwin', 'RT', 'TrialPoints']
    existing_cols = [c for c in cols_to_keep if c in df.columns]
    missing_cols = set(cols_to_keep) - set(existing_cols)
    if missing_cols:
        print(f"Warning: missing columns {missing_cols} in file {filename}")
        
    all_data.append(df[existing_cols])

print(f"\nNumber of unique subs: {len(sub_file_counts)}")

multi_files = [sub for sub, count in sub_file_counts.items() if count > 1]
if multi_files:
    print(f"Subs with more than 1 file: {multi_files}")
else:
    print("No sub has more than 1 file.")

trial_dist = Counter(sub_trial_counts.values())
print("\nTrial count distribution:")
for trials, count in trial_dist.items():
    print(f"  {count} subs had {trials} trials.")

if trial_dist:
    most_common_trials = trial_dist.most_common(1)[0][0]
    abnormal_subs = {sub: count for sub, count in sub_trial_counts.items() if count != most_common_trials}
    if abnormal_subs:
        print(f"\nSubs with trials != {most_common_trials}:")
        for sub, count in abnormal_subs.items():
            print(f"  Sub {sub}: {count} trials")
    else:
        print(f"\nAll subs had exactly {most_common_trials} trials.")

if pwin_violations:
    print("\nViolations of W1_Pwin or L2_Pwin == 0.75:")
    unique_viols = list(set(pwin_violations))
    for v in unique_viols:
         print(f"  Sub {v[0]}: {v[1]} had value {v[2]}")
else:
    print("\nW1_Pwin and L2_Pwin are always 0.75.")

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    out_path = '/Users/nachshonkorem/Documents/AntiGrav/Anxiety/data/allsubs_day1.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    final_df.to_csv(out_path, index=False)
    print(f"\nSaved concatenated data to {out_path}")
