import glob
import pandas as pd
import os
import re

base_dir = '/Volumes/Current_Projects2-CC0931-Comparative_Medicine/Lab_Projects/Belief_Updating_Anxiety/Reversal_Data'
sub_dirs = sorted(glob.glob(f'{base_dir}/AX_*'))

# Excluded subjects:
# Reversal on Day 1: 1, 3, 4, 5, 6
# User exclusions: 2, 26, 40, 50, 53
# Incomplete (no Day 2 yet): 78, 79
EXCLUDED_SUBS = {'1', '2', '3', '4', '5', '6', '26', '40', '50', '53', '78', '79'}

results = []

for d in sub_dirs:
    folder = os.path.basename(d)
    match = re.search(r'AX_(\w+)', folder)
    if not match:
        continue
    sub_id = match.group(1)

    if sub_id in EXCLUDED_SUBS:
        continue

    f1 = os.path.join(d, f'AX_{sub_id}_day1.csv')
    f2 = os.path.join(d, f'AX_{sub_id}_day2.csv')

    # Handle AX_47 where Day 2 file is named AX_46_day2.csv inside AX_47 folder
    if sub_id == '47' and not os.path.exists(f2):
        alt_f2 = os.path.join(d, 'AX_46_day2.csv')
        if os.path.exists(alt_f2):
            f2 = alt_f2

    if not os.path.exists(f1) or not os.path.exists(f2):
        print(f"Skipping AX_{sub_id}: missing file (day1={os.path.exists(f1)}, day2={os.path.exists(f2)})")
        continue

    try:
        df1 = pd.read_csv(f1)
        df2 = pd.read_csv(f2)
    except Exception as e:
        print(f"Error reading {sub_id}: {e}")
        continue

    # Determine correct choice from the END of Day 1
    win_trials = df1[df1['TrialType'] == 'W'].dropna(subset=['W1_Pwin', 'W2_Pwin'])
    if not win_trials.empty:
        last_win = win_trials.iloc[-1]
        correct_win_id = 'W1' if last_win['W1_Pwin'] > 0.5 else 'W2'
    else:
        correct_win_id = 'W1'

    lose_trials = df1[df1['TrialType'] == 'L'].dropna(subset=['L1_Pwin', 'L2_Pwin'])
    if not lose_trials.empty:
        last_lose = lose_trials.iloc[-1]
        correct_lose_id = 'L1' if last_lose['L1_Pwin'] > 0.5 else 'L2'
    else:
        correct_lose_id = 'L2'

    valid_choices = df2.dropna(subset=['ChosenID'])

    w_trials = valid_choices[valid_choices['TrialType'] == 'W']
    l_trials = valid_choices[valid_choices['TrialType'] == 'L']

    w_pct = (w_trials['ChosenID'] == correct_win_id).mean() if len(w_trials) > 0 else None
    l_pct = (l_trials['ChosenID'] == correct_lose_id).mean() if len(l_trials) > 0 else None

    results.append({
        'sub': sub_id,
        'Win': w_pct,
        'Lose': l_pct,
        'Correct_Win_ID': correct_win_id,
        'Correct_Lose_ID': correct_lose_id
    })

res_df = pd.DataFrame(results)
res_df['sub_num'] = pd.to_numeric(res_df['sub'], errors='coerce')
res_df = res_df.sort_values('sub_num').drop(columns=['sub_num'])
res_df = res_df[['sub', 'Win', 'Lose', 'Correct_Win_ID', 'Correct_Lose_ID']]

out_path = '/Users/nachshonkorem/Documents/AntiGrav/Anxiety/data/allsubs_day2.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
res_df.to_csv(out_path, index=False)
print(f"Saved results for {len(res_df)} subjects to {out_path}")
print(res_df.head())
