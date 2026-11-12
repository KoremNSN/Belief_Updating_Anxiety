import glob
import pandas as pd
import os

base_ass = '/Volumes/Current_Projects2-CC0931-Comparative_Medicine/Lab_Projects/Belief_Updating_Anxiety/Assessment Data'
files = [
    f'{base_ass}/AX_34/AX_34_day1.csv',
    f'{base_ass}/AX_58/AX_58_day1.csv',
    f'{base_ass}/AX_79/AX_79_day1.csv',
]

EXCLUDED_SUBS = {'1', '2', '3', '4', '5', '6', '26', '40', '50', '53', '78', '79'}

mapping = {
    'Not at all': 0,
    'Several days': 1,
    'More than half the days': 2,
    'Nearly every day': 3
}

# Collect all unique responses by ResponseId across Day 1 files
responses = {}
for f in sorted(files):
    try:
        df = pd.read_csv(f, skiprows=[1, 2], encoding_errors='replace')
        for idx, row in df.iterrows():
            rid = row.get('ResponseId')
            if pd.isna(rid):
                continue
            responses[rid] = row
    except Exception as e:
        print(f"Error processing {f}: {e}")

results = []
for rid, row in responses.items():
    if pd.isna(row.get('ID')):
        continue

    raw_id = str(row['ID']).strip()

    # Drop AX_11 first attempt, keep AX_11 (2) as 11
    if raw_id == 'AX_11':
        continue
    if raw_id == 'AX_11 (2)':
        sub_id = '11'
    elif raw_id == 'AX2':
        sub_id = '2'
    else:
        sub_id = raw_id.replace('AX_', '').strip()

    # Keep only numeric subject IDs in 1..79 that are not excluded
    if not sub_id.isdigit() or not (1 <= int(sub_id) <= 79):
        continue
    if sub_id in EXCLUDED_SUBS:
        continue

    gad_sum = 0
    valid_gad = True
    for i in range(1, 8):
        col = f'Pre GAD-7_{i}'
        if col in row and not pd.isna(row[col]):
            val_str = str(row[col]).strip()
            if val_str in mapping:
                gad_sum += mapping[val_str]
            else:
                valid_gad = False
                break
        else:
            valid_gad = False
            break

    if valid_gad:
        results.append({'Sub': sub_id, 'GAD7': gad_sum})

res_df = pd.DataFrame(results)

if not res_df.empty:
    res_df['sub_num'] = pd.to_numeric(res_df['Sub'], errors='coerce')
    res_df = res_df.sort_values('sub_num').drop(columns=['sub_num'])
    res_df = res_df.drop_duplicates(subset=['Sub'], keep='first')

out_path = '/Users/nachshonkorem/Documents/AntiGrav/Anxiety/data/survey.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
res_df.to_csv(out_path, index=False)
print(f"Saved {len(res_df)} survey results to {out_path}")
if not res_df.empty:
    print(res_df.head())
