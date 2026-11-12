import glob
import pandas as pd
import os

base_ass = '/Volumes/Current_Projects2-CC0931-Comparative_Medicine/Lab_Projects/Belief_Updating_Anxiety/Assessment Data'
files_day1 = [
    f'{base_ass}/AX_34/AX_34_day1.csv',
    f'{base_ass}/AX_58/AX_58_day1.csv',
    f'{base_ass}/AX_79/AX_79_day1.csv',
]

EXCLUDED_SUBS = {'1', '2', '3', '4', '5', '6', '26', '40', '50', '53', '78', '79'}

# Use the most complete cumulative Day 1 and Day 2 files, plus any extra ResponseIds
d1_rows = {}
for f in sorted(files_day1):
    try:
        df = pd.read_csv(f, skiprows=[1, 2], encoding_errors='replace')
        for _, row in df.iterrows():
            rid = row.get('ResponseId')
            if not pd.isna(rid):
                d1_rows[rid] = row
    except Exception as e:
        print(f"Error reading Day 1 file {f}: {e}")

# Use AX_77_day2.csv as the canonical Day 2 table so raw Qualtrics IDs are preserved prior to explicit rules
canonical_d2 = '/Volumes/Current_Projects2-CC0931-Comparative_Medicine/Lab_Projects/Belief_Updating_Anxiety/Assessment Data/AX_77/AX_77_day2.csv'
df_d2_raw = pd.read_csv(canonical_d2, skiprows=[1, 2], encoding_errors='replace')

results_day1 = {}
for rid, row in d1_rows.items():
    if pd.isna(row.get('ID')):
        continue
    raw_id = str(row['ID']).strip()

    # AX_11 rule: drop first attempt ('AX_11'), keep 'AX_11 (2)' as '11'
    if raw_id == 'AX_11':
        continue
    elif raw_id == 'AX_11 (2)':
        sub_id = '11'
    elif raw_id == 'AX2':
        sub_id = '2'
    else:
        sub_id = raw_id.replace('AX_', '').strip()

    if not sub_id.isdigit() or not (1 <= int(sub_id) <= 79):
        continue
    if sub_id in EXCLUDED_SUBS:
        continue

    pre_text = row.get('Pre Free Recall', '')
    post_text = row.get('Post Free Recall', '')
    pre_text = "" if pd.isna(pre_text) else str(pre_text).strip()
    post_text = "" if pd.isna(post_text) else str(post_text).strip()

    if sub_id not in results_day1:
        results_day1[sub_id] = {'Sub': sub_id, 'pre': pre_text, 'post': post_text}

results_day2 = {}
post_overrides_day2 = {}

for _, row in df_d2_raw.iterrows():
    if pd.isna(row.get('ID')):
        continue
    raw_id = str(row['ID']).strip()
    rid = str(row.get('ResponseId', '')).strip()

    day2_pre_text = row.get('Free Recall', '')
    day2_post_text = row.get('Post Free Recall', '')
    day2_pre_text = "" if pd.isna(day2_pre_text) else str(day2_pre_text).strip()
    day2_post_text = "" if pd.isna(day2_post_text) else str(day2_post_text).strip()

    # 1. AX_10 on Day 2: drop first AX_10, keep AX_10 (2) as '10'
    if raw_id == 'AX_10':
        continue
    elif raw_id == 'AX_10 (2)':
        sub_id = '10'
    # 2. AX_30_2 and AX_49 (2): store their Post Free Recall to overwrite AX_30 and AX_49
    elif raw_id == 'AX_30_2':
        post_overrides_day2['30'] = day2_post_text
        continue
    elif raw_id == 'AX_49 (2)':
        post_overrides_day2['49'] = day2_post_text
        continue
    # 3. AX_39 unfinished attempt (Progress=60, R_3oz4ciDMh2YcCw4): skip
    elif raw_id == 'AX_39' and rid == 'R_3oz4ciDMh2YcCw4':
        continue
    # 4. AX_52 first response (07:25, R_6cqnpVHFOgifuHC) -> change to AX_51 ('51'); second response (11:47, R_1tAGPyV5nXOcw25) -> keep as '52'
    elif raw_id == 'AX_52' and rid == 'R_6cqnpVHFOgifuHC':
        sub_id = '51'
    elif raw_id == 'AX_52' and rid == 'R_1tAGPyV5nXOcw25':
        sub_id = '52'
    else:
        sub_id = raw_id.replace('AX_', '').strip()

    if not sub_id.isdigit() or not (1 <= int(sub_id) <= 79):
        continue
    if sub_id in EXCLUDED_SUBS:
        continue

    if sub_id not in results_day2:
        results_day2[sub_id] = {
            'Sub': sub_id,
            'day2_pre': day2_pre_text,
            'day2_post': day2_post_text
        }

for sub_id, override_post in post_overrides_day2.items():
    if sub_id in results_day2 and override_post:
        results_day2[sub_id]['day2_post'] = override_post

df_day1 = pd.DataFrame(list(results_day1.values()))
df_day2 = pd.DataFrame(list(results_day2.values()))

res_df = pd.merge(df_day1, df_day2, on='Sub', how='outer').fillna("")
res_df['sub_num'] = pd.to_numeric(res_df['Sub'], errors='coerce')
res_df = res_df.sort_values('sub_num').drop(columns=['sub_num'])

out_path = '/Users/nachshonkorem/Documents/AntiGrav/Anxiety/data/free_recall.csv'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
res_df.to_csv(out_path, index=False)
print(f"Saved {len(res_df)} free recall responses to {out_path}")
print(res_df.head())
