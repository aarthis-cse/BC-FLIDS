import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# BASE PATH — your dataset folder
# ─────────────────────────────────────────────
base = r'E:\Journals\Journal 2026\July 26\IoV 14.7.26\Dataset'

# ─────────────────────────────────────────────
# STEP 1 — Load CICIDS2018 (3 selected files)
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 1: Loading CICIDS2018 dataset...")
print("=" * 50)

ids2018_files = [
    r'archive\02-14-2018.csv',
    r'archive\02-22-2018.csv',
    r'archive\03-02-2018.csv'
]

df2018_list = []
for f in ids2018_files:
    path = os.path.join(base, f)
    print(f"  Reading: {f}")
    df_temp = pd.read_csv(path, low_memory=False)
    df_temp.columns = df_temp.columns.str.strip()
    df2018_list.append(df_temp)

df2018 = pd.concat(df2018_list, ignore_index=True)
df2018['dataset_source'] = 'CICIDS2018'
print(f"  CICIDS2018 loaded: {df2018.shape[0]:,} rows | {df2018.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 2 — Load CICIDS2017 (3 selected files)
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 2: Loading CICIDS2017 dataset...")
print("=" * 50)

ids2017_files = [
    r'archive (1)\Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv',
    r'archive (1)\Tuesday-WorkingHours.pcap_ISCX.csv',
    r'archive (1)\Wednesday-workingHours.pcap_ISCX.csv'
]

df2017_list = []
for f in ids2017_files:
    path = os.path.join(base, f)
    print(f"  Reading: {os.path.basename(f)}")
    df_temp = pd.read_csv(path, low_memory=False)
    df_temp.columns = df_temp.columns.str.strip()
    df2017_list.append(df_temp)

df2017 = pd.concat(df2017_list, ignore_index=True)
df2017['dataset_source'] = 'CICIDS2017'
print(f"  CICIDS2017 loaded: {df2017.shape[0]:,} rows | {df2017.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 3 — Load UNSW-NB15 (all 4 parts)
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 3: Loading UNSW-NB15 dataset...")
print("=" * 50)

unsw_files = [
    r'OneDrive_1_14-07-2026\CSV Files\UNSW-NB15_1.csv',
    r'OneDrive_1_14-07-2026\CSV Files\UNSW-NB15_2.csv',
    r'OneDrive_1_14-07-2026\CSV Files\UNSW-NB15_3.csv',
    r'OneDrive_1_14-07-2026\CSV Files\UNSW-NB15_4.csv'
]

# UNSW-NB15 has no header row, so we add column names manually
unsw_columns = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur',
    'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service',
    'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb',
    'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
    'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt',
    'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl',
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src',
    'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label'
]

unsw_list = []
for f in unsw_files:
    path = os.path.join(base, f)
    print(f"  Reading: {os.path.basename(f)}")
    df_temp = pd.read_csv(path, low_memory=False, header=None, names=unsw_columns)
    unsw_list.append(df_temp)

df_unsw = pd.concat(unsw_list, ignore_index=True)
df_unsw['dataset_source'] = 'UNSW-NB15'
print(f"  UNSW-NB15 loaded: {df_unsw.shape[0]:,} rows | {df_unsw.shape[1]} columns\n")

# ─────────────────────────────────────────────
# STEP 4 — Find Common Columns
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 4: Finding common columns across datasets...")
print("=" * 50)

cols2018 = set(df2018.columns)
cols2017 = set(df2017.columns)
cols_unsw = set(df_unsw.columns)

# Common between 2017 and 2018 (they share 80 CICFlowMeter features)
common_cic = cols2018.intersection(cols2017)
print(f"  Common columns between CICIDS2017 and CICIDS2018: {len(common_cic)}")

# Add dataset_source and Label to keep
keep_cols = list(common_cic) + ['dataset_source']
keep_cols = list(dict.fromkeys(keep_cols))  # remove duplicates

# ─────────────────────────────────────────────
# STEP 5 — Standardise Label Column
# ─────────────────────────────────────────────
print("\nSTEP 5: Standardising label columns...")

# CICIDS2017 and 2018 use ' Label' (with space) — already stripped above
# Rename to standard 'Label'
if 'Label' in df2018.columns:
    df2018 = df2018.rename(columns={'Label': 'Label'})
if 'Label' in df2017.columns:
    df2017 = df2017.rename(columns={'Label': 'Label'})

# UNSW-NB15: use 'Label' column (0 = benign, 1 = attack) and attack_cat
df_unsw['Label'] = df_unsw['Label'].astype(str).replace({'0': 'Benign', '1': 'Attack'})

print("  Labels standardised.\n")

# ─────────────────────────────────────────────
# STEP 6 — Merge All Three Datasets
# ─────────────────────────────────────────────
print("=" * 50)
print("STEP 6: Merging all three datasets...")
print("=" * 50)

df2018_slim = df2018[keep_cols] if all(c in df2018.columns for c in keep_cols) else df2018[list(common_cic) + ['dataset_source']]
df2017_slim = df2017[keep_cols] if all(c in df2017.columns for c in keep_cols) else df2017[list(common_cic) + ['dataset_source']]

# For UNSW keep only Label and dataset_source to align
unsw_slim_cols = [c for c in keep_cols if c in df_unsw.columns]
df_unsw_slim = df_unsw[unsw_slim_cols]

# Stack all three
merged = pd.concat([df2018_slim, df2017_slim, df_unsw_slim], ignore_index=True, sort=False)

print(f"  Total merged rows : {merged.shape[0]:,}")
print(f"  Total columns     : {merged.shape[1]}")

# ─────────────────────────────────────────────
# STEP 7 — Clean Up
# ─────────────────────────────────────────────
print("\nSTEP 7: Cleaning merged dataset...")

# Drop rows where all values are NaN
merged.dropna(how='all', inplace=True)

# Drop duplicate rows
before = merged.shape[0]
merged.drop_duplicates(inplace=True)
after = merged.shape[0]
print(f"  Duplicates removed: {before - after:,} rows")
print(f"  Final clean rows  : {after:,}")

# ─────────────────────────────────────────────
# STEP 8 — Save Output
# ─────────────────────────────────────────────
print("\nSTEP 8: Saving merged dataset...")

output_path = os.path.join(base, 'IoV_SecFusion_Dataset.csv')
merged.to_csv(output_path, index=False)

print(f"\n{'=' * 50}")
print("MERGE COMPLETE!")
print(f"{'=' * 50}")
print(f"  Saved at : {output_path}")
print(f"  Rows     : {merged.shape[0]:,}")
print(f"  Columns  : {merged.shape[1]}")
print(f"\n  Dataset source breakdown:")
print(merged['dataset_source'].value_counts().to_string())
print("=" * 50)
