# ============================================================
# Label Cleaner — Fix before running bc_flids_finetuned.py
# Fixes: duplicate label names, tiny classes, label encoding
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

dataset_path = r"E:\Journals\Journal 2026\July 26\IoV 14.7.26\Dataset\IoV_SecFusion_Dataset.csv"
output_path  = r"E:\Journals\Journal 2026\July 26\IoV 14.7.26\Dataset\IoV_SecFusion_Cleaned.csv"

print("=" * 55)
print("LABEL CLEANING SCRIPT")
print("=" * 55)

# ── Load ────────────────────────────────────────────────────
print("\n[1] Loading dataset...")
df = pd.read_csv(dataset_path, low_memory=False)
df.columns = df.columns.str.strip()
print(f"    Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")

# ── Find label column ────────────────────────────────────────
label_col = None
for lc in ['Label', 'label', ' Label']:
    if lc in df.columns:
        label_col = lc
        break
print(f"    Label column: '{label_col}'")

# ── Step 1: Strip whitespace and fix case ───────────────────
print("\n[2] Fixing label formatting...")
df[label_col] = df[label_col].astype(str).str.strip()

# ── Step 2: Merge duplicate/similar labels ──────────────────
print("\n[3] Merging duplicate labels...")

label_mapping = {
    # Benign variants → unified 'Benign'
    'BENIGN'            : 'Benign',
    'benign'            : 'Benign',
    'Attack'            : 'Benign',   # UNSW-NB15 'Attack'=0 means benign in context

    # Brute force variants
    'FTP-Patator'       : 'FTP-BruteForce',
    'SSH-Patator'       : 'SSH-BruteForce',

    # DoS variants — keep as-is (enough samples)
    'DoS GoldenEye'     : 'DoS-GoldenEye',
    'DoS Hulk'          : 'DoS-Hulk',
    'DoS Slowhttptest'  : 'DoS-Slowhttptest',
    'DoS slowloris'     : 'DoS-Slowloris',

    # Web attacks
    'Brute Force -Web'  : 'WebAttack-BruteForce',
    'Brute Force -XSS'  : 'WebAttack-XSS',
}

before_classes = df[label_col].nunique()
df[label_col] = df[label_col].replace(label_mapping)
after_classes  = df[label_col].nunique()
print(f"    Before: {before_classes} classes → After: {after_classes} classes")

# ── Step 3: Show class distribution ─────────────────────────
print("\n[4] Class distribution after cleaning:")
vc = df[label_col].value_counts()
for cls, count in vc.items():
    pct = count / len(df) * 100
    flag = " ⚠ TOO FEW" if count < 10 else ""
    print(f"    {cls:<30} {count:>8,}  ({pct:.2f}%){flag}")

# ── Step 4: Remove classes with less than 10 samples ────────
print("\n[5] Removing classes with fewer than 10 samples...")
min_samples   = 10
class_counts  = df[label_col].value_counts()
valid_classes = class_counts[class_counts >= min_samples].index
removed       = class_counts[class_counts < min_samples]

if len(removed) > 0:
    print(f"    Removing {len(removed)} class(es):")
    for cls, cnt in removed.items():
        print(f"      '{cls}' — only {cnt} sample(s)")
    df = df[df[label_col].isin(valid_classes)]
else:
    print("    All classes have enough samples — nothing removed.")

print(f"    Remaining rows: {df.shape[0]:,}")
print(f"    Remaining classes: {df[label_col].nunique()}")

# ── Step 5: Fix missing and infinite values ──────────────────
print("\n[6] Fixing missing and infinite values...")
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print(f"    Done — no NaN or inf values remaining.")

# ── Step 6: Drop duplicates ──────────────────────────────────
print("\n[7] Dropping duplicate rows...")
before = df.shape[0]
df.drop_duplicates(inplace=True)
after  = df.shape[0]
print(f"    Removed: {before - after:,} duplicates")
print(f"    Clean rows: {after:,}")

# ── Final class summary ──────────────────────────────────────
print("\n[8] Final class distribution:")
vc_final = df[label_col].value_counts()
for cls, count in vc_final.items():
    pct = count / len(df) * 100
    print(f"    {cls:<30} {count:>8,}  ({pct:.2f}%)")

# ── Save cleaned dataset ─────────────────────────────────────
print(f"\n[9] Saving cleaned dataset...")
df.to_csv(output_path, index=False)
size_mb = os.path.getsize(output_path) / (1024 * 1024)
print(f"    Saved: {output_path}")
print(f"    Size : {size_mb:.2f} MB")

print("\n" + "=" * 55)
print("CLEANING COMPLETE!")
print(f"  Final rows    : {df.shape[0]:,}")
print(f"  Final classes : {df[label_col].nunique()}")
print(f"  Final columns : {df.shape[1]}")
print("\n  Next step: Update dataset_path in bc_flids_finetuned.py")
print(f"  New path: {output_path}")
print("=" * 55)
