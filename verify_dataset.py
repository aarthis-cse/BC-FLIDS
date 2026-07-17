# ============================================================
# Dataset Verification Script
# Run this BEFORE the model to catch any issues early
# ============================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

dataset_path = r"E:\Journals\Journal 2026\July 26\IoV 14.7.26\Dataset\IoV_SecFusion_Dataset.csv"

print("=" * 55)
print("DATASET VERIFICATION REPORT")
print("=" * 55)

# ── Load ────────────────────────────────────────────────────
print("\n[1] Loading dataset...")
df = pd.read_csv(dataset_path, low_memory=False)
df.columns = df.columns.str.strip()
print(f"    Rows    : {df.shape[0]:,}")
print(f"    Columns : {df.shape[1]}")

# ── File size ───────────────────────────────────────────────
size_mb = os.path.getsize(dataset_path) / (1024 * 1024)
print(f"    File size: {size_mb:.2f} MB")

# ── Dataset source breakdown ────────────────────────────────
print("\n[2] Dataset source breakdown:")
if 'dataset_source' in df.columns:
    print(df['dataset_source'].value_counts().to_string())
else:
    print("    WARNING: 'dataset_source' column not found!")

# ── Label column check ──────────────────────────────────────
print("\n[3] Label column check:")
label_candidates = ['Label', 'label', ' Label', 'label_col']
label_col = None
for lc in label_candidates:
    if lc in df.columns:
        label_col = lc
        break

if label_col:
    df[label_col] = df[label_col].astype(str).str.strip()
    print(f"    Label column found: '{label_col}'")
    print(f"    Unique classes ({df[label_col].nunique()}):")
    vc = df[label_col].value_counts()
    for cls, count in vc.items():
        pct = count / len(df) * 100
        print(f"      {cls:<30} {count:>8,}  ({pct:.1f}%)")
else:
    print("    ERROR: No label column found! Check column names.")
    print(f"    Available columns: {list(df.columns[:10])}...")

# ── Missing values ──────────────────────────────────────────
print("\n[4] Missing values:")
total_missing = df.isnull().sum().sum()
cols_with_nan = (df.isnull().sum() > 0).sum()
print(f"    Total missing cells : {total_missing:,}")
print(f"    Columns with NaN    : {cols_with_nan}")
if total_missing > 0:
    top_missing = df.isnull().sum().sort_values(ascending=False).head(5)
    print(f"    Top 5 columns with most NaN:")
    for col, cnt in top_missing.items():
        if cnt > 0:
            print(f"      {col:<35} {cnt:>8,}")

# ── Infinite values ─────────────────────────────────────────
print("\n[5] Infinite values:")
numeric_df = df.select_dtypes(include=[np.number])
inf_count  = np.isinf(numeric_df).sum().sum()
print(f"    Total inf cells : {inf_count:,}")

# ── Duplicate rows ──────────────────────────────────────────
print("\n[6] Duplicate rows:")
dup_count = df.duplicated().sum()
print(f"    Duplicate rows : {dup_count:,}")

# ── Data types ──────────────────────────────────────────────
print("\n[7] Column data types:")
dtype_counts = df.dtypes.value_counts()
for dtype, count in dtype_counts.items():
    print(f"    {str(dtype):<15} : {count} columns")

non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
if label_col in non_numeric:
    non_numeric.remove(label_col)
if 'dataset_source' in non_numeric:
    non_numeric.remove('dataset_source')
if non_numeric:
    print(f"\n    Non-numeric columns (will be dropped in model):")
    for c in non_numeric[:10]:
        print(f"      {c}")

# ── Numeric feature count ───────────────────────────────────
print("\n[8] Numeric features available for model:")
feature_cols = [c for c in numeric_df.columns
                if c not in ['Label', 'label', ' Label']]
print(f"    Count : {len(feature_cols)}")

# ── Class imbalance check ───────────────────────────────────
print("\n[9] Class imbalance check:")
if label_col:
    vc   = df[label_col].value_counts()
    ratio = vc.max() / vc.min() if vc.min() > 0 else float('inf')
    print(f"    Max/Min class ratio : {ratio:.1f}x")
    if ratio > 10:
        print("    WARNING: High imbalance — SMOTE in model will handle this")
    else:
        print("    OK: Class balance is acceptable")

# ── Sample preview ──────────────────────────────────────────
print("\n[10] First 3 rows preview:")
print(df.head(3).to_string(max_cols=8))

# ── Final verdict ───────────────────────────────────────────
print("\n" + "=" * 55)
print("VERIFICATION COMPLETE")
print("=" * 55)

issues = []
if total_missing > 0:
    issues.append(f"Missing values: {total_missing:,} cells")
if inf_count > 0:
    issues.append(f"Infinite values: {inf_count:,} cells")
if dup_count > 0:
    issues.append(f"Duplicate rows: {dup_count:,}")
if not label_col:
    issues.append("Label column not found!")

if issues:
    print("\n  Issues found (model will auto-fix these):")
    for i in issues:
        print(f"    - {i}")
else:
    print("\n  Dataset looks clean and ready for training!")

print("\n  Next step: Run bc_flids_finetuned.py")
print("=" * 55)
