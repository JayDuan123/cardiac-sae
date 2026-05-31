"""
Stage 13: Quantitative atom characterization via ECG measurements.

Maps each SAE atom to the objective ECG features it encodes:
  - Heart rate (from rr_interval)
  - PR interval (qrs_onset - p_onset)
  - QRS duration (qrs_end - qrs_onset)
  - QT interval (t_end - qrs_onset)
  - P/QRS/T axes
  - ST elevation/depression (regex from reports)
  - Arrhythmia (from Stage 12 lift labels)

Outputs three views:
  1. atom_feature_correlation.csv     # full atom x feature correlation matrix
  2. atom_profiles.csv                # per-atom top-50 distribution of each feature
  3. feature_top_atoms.csv            # for each feature, the atoms that best predict it
"""
import sys, re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

mpl.rcParams.update({'font.size': 9, 'figure.dpi': 100, 'savefig.dpi': 150,
                     'savefig.bbox': 'tight'})

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
out_dir = sae_dir / "atom_features"
out_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Step 1: Load data + build feature table
# ============================================================
print("=" * 60)
print("Step 1: Building feature table from machine_measurements")
print("=" * 60)

# Meta to map record_idx -> study_id
# meta has only 'path' column; record_idx is just the row number
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
print(f"  meta columns: {meta.columns.tolist()}")
meta = meta.reset_index().rename(columns={'index': 'record_idx'})
meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')
N = len(meta)
print(f"  N records: {N:,}")
print(f"  study_id non-null: {meta['study_id'].notna().sum():,}")

# Load machine measurements
mm_cols_num = ['study_id', 'rr_interval', 'p_onset', 'p_end',
               'qrs_onset', 'qrs_end', 't_end',
               'p_axis', 'qrs_axis', 't_axis']
mm_cols_rep = [f'report_{i}' for i in range(18)]
mm = pd.read_csv("/workspace/data/mimic-iv-ecg-aws/machine_measurements.csv",
                 usecols=mm_cols_num + mm_cols_rep)
print(f"  loaded {len(mm):,} machine_measurements rows")

# Derive numerical features (in ms, axes in degrees)
mm['heart_rate'] = 60000 / mm['rr_interval'].replace(0, np.nan)
mm['pr_interval'] = mm['qrs_onset'] - mm['p_onset']
mm['qrs_duration'] = mm['qrs_end'] - mm['qrs_onset']
mm['qt_interval'] = mm['t_end'] - mm['qrs_onset']
# QTc (Bazett's formula, RR in seconds)
mm['qtc'] = mm['qt_interval'] / np.sqrt(mm['rr_interval'] / 1000)

# ST elevation / depression from report text (regex)
def extract_st(stmts):
    txt = ' '.join(str(s) for s in stmts if pd.notna(s)).lower()
    has_elev = bool(re.search(r'\bst\s*elevation\b', txt))
    has_depr = bool(re.search(r'\bst\s*depression\b', txt))
    return has_elev, has_depr

st_elev, st_depr = [], []
for _, row in mm[mm_cols_rep].iterrows():
    e, d = extract_st(row.values.tolist())
    st_elev.append(int(e))
    st_depr.append(int(d))
mm['st_elevation'] = st_elev
mm['st_depression'] = st_depr

# Tachycardia / bradycardia binary labels from HR (more robust than report)
mm['tachycardia'] = (mm['heart_rate'] > 100).astype(int)
mm['bradycardia'] = (mm['heart_rate'] < 60).astype(int)

# Wide QRS (>= 120 ms, signals BBB or VT)
mm['wide_qrs'] = (mm['qrs_duration'] >= 120).astype(int)

# Long QT (QTc > 460 women / 450 men, simplified threshold 460)
mm['long_qt'] = (mm['qtc'] > 460).astype(int)

# Left axis (< -30), right axis (> +90)
mm['left_axis'] = (mm['qrs_axis'] < -30).astype(int)
mm['right_axis'] = (mm['qrs_axis'] > 90).astype(int)

FEATURES_NUM = ['heart_rate', 'pr_interval', 'qrs_duration', 'qt_interval', 'qtc',
                'p_axis', 'qrs_axis', 't_axis']
FEATURES_BIN = ['tachycardia', 'bradycardia', 'wide_qrs', 'long_qt',
                'left_axis', 'right_axis', 'st_elevation', 'st_depression']
ALL_FEATURES = FEATURES_NUM + FEATURES_BIN

print(f"\n  Features extracted:")
print(f"    Numerical: {FEATURES_NUM}")
print(f"    Binary:    {FEATURES_BIN}")
print(f"\n  Feature stats:")
print(mm[ALL_FEATURES].describe().T[['count', 'mean', 'std', 'min', 'max']].round(1))

# Merge to record level (one row per ECG record in our embedding/SAE)
# meta has 'study_id' for each record. Some studies may have multiple records, take first.
print("\n  Joining to records ...")
feat_df = meta[['record_idx', 'study_id']].merge(
    mm[['study_id'] + ALL_FEATURES],
    on='study_id', how='left'
)
feat_df = feat_df.set_index('record_idx').reindex(range(N))
print(f"  feature table shape: {feat_df.shape}")
print(f"  records with HR available: {feat_df['heart_rate'].notna().sum():,}/{N:,}")

# ============================================================
# Step 2: Load SAE activations
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Load SAE activations")
print("=" * 60)
acts = load_npz(sae_dir / "activations_all.npz").tocsc()  # column access faster
D = acts.shape[1]
print(f"  activations: {acts.shape}, nnz={acts.nnz:,}")

# ============================================================
# Step 3: Atom-feature correlation matrix
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Computing atom x feature correlations")
print("=" * 60)
print("This computes Spearman r for each (atom, feature) pair.")
print("Only considers records where atom fires (acts > 0)\n")

corr_matrix = np.full((D, len(ALL_FEATURES)), np.nan)
n_records_used = np.zeros((D, len(ALL_FEATURES)), dtype=int)

import time
t0 = time.time()
for atom_id in range(D):
    if atom_id % 200 == 0:
        print(f"  atom {atom_id}/{D} ({time.time()-t0:.0f}s elapsed)")
    # get records where this atom fires
    st, en = acts.indptr[atom_id], acts.indptr[atom_id+1]
    if en - st < 30:  # too few activations to compute meaningful correlation
        continue
    rec_idx = acts.indices[st:en]
    act_vals = acts.data[st:en]
    sub = feat_df.iloc[rec_idx]
    for f_i, feat in enumerate(ALL_FEATURES):
        y = sub[feat].values
        valid = ~np.isnan(y)
        if valid.sum() < 30:
            continue
        try:
            r, _ = spearmanr(act_vals[valid], y[valid])
            corr_matrix[atom_id, f_i] = r
            n_records_used[atom_id, f_i] = valid.sum()
        except Exception:
            pass

print(f"  done in {time.time()-t0:.0f}s")

# Save correlation matrix
corr_df = pd.DataFrame(corr_matrix, columns=ALL_FEATURES)
corr_df.index.name = 'atom_id'
corr_df.to_csv(out_dir / "atom_feature_correlation.csv")
print(f"  saved: {out_dir}/atom_feature_correlation.csv")

# ============================================================
# Step 4: Top atoms per feature (single-atom predictor)
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Top atoms for each feature")
print("=" * 60)

# For each feature, rank atoms by abs(correlation), report top 10
top_per_feat = []
for feat in ALL_FEATURES:
    if feat not in corr_df.columns:
        continue
    r = corr_df[feat]
    valid = r.notna()
    if valid.sum() == 0:
        continue
    # Use abs(r) but keep sign
    abs_r = r.abs()
    top_atoms = abs_r.nlargest(10).index.tolist()
    for rank, atom_id in enumerate(top_atoms, 1):
        top_per_feat.append({
            'feature': feat,
            'rank': rank,
            'atom_id': int(atom_id),
            'spearman_r': float(r[atom_id]),
            'abs_r': float(abs_r[atom_id]),
            'n_records': int(n_records_used[atom_id, ALL_FEATURES.index(feat)]),
        })

feat_top = pd.DataFrame(top_per_feat)
feat_top.to_csv(out_dir / "feature_top_atoms.csv", index=False)
print(f"  saved: {out_dir}/feature_top_atoms.csv")

# Print top-3 per feature for quick view
print("\n=== Top-3 atoms per feature ===")
for feat in ALL_FEATURES:
    sub = feat_top[feat_top['feature'] == feat].head(3)
    print(f"\n  {feat}:")
    for _, r in sub.iterrows():
        sign = '+' if r['spearman_r'] > 0 else '-'
        print(f"    atom {r['atom_id']:4d}  r={sign}{abs(r['spearman_r']):.3f}  "
              f"(n={r['n_records']:,})")

# ============================================================
# Step 5: Per-atom profile (top-50 distribution of features)
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Per-atom feature profile (top-50 activations)")
print("=" * 60)

TOP_N = 50
profiles = []
acts_csc = acts.tocsc()
for atom_id in range(D):
    st, en = acts_csc.indptr[atom_id], acts_csc.indptr[atom_id+1]
    if en - st < TOP_N:
        # use all available if fewer than TOP_N
        n_avail = en - st
        if n_avail < 5:
            continue
        rec_idx = acts_csc.indices[st:en]
    else:
        rec_idx = acts_csc.indices[st:en]
        vals = acts_csc.data[st:en]
        rec_idx = rec_idx[np.argpartition(vals, -TOP_N)[-TOP_N:]]
    sub = feat_df.iloc[rec_idx]
    row = {'atom_id': atom_id, 'n_records_top': len(rec_idx)}
    for feat in FEATURES_NUM:
        v = sub[feat].dropna()
        if len(v) >= 5:
            row[f'{feat}_median'] = float(v.median())
            row[f'{feat}_mean'] = float(v.mean())
            row[f'{feat}_std'] = float(v.std())
    for feat in FEATURES_BIN:
        v = sub[feat].dropna()
        if len(v) >= 5:
            row[f'{feat}_frac'] = float(v.mean())  # fraction of top-50 with this label
    profiles.append(row)

prof_df = pd.DataFrame(profiles)
prof_df.to_csv(out_dir / "atom_profiles.csv", index=False)
print(f"  saved: {out_dir}/atom_profiles.csv")

# ============================================================
# Step 6: Compute baselines for comparison
# ============================================================
print("\n=== Global feature statistics (for comparison) ===")
print(f"  HR        median={feat_df['heart_rate'].median():.0f}  "
      f"IQR=[{feat_df['heart_rate'].quantile(.25):.0f}, "
      f"{feat_df['heart_rate'].quantile(.75):.0f}]")
print(f"  PR        median={feat_df['pr_interval'].median():.0f}  "
      f"IQR=[{feat_df['pr_interval'].quantile(.25):.0f}, "
      f"{feat_df['pr_interval'].quantile(.75):.0f}]")
print(f"  QRS       median={feat_df['qrs_duration'].median():.0f}  "
      f"IQR=[{feat_df['qrs_duration'].quantile(.25):.0f}, "
      f"{feat_df['qrs_duration'].quantile(.75):.0f}]")
print(f"  QTc       median={feat_df['qtc'].median():.0f}  "
      f"IQR=[{feat_df['qtc'].quantile(.25):.0f}, "
      f"{feat_df['qtc'].quantile(.75):.0f}]")
print(f"  QRS axis  median={feat_df['qrs_axis'].median():.0f}  "
      f"IQR=[{feat_df['qrs_axis'].quantile(.25):.0f}, "
      f"{feat_df['qrs_axis'].quantile(.75):.0f}]")
print(f"  Tachy %:  {100*feat_df['tachycardia'].mean():.1f}%")
print(f"  Brady %:  {100*feat_df['bradycardia'].mean():.1f}%")
print(f"  Wide QRS %: {100*feat_df['wide_qrs'].mean():.1f}%")
print(f"  Long QT %:  {100*feat_df['long_qt'].mean():.1f}%")

# ============================================================
# Step 7: Cross-reference with Stage 12 report labels
# ============================================================
report_labels_path = sae_dir / "atom_reports" / "atom_report_labels_v2.csv"
if report_labels_path.exists():
    print("\n" + "=" * 60)
    print("Step 7: Combined view -- report label + measurement profile")
    print("=" * 60)
    rep = pd.read_csv(report_labels_path)
    merged = rep.merge(prof_df, on='atom_id', how='inner')
    # Show top strong-lift atoms with their measurement profile
    strong = merged[merged['lift'] > 100].nlargest(15, 'lift')
    print("\n=== Top 15 strong-lift atoms (Stage 12) with measurements ===")
    for _, r in strong.iterrows():
        hr = r.get('heart_rate_median', np.nan)
        qrs = r.get('qrs_duration_median', np.nan)
        qrsax = r.get('qrs_axis_median', np.nan)
        print(f"  atom {int(r['atom_id']):4d}  lift={r['lift']:.0f}  "
              f"'{r['label'][:50]}'")
        print(f"     HR={hr:.0f}  QRS={qrs:.0f}ms  QRS_axis={qrsax:.0f}°")

    merged.to_csv(out_dir / "atom_combined_report_and_features.csv", index=False)
    print(f"\n  saved: {out_dir}/atom_combined_report_and_features.csv")

print("\n" + "=" * 60)
print(f"All outputs in: {out_dir}")
print("=" * 60)
