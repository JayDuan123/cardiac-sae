"""
Explore the clinical cohort: age, sex, ICD distributions.
Save plots and a Table 1 summary for the paper.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

CLINICAL_DIR = cfg.EMBEDDING_DIR.parent / "clinical"
FIG_DIR = CLINICAL_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Load
print("Loading clinical join ...")
df = pd.read_csv(CLINICAL_DIR / "record_with_clinical.csv")
df['ecg_time'] = pd.to_datetime(df['ecg_time'])
print(f"  shape: {df.shape}")

# Also load ICD code → name dictionary
print("Loading ICD dictionary ...")
icd_dict = pd.read_csv("/workspace/data/mimic-iv/hosp/d_icd_diagnoses.csv.gz")
icd_dict['icd_code'] = icd_dict['icd_code'].str.strip().str.upper()
# Make a dict: (icd_code, version) -> title
icd_lookup = {(row['icd_code'], row['icd_version']): row['long_title']
              for _, row in icd_dict.iterrows()}

# ============================================================
# Table 1: Cohort summary
# ============================================================
print("\n" + "=" * 60)
print("Table 1: Cohort Summary")
print("=" * 60)

n_total = len(df)
n_subjects = df['subject_id'].nunique()
print(f"Total ECG records:        {n_total:,}")
print(f"Unique subjects:          {n_subjects:,}")
print(f"Mean records per subject: {n_total/n_subjects:.1f}")

# Age
valid_age = df['age_at_ecg'].dropna()
print(f"\nAge (years):")
print(f"  mean ± std:   {valid_age.mean():.1f} ± {valid_age.std():.1f}")
print(f"  median (IQR): {valid_age.median():.0f}  ({valid_age.quantile(0.25):.0f}–{valid_age.quantile(0.75):.0f})")
print(f"  range:        {valid_age.min():.0f} – {valid_age.max():.0f}")

# Sex
sex_counts = df['gender'].value_counts()
print(f"\nSex: F={sex_counts.get('F', 0):,} ({100*sex_counts.get('F', 0)/n_total:.1f}%)  "
      f"M={sex_counts.get('M', 0):,} ({100*sex_counts.get('M', 0)/n_total:.1f}%)")

# Admission context
print(f"\nAdmission context:")
for k, v in df['hadm_match_type'].value_counts().items():
    print(f"  {k:10s}: {v:,} ({100*v/n_total:.1f}%)")

# Mortality
print(f"\nMortality:")
print(f"  All-cause death (any time): {df['has_dod'].sum():,} ({100*df['has_dod'].mean():.1f}%)")
in_30d = ((df['days_to_death'] >= 0) & (df['days_to_death'] <= 30)).sum()
in_1y = ((df['days_to_death'] >= 0) & (df['days_to_death'] <= 365)).sum()
print(f"  Death within 30 days of ECG:  {in_30d:,} ({100*in_30d/n_total:.1f}%)")
print(f"  Death within 365 days of ECG: {in_1y:,} ({100*in_1y/n_total:.1f}%)")

# Diagnoses
n_with_dx = (df['n_diagnoses'] > 0).sum()
print(f"\nDiagnoses:")
print(f"  ECGs with ICD codes: {n_with_dx:,} ({100*n_with_dx/n_total:.1f}%)")
print(f"  Mean ICDs per record: {df['n_diagnoses'].mean():.1f}")

# ============================================================
# Figure A: Age histogram
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(valid_age, bins=80, color='steelblue', edgecolor='k', alpha=0.7)
axes[0].axvline(valid_age.median(), color='r', linestyle='--', label=f'median={valid_age.median():.0f}')
axes[0].set_xlabel('Age at ECG (years)')
axes[0].set_ylabel('Number of ECGs')
axes[0].set_title(f'Age distribution (n={len(valid_age):,})')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Age by sex
for sex, color in [('F', 'crimson'), ('M', 'steelblue')]:
    sub = df[df['gender'] == sex]['age_at_ecg'].dropna()
    axes[1].hist(sub, bins=80, color=color, alpha=0.5, label=f'{sex} (n={len(sub):,})')
axes[1].set_xlabel('Age at ECG (years)')
axes[1].set_ylabel('Number of ECGs')
axes[1].set_title('Age distribution by sex')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / "age_distribution.png", dpi=120)
plt.close()
print(f"\nSaved: {FIG_DIR}/age_distribution.png")

# ============================================================
# Figure B: Top ICD codes
# ============================================================
print("\nCounting top ICD codes ...")
icd_counter = Counter()
icd_counter_by_v = {9: Counter(), 10: Counter()}

# Only iterate over rows that have diagnoses
df_dx = df[df['icd_codes_str'].notna()]
for _, row in df_dx.iterrows():
    codes = row['icd_codes_str'].split(',')
    versions = row['icd_codes_v_str'].split(',')
    for c, v in zip(codes, versions):
        icd_counter[(c, int(v))] += 1
        icd_counter_by_v[int(v)][c] += 1

top30_all = icd_counter.most_common(30)
top20_v10 = icd_counter_by_v[10].most_common(20)
top20_v9 = icd_counter_by_v[9].most_common(20)

print(f"\nTop 30 ICD codes overall:")
for (code, ver), count in top30_all:
    title = icd_lookup.get((code, ver), '?')
    print(f"  v{ver}  {code:8s}  n={count:7,}  {title[:70]}")

# Save as csv
top_df_data = []
for (code, ver), count in icd_counter.most_common(500):
    title = icd_lookup.get((code, ver), '?')
    top_df_data.append({
        'icd_code': code, 'icd_version': ver,
        'count': count, 'pct': 100*count/n_total,
        'title': title,
    })
top_icd_df = pd.DataFrame(top_df_data)
top_icd_df.to_csv(CLINICAL_DIR / "icd_top500.csv", index=False)
print(f"\nSaved top-500 ICD codes to {CLINICAL_DIR}/icd_top500.csv")

# Plot top-20 ICD-10
fig, ax = plt.subplots(figsize=(12, 7))
codes = [f"{c}\n{icd_lookup.get((c, 10), '?')[:35]}" for c, _ in top20_v10]
counts = [n for _, n in top20_v10]
ax.barh(range(len(codes)), counts, color='steelblue')
ax.set_yticks(range(len(codes)))
ax.set_yticklabels(codes, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Number of ECGs with this code')
ax.set_title('Top 20 ICD-10 codes')
ax.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(FIG_DIR / "top20_icd10.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}/top20_icd10.png")

# ============================================================
# Figure C: ICDs of interest for cardiology
# ============================================================
print("\n" + "=" * 60)
print("Cardiology-relevant phenotype counts")
print("=" * 60)

# Define groups (ICD-10 and ICD-9 codes that map to common cardiac phenotypes)
# Use prefix matching for parent codes
PHENOTYPES = {
    "Atrial fibrillation":     {"icd10_prefix": ["I48"],            "icd9_prefix": ["427.3"]},
    "Heart failure":           {"icd10_prefix": ["I50"],            "icd9_prefix": ["428"]},
    "Hypertension (primary)":  {"icd10_prefix": ["I10"],            "icd9_prefix": ["401"]},
    "Diabetes mellitus":       {"icd10_prefix": ["E10", "E11"],     "icd9_prefix": ["250"]},
    "MI / ischemic heart":     {"icd10_prefix": ["I21", "I25"],     "icd9_prefix": ["410", "414"]},
    "Stroke / cerebrovascular":{"icd10_prefix": ["I60","I61","I63","I64"], "icd9_prefix": ["430","431","434","436"]},
    "COPD":                    {"icd10_prefix": ["J44"],            "icd9_prefix": ["491", "492", "496"]},
    "CKD":                     {"icd10_prefix": ["N18"],            "icd9_prefix": ["585"]},
    "Sepsis":                  {"icd10_prefix": ["A41", "R65"],     "icd9_prefix": ["995.91", "995.92"]},
}

def matches(code, version, prefixes):
    """Check if ICD code starts with any prefix (after removing periods)."""
    c = code.replace('.', '')
    for p in prefixes:
        if c.startswith(p.replace('.', '')):
            return True
    return False

# For each phenotype, count how many ECGs have it
print(f"\n{'Phenotype':<28s}{'count':>10s}{'pct':>8s}")
print("-" * 46)
phenotype_counts = {}
for pheno_name, prefix_dict in PHENOTYPES.items():
    icd10_pre = prefix_dict['icd10_prefix']
    icd9_pre = prefix_dict['icd9_prefix']
    has_pheno = np.zeros(len(df), dtype=bool)
    for i, row in enumerate(df_dx.itertuples()):
        codes = row.icd_codes_str.split(',')
        versions = row.icd_codes_v_str.split(',')
        for c, v in zip(codes, versions):
            v = int(v)
            pre = icd10_pre if v == 10 else icd9_pre
            if matches(c, v, pre):
                has_pheno[df_dx.index[i]] = True   # mark original df index
                break
    phenotype_counts[pheno_name] = int(has_pheno.sum())
    print(f"{pheno_name:<28s}{has_pheno.sum():>10,d}{100*has_pheno.sum()/n_total:>7.1f}%")

# Save phenotype flags as a separate file
print("\nBuilding phenotype flag table (this takes a minute) ...")
pheno_flags = pd.DataFrame({'record_idx': df['record_idx']})
for pheno_name, prefix_dict in PHENOTYPES.items():
    icd10_pre = prefix_dict['icd10_prefix']
    icd9_pre = prefix_dict['icd9_prefix']
    flags = np.zeros(len(df), dtype=bool)
    mask = df['icd_codes_str'].notna()
    for idx in df.index[mask]:
        codes = df.loc[idx, 'icd_codes_str'].split(',')
        versions = df.loc[idx, 'icd_codes_v_str'].split(',')
        for c, v in zip(codes, versions):
            v = int(v)
            pre = icd10_pre if v == 10 else icd9_pre
            if matches(c, v, pre):
                flags[idx] = True
                break
    col_name = pheno_name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    pheno_flags[col_name] = flags

pheno_flags.to_csv(CLINICAL_DIR / "phenotype_flags.csv", index=False)
print(f"Saved: {CLINICAL_DIR}/phenotype_flags.csv")

# Plot phenotypes bar chart
fig, ax = plt.subplots(figsize=(10, 5))
names = list(phenotype_counts.keys())
counts = list(phenotype_counts.values())
ax.barh(range(len(names)), counts, color='coral')
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.invert_yaxis()
ax.set_xlabel('Number of ECGs')
ax.set_title('Cardiology phenotype prevalence in MIMIC-IV-ECG cohort')
ax.grid(alpha=0.3, axis='x')
for i, c in enumerate(counts):
    ax.text(c, i, f' {c:,} ({100*c/n_total:.1f}%)', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(FIG_DIR / "phenotype_prevalence.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"Saved: {FIG_DIR}/phenotype_prevalence.png")

print("\nDone.")
