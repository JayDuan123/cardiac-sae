"""
Join ECG records with MIMIC-IV clinical data:
- patients.csv: age, gender
- admissions.csv: match each ECG to nearest hospital admission
- diagnoses_icd.csv: ICD codes per admission

Output: record_with_clinical.parquet
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

HOSP = Path("/workspace/data/mimic-iv/hosp")
OUT_DIR = cfg.EMBEDDING_DIR.parent / "clinical"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # ============================================================
    # 1. Load patients
    # ============================================================
    print("Loading patients.csv.gz ...")
    patients = pd.read_csv(HOSP / "patients.csv.gz",
                          usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year', 'dod'])
    patients['dod'] = pd.to_datetime(patients['dod'], errors='coerce')
    print(f"  {len(patients):,} patients")

    # ============================================================
    # 2. Load record_list (ECGs)
    # ============================================================
    print("\nLoading record_list.csv ...")
    records = pd.read_csv(cfg.DATA_ROOT / "record_list.csv")
    records['path'] = records['path'].str.rstrip('/')
    records['ecg_time'] = pd.to_datetime(records['ecg_time'])
    records['record_idx'] = np.arange(len(records))  # original row order
    print(f"  {len(records):,} ECG records")

    # ============================================================
    # 3. Join age & gender
    # ============================================================
    print("\nJoining patient demographics ...")
    records = records.merge(patients, on='subject_id', how='left')

    # Compute age at ECG (MIMIC-IV trick: anchor_age + (ecg_time.year - anchor_year))
    records['age_at_ecg'] = (records['anchor_age']
                             + (records['ecg_time'].dt.year - records['anchor_year']))

    n_with_age = records['age_at_ecg'].notna().sum()
    print(f"  ECGs with age:    {n_with_age:,} ({100*n_with_age/len(records):.2f}%)")
    print(f"  ECGs with gender: {records['gender'].notna().sum():,}")

    # Sanity check ages
    valid_age = records['age_at_ecg'].dropna()
    print(f"  age stats: min={valid_age.min():.0f}, max={valid_age.max():.0f}, "
          f"mean={valid_age.mean():.1f}, median={valid_age.median():.0f}")

    # ============================================================
    # 4. Load admissions
    # ============================================================
    print("\nLoading admissions.csv.gz ...")
    adm = pd.read_csv(HOSP / "admissions.csv.gz",
                      usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime',
                               'admission_type', 'admission_location',
                               'edregtime', 'edouttime', 'hospital_expire_flag'])
    adm['admittime'] = pd.to_datetime(adm['admittime'])
    adm['dischtime'] = pd.to_datetime(adm['dischtime'])
    print(f"  {len(adm):,} admissions for {adm['subject_id'].nunique():,} subjects")

    # ============================================================
    # 5. Match each ECG to its admission (use ecg_time)
    # ============================================================
    print("\nMatching ECGs to admissions (using ecg_time) ...")

    # Sort admissions by (subject_id, admittime) for efficient lookup
    adm_sorted = adm.sort_values(['subject_id', 'admittime']).reset_index(drop=True)

    # Group admissions by subject_id
    adm_groups = {sid: g for sid, g in adm_sorted.groupby('subject_id')}
    print(f"  {len(adm_groups):,} subjects have admissions")

    # For each ECG, find admission such that admittime <= ecg_time <= dischtime
    # If none, find nearest admission within ±30 days
    matched_hadm = np.full(len(records), -1, dtype=np.int64)
    match_type = np.full(len(records), 'none', dtype=object)
    days_from_admit = np.full(len(records), np.nan, dtype=np.float32)

    for i in tqdm(range(len(records)), desc="Matching"):
        sid = records.iloc[i]['subject_id']
        et = records.iloc[i]['ecg_time']
        if sid not in adm_groups:
            continue
        g = adm_groups[sid]
        # During admission?
        during = g[(g['admittime'] <= et) & (g['dischtime'] >= et)]
        if len(during) > 0:
            row = during.iloc[0]  # first one if overlap
            matched_hadm[i] = row['hadm_id']
            match_type[i] = 'during'
            days_from_admit[i] = (et - row['admittime']).total_seconds() / 86400
            continue
        # Nearest within ±30 days
        diffs = (g['admittime'] - et).dt.total_seconds() / 86400
        # Take admissions whose admittime is within ±30 days of ecg_time
        within = g[diffs.abs() <= 30]
        if len(within) > 0:
            within_diffs = (within['admittime'] - et).dt.total_seconds() / 86400
            nearest_idx = within_diffs.abs().idxmin()
            row = within.loc[nearest_idx]
            matched_hadm[i] = row['hadm_id']
            match_type[i] = 'nearby'
            days_from_admit[i] = (et - row['admittime']).total_seconds() / 86400

    records['hadm_id'] = matched_hadm
    records['hadm_match_type'] = match_type
    records['days_from_admit'] = days_from_admit

    print(f"\n  Match types:")
    print(records['hadm_match_type'].value_counts().to_string())

    # ============================================================
    # 6. Load diagnoses and join
    # ============================================================
    print("\nLoading diagnoses_icd.csv.gz ...")
    dx = pd.read_csv(HOSP / "diagnoses_icd.csv.gz",
                     usecols=['hadm_id', 'icd_code', 'icd_version', 'seq_num'])
    print(f"  {len(dx):,} diagnoses for {dx['hadm_id'].nunique():,} admissions")
    print(f"  ICD version distribution:")
    print(dx['icd_version'].value_counts().to_string())

    # Normalize icd_code: strip whitespace, make uppercase, prefix with version
    dx['icd_code'] = dx['icd_code'].str.strip().str.upper()
    dx['icd_full'] = dx['icd_version'].astype(str) + '|' + dx['icd_code']

    # For each hadm, aggregate ICD codes into a list
    print("\nAggregating ICDs per admission ...")
    dx_by_hadm = dx.groupby('hadm_id').agg(
        icd_codes_v=('icd_version', lambda x: list(x)),
        icd_codes=('icd_code', lambda x: list(x)),
    ).reset_index()
    # Store as a comma-separated string for parquet compat (lists OK in parquet but slower)
    dx_by_hadm['icd_codes_str'] = dx_by_hadm['icd_codes'].apply(
        lambda lst: ','.join(str(c) for c in lst))
    dx_by_hadm['icd_codes_v_str'] = dx_by_hadm['icd_codes_v'].apply(
        lambda lst: ','.join(str(v) for v in lst))
    dx_by_hadm['n_diagnoses'] = dx_by_hadm['icd_codes'].apply(len)

    records = records.merge(
        dx_by_hadm[['hadm_id', 'icd_codes_str', 'icd_codes_v_str', 'n_diagnoses']],
        on='hadm_id', how='left'
    )
    records['n_diagnoses'] = records['n_diagnoses'].fillna(0).astype(int)

    n_with_dx = (records['n_diagnoses'] > 0).sum()
    print(f"  ECGs with diagnoses: {n_with_dx:,} ({100*n_with_dx/len(records):.1f}%)")

    # ============================================================
    # 7. Death flag (if dod is set and within reasonable window)
    # ============================================================
    print("\nDeath info ...")
    records['has_dod'] = records['dod'].notna()
    # Did patient die during this admission?
    records['days_to_death'] = (records['dod'] - records['ecg_time']).dt.total_seconds() / 86400
    n_died = records['has_dod'].sum()
    print(f"  ECGs from patients who eventually died: {n_died:,} ({100*n_died/len(records):.1f}%)")

    # ============================================================
    # 8. Final cleanup and save
    # ============================================================
    final_cols = [
        'record_idx', 'subject_id', 'study_id', 'ecg_time', 'path',
        'age_at_ecg', 'gender', 'anchor_age', 'anchor_year',
        'hadm_id', 'hadm_match_type', 'days_from_admit', 'n_diagnoses',
        'icd_codes_str', 'icd_codes_v_str',
        'has_dod', 'days_to_death',
    ]
    out = records[final_cols].copy()
    # hadm_id: -1 means no match → use NaN
    out.loc[out['hadm_id'] == -1, 'hadm_id'] = np.nan

    out_csv = OUT_DIR / "record_with_clinical.csv"
    out_parquet = OUT_DIR / "record_with_clinical.parquet"

    print(f"\nSaving to:")
    print(f"  {out_csv}")
    out.to_csv(out_csv, index=False)

    try:
        print(f"  {out_parquet}")
        out.to_parquet(out_parquet, index=False)
    except Exception as e:
        print(f"  (parquet failed: {e})")

    print(f"\nDone. Shape: {out.shape}")
    print(f"\nColumn summary:")
    print(out.dtypes.to_string())
    print(f"\nMissing per column:")
    print(out.isna().sum().to_string())

    print(f"\nSize on disk:")
    print(f"  CSV:     {out_csv.stat().st_size / 1e6:.1f} MB")
    if out_parquet.exists():
        print(f"  Parquet: {out_parquet.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
