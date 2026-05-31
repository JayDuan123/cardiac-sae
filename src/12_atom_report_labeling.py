"""
Category (2): Auto-label SAE atoms via ECG report statements.
v2: filter boilerplate stopwords + rank by enrichment (lift), not raw frequency.
"""
import sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
out_dir = sae_dir / "atom_reports"
out_dir.mkdir(parents=True, exist_ok=True)

# Boilerplate / non-specific statements to ignore as labels
STOPWORDS = {
    "Abnormal ECG", "Borderline ECG", "Normal ECG", "Sinus rhythm",
    "Sinus rhythm.", "Normal ECG except for rate", "Otherwise normal ECG",
    "Analysis error", "",
}
def is_stopword(s):
    if s in STOPWORDS:
        return True
    if s.startswith("---"):          # warnings / lead reversal notices
        return True
    if "unsuitable for analysis" in s.lower():
        return True
    if "data quality" in s.lower():
        return True
    return False

# ============================================================
# Load
# ============================================================
print("Loading machine_measurements reports ...")
report_cols = [f"report_{i}" for i in range(18)]
mm = pd.read_csv("/workspace/data/mimic-iv-ecg-aws/machine_measurements.csv",
                 usecols=['study_id'] + report_cols)

mm_reports = {}
for row in mm.itertuples(index=False):
    sid = row[0]
    stmts = [str(getattr(row, c)).strip() for c in report_cols]
    stmts = [s for s in stmts if s and s != 'nan' and not is_stopword(s)]
    mm_reports[sid] = stmts
print(f"  built reports for {len(mm_reports):,} studies (stopwords filtered)")

# Global phrase frequency (for lift computation)
print("Computing global phrase frequencies ...")
global_counter = Counter()
total_studies_with_phrase = 0
for stmts in mm_reports.values():
    for s in set(stmts):   # count each study once per phrase
        global_counter[s] += 1
n_studies = len(mm_reports)
global_freq = {p: c / n_studies for p, c in global_counter.items()}

# meta + top records
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')
top = np.load(sae_dir / "top_records.npz")
top_indices = top['indices']
D = top_indices.shape[0]
use_n = top_indices.shape[1]
atom_stats = pd.read_csv(sae_dir / "atom_stats.csv")

# ============================================================
# Label each atom: pick the phrase with highest LIFT
# ============================================================
print(f"\nLabeling {D} atoms (top-{use_n}, lift-ranked) ...")
results = []
for atom_id in range(D):
    rec_idx = top_indices[atom_id]
    rec_idx = rec_idx[rec_idx >= 0][:use_n]
    if len(rec_idx) == 0:
        results.append({'atom_id': atom_id, 'label': 'NO_RECORDS',
                        'lift': 0, 'local_freq': 0, 'global_freq': 0,
                        'freq_pct': float(atom_stats.iloc[atom_id]['freq_pct'])})
        continue

    local = Counter()
    n_rep = 0
    for ri in rec_idx:
        sid = meta.iloc[ri]['study_id']
        if pd.isna(sid):
            continue
        stmts = mm_reports.get(int(sid), [])
        if stmts:
            n_rep += 1
            for s in set(stmts):
                local[s] += 1

    if not local:
        results.append({'atom_id': atom_id, 'label': 'NO_SPECIFIC_DX',
                        'lift': 0, 'local_freq': 0, 'global_freq': 0,
                        'freq_pct': float(atom_stats.iloc[atom_id]['freq_pct'])})
        continue

    # Rank phrases by lift = local_freq / global_freq, require local count >= 3
    best_phrase, best_lift = None, 0
    cand = []
    for phrase, cnt in local.items():
        local_f = cnt / len(rec_idx)
        gf = global_freq.get(phrase, 1e-6)
        lift = local_f / gf
        if cnt >= 3:                       # need at least 3 supporting records
            cand.append((phrase, local_f, gf, lift, cnt))
    cand.sort(key=lambda x: -x[3])         # sort by lift
    if cand:
        best_phrase, local_f, gf, best_lift, cnt = cand[0]
        top3 = " | ".join(f"{p}(lift {l:.1f},n{c})" for p, lf, g, l, c in cand[:3])
    else:
        best_phrase, local_f, gf, best_lift = 'NO_SPECIFIC_DX', 0, 0, 0
        top3 = ''

    results.append({
        'atom_id': atom_id, 'label': best_phrase,
        'lift': round(best_lift, 2), 'local_freq': round(local_f, 3),
        'global_freq': round(gf, 4),
        'n_with_report': n_rep,
        'top3_by_lift': top3,
        'freq_pct': float(atom_stats.iloc[atom_id]['freq_pct']),
    })

df = pd.DataFrame(results)
df.to_csv(out_dir / "atom_report_labels_v2.csv", index=False)
print(f"Saved: {out_dir}/atom_report_labels_v2.csv")

# ============================================================
# Summary
# ============================================================
labeled = df[~df['label'].isin(['NO_RECORDS', 'NO_SPECIFIC_DX'])]
print(f"\n=== Summary ===")
print(f"Atoms with specific dx label: {len(labeled)}/{D}")
print(f"  strong (lift>5):   {(labeled['lift'] > 5).sum()}")
print(f"  moderate (lift 2-5): {((labeled['lift'] >= 2) & (labeled['lift'] <= 5)).sum()}")
print(f"  weak (lift<2):     {(labeled['lift'] < 2).sum()}")

print("\n=== Most common SPECIFIC labels (boilerplate removed) ===")
for lbl, cnt in labeled['label'].value_counts().head(25).items():
    print(f"  {cnt:3d} atoms: {lbl}")

print("\n=== Top 25 strongest atom labels (highest lift, freq>0.5%) ===")
strong = labeled[labeled['freq_pct'] > 0.5].nlargest(25, 'lift')
for _, r in strong.iterrows():
    print(f"  atom {r['atom_id']:4d} (freq {r['freq_pct']:.1f}%) "
          f"lift={r['lift']:.1f}: '{r['label']}'")

print("\nDone.")
