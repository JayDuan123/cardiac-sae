"""
Stage 21: Distributed encoding analysis.

Question:
  For each concept, can a small set of atoms jointly predict it,
  even when no single atom can?

Method:
  For each concept C:
    1. Build positive set (same as Stage 15)
    2. top1_auroc  = max single-atom test AUROC
    3. multi_auroc = test AUROC of L1-regularized logistic regression
                     over all atoms, using CV-selected regularization
    4. Both measured on subject-disjoint train/test split

Classification of concept encoding type:
  - 'separable':    top1 >= 0.70                              (already captured)
  - 'distributed':  multi >= 0.75 AND top1 < 0.65 AND
                    (multi - top1) >= 0.10                    (truly distributed)
  - 'weak':         multi >= 0.65 but neither threshold met   (partially captured)
  - 'not_encoded':  multi < 0.65                              (not detectable)

For distributed concepts, identify "contributing atoms" as those with
nonzero L1 coefficients (limited to top-K by |coef|).

Atom-level taxonomy update:
  Original Uninformative atoms that contribute to >=1 distributed concept
  → re-classified as 'Contributing'.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

# ============================================================
# Config
# ============================================================
SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
tax_dir = sae_dir / "taxonomy"
out_dir = sae_dir / "taxonomy_distributed"
out_dir.mkdir(parents=True, exist_ok=True)

# Thresholds (conservative; tunable)
TOP1_CAPTURED = 0.70   # if any atom hits this, concept is "captured"
TOP1_WEAK_MAX = 0.65   # max top1 for "truly distributed"
MULTI_DISTRIB = 0.75   # multi must hit this for "distributed"
MULTI_WEAK_MIN = 0.65  # multi must hit this for at least "weak"
DELTA_MIN = 0.10       # multi must exceed top1 by this much

# Subsampling for speed (L1 logistic on full 800k × 1536 is slow)
TRAIN_N = 100_000
TEST_N = 50_000
MAX_CONTRIBUTING_ATOMS = 10

# ============================================================
# Load data
# ============================================================
print("=" * 70)
print("Stage 21: Distributed concept encoding analysis")
print("=" * 70)

print("\nLoading data ...")
acts_csc = load_npz(sae_dir / "activations_all.npz").tocsc()
acts = acts_csc.tocsr()   # CSR for row-wise slicing
N, D = acts.shape
print(f"  activations: {acts.shape}")

tax = pd.read_csv(tax_dir / "atom_taxonomy.csv")

# Meta + clinical
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
meta = meta.reset_index().rename(columns={'index': 'record_idx'})
meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')

CLINICAL_DIR = cfg.EMBEDDING_DIR.parent / "clinical"
clin = pd.read_csv(CLINICAL_DIR / "record_with_clinical.csv")
clin = clin.set_index('record_idx').reindex(range(N))

# Machine measurements
mm = pd.read_csv("/workspace/data/mimic-iv-ecg-aws/machine_measurements.csv",
                 usecols=['study_id', 'rr_interval', 'qrs_onset', 'qrs_end',
                          'p_axis', 'qrs_axis', 't_axis'] +
                         [f'report_{i}' for i in range(18)],
                 low_memory=False)
mm['study_id'] = mm['study_id'].astype('Int64')
for col in ['rr_interval', 'qrs_onset', 'qrs_end']:
    mm[col] = mm[col].where((mm[col] >= 0) & (mm[col] < 2000), np.nan)
mm['heart_rate'] = 60000 / mm['rr_interval'].replace(0, np.nan)
mm['qrs_duration'] = mm['qrs_end'] - mm['qrs_onset']

feat = meta[['record_idx', 'study_id']].merge(mm, on='study_id', how='left')
feat = feat.set_index('record_idx').reindex(range(N))

rep_cols = [f'report_{i}' for i in range(18)]
feat['report_lower'] = feat[rep_cols].apply(
    lambda r: ' || '.join(str(s) for s in r.values if pd.notna(s)).lower(), axis=1
)

# ============================================================
# Subject-level train/test split (CRITICAL for honest AUROC)
# ============================================================
print("\nBuilding subject-level train/test split ...")
subject_ids = clin['subject_id'].values
valid_subj = pd.notna(subject_ids)
unique_subj = np.unique(subject_ids[valid_subj])
rng = np.random.RandomState(42)
rng.shuffle(unique_subj)
n_train_subj = int(len(unique_subj) * 0.7)
train_subj = set(unique_subj[:n_train_subj].tolist())
test_subj = set(unique_subj[n_train_subj:].tolist())

train_mask = np.array([s in train_subj if pd.notna(s) else False for s in subject_ids])
test_mask = np.array([s in test_subj if pd.notna(s) else False for s in subject_ids])
print(f"  train: {train_mask.sum():,} records / {len(train_subj):,} subjects")
print(f"  test:  {test_mask.sum():,} records / {len(test_subj):,} subjects")

# Subsample for L1 logistic speed
train_idx = np.where(train_mask)[0]
test_idx = np.where(test_mask)[0]
if len(train_idx) > TRAIN_N:
    train_idx = rng.choice(train_idx, TRAIN_N, replace=False)
if len(test_idx) > TEST_N:
    test_idx = rng.choice(test_idx, TEST_N, replace=False)
print(f"  using {len(train_idx):,} train + {len(test_idx):,} test for L1 logistic")

# Pre-extract activation matrices (these become dense, but on subsample it's ok)
X_train = acts[train_idx].toarray().astype(np.float32)  # (TRAIN_N, 1536)
X_test = acts[test_idx].toarray().astype(np.float32)
print(f"  X_train: {X_train.shape} ({X_train.nbytes/1e9:.2f} GB)")

# Standardize
scaler = StandardScaler(with_mean=False)  # keep sparse-like structure
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# Concept positive sets (mirror Stage 15)
# ============================================================
print("\nBuilding concept positive sets ...")
hr = feat['heart_rate'].values
qrs = feat['qrs_duration'].values
qrs_ax = feat['qrs_axis'].values
report = feat['report_lower'].values

ICD_CONCEPTS = {
    'ICD:AF': 'atrial_fibrillation', 'ICD:HF': 'heart_failure',
    'ICD:MI': 'mi___ischemic_heart', 'ICD:DM': 'diabetes_mellitus',
    'ICD:HTN': 'hypertension_primary',
}

concept_masks = {}
# ICD
for cn, col in ICD_CONCEPTS.items():
    if col in clin.columns:
        concept_masks[cn] = clin[col].fillna(0).astype(int).values.astype(bool)

# NUM
concept_masks['NUM:tachycardia'] = (hr > 100)
concept_masks['NUM:bradycardia'] = (hr < 60)
concept_masks['NUM:wide_qrs'] = (qrs >= 120)
concept_masks['NUM:left_axis'] = (qrs_ax < -30)
concept_masks['NUM:right_axis'] = (qrs_ax > 90)

# TXT: reuse Stage 15's set
enr = pd.read_csv(tax_dir / "enrichment_tests.csv")
txt_concepts = sorted(set(c for c in enr['concept'].unique() if c.startswith('TXT:')))
print(f"  ICD: {sum(1 for k in concept_masks if k.startswith('ICD'))}")
print(f"  NUM: {sum(1 for k in concept_masks if k.startswith('NUM'))}")
print(f"  TXT: {len(txt_concepts)}")

ARTIFACT = ['12 leads are missing', 'leads are missing',
            'based on available leads', 'poor quality data']

for c in txt_concepts:
    phrase = c[4:].lower().strip().rstrip('.').strip()
    if any(a in phrase for a in ARTIFACT):
        continue
    mask = np.array([phrase in (r or '') for r in report])
    if mask.sum() < 200:
        continue
    concept_masks[c] = mask

print(f"\n  Total concepts to test: {len(concept_masks)}")

# ============================================================
# Per-concept: top1 vs multi AUROC
# ============================================================
print("\nRunning per-concept top1 vs multi AUROC ...")
print(f"{'concept':<48} {'n_pos':>6}  {'top1':>5}  {'multi':>5}  {'verdict':<15}")
print("-" * 90)

results = []
contributing_atoms = {}   # concept → list of atom_id with nonzero coef

t0 = time.time()
for i, (cname, mask_full) in enumerate(concept_masks.items()):
    mask_full = np.asarray(mask_full).astype(bool)
    y_train = mask_full[train_idx]
    y_test = mask_full[test_idx]
    n_pos_train = int(y_train.sum())
    n_pos_test = int(y_test.sum())

    if n_pos_train < 30 or n_pos_test < 30:
        results.append({'concept': cname, 'n_pos_train': n_pos_train,
                        'n_pos_test': n_pos_test, 'top1_auroc': np.nan,
                        'multi_auroc': np.nan, 'verdict': 'too_few_positives',
                        'n_contributing': 0})
        continue

    # ---- top1 AUROC: best single atom (subsample for speed) ----
    # Use mean-difference to pick candidates, then exact AUROC
    pos_mean = X_train[y_train].mean(axis=0)
    neg_mean = X_train[~y_train].mean(axis=0)
    md = np.abs(pos_mean - neg_mean)
    candidates = np.argsort(md)[::-1][:50]
    top1 = 0.5
    best_atom = -1
    for a in candidates:
        s = X_test[:, a]
        try:
            auc = roc_auc_score(y_test, s)
            auc = max(auc, 1 - auc)
            if auc > top1:
                top1 = auc; best_atom = int(a)
        except Exception:
            pass

    # ---- multi AUROC: L1 logistic with CV ----
    try:
        # L1 logistic, small Cs grid for speed
        clf = LogisticRegressionCV(
            Cs=[0.01, 0.1, 1.0], penalty='l1', solver='liblinear',
            cv=3, scoring='roc_auc', max_iter=200, n_jobs=4,
            class_weight='balanced'
        )
        clf.fit(X_train_s, y_train)
        probs = clf.predict_proba(X_test_s)[:, 1]
        multi = roc_auc_score(y_test, probs)
        coefs = clf.coef_[0]
        nonzero = np.where(np.abs(coefs) > 1e-6)[0]
        # Limit to top-K by |coef|
        if len(nonzero) > MAX_CONTRIBUTING_ATOMS:
            top_k = nonzero[np.argsort(np.abs(coefs[nonzero]))[::-1][:MAX_CONTRIBUTING_ATOMS]]
            contributing = sorted(top_k.tolist())
        else:
            contributing = sorted(nonzero.tolist())
    except Exception as e:
        multi = np.nan; contributing = []

    # ---- Classify ----
    delta = multi - top1 if not np.isnan(multi) else 0
    if top1 >= TOP1_CAPTURED:
        verdict = 'captured'           # Stage 15 should have found this
    elif multi >= MULTI_DISTRIB and top1 < TOP1_WEAK_MAX and delta >= DELTA_MIN:
        verdict = 'distributed'
        contributing_atoms[cname] = contributing
    elif multi >= MULTI_WEAK_MIN:
        verdict = 'weak'
    else:
        verdict = 'not_encoded'

    results.append({
        'concept': cname, 'n_pos_train': n_pos_train, 'n_pos_test': n_pos_test,
        'top1_auroc': top1, 'top1_atom': best_atom,
        'multi_auroc': multi, 'verdict': verdict,
        'n_contributing': len(contributing),
        'contributing_atoms': ';'.join(str(a) for a in contributing) if contributing else '',
    })

    print(f"{cname[:48]:<48} {n_pos_train:>6}  {top1:>5.3f}  {multi:>5.3f}  {verdict:<15}")

print(f"\nElapsed: {time.time() - t0:.0f}s")

# ============================================================
# Save concept-level results
# ============================================================
res_df = pd.DataFrame(results)
res_df.to_csv(out_dir / "concept_encoding_types.csv", index=False)

print("\n" + "=" * 70)
print("Concept encoding verdicts:")
print("=" * 70)
print(res_df['verdict'].value_counts().to_string())

# ============================================================
# Atom-level update: mark Contributing atoms
# ============================================================
print("\n" + "=" * 70)
print("Updating atom taxonomy with Contributing class")
print("=" * 70)

all_contributing = set()
for atoms in contributing_atoms.values():
    all_contributing.update(atoms)
print(f"  unique contributing atoms across all distributed concepts: {len(all_contributing)}")

tax_new = tax.copy()
# Only upgrade atoms that are CURRENTLY Uninformative (don't override Separable/Entangled)
upgrade_mask = (tax_new['atom_id'].isin(all_contributing)) & (tax_new['category'] == 'Uninformative')
n_upgrade = upgrade_mask.sum()
print(f"  Uninformative atoms upgrading to Contributing: {n_upgrade}")
tax_new.loc[upgrade_mask, 'category'] = 'Contributing'

# Add a column: which distributed concepts does this atom contribute to?
contrib_by_atom = {a: [] for a in all_contributing}
for cn, atoms in contributing_atoms.items():
    for a in atoms:
        contrib_by_atom[a].append(cn)
tax_new['contributes_to'] = tax_new['atom_id'].apply(
    lambda a: '; '.join(contrib_by_atom.get(a, [])) if a in contrib_by_atom else ''
)

print("\nFinal taxonomy:")
print(tax_new['category'].value_counts().to_string())

D = len(tax_new)
for cat in ['Separable', 'Entangled', 'Contributing', 'Uninformative', 'Dead']:
    n = (tax_new['category'] == cat).sum()
    print(f"  {cat:<15}: {n:>5} ({100*n/D:.1f}%)")

tax_new.to_csv(out_dir / "atom_taxonomy_v3.csv", index=False)
print(f"\nOutputs in: {out_dir}")
print(f"  concept_encoding_types.csv  - per-concept encoding type")
print(f"  atom_taxonomy_v3.csv        - updated atom taxonomy with Contributing")

# Sample distributed concepts
print("\n" + "=" * 70)
print("Sample distributed concepts (multi >> top1):")
print("=" * 70)
dist = res_df[res_df['verdict'] == 'distributed'].sort_values(
    'multi_auroc', ascending=False).head(10)
for _, r in dist.iterrows():
    delta = r['multi_auroc'] - r['top1_auroc']
    print(f"  {r['concept'][:45]:<45}  top1={r['top1_auroc']:.3f}  "
          f"multi={r['multi_auroc']:.3f}  Δ={delta:+.3f}  "
          f"({r['n_contributing']} contributing atoms)")
