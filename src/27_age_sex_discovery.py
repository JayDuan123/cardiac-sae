"""
Stage 27: Discover age/sex encoding in the SAE dictionary.

Age and sex are NOT in our concept inventory but are known to systematically
affect ECG morphology (QT length with age, QRS amplitude with sex, etc.).
If the foundation model learned these attributes, the SAE may contain atoms
encoding them—atoms that Stage 15/17 cannot identify because age/sex are
not in report text or numerical measurements.

Pipeline (mirrors Stage 21 distributed analysis):
  1. Extract age/sex from clinical data (record_with_clinical.csv)
  2. For each attribute:
     - Compute per-atom Spearman r (continuous)
     - Compute per-atom AUROC (binary: age>=65, M vs F)
     - Compute multi-atom L1-logistic AUROC
  3. Classify atom encoding type per attribute:
     captured / distributed / weak / not_encoded
  4. Identify "age atoms" and "sex atoms" — flag them
     as potential confounders for other atom interpretations
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
out_dir = sae_dir / "age_sex_discovery"
out_dir.mkdir(parents=True, exist_ok=True)

TRAIN_N = 100_000
TEST_N = 50_000
TOP_K_TO_REPORT = 20

# ============================================================
# Load
# ============================================================
print("=" * 70)
print("Stage 27: Age & Sex encoding in SAE dictionary")
print("=" * 70)

acts = load_npz(sae_dir / "activations_all.npz").tocsr()
N, D = acts.shape
print(f"  activations: {acts.shape}")

# Clinical: subject_id, age, sex
CLINICAL_DIR = cfg.EMBEDDING_DIR.parent / "clinical"
clin = pd.read_csv(CLINICAL_DIR / "record_with_clinical.csv")
clin = clin.set_index('record_idx').reindex(range(N))

# Check what columns we have
print(f"\nClinical columns: {[c for c in clin.columns if c in ['anchor_age', 'age', 'gender', 'sex', 'subject_id']]}")

# Determine age/sex column names
age_col = 'anchor_age' if 'anchor_age' in clin.columns else 'age'
sex_col = 'gender' if 'gender' in clin.columns else 'sex'

age = clin[age_col].values if age_col in clin.columns else None
sex_raw = clin[sex_col].values if sex_col in clin.columns else None

if age is None or sex_raw is None:
    print(f"  ⚠ missing age/sex columns. age col exists: {age_col in clin.columns}, "
          f"sex col exists: {sex_col in clin.columns}")
    sys.exit(1)

# Convert sex to binary (M=1, F=0)
sex_bin = np.where(pd.Series(sex_raw).str.upper() == 'M', 1,
          np.where(pd.Series(sex_raw).str.upper() == 'F', 0, np.nan)).astype(float)

print(f"\n  Age: n={(~pd.isna(age)).sum():,}, median={np.nanmedian(age):.0f}, "
      f"range=[{np.nanmin(age):.0f}, {np.nanmax(age):.0f}]")
print(f"  Sex: M={int((sex_bin==1).sum()):,}, F={int((sex_bin==0).sum()):,}")

# ============================================================
# Subject-level train/test split
# ============================================================
print("\nSubject-level split ...")
subj = clin['subject_id'].values
valid = ~pd.isna(subj) & ~pd.isna(age) & ~pd.isna(sex_bin)
unique_subj = np.unique(subj[valid])
rng = np.random.RandomState(42); rng.shuffle(unique_subj)
n_train_subj = int(len(unique_subj) * 0.7)
train_subj = set(unique_subj[:n_train_subj].tolist())
test_subj = set(unique_subj[n_train_subj:].tolist())

train_mask = np.array([s in train_subj if pd.notna(s) else False for s in subj]) & valid
test_mask = np.array([s in test_subj if pd.notna(s) else False for s in subj]) & valid

train_idx = np.where(train_mask)[0]
test_idx = np.where(test_mask)[0]
if len(train_idx) > TRAIN_N:
    train_idx = rng.choice(train_idx, TRAIN_N, replace=False)
if len(test_idx) > TEST_N:
    test_idx = rng.choice(test_idx, TEST_N, replace=False)
print(f"  train: {len(train_idx):,}, test: {len(test_idx):,}")

X_train = acts[train_idx].toarray().astype(np.float32)
X_test = acts[test_idx].toarray().astype(np.float32)
age_train, age_test = age[train_idx], age[test_idx]
sex_train, sex_test = sex_bin[train_idx], sex_bin[test_idx]

scaler = StandardScaler(with_mean=False)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ============================================================
# Per-atom Spearman r with age
# ============================================================
print("\n[Age analysis]")
print("  computing per-atom Spearman r with age (top atoms only) ...")
# Pre-rank age once
from scipy.stats import rankdata
age_rank = rankdata(age_train)
# For speed, compute corr on training set only
spearman_age = np.zeros(D)
for a in range(D):
    col = X_train[:, a]
    if (col > 0).sum() < 100:
        continue
    col_rank = rankdata(col)
    spearman_age[a], _ = spearmanr(col, age_train)
print(f"  Spearman done. Top atoms by |r| with age:")
top_age = np.argsort(-np.abs(spearman_age))[:TOP_K_TO_REPORT]
for a in top_age[:10]:
    print(f"    atom {a:4d}: Spearman r={spearman_age[a]:+.3f}")

# ============================================================
# Per-atom top1 AUROC: age >= 65
# ============================================================
print("\n  Per-atom AUROC (age >= 65) ...")
age65_train = (age_train >= 65).astype(int)
age65_test = (age_test >= 65).astype(int)
print(f"    train age>=65: {age65_train.sum()}/{len(age65_train)} ({100*age65_train.mean():.0f}%)")

# Pick top atoms by mean-diff on train, then test AUROC on test
pos_mean = X_train[age65_train == 1].mean(axis=0)
neg_mean = X_train[age65_train == 0].mean(axis=0)
mean_diff = np.abs(pos_mean - neg_mean)
cands = np.argsort(-mean_diff)[:100]
top1_age = 0.5
best_atom_age = -1
top10_age_atoms = []
for a in cands:
    try:
        auc = roc_auc_score(age65_test, X_test[:, a])
        auc = max(auc, 1 - auc)
        top10_age_atoms.append((a, auc))
        if auc > top1_age:
            top1_age = auc
            best_atom_age = int(a)
    except Exception:
        pass
top10_age_atoms.sort(key=lambda x: -x[1])
print(f"  Top-1 atom for age>=65: atom {best_atom_age}, AUROC={top1_age:.3f}")
print(f"  Top-10 age atoms:")
for a, auc in top10_age_atoms[:10]:
    print(f"    atom {a:4d}: AUROC={auc:.3f}")

# ============================================================
# Multi-atom L1 logistic for age
# ============================================================
print("\n  Multi-atom L1 logistic for age >= 65 ...")
clf_age = LogisticRegressionCV(
    Cs=[0.01, 0.1, 1.0], penalty='l1', solver='liblinear',
    cv=3, scoring='roc_auc', max_iter=200, n_jobs=4,
    class_weight='balanced')
clf_age.fit(X_train_s, age65_train)
probs = clf_age.predict_proba(X_test_s)[:, 1]
multi_age = roc_auc_score(age65_test, probs)
nz_age = np.where(np.abs(clf_age.coef_[0]) > 1e-6)[0]
print(f"  Multi-atom AUROC (age >= 65): {multi_age:.3f}")
print(f"  L1 selected {len(nz_age)} atoms (non-zero coefs)")

# Continuous age via Ridge
print("\n  Ridge regression for continuous age ...")
clf_age_reg = RidgeCV(alphas=[0.1, 1.0, 10.0])
clf_age_reg.fit(X_train_s, age_train)
age_pred = clf_age_reg.predict(X_test_s)
r2_age = r2_score(age_test, age_pred)
mae_age = np.abs(age_pred - age_test).mean()
print(f"  R² = {r2_age:.3f}, MAE = {mae_age:.1f} years")

# ============================================================
# Sex analysis (same structure)
# ============================================================
print("\n[Sex analysis]")
print(f"    train M: {int(sex_train.sum())}/{len(sex_train)} ({100*sex_train.mean():.0f}%)")
# Spearman with sex (point-biserial really)
print("  Per-atom correlation with sex ...")
biserial_sex = np.zeros(D)
for a in range(D):
    col = X_train[:, a]
    if (col > 0).sum() < 100:
        continue
    biserial_sex[a], _ = spearmanr(col, sex_train)
top_sex = np.argsort(-np.abs(biserial_sex))[:TOP_K_TO_REPORT]
print(f"  Top atoms by |r| with sex:")
for a in top_sex[:10]:
    print(f"    atom {a:4d}: r={biserial_sex[a]:+.3f}")

# Top-1 AUROC for sex
print("\n  Per-atom AUROC (Male vs Female) ...")
pos_mean = X_train[sex_train == 1].mean(axis=0)
neg_mean = X_train[sex_train == 0].mean(axis=0)
mean_diff = np.abs(pos_mean - neg_mean)
cands = np.argsort(-mean_diff)[:100]
top1_sex = 0.5
best_atom_sex = -1
top10_sex_atoms = []
for a in cands:
    try:
        auc = roc_auc_score(sex_test, X_test[:, a])
        auc = max(auc, 1 - auc)
        top10_sex_atoms.append((a, auc))
        if auc > top1_sex:
            top1_sex = auc
            best_atom_sex = int(a)
    except Exception:
        pass
top10_sex_atoms.sort(key=lambda x: -x[1])
print(f"  Top-1 atom for sex: atom {best_atom_sex}, AUROC={top1_sex:.3f}")
print(f"  Top-10 sex atoms:")
for a, auc in top10_sex_atoms[:10]:
    print(f"    atom {a:4d}: AUROC={auc:.3f}")

# Multi-atom for sex
print("\n  Multi-atom L1 logistic for sex ...")
clf_sex = LogisticRegressionCV(
    Cs=[0.01, 0.1, 1.0], penalty='l1', solver='liblinear',
    cv=3, scoring='roc_auc', max_iter=200, n_jobs=4)
clf_sex.fit(X_train_s, sex_train)
probs_sex = clf_sex.predict_proba(X_test_s)[:, 1]
multi_sex = roc_auc_score(sex_test, probs_sex)
nz_sex = np.where(np.abs(clf_sex.coef_[0]) > 1e-6)[0]
print(f"  Multi-atom AUROC (sex): {multi_sex:.3f}")
print(f"  L1 selected {len(nz_sex)} atoms")

# ============================================================
# Cross-check with existing taxonomy
# ============================================================
print("\n[Cross-check with Stage 22 taxonomy]")
tax_path = sae_dir / "taxonomy_grouped" / "atom_taxonomy_grouped.csv"
if tax_path.exists():
    tax = pd.read_csv(tax_path)
    cat_col = 'category_grouped' if 'category_grouped' in tax.columns else 'category'
    
    print(f"\n  Top-10 age atoms categories:")
    for a, auc in top10_age_atoms[:10]:
        cat = tax[tax['atom_id'] == a].iloc[0][cat_col] if (tax['atom_id'] == a).any() else 'N/A'
        print(f"    atom {a:4d} (AUROC={auc:.3f}): {cat}")
    
    print(f"\n  Top-10 sex atoms categories:")
    for a, auc in top10_sex_atoms[:10]:
        cat = tax[tax['atom_id'] == a].iloc[0][cat_col] if (tax['atom_id'] == a).any() else 'N/A'
        print(f"    atom {a:4d} (AUROC={auc:.3f}): {cat}")

# ============================================================
# Save
# ============================================================
results = {
    'age_top1_auroc': float(top1_age),
    'age_multi_auroc': float(multi_age),
    'age_R2': float(r2_age),
    'age_MAE': float(mae_age),
    'age_n_contributing_atoms': int(len(nz_age)),
    'age_top1_atom': int(best_atom_age),
    'sex_top1_auroc': float(top1_sex),
    'sex_multi_auroc': float(multi_sex),
    'sex_n_contributing_atoms': int(len(nz_sex)),
    'sex_top1_atom': int(best_atom_sex),
}

pd.DataFrame([results]).to_csv(out_dir / "summary.csv", index=False)

# Per-atom details
pd.DataFrame({
    'atom_id': range(D),
    'spearman_age': spearman_age,
    'biserial_sex': biserial_sex,
}).to_csv(out_dir / "per_atom_correlations.csv", index=False)

# Top atoms
pd.DataFrame(top10_age_atoms[:TOP_K_TO_REPORT], 
             columns=['atom_id', 'auroc_age65']).to_csv(out_dir / "top_age_atoms.csv", index=False)
pd.DataFrame(top10_sex_atoms[:TOP_K_TO_REPORT],
             columns=['atom_id', 'auroc_sex']).to_csv(out_dir / "top_sex_atoms.csv", index=False)

# ============================================================
# Final verdict
# ============================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

def classify(top1, multi):
    if top1 >= 0.70:
        return f"CAPTURED  (top1={top1:.3f}, single atom suffices)"
    elif multi >= 0.75 and top1 < 0.65 and (multi - top1) >= 0.10:
        return f"DISTRIBUTED  (top1={top1:.3f}, multi={multi:.3f})"
    elif multi >= 0.65:
        return f"WEAK  (top1={top1:.3f}, multi={multi:.3f})"
    else:
        return f"NOT ENCODED  (top1={top1:.3f}, multi={multi:.3f})"

print(f"\n  Age (>=65): {classify(top1_age, multi_age)}")
print(f"    continuous age R² = {r2_age:.3f}, MAE = {mae_age:.1f} years")
print(f"  Sex (M/F):  {classify(top1_sex, multi_sex)}")

print(f"\nResults in: {out_dir}")
