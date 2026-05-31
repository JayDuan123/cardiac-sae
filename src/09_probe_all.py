"""
Run linear/logistic probes on all 3 representations × 7 tasks.

Representations:
  - dense CSFM (768)
  - sparse SAE K=1536
  - sparse SAE K=12288

Tasks:
  - age (regression)
  - sex / AF / HF / MI / DM / HTN (binary classification)

Output:
  - probe_results.csv: full table
  - figures/probe_recovery.png: Probe Recovery Score (PRS) bar chart
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix, vstack
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    roc_auc_score, average_precision_score, r2_score, mean_absolute_error
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

CLINICAL_DIR = cfg.EMBEDDING_DIR.parent / "clinical"
PROBE_DIR = cfg.EMBEDDING_DIR.parent / "probes"
PROBE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = PROBE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ============================================================
# Load all data
# ============================================================
print("Loading clinical labels ...")
clin = pd.read_csv(CLINICAL_DIR / "record_with_clinical.csv")
flags = pd.read_csv(CLINICAL_DIR / "phenotype_flags.csv")
df = clin.merge(flags, on='record_idx')
N_total = len(df)
print(f"  {N_total:,} records")

# Build label vectors. For binary tasks, restrict to records with ICD data
# (or treat missing as 0 — we'll be explicit per task)
print("Loading dense embeddings ...")
emb_mmap = np.memmap(
    cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_embeddings.npy",
    dtype=np.float16, mode='r',
    shape=(N_total, cfg.CSFM_DIM),
)
# Convert to float32 for sklearn
print("  Converting to float32 (one-time, ~2 GB) ...")
dense_emb = np.asarray(emb_mmap, dtype=np.float32)
print(f"  dense: {dense_emb.shape}")

print("\nLoading sparse activations ...")
sae_acts = {}
for sae_name in ["batchtopk_tiny_aws_k32_d1536", "batchtopk_tiny_aws_k32_d12288"]:
    p = cfg.SAE_DIR / sae_name / "activations_all.npz"
    a = load_npz(p)
    sae_acts[sae_name] = a
    print(f"  {sae_name}: {a.shape}, nnz={a.nnz:,}")

# ============================================================
# Subject-level split (avoid leak)
# ============================================================
print("\nBuilding subject-level split ...")
np.random.seed(42)
subjects = df['subject_id'].unique()
np.random.shuffle(subjects)
n_test = int(0.1 * len(subjects))
n_val = int(0.1 * len(subjects))
test_subjects = set(subjects[:n_test])
val_subjects = set(subjects[n_test:n_test + n_val])
train_subjects = set(subjects[n_test + n_val:])

df['split'] = 'train'
df.loc[df['subject_id'].isin(val_subjects), 'split'] = 'val'
df.loc[df['subject_id'].isin(test_subjects), 'split'] = 'test'

split_counts = df['split'].value_counts()
print(f"  train: {split_counts['train']:,}  val: {split_counts['val']:,}  test: {split_counts['test']:,}")

train_idx = df.index[df['split'] == 'train'].values
val_idx = df.index[df['split'] == 'val'].values
test_idx = df.index[df['split'] == 'test'].values

# ============================================================
# Helper: prepare X/y for a task
# ============================================================
def get_X(rep_name, indices):
    """Get features for given record indices, dense ndarray or csr matrix."""
    if rep_name == 'dense':
        return dense_emb[indices]
    else:
        return sae_acts[rep_name][indices].tocsr()


# ============================================================
# Task definitions
# ============================================================
TASKS = [
    # (name, label_column, type, restrict_to_records_with_label)
    ('age',  'age_at_ecg',          'regression',     False),
    ('sex',  'gender',               'binary',         False),
    ('af',   'atrial_fibrillation',  'binary',         True),
    ('hf',   'heart_failure',        'binary',         True),
    ('mi',   'mi___ischemic_heart',  'binary',         True),
    ('dm',   'diabetes_mellitus',    'binary',         True),
    ('htn',  'hypertension_primary', 'binary',         True),
]

REPRESENTATIONS = [
    'dense',
    'batchtopk_tiny_aws_k32_d1536',
    'batchtopk_tiny_aws_k32_d12288',
]


def prepare_task(task_name, label_col, task_type, restrict_labeled):
    """Build labels and filter indices."""
    y_all = df[label_col].copy()

    if task_type == 'binary' and label_col == 'gender':
        # F=1, M=0
        y_all = (y_all == 'F').astype(int).values
        valid = df['gender'].isin(['F', 'M']).values
    elif task_type == 'binary':
        y_all = y_all.astype(int).values
        if restrict_labeled:
            # Only records with ICD info (n_diagnoses > 0)
            valid = df['n_diagnoses'].values > 0
        else:
            valid = np.ones(N_total, dtype=bool)
    else:  # regression
        y_all = y_all.values
        valid = ~np.isnan(y_all)

    return y_all, valid


def run_probe(X_train, y_train, X_val, y_val, X_test, y_test, task_type, C_grid):
    """Train probe with multiple Cs (regularization), pick best on val, eval on test."""
    best_val = -np.inf if task_type == 'binary' else np.inf
    best = None
    for C in C_grid:
        if task_type == 'binary':
            clf = LogisticRegression(
                C=C, max_iter=200, solver='lbfgs',
                n_jobs=-1, random_state=42,
            )
            clf.fit(X_train, y_train)
            pred_val = clf.predict_proba(X_val)[:, 1]
            if len(np.unique(y_val)) < 2:
                continue
            val_score = roc_auc_score(y_val, pred_val)
            if val_score > best_val:
                best_val = val_score
                best = (clf, C)
        else:
            clf = Ridge(alpha=1.0/C, random_state=42)
            clf.fit(X_train, y_train)
            pred_val = clf.predict(X_val)
            val_score = mean_absolute_error(y_val, pred_val)
            if val_score < best_val or best is None:
                best_val = val_score
                best = (clf, C)

    clf, best_C = best
    if task_type == 'binary':
        pred_test = clf.predict_proba(X_test)[:, 1]
        return {
            'auroc': roc_auc_score(y_test, pred_test),
            'auprc': average_precision_score(y_test, pred_test),
            'val_auroc': best_val,
            'best_C': best_C,
        }
    else:
        pred_test = clf.predict(X_test)
        return {
            'mae': mean_absolute_error(y_test, pred_test),
            'r2':  r2_score(y_test, pred_test),
            'val_mae': best_val,
            'best_C': best_C,
        }


# ============================================================
# Main loop
# ============================================================
results = []
C_GRID = [0.01, 0.1, 1.0, 10.0]   # smaller = more reg

for task_name, label_col, task_type, restrict_labeled in TASKS:
    print(f"\n{'='*60}\nTask: {task_name} ({task_type})\n{'='*60}")

    y_all, valid = prepare_task(task_name, label_col, task_type, restrict_labeled)

    tr_idx = np.intersect1d(train_idx, np.where(valid)[0])
    va_idx = np.intersect1d(val_idx, np.where(valid)[0])
    te_idx = np.intersect1d(test_idx, np.where(valid)[0])
    y_tr, y_va, y_te = y_all[tr_idx], y_all[va_idx], y_all[te_idx]

    print(f"  train: {len(tr_idx):,}  val: {len(va_idx):,}  test: {len(te_idx):,}")
    if task_type == 'binary':
        print(f"  positive rate (test): {y_te.mean():.3f}")

    for rep_name in REPRESENTATIONS:
        print(f"  ► {rep_name}...", end='', flush=True)
        t0 = time.time()
        Xtr = get_X(rep_name, tr_idx)
        Xva = get_X(rep_name, va_idx)
        Xte = get_X(rep_name, te_idx)
        m = run_probe(Xtr, y_tr, Xva, y_va, Xte, y_te, task_type, C_GRID)
        elapsed = time.time() - t0

        row = {
            'task': task_name, 'task_type': task_type,
            'representation': rep_name,
            'n_train': len(tr_idx), 'n_val': len(va_idx), 'n_test': len(te_idx),
            'pos_rate_test': float(y_te.mean()) if task_type == 'binary' else None,
            **m,
            'time_s': elapsed,
        }
        results.append(row)
        if task_type == 'binary':
            print(f" AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} ({elapsed:.0f}s)")
        else:
            print(f" MAE={m['mae']:.3f} R²={m['r2']:.4f} ({elapsed:.0f}s)")

# ============================================================
# Save and summarize
# ============================================================
res_df = pd.DataFrame(results)
res_df.to_csv(PROBE_DIR / "probe_results.csv", index=False)
print(f"\n\nSaved: {PROBE_DIR}/probe_results.csv")

# Pivot for easy reading
print("\n=== Summary ===\n")
for task_type in ['binary', 'regression']:
    sub = res_df[res_df['task_type'] == task_type]
    if len(sub) == 0:
        continue
    metric = 'auroc' if task_type == 'binary' else 'r2'
    print(f"\n{task_type} tasks (metric: {metric}):")
    pivot = sub.pivot(index='task', columns='representation', values=metric)
    # Rename for readability
    pivot = pivot.rename(columns={
        'batchtopk_tiny_aws_k32_d1536': 'SAE_K=1536',
        'batchtopk_tiny_aws_k32_d12288': 'SAE_K=12288',
    })
    print(pivot.round(4).to_string())

    # Probe Recovery Score: SAE / dense
    print(f"\n{task_type} tasks: Probe Recovery Score (PRS)")
    prs = pivot.copy()
    for col in pivot.columns:
        if col != 'dense':
            prs[f'PRS({col})'] = pivot[col] / pivot['dense']
    print(prs[[c for c in prs.columns if 'PRS' in c]].round(4).to_string())

# ============================================================
# Plot
# ============================================================
binary_tasks = res_df[res_df['task_type'] == 'binary'].copy()
fig, ax = plt.subplots(figsize=(11, 5))

tasks_order = ['sex', 'af', 'hf', 'mi', 'dm', 'htn']
x = np.arange(len(tasks_order))
width = 0.25

for i, rep in enumerate(REPRESENTATIONS):
    aurocs = []
    for t in tasks_order:
        v = binary_tasks[(binary_tasks['task'] == t) &
                         (binary_tasks['representation'] == rep)]['auroc'].values
        aurocs.append(v[0] if len(v) > 0 else 0)
    label = {'dense': 'Dense CSFM (768)',
             'batchtopk_tiny_aws_k32_d1536': 'SAE K=1536',
             'batchtopk_tiny_aws_k32_d12288': 'SAE K=12288'}[rep]
    ax.bar(x + (i-1)*width, aurocs, width, label=label)

ax.set_xticks(x)
ax.set_xticklabels([t.upper() for t in tasks_order])
ax.set_ylabel('AUROC')
ax.set_ylim(0.5, 1.0)
ax.axhline(0.5, color='k', linestyle=':', alpha=0.5)
ax.set_title('Probe AUROC: dense vs sparse SAE representations')
ax.legend()
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / "probe_auroc.png", dpi=120, bbox_inches='tight')
plt.close()
print(f"\nSaved: {FIG_DIR}/probe_auroc.png")

print("\nDone.")
