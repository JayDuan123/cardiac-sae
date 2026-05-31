"""
For each (atom, phenotype) pair, compute association strength:
  - Point-biserial correlation (continuous atom × binary phenotype)
  - Mean activation in positive vs negative cohort
  - AUROC of single-atom classifier (atom value as score for the phenotype)

Output:
  - atom_phenotype_assoc.csv: full table (atoms × phenotypes × metrics)
  - figures/heatmap_top_atoms.png: heatmap of top differentially-associated atoms
  - figures/atom_rankings.csv: per-phenotype top-K atoms
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

# Use K=1536 (cleaner)
SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
out_dir = sae_dir / "atom_phenotype"
out_dir.mkdir(parents=True, exist_ok=True)
fig_dir = out_dir / "figures"
fig_dir.mkdir(exist_ok=True)

CLINICAL_DIR = cfg.EMBEDDING_DIR.parent / "clinical"

# ============================================================
# Load
# ============================================================
print("Loading activations ...")
acts = load_npz(sae_dir / "activations_all.npz")  # (800k, 1536) CSR
N, D = acts.shape
print(f"  shape: {acts.shape}")

print("Loading clinical labels ...")
clin = pd.read_csv(CLINICAL_DIR / "record_with_clinical.csv")
flags = pd.read_csv(CLINICAL_DIR / "phenotype_flags.csv")
df = clin.merge(flags, on='record_idx')

PHENOTYPES = [
    'atrial_fibrillation',
    'heart_failure',
    'mi___ischemic_heart',
    'diabetes_mellitus',
    'hypertension_primary',
    'ckd',
    'copd',
    'sepsis',
]

# Only use records with diagnoses (otherwise negatives are unreliable)
valid_mask = (df['n_diagnoses'].values > 0)
print(f"  records with diagnoses: {valid_mask.sum():,}")

# ============================================================
# Compute per-atom per-phenotype association
# ============================================================
print("\nComputing atom × phenotype associations ...")

# Convert sparse acts to CSC for fast column slicing
acts_csc = acts.tocsc()

results = []

for pheno in tqdm(PHENOTYPES, desc="Phenotypes"):
    y_all = df[pheno].values.astype(int)
    valid = valid_mask & ~np.isnan(y_all.astype(float))
    y = y_all[valid]
    if y.sum() < 100 or (1 - y).sum() < 100:
        print(f"  Skipping {pheno}: too few labels (pos={y.sum()}, neg={(1-y).sum()})")
        continue

    pos_rate = y.mean()
    pos_idx = np.where(valid & (y_all == 1))[0]
    neg_idx = np.where(valid & (y_all == 0))[0]

    for atom_id in range(D):
        col = acts_csc[:, atom_id].toarray().ravel()  # (N,) atom activations
        col_valid = col[valid]

        # Mean activation in pos vs neg
        mean_pos = col[pos_idx].mean() if len(pos_idx) > 0 else 0
        mean_neg = col[neg_idx].mean() if len(neg_idx) > 0 else 0

        # Activation frequency in pos vs neg
        freq_pos = (col[pos_idx] > 0).mean() if len(pos_idx) > 0 else 0
        freq_neg = (col[neg_idx] > 0).mean() if len(neg_idx) > 0 else 0

        # AUROC of single-atom classifier (atom value as score)
        # Skip if atom never activates in valid records
        if col_valid.sum() == 0:
            auroc = 0.5
        else:
            try:
                auroc = roc_auc_score(y, col_valid)
            except ValueError:
                auroc = 0.5

        results.append({
            'phenotype': pheno,
            'atom_id': atom_id,
            'n_pos': int(y.sum()),
            'n_neg': int((1-y).sum()),
            'pos_rate': float(pos_rate),
            'mean_pos': float(mean_pos),
            'mean_neg': float(mean_neg),
            'mean_diff': float(mean_pos - mean_neg),
            'freq_pos': float(freq_pos),
            'freq_neg': float(freq_neg),
            'freq_lift': float(freq_pos / (freq_neg + 1e-6)),
            'auroc': float(auroc),
        })

assoc = pd.DataFrame(results)
assoc.to_csv(out_dir / "atom_phenotype_assoc.csv", index=False)
print(f"\nSaved: {out_dir}/atom_phenotype_assoc.csv  ({len(assoc):,} rows)")

# ============================================================
# Find top atoms per phenotype
# ============================================================
print("\n=== Top 10 atoms per phenotype (by single-atom AUROC) ===")
for pheno in PHENOTYPES:
    sub = assoc[assoc['phenotype'] == pheno]
    if len(sub) == 0:
        continue
    print(f"\n--- {pheno} ---")
    top10 = sub.nlargest(10, 'auroc')[
        ['atom_id', 'auroc', 'mean_pos', 'mean_neg', 'freq_pos', 'freq_neg', 'freq_lift']
    ]
    print(top10.to_string(index=False))

# Save top-50 per phenotype
top_per_pheno = []
for pheno in PHENOTYPES:
    sub = assoc[assoc['phenotype'] == pheno]
    if len(sub) == 0:
        continue
    top50 = sub.nlargest(50, 'auroc').copy()
    top_per_pheno.append(top50)
pd.concat(top_per_pheno).to_csv(out_dir / "top50_atoms_per_phenotype.csv", index=False)
print(f"\nSaved: {out_dir}/top50_atoms_per_phenotype.csv")

# ============================================================
# Heatmap: top-N atoms across phenotypes
# ============================================================
print("\nBuilding heatmap ...")
TOP_N = 30  # show top-30 atoms per phenotype (union)

top_atoms = set()
for pheno in PHENOTYPES:
    sub = assoc[assoc['phenotype'] == pheno]
    if len(sub) == 0:
        continue
    top_atoms.update(sub.nlargest(TOP_N, 'auroc')['atom_id'].tolist())

top_atoms = sorted(top_atoms)
print(f"  Total unique top atoms: {len(top_atoms)}")

# Build matrix: rows = atoms, cols = phenotypes, values = AUROC
heatmap_data = np.full((len(top_atoms), len(PHENOTYPES)), 0.5)
atom_to_row = {a: i for i, a in enumerate(top_atoms)}
for pheno_idx, pheno in enumerate(PHENOTYPES):
    sub = assoc[assoc['phenotype'] == pheno]
    if len(sub) == 0:
        continue
    sub_top = sub[sub['atom_id'].isin(top_atoms)]
    for _, row in sub_top.iterrows():
        heatmap_data[atom_to_row[row['atom_id']], pheno_idx] = row['auroc']

# Sort atoms by max AUROC for clean visualization
max_aurocs = heatmap_data.max(axis=1)
sort_order = np.argsort(-max_aurocs)
heatmap_sorted = heatmap_data[sort_order]
atom_labels = [f"atom_{top_atoms[i]}" for i in sort_order]

# Show only top-50 most discriminative for readability
DISPLAY_N = 50
display_data = heatmap_sorted[:DISPLAY_N]
display_labels = atom_labels[:DISPLAY_N]

fig, ax = plt.subplots(figsize=(8, max(12, DISPLAY_N * 0.25)))
sns.heatmap(display_data, ax=ax,
            xticklabels=[p.replace('_', ' ').replace('___', ' / ') for p in PHENOTYPES],
            yticklabels=display_labels,
            cmap='RdYlBu_r', center=0.5, vmin=0.3, vmax=0.85,
            cbar_kws={'label': 'single-atom AUROC'},
            linewidths=0.3, linecolor='white')
ax.set_title(f'Top-{DISPLAY_N} atoms × phenotypes (single-atom AUROC)')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(fig_dir / "heatmap_top_atoms.png", dpi=140, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_dir}/heatmap_top_atoms.png")

# ============================================================
# Histogram of max AUROC per atom (which atoms are "informative")
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4))
max_per_atom = assoc.groupby('atom_id')['auroc'].max().values
ax.hist(max_per_atom, bins=80, color='steelblue', edgecolor='k', alpha=0.7)
ax.axvline(0.5, color='k', linestyle=':', label='random=0.5')
ax.axvline(0.7, color='r', linestyle='--', label='AUROC=0.7')
ax.set_xlabel('Max single-atom AUROC across all phenotypes')
ax.set_ylabel('Number of atoms')
ax.set_title(f'How many atoms have phenotype-predictive value? (K={D} atoms)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "atom_predictive_distribution.png", dpi=120)
plt.close()
print(f"Saved: {fig_dir}/atom_predictive_distribution.png")

# Stats
print(f"\n=== Atom predictive value distribution ===")
print(f"Atoms with max AUROC > 0.6: {(max_per_atom > 0.6).sum()} ({100*(max_per_atom > 0.6).mean():.1f}%)")
print(f"Atoms with max AUROC > 0.7: {(max_per_atom > 0.7).sum()} ({100*(max_per_atom > 0.7).mean():.1f}%)")
print(f"Atoms with max AUROC > 0.8: {(max_per_atom > 0.8).sum()} ({100*(max_per_atom > 0.8).mean():.1f}%)")

print("\nDone.")
