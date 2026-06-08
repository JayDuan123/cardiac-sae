"""
Stage 20: HR-stratified confounder check on Separable atoms.

After Stage 15 (with MIN_EFFECT=0.70) declares an atom Separable to concept C,
we ask:
  Is the enrichment still significant within an HR-matched stratum?
  (i.e., does it survive when we control for heart rate?)

If a "Separable" atom only achieves AUROC>0.70 because both the atom and
the concept correlate with HR (like our atom 29 example), the stratified
test will fail and we downgrade the atom.

Method:
  For each (atom, concept) that passed Stage 15:
    Split data into HR bins: [0,60), [60,80), [80,100), [100,120), [120,250)
    Run Mann-Whitney WITHIN each bin where n_pos >= 30
    Compute per-bin AUROC
    Aggregate: take median AUROC across bins with sufficient data
    
    If median bin-AUROC < 0.65 → flag as HR-confounded, downgrade to Entangled
    If median bin-AUROC >= 0.65 → confirm as truly Separable
    If too few bins have data → keep as Separable (cannot test)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import mannwhitneyu

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

# ============================================================
# Config
# ============================================================
SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
tax_dir = sae_dir / "taxonomy"
out_dir = sae_dir / "taxonomy_stratified"
out_dir.mkdir(parents=True, exist_ok=True)

HR_BINS = [(0, 60), (60, 80), (80, 100), (100, 120), (120, 250)]
MIN_POS_PER_BIN = 30                 # min positives per bin to test
MIN_BINS_TO_AGGREGATE = 2            # need >= 2 bins to make a call
CONFOUND_THRESHOLD = 0.65            # if median bin-AUROC < this, mark as confounded

# ============================================================
# Load data
# ============================================================
print("=" * 70)
print("Stage 20: HR-stratified confounder test on Separable atoms")
print("=" * 70)

acts = load_npz(sae_dir / "activations_all.npz").tocsc()
N, D = acts.shape
print(f"\n  activations: {acts.shape}")

tax = pd.read_csv(tax_dir / "atom_taxonomy.csv")
enr = pd.read_csv(tax_dir / "enrichment_tests.csv")
print(f"  Stage 15 taxonomy: {len(tax)} atoms")
print(f"    Separable: {(tax['category']=='Separable').sum()}")

# Meta + HR
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
meta = meta.reset_index().rename(columns={'index': 'record_idx'})
meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')

mm = pd.read_csv("/workspace/data/mimic-iv-ecg-aws/machine_measurements.csv",
                 usecols=['study_id', 'rr_interval'] + [f'report_{i}' for i in range(18)],
                 low_memory=False)
mm['study_id'] = mm['study_id'].astype('Int64')
mm['heart_rate'] = 60000 / mm['rr_interval'].replace(0, np.nan)

feat = meta[['record_idx', 'study_id']].merge(mm, on='study_id', how='left')
feat = feat.set_index('record_idx').reindex(range(N))

rep_cols = [f'report_{i}' for i in range(18)]
def all_reports(row):
    return ' || '.join(str(s) for s in row.values if pd.notna(s) and str(s).strip())
feat['full_report'] = feat[rep_cols].apply(all_reports, axis=1)
feat['report_lower'] = feat['full_report'].str.lower()

hr_all = feat['heart_rate'].values
print(f"  HR available for {(~np.isnan(hr_all)).sum():,} / {N:,} records")

# ============================================================
# Helper: build dense activation vector for an atom
# ============================================================
def dense_act(atom_id):
    a = np.zeros(N, dtype=np.float32)
    st, en = acts.indptr[atom_id], acts.indptr[atom_id + 1]
    a[acts.indices[st:en]] = acts.data[st:en]
    return a

# ============================================================
# Helper: build positive mask for a concept name
# ============================================================
ARTIFACT_PHRASES = ['12 leads are missing', 'leads are missing',
                    'based on available leads', 'poor quality data']

def concept_positive_mask(concept_name):
    """Reconstruct the positive set used by Stage 15."""
    if concept_name.startswith('NUM:'):
        feat_name = concept_name[4:]
        # Recompute the same binary definitions used in Stage 15
        if feat_name == 'tachycardia':  return (hr_all > 100)
        if feat_name == 'bradycardia':  return (hr_all < 60)
        # Wide QRS, long QT, axes — need their underlying numerical
        # For stratification we mainly care about HR-coupled concepts;
        # if not implemented just skip (return None)
        return None
    if concept_name.startswith('TXT:'):
        phrase = concept_name[4:].lower().strip().rstrip('.').strip()
        if any(a in phrase for a in ARTIFACT_PHRASES):
            return None
        mask = feat['report_lower'].str.contains(phrase, na=False, regex=False).values
        return mask
    if concept_name.startswith('ICD:'):
        # ICD masks aren't easily reproducible here; skip
        return None
    return None

# ============================================================
# Main: for each Separable atom, run stratified test
# ============================================================
separable = tax[tax['category'] == 'Separable'].copy()
print(f"\nRunning stratified test on {len(separable)} Separable atoms ...")

# Find each Separable atom's enriched concept(s)
enr_pass = enr[enr['enriched']].copy()  # already AUROC>0.70 in new run

results = []
for ai, (_, row) in enumerate(separable.iterrows()):
    atom_id = int(row['atom_id'])
    # Look up enriched concept(s) for this atom
    atom_enr = enr_pass[enr_pass['atom_id'] == atom_id]
    if len(atom_enr) == 0:
        continue

    a_vec = dense_act(atom_id)

    for _, er in atom_enr.iterrows():
        concept = er['concept']
        global_auroc = er['auroc']

        pos_mask = concept_positive_mask(concept)
        if pos_mask is None:
            # Cannot stratify (ICD or NUM-non-HR concepts); keep as-is
            results.append({
                'atom_id': atom_id, 'concept': concept,
                'global_auroc': global_auroc,
                'stratifiable': False,
                'bin_aurocs': '',
                'median_bin_auroc': np.nan,
                'verdict': 'unstratified'
            })
            continue

        # Compute per-bin AUROC
        bin_aurocs = []
        bin_ns = []
        for lo, hi in HR_BINS:
            in_bin = (hr_all >= lo) & (hr_all < hi) & ~np.isnan(hr_all)
            pos_in_bin = in_bin & pos_mask
            neg_in_bin = in_bin & ~pos_mask
            n_pos = pos_in_bin.sum()
            n_neg = neg_in_bin.sum()
            if n_pos < MIN_POS_PER_BIN or n_neg < MIN_POS_PER_BIN:
                continue
            try:
                pos_vals = a_vec[pos_in_bin]
                neg_vals = a_vec[neg_in_bin]
                # Subsample negatives for speed
                if len(neg_vals) > 5000:
                    rng = np.random.RandomState(atom_id)
                    neg_vals = rng.choice(neg_vals, 5000, replace=False)
                U, _ = mannwhitneyu(pos_vals, neg_vals, alternative='greater')
                auroc_bin = U / (len(pos_vals) * len(neg_vals))
                bin_aurocs.append(auroc_bin)
                bin_ns.append(n_pos)
            except Exception:
                continue

        if len(bin_aurocs) < MIN_BINS_TO_AGGREGATE:
            results.append({
                'atom_id': atom_id, 'concept': concept,
                'global_auroc': global_auroc,
                'stratifiable': False,
                'bin_aurocs': ';'.join(f'{x:.2f}' for x in bin_aurocs),
                'median_bin_auroc': np.nan,
                'verdict': 'too_few_bins'
            })
            continue

        median_bin_auroc = float(np.median(bin_aurocs))
        if median_bin_auroc < CONFOUND_THRESHOLD:
            verdict = 'HR-confounded'
        else:
            verdict = 'confirmed'
        results.append({
            'atom_id': atom_id, 'concept': concept,
            'global_auroc': global_auroc,
            'stratifiable': True,
            'bin_aurocs': ';'.join(f'{x:.2f}' for x in bin_aurocs),
            'median_bin_auroc': median_bin_auroc,
            'verdict': verdict
        })

    if (ai + 1) % 20 == 0:
        print(f"  processed {ai+1}/{len(separable)} atoms")

res_df = pd.DataFrame(results)
res_df.to_csv(out_dir / "stratified_results.csv", index=False)

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("Stratified test verdicts:")
print("=" * 70)
print(res_df['verdict'].value_counts().to_string())

# Update taxonomy: atoms with any HR-confounded verdict → downgrade
confounded_atoms = set(res_df.loc[res_df['verdict'] == 'HR-confounded', 'atom_id'].unique())
print(f"\nAtoms with at least one HR-confounded concept: {len(confounded_atoms)}")

tax_new = tax.copy()
# Downgrade confounded Separable atoms to Entangled
# (because they're "enriched" but the enrichment is spurious through HR)
mask = tax_new['atom_id'].isin(confounded_atoms) & (tax_new['category'] == 'Separable')
print(f"Downgrading {mask.sum()} Separable atoms → Entangled (HR-confounded)")
tax_new.loc[mask, 'category'] = 'Entangled-Confounded'

print("\nFinal taxonomy:")
print(tax_new['category'].value_counts().to_string())

tax_new.to_csv(out_dir / "atom_taxonomy_stratified.csv", index=False)

# ============================================================
# Show top examples
# ============================================================
print("\n" + "=" * 70)
print("Examples of HR-confounded Separable atoms (incl. atom 29 if present):")
print("=" * 70)
conf = res_df[res_df['verdict'] == 'HR-confounded'].head(10)
for _, r in conf.iterrows():
    print(f"  atom {int(r['atom_id']):>4}  concept '{r['concept'][:35]:35s}'  "
          f"global={r['global_auroc']:.3f}  median_bin={r['median_bin_auroc']:.3f}  "
          f"bins=[{r['bin_aurocs']}]")

print("\n" + "=" * 70)
print(f"Outputs in: {out_dir}")
print(f"  stratified_results.csv         - per (atom, concept) verdict")
print(f"  atom_taxonomy_stratified.csv   - updated taxonomy")
