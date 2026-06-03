"""
Stage 15: Rigorous atom taxonomy via enrichment significance testing.

Replaces the flawed single-atom-AUROC>0.65 threshold with the EEG-SAE
methodology: per-(atom, concept) enrichment test + Benjamini-Hochberg
FDR correction (q < 0.05).

Concept sources (union):
  (a) ICD phenotypes (5): AF, HF, MI, DM, HTN
  (b) Numerical ECG binary features (8): tachycardia, bradycardia,
      wide_qrs, long_qt, left_axis, right_axis, st_elevation, st_depression
  (c) Top report phrases (from Stage 12)

Test: one-sided Mann-Whitney U (atom higher in concept-positive ECGs).
Correction: Benjamini-Hochberg across ALL (atom x concept) pairs.

Classification per atom:
  Dead          : never fires
  Uninformative : fires but no concept enriched (q >= 0.05 for all)
  Separable     : exactly 1 concept enriched (monosemantic)
  Entangled     : >= 2 concepts enriched (polysemantic)
"""
import sys, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

mpl.rcParams.update({'font.size': 9, 'figure.dpi': 100, 'savefig.dpi': 150,
                     'savefig.bbox': 'tight'})

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
out_dir = sae_dir / "taxonomy"
out_dir.mkdir(parents=True, exist_ok=True)

Q_THRESHOLD = 0.05          # BH-corrected significance
MIN_EFFECT = 0.60           # minimum AUROC effect size (large N makes
                            # trivial differences statistically significant)
SENTINEL_MAX = 1000         # values beyond physiological range are device sentinels
MIN_POS = 30                # minimum concept-positive samples to test
MIN_ATOM_FIRES = 30         # minimum activations to consider an atom
TOP_REPORT_PHRASES = 40     # number of report phrases to include as concepts

# ============================================================
# Load activations
# ============================================================
print("=" * 60)
print("Stage 15: Rigorous Atom Taxonomy (z-test + BH correction)")
print("=" * 60)
print("\nLoading activations ...")
acts = load_npz(sae_dir / "activations_all.npz").tocsc()
N, D = acts.shape
print(f"  activations: {acts.shape}, nnz={acts.nnz:,}")

# Which atoms ever fire
atom_fires = np.diff(acts.indptr) > 0   # nnz per column > 0
atom_fire_count = np.diff(acts.indptr)
print(f"  atoms that fire: {atom_fires.sum()}/{D}")

# ============================================================
# Build concept matrix (N x n_concepts), boolean
# ============================================================
print("\nBuilding concept matrix ...")
concept_cols = {}   # concept_name -> boolean array of length N

# ---- (a) ICD phenotypes ----
CLINICAL_DIR = cfg.EMBEDDING_DIR.parent / "clinical"
clin = pd.read_csv(CLINICAL_DIR / "record_with_clinical.csv")
flags = pd.read_csv(CLINICAL_DIR / "phenotype_flags.csv")
df_clin = clin.merge(flags, on='record_idx').set_index('record_idx').reindex(range(N))
has_dx = (df_clin['n_diagnoses'].values > 0)

ICD_MAP = {
    'AF': 'atrial_fibrillation', 'HF': 'heart_failure',
    'MI': 'mi___ischemic_heart', 'DM': 'diabetes_mellitus',
    'HTN': 'hypertension_primary',
}
for name, col in ICD_MAP.items():
    if col in df_clin.columns:
        # only valid where patient has diagnoses recorded
        v = df_clin[col].values
        pos = (v == 1) & has_dx
        concept_cols[f'ICD:{name}'] = pos
print(f"  + {len(ICD_MAP)} ICD phenotype concepts")

# ---- (b) Numerical ECG binary features ----
# Recompute from machine_measurements (same as Stage 13)
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
meta = meta.reset_index().rename(columns={'index': 'record_idx'})
meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')

mm_num = ['study_id', 'rr_interval', 'p_onset', 'qrs_onset', 'qrs_end', 't_end', 'qrs_axis']
mm_rep = [f'report_{i}' for i in range(18)]
mm = pd.read_csv("/workspace/data/mimic-iv-ecg-aws/machine_measurements.csv",
                 usecols=mm_num + mm_rep)
# Clean device sentinel values (e.g. 32767 = int16 max) before deriving features
for col in ['rr_interval', 'p_onset', 'qrs_onset', 'qrs_end', 't_end', 'qrs_axis']:
    if col in mm.columns:
        if col == 'qrs_axis':
            mm[col] = mm[col].where((mm[col] >= -180) & (mm[col] <= 180), np.nan)
        else:
            # intervals/onsets: physiological range 0..2000 ms; mask sentinels
            mm[col] = mm[col].where((mm[col] >= 0) & (mm[col] < 2000), np.nan)
mm['heart_rate'] = 60000 / mm['rr_interval'].replace(0, np.nan)
mm['heart_rate'] = mm['heart_rate'].where((mm['heart_rate'] >= 20) & (mm['heart_rate'] <= 300), np.nan)
mm['qrs_duration'] = mm['qrs_end'] - mm['qrs_onset']
mm['qrs_duration'] = mm['qrs_duration'].where((mm['qrs_duration'] > 0) & (mm['qrs_duration'] < 400), np.nan)
mm['qt_interval'] = mm['t_end'] - mm['qrs_onset']
mm['qt_interval'] = mm['qt_interval'].where((mm['qt_interval'] > 0) & (mm['qt_interval'] < 800), np.nan)
mm['qtc'] = mm['qt_interval'] / np.sqrt(mm['rr_interval'] / 1000)
mm['qtc'] = mm['qtc'].where((mm['qtc'] > 0) & (mm['qtc'] < 800), np.nan)

def st_flags(stmts):
    txt = ' '.join(str(s) for s in stmts if pd.notna(s)).lower()
    return int(bool(re.search(r'\bst\s*elevation\b', txt))), \
           int(bool(re.search(r'\bst\s*depression\b', txt)))
st_e, st_d = zip(*[st_flags(r) for r in mm[mm_rep].values.tolist()])
mm['st_elevation'] = st_e
mm['st_depression'] = st_d

feat_df = meta[['record_idx', 'study_id']].merge(
    mm[['study_id', 'heart_rate', 'qrs_duration', 'qtc', 'qrs_axis',
        'st_elevation', 'st_depression']],
    on='study_id', how='left'
).set_index('record_idx').reindex(range(N))

# Binary concepts (with validity masks where measurement exists)
def make_binary(series, cond):
    valid = series.notna().values
    pos = np.zeros(N, dtype=bool)
    pos[valid] = cond(series.values[valid])
    return pos, valid

bin_specs = {
    'NUM:tachycardia':  (feat_df['heart_rate'],   lambda x: x > 100),
    'NUM:bradycardia':  (feat_df['heart_rate'],   lambda x: x < 60),
    'NUM:wide_qrs':     (feat_df['qrs_duration'], lambda x: x >= 120),
    'NUM:long_qt':      (feat_df['qtc'],          lambda x: x > 460),
    'NUM:left_axis':    (feat_df['qrs_axis'],     lambda x: x < -30),
    'NUM:right_axis':   (feat_df['qrs_axis'],     lambda x: x > 90),
}
num_valid = {}
for name, (series, cond) in bin_specs.items():
    pos, valid = make_binary(series, cond)
    concept_cols[name] = pos
    num_valid[name] = valid

# ST flags (already 0/1, valid everywhere a report exists)
concept_cols['NUM:st_elevation'] = (feat_df['st_elevation'].fillna(0).values == 1)
concept_cols['NUM:st_depression'] = (feat_df['st_depression'].fillna(0).values == 1)
print(f"  + 8 numerical binary concepts")

# ---- (c) Top report phrases ----
report_labels_path = sae_dir / "atom_reports" / "atom_report_labels_v2.csv"
phrase_concepts = []
if report_labels_path.exists():
    rep = pd.read_csv(report_labels_path)
    # Take the most common high-lift phrases as concepts
    top_phrases = (rep[rep['lift'] > 50]['label']
                   .value_counts().head(TOP_REPORT_PHRASES).index.tolist())
    # Build per-ECG phrase presence from machine reports
    all_reports = mm[mm_rep].apply(
        lambda r: ' '.join(str(s) for s in r if pd.notna(s)).lower(), axis=1)
    report_by_record = meta[['record_idx', 'study_id']].merge(
        pd.DataFrame({'study_id': mm['study_id'], 'rep_text': all_reports}),
        on='study_id', how='left'
    ).set_index('record_idx').reindex(range(N))['rep_text'].fillna('')

    # Artifact phrases: data-quality / acquisition status, NOT clinical concepts.
    ARTIFACT_PHRASES = [
        '12 leads are missing', 'leads are missing',
        'based on available leads', 'poor quality data',
        'interpretation may be', 'available leads',
    ]
    def is_artifact(ph):
        pl = ph.lower()
        return any(a in pl for a in ARTIFACT_PHRASES)

    seen_normalized = set()
    for phrase in top_phrases:
        if is_artifact(phrase):
            continue
        # Normalize: strip trailing period/space so 'X' and 'X.' collapse to one concept
        normalized = phrase.strip().rstrip('.').strip()
        p_low = normalized.lower()
        if len(p_low) < 4:
            continue
        if normalized in seen_normalized:
            continue   # already added this concept (dedup period-variants)
        seen_normalized.add(normalized)
        # Match the normalized phrase as a substring in reports
        pos = report_by_record.str.contains(re.escape(p_low), regex=True).values
        if pos.sum() >= MIN_POS:
            concept_cols[f'TXT:{normalized[:40]}'] = pos
            phrase_concepts.append(normalized)
    print(f"  + {len(phrase_concepts)} report-phrase concepts")
else:
    print("  (no Stage 12 report labels; skipping phrase concepts)")

CONCEPTS = list(concept_cols.keys())
print(f"\n  Total concepts: {len(CONCEPTS)}")

# ============================================================
# Enrichment test: for each (atom, concept), Mann-Whitney U
# ============================================================
print("\n" + "=" * 60)
print("Running enrichment tests (Mann-Whitney U, one-sided)")
print("=" * 60)

# Pre-extract concept positive masks
concept_pos = {c: concept_cols[c] for c in CONCEPTS}
concept_npos = {c: int(concept_cols[c].sum()) for c in CONCEPTS}
# Drop concepts with too few positives
CONCEPTS = [c for c in CONCEPTS if concept_npos[c] >= MIN_POS]
print(f"  concepts with >= {MIN_POS} positives: {len(CONCEPTS)}")

# For each atom, get its dense activation vector once (from CSC)
# Vectorized Mann-Whitney U via GPU ranks. For each atom, compute the
# AUROC (= U/(n1*n2)) and z-based p-value against each concept at once.
import torch
from scipy.stats import norm as _norm
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  using device: {DEVICE}")

# Pre-build concept positive masks as a GPU boolean matrix (n_concepts x N)
# and validity masks.
conc_pos_mat = np.zeros((len(CONCEPTS), N), dtype=bool)
conc_valid_mat = np.ones((len(CONCEPTS), N), dtype=bool)
for ci, c in enumerate(CONCEPTS):
    conc_pos_mat[ci] = concept_pos[c]
    if c in num_valid:
        conc_valid_mat[ci] = num_valid[c]
conc_pos_t = torch.from_numpy(conc_pos_mat).to(DEVICE)        # (C, N)
conc_valid_t = torch.from_numpy(conc_valid_mat).to(DEVICE)    # (C, N)
n_pos_per_c = (conc_pos_t & conc_valid_t).sum(dim=1)          # (C,)

results = []
t0 = time.time()
for atom_id in range(D):
    if atom_id % 100 == 0:
        print(f"  atom {atom_id}/{D} ({time.time()-t0:.0f}s, {len(results)} tests)", flush=True)
    if not atom_fires[atom_id]:
        continue
    st, en = acts.indptr[atom_id], acts.indptr[atom_id + 1]
    rec_idx = acts.indices[st:en]
    act_vals = acts.data[st:en]
    a_dense = torch.zeros(N, device=DEVICE)
    a_dense[torch.from_numpy(rec_idx).to(DEVICE)] = torch.from_numpy(act_vals).to(DEVICE)

    # Global ranks of this atom's activations (ties -> average rank approx via argsort)
    ranks = a_dense.argsort().argsort().float() + 1.0   # (N,)

    for ci, c in enumerate(CONCEPTS):
        valid = conc_valid_t[ci]
        pos = conc_pos_t[ci] & valid
        neg = (~conc_pos_t[ci]) & valid
        n1 = pos.sum().item()
        n2 = neg.sum().item()
        if n1 < MIN_POS or n2 < MIN_POS:
            continue
        # Sum of ranks in positive group, but ranks must be computed within valid subset.
        # Approximate using global ranks restricted to valid (good enough for large N).
        valid_idx = valid
        # recompute ranks within valid set for correctness
        a_valid = a_dense[valid_idx]
        r_valid = a_valid.argsort().argsort().float() + 1.0
        pos_in_valid = pos[valid_idx]
        sum_rank_pos = r_valid[pos_in_valid].sum().item()
        U = sum_rank_pos - n1 * (n1 + 1) / 2.0
        auroc = U / (n1 * n2)
        # skip if no signal
        if a_dense[pos].max().item() == 0 and a_dense[neg].max().item() == 0:
            continue
        # Normal approximation for one-sided p (atom higher in positive => auroc > 0.5)
        mu = n1 * n2 / 2.0
        sigma = (n1 * n2 * (n1 + n2 + 1) / 12.0) ** 0.5
        if sigma == 0:
            continue
        z = (U - mu) / sigma
        p = 1.0 - _norm.cdf(z)   # one-sided: greater
        results.append((atom_id, c, p, auroc, int((a_dense[pos] > 0).sum().item())))

print(f"  done: {len(results)} (atom,concept) tests in {time.time()-t0:.0f}s", flush=True)

res_df = pd.DataFrame(results, columns=['atom_id', 'concept', 'p_value', 'auroc', 'n_pos_fired'])

# ============================================================
# Benjamini-Hochberg correction across ALL tests
# ============================================================
print("\n" + "=" * 60)
print("Benjamini-Hochberg FDR correction")
print("=" * 60)
reject, q_values, _, _ = multipletests(res_df['p_value'].values, alpha=Q_THRESHOLD, method='fdr_bh')
res_df['q_value'] = q_values
res_df['sig'] = reject
# Require BOTH statistical significance AND meaningful effect size.
# With N=800k, Mann-Whitney is significant even for AUROC~0.51 (trivial),
# so effect-size gating is essential.
res_df['enriched'] = res_df['sig'] & (res_df['auroc'] > MIN_EFFECT)
print(f"  significant (q<{Q_THRESHOLD}):              {res_df['sig'].sum():,}")
print(f"  + effect size (AUROC>{MIN_EFFECT}): {res_df['enriched'].sum():,}  <- used for taxonomy")
n_enriched_pairs = res_df['enriched'].sum()

res_df.to_csv(out_dir / "enrichment_tests.csv", index=False)
print(f"  saved: {out_dir}/enrichment_tests.csv")

# ============================================================
# Classify atoms
# ============================================================
print("\n" + "=" * 60)
print("Atom classification")
print("=" * 60)

enriched_per_atom = res_df[res_df['enriched']].groupby('atom_id')['concept'].apply(list)

taxonomy = []
for atom_id in range(D):
    if not atom_fires[atom_id]:
        cat = 'Dead'
        concepts = []
    else:
        concepts = enriched_per_atom.get(atom_id, [])
        n = len(concepts)
        if n == 0:
            cat = 'Uninformative'
        elif n == 1:
            cat = 'Separable'
        else:
            cat = 'Entangled'
    taxonomy.append({
        'atom_id': atom_id,
        'category': cat,
        'n_enriched_concepts': len(concepts),
        'enriched_concepts': '; '.join(concepts) if concepts else '',
        'fire_count': int(atom_fire_count[atom_id]),
        'fire_pct': 100.0 * atom_fire_count[atom_id] / N,
    })

tax_df = pd.DataFrame(taxonomy)
tax_df.to_csv(out_dir / "atom_taxonomy.csv", index=False)
print(f"  saved: {out_dir}/atom_taxonomy.csv")

# Summary
counts = tax_df['category'].value_counts()
print("\n=== Taxonomy summary ===")
for cat in ['Separable', 'Entangled', 'Uninformative', 'Dead']:
    n = counts.get(cat, 0)
    print(f"  {cat:15s}: {n:5d}  ({100*n/D:.1f}%)")

# ============================================================
# Visualization
# ============================================================
print("\nGenerating figures ...")

# Fig 1: category pie + bar
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cat_order = ['Separable', 'Entangled', 'Uninformative', 'Dead']
cat_colors = {'Separable': '#2ca02c', 'Entangled': '#ff7f0e',
              'Uninformative': '#7f7f7f', 'Dead': '#d62728'}
sizes = [counts.get(c, 0) for c in cat_order]
colors = [cat_colors[c] for c in cat_order]

ax = axes[0]
ax.pie(sizes, labels=[f'{c}\n{s} ({100*s/D:.1f}%)' for c, s in zip(cat_order, sizes)],
       colors=colors, autopct='', startangle=90,
       wedgeprops=dict(edgecolor='white', linewidth=1.5))
ax.set_title(f'Atom Taxonomy (K={D})\nz-test + BH correction, q < {Q_THRESHOLD}',
             fontweight='bold')

# Distribution of n_enriched_concepts for non-dead atoms
ax = axes[1]
nonzero = tax_df[tax_df['category'].isin(['Separable', 'Entangled'])]
ax.hist(nonzero['n_enriched_concepts'], bins=range(1, nonzero['n_enriched_concepts'].max()+2),
        color='steelblue', edgecolor='black', align='left')
ax.set_xlabel('# enriched concepts per atom')
ax.set_ylabel('# atoms')
ax.set_title('Concept multiplicity\n(Separable=1, Entangled>=2)')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(out_dir / 'taxonomy_overview.png')
plt.close()
print(f"  saved: {out_dir}/taxonomy_overview.png")

# Fig 2: which concepts have the most separable atoms?
sep_atoms = res_df[res_df['enriched']].merge(
    tax_df[tax_df['category'] == 'Separable'][['atom_id']], on='atom_id')
concept_sep_counts = sep_atoms['concept'].value_counts().head(25)

fig, ax = plt.subplots(figsize=(10, 8))
colors_c = ['#2ca02c' if c.startswith('ICD') else '#1f77b4' if c.startswith('NUM') else '#9467bd'
            for c in concept_sep_counts.index]
ax.barh(range(len(concept_sep_counts)), concept_sep_counts.values, color=colors_c)
ax.set_yticks(range(len(concept_sep_counts)))
ax.set_yticklabels(concept_sep_counts.index, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('# separable (monosemantic) atoms')
ax.set_title('Concepts with dedicated monosemantic atoms\n(green=ICD, blue=numerical, purple=report phrase)',
             fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(out_dir / 'separable_by_concept.png')
plt.close()
print(f"  saved: {out_dir}/separable_by_concept.png")

# ============================================================
# Print top separable atoms
# ============================================================
print("\n=== Sample Separable atoms (monosemantic) ===")
sep = tax_df[tax_df['category'] == 'Separable'].head(20)
for _, r in sep.iterrows():
    print(f"  atom {int(r['atom_id']):4d}  fire={r['fire_pct']:.2f}%  -> {r['enriched_concepts']}")

print(f"\n{'='*60}")
print(f"Outputs in: {out_dir}")
print(f"{'='*60}")
