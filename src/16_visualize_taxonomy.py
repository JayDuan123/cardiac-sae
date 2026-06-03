"""
Stage 16: Visualize rigorous taxonomy results.
Reads Stage 15 outputs (enrichment_tests.csv, atom_taxonomy.csv).

Figures:
  fig1_taxonomy_composition  : pie + concept-multiplicity histogram
  fig2_separable_by_concept  : which concepts have monosemantic atoms
  fig3_effect_vs_significance: AUROC vs q-value scatter (why effect-size gating matters)
  fig4_sensitivity           : separable% across effect-size thresholds
  fig5_concept_atom_matrix   : top atoms x top concepts enrichment heatmap
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

mpl.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 100,
                     'savefig.dpi': 150, 'savefig.bbox': 'tight'})

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
tax_dir = cfg.SAE_DIR / SAE_NAME / "taxonomy"
fig_dir = tax_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

MIN_EFFECT = 0.60
Q_THRESHOLD = 0.05

print("Loading taxonomy outputs ...")
enr = pd.read_csv(tax_dir / "enrichment_tests.csv")
tax = pd.read_csv(tax_dir / "atom_taxonomy.csv")
D = len(tax)
print(f"  {len(enr)} enrichment tests, {D} atoms")

cat_colors = {'Separable': '#2ca02c', 'Entangled': '#ff7f0e',
              'Uninformative': '#7f7f7f', 'Dead': '#d62728'}
cat_order = ['Separable', 'Entangled', 'Uninformative', 'Dead']
counts = tax['category'].value_counts()

# ============================================================
# Fig 1: Composition pie + concept multiplicity
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sizes = [counts.get(c, 0) for c in cat_order]
colors = [cat_colors[c] for c in cat_order]
ax = axes[0]
wedges, _, autotexts = ax.pie(
    sizes, labels=[f'{c}\n{s} ({100*s/D:.1f}%)' for c, s in zip(cat_order, sizes)],
    colors=colors, autopct='', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=1.5))
ax.set_title(f'Atom Taxonomy (K={D})\nMann-Whitney + BH (q<{Q_THRESHOLD}) AND AUROC>{MIN_EFFECT}',
             fontweight='bold')

ax = axes[1]
nz = tax[tax['category'].isin(['Separable', 'Entangled'])]
if len(nz) > 0:
    maxc = int(nz['n_enriched_concepts'].max())
    ax.hist(nz['n_enriched_concepts'], bins=range(1, maxc + 2),
            color='steelblue', edgecolor='black', align='left')
    ax.set_xlabel('# enriched concepts per atom')
    ax.set_ylabel('# atoms')
    ax.set_title('Concept multiplicity\n(Separable=1, Entangled>=2)')
    ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(fig_dir / 'fig1_taxonomy_composition.png')
plt.close()
print("  fig1 saved")

# ============================================================
# Fig 2: Separable by concept
# ============================================================
sep_ids = tax[tax['category'] == 'Separable']['atom_id']
sep_pairs = enr[(enr['enriched']) & (enr['atom_id'].isin(sep_ids))]
csc = sep_pairs['concept'].value_counts().head(25)

fig, ax = plt.subplots(figsize=(10, 8))
colors_c = ['#2ca02c' if c.startswith('ICD') else '#1f77b4' if c.startswith('NUM')
            else '#9467bd' for c in csc.index]
ax.barh(range(len(csc)), csc.values, color=colors_c)
ax.set_yticks(range(len(csc)))
ax.set_yticklabels(csc.index, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('# separable (monosemantic) atoms')
ax.set_title('Concepts with dedicated monosemantic atoms\n(green=ICD, blue=numerical, purple=report phrase)',
             fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(csc.values):
    ax.text(v + 0.3, i, str(v), va='center', fontsize=8)
plt.tight_layout()
plt.savefig(fig_dir / 'fig2_separable_by_concept.png')
plt.close()
print("  fig2 saved")

# ============================================================
# Fig 3: Why effect-size gating matters (AUROC vs significance)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
sig = enr[enr['sig']] if 'sig' in enr.columns else enr[enr['q_value'] < Q_THRESHOLD]
ax.hist(sig['auroc'], bins=60, color='salmon', edgecolor='black', alpha=0.8)
ax.axvline(MIN_EFFECT, color='green', linestyle='--', linewidth=2, label=f'effect threshold {MIN_EFFECT}')
ax.axvline(0.5, color='gray', linestyle=':', label='random (0.5)')
ax.set_xlabel('AUROC (effect size)')
ax.set_ylabel('# (atom,concept) pairs')
ax.set_title(f'AUROC of statistically significant pairs\n(median={sig["auroc"].median():.3f} -- mostly trivial!)')
ax.legend()
ax.grid(True, alpha=0.3)
n_sig = len(sig)
n_eff = (sig['auroc'] > MIN_EFFECT).sum()
ax.text(0.98, 0.97, f'significant: {n_sig:,}\npass effect: {n_eff:,}\n({100*n_eff/max(n_sig,1):.1f}%)',
        transform=ax.transAxes, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

ax = axes[1]
samp = enr.sample(min(20000, len(enr)), random_state=0)
ax.scatter(samp['auroc'], -np.log10(samp['q_value'].clip(lower=1e-300)),
           s=3, alpha=0.3, c='steelblue')
ax.axvline(MIN_EFFECT, color='green', linestyle='--', label=f'AUROC={MIN_EFFECT}')
ax.axhline(-np.log10(Q_THRESHOLD), color='red', linestyle='--', label=f'q={Q_THRESHOLD}')
ax.set_xlabel('AUROC (effect size)')
ax.set_ylabel('-log10(q-value)')
ax.set_title('Significance vs effect size\n(top-right quadrant = enriched)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'fig3_effect_vs_significance.png')
plt.close()
print("  fig3 saved")

# ============================================================
# Fig 4: Sensitivity of separable% to effect-size threshold
# ============================================================
thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
sep_pcts, ent_pcts = [], []
sig_pairs = enr[enr['sig']] if 'sig' in enr.columns else enr[enr['q_value'] < Q_THRESHOLD]
for th in thresholds:
    enriched = sig_pairs[sig_pairs['auroc'] > th]
    per_atom = enriched.groupby('atom_id')['concept'].count()
    n_sep = (per_atom == 1).sum()
    n_ent = (per_atom >= 2).sum()
    sep_pcts.append(100 * n_sep / D)
    ent_pcts.append(100 * n_ent / D)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(thresholds, sep_pcts, 'o-', color='#2ca02c', linewidth=2, markersize=8, label='Separable %')
ax.plot(thresholds, ent_pcts, 's-', color='#ff7f0e', linewidth=2, markersize=8, label='Entangled %')
ax.axvline(MIN_EFFECT, color='gray', linestyle='--', alpha=0.6, label=f'chosen ({MIN_EFFECT})')
ax.set_xlabel('Effect-size threshold (AUROC)')
ax.set_ylabel('% of dictionary')
ax.set_title('Sensitivity of taxonomy to effect-size threshold\n(all points require q<0.05)',
             fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
for x, y in zip(thresholds, sep_pcts):
    ax.annotate(f'{y:.1f}', (x, y), textcoords='offset points', xytext=(0, 8), fontsize=7, ha='center')
plt.tight_layout()
plt.savefig(fig_dir / 'fig4_sensitivity.png')
plt.close()
print("  fig4 saved")

# ============================================================
# Fig 5: Concept-atom enrichment matrix (top atoms x top concepts)
# ============================================================
enriched = enr[enr['enriched']]
top_concepts = enriched['concept'].value_counts().head(20).index.tolist()
# top atoms = separable/entangled atoms with most enrichments
top_atom_ids = (enriched.groupby('atom_id')['auroc'].max()
                .sort_values(ascending=False).head(40).index.tolist())

M = np.full((len(top_atom_ids), len(top_concepts)), np.nan)
for i, aid in enumerate(top_atom_ids):
    for j, c in enumerate(top_concepts):
        row = enriched[(enriched['atom_id'] == aid) & (enriched['concept'] == c)]
        if len(row) > 0:
            M[i, j] = row['auroc'].iloc[0]

fig, ax = plt.subplots(figsize=(12, 11))
im = ax.imshow(M, cmap='YlOrRd', aspect='auto', vmin=MIN_EFFECT, vmax=1.0)
ax.set_xticks(range(len(top_concepts)))
ax.set_xticklabels(top_concepts, rotation=45, ha='right', fontsize=7)
ax.set_yticks(range(len(top_atom_ids)))
ax.set_yticklabels([f'atom {a}' for a in top_atom_ids], fontsize=7)
ax.set_title('Enriched (atom x concept) AUROC\n(only pairs passing q<0.05 AND AUROC>%.2f)' % MIN_EFFECT,
             fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='AUROC')
for i in range(len(top_atom_ids)):
    for j in range(len(top_concepts)):
        if not np.isnan(M[i, j]):
            ax.text(j, i, f'{M[i,j]:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if M[i, j] > 0.75 else 'black')
plt.tight_layout()
plt.savefig(fig_dir / 'fig5_concept_atom_matrix.png')
plt.close()
print("  fig5 saved")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Taxonomy summary:")
for c in cat_order:
    n = counts.get(c, 0)
    print(f"  {c:15s}: {n:5d}  ({100*n/D:.1f}%)")
print(f"\nFigures in: {fig_dir}")
for f in sorted(fig_dir.glob('*.png')):
    print(f"  {f.name}")
