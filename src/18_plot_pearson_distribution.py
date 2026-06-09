"""
Stage 18 (post-17d analysis): Pearson r distribution histogram.

Plots:
  Panel 1: Overall r distribution (200 atoms) with InterPLM median reference
  Panel 2: r distribution stratified by Stage 15 category
  Panel 3: r vs Stage 15 max AUROC scatter (independence test)
  Panel 4: r > 0 vs r <= 0 by category (bar chart)
"""
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

sys.path.insert(0, '/workspace/jay/stsae_project/Cardiac-Sensing-FM')
import config as cfg

mpl.rcParams.update({
    'font.size': 10, 'axes.titlesize': 11, 'figure.dpi': 100,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

SAE = cfg.SAE_DIR / "batchtopk_tiny_aws_k32_d1536"
results_path = SAE / "claude_interp_random200" / "atom_descriptions_random200.csv"
out_dir = SAE / "claude_interp_random200"

# Reference: InterPLM and previous runs
INTERPLM_MEDIAN = 0.72
STAGE17_UNINF_MEDIAN = 0.30  # 旧 Stage 17 在 Uninf-only 上的 median

if not results_path.exists():
    print(f"⚠ Not found: {results_path}")
    print("Run stage 17d first, or check path")
    sys.exit(1)

df = pd.read_csv(results_path)
print(f"Loaded {len(df)} atoms from 17d")

valid = df.dropna(subset=['pearson_r']).copy()
print(f"  with valid r: {len(valid)}")

# ============================================================
# Stage 15 max AUROC for independence test
# ============================================================
enr = pd.read_csv(SAE / "taxonomy" / "enrichment_tests.csv")
max_auc = enr.groupby('atom_id')['auroc'].max().reset_index()
max_auc.columns = ['atom_id', 'stage15_max_auroc']
valid = valid.merge(max_auc, on='atom_id', how='left')

# ============================================================
# Stats
# ============================================================
overall_median = valid['pearson_r'].median()
overall_mean = valid['pearson_r'].mean()
pct_above_05 = 100 * (valid['pearson_r'] > 0.5).mean()
pct_above_03 = 100 * (valid['pearson_r'] > 0.3).mean()
pct_negative = 100 * (valid['pearson_r'] < 0).mean()

print(f"\nOverall: median r = {overall_median:+.3f}, mean = {overall_mean:+.3f}")
print(f"  r > 0.5: {pct_above_05:.0f}%, r > 0.3: {pct_above_03:.0f}%, r < 0: {pct_negative:.0f}%")

# ============================================================
# Figure
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

CAT_COLORS = {
    'Separable': '#2ca02c',
    'Entangled-Related': '#ff7f0e',
    'Entangled-Mixed': '#d62728',
    'Uninformative': '#7f7f7f',
    'Contributing': '#1f77b4',
}
CAT_ORDER = ['Separable', 'Entangled-Related', 'Entangled-Mixed',
             'Uninformative', 'Contributing']

# ===== Panel 1: Overall histogram =====
ax = axes[0, 0]
bins = np.arange(-0.6, 1.05, 0.05)
ax.hist(valid['pearson_r'], bins=bins, color='steelblue',
        edgecolor='black', alpha=0.85)
ax.axvline(0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
ax.axvline(overall_median, color='red', linestyle='--', linewidth=2,
           label=f'Our median = {overall_median:+.3f}')
ax.axvline(INTERPLM_MEDIAN, color='green', linestyle=':', linewidth=2,
           label=f'InterPLM (protein) = {INTERPLM_MEDIAN:.2f}')
ax.axvline(STAGE17_UNINF_MEDIAN, color='purple', linestyle=':', linewidth=2,
           label=f'Stage 17 (Uninf-only) = {STAGE17_UNINF_MEDIAN:.2f}')
ax.set_xlabel('Pearson r (held-out)')
ax.set_ylabel('# atoms')
ax.set_title(f'(a) Overall r distribution (n={len(valid)} random atoms)\n'
             f'median={overall_median:+.3f}, mean={overall_mean:+.3f}, '
             f'>0.5: {pct_above_05:.0f}%, <0: {pct_negative:.0f}%',
             fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(-0.6, 1.05)

# ===== Panel 2: By Stage 15 category =====
ax = axes[0, 1]

# Stacked / overlay histograms
for cat in CAT_ORDER:
    sub = valid[valid['stage15_category'] == cat]['pearson_r']
    if len(sub) < 3: continue
    ax.hist(sub, bins=bins, color=CAT_COLORS[cat], alpha=0.55,
            edgecolor='black', linewidth=0.5,
            label=f'{cat} (n={len(sub)}, med={sub.median():+.2f})')

ax.axvline(0, color='gray', linestyle='-', alpha=0.4, linewidth=1)
ax.axvline(overall_median, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.set_xlabel('Pearson r (held-out)')
ax.set_ylabel('# atoms')
ax.set_title(f'(b) Stratified by Stage 15 category',
             fontweight='bold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(-0.6, 1.05)

# ===== Panel 3: Independence scatter (r vs Stage 15 AUROC) =====
ax = axes[1, 0]
v3 = valid.dropna(subset=['stage15_max_auroc'])
for cat in CAT_ORDER:
    sub = v3[v3['stage15_category'] == cat]
    if len(sub) < 1: continue
    ax.scatter(sub['stage15_max_auroc'], sub['pearson_r'],
               c=CAT_COLORS[cat], s=40, alpha=0.75,
               edgecolor='black', linewidth=0.4, label=f'{cat}')

# Compute correlation
if len(v3) > 10:
    r_ind, p_ind = pearsonr(v3['stage15_max_auroc'], v3['pearson_r'])
    rho_ind, p_rho = spearmanr(v3['stage15_max_auroc'], v3['pearson_r'])
    # Trend line
    z = np.polyfit(v3['stage15_max_auroc'], v3['pearson_r'], 1)
    xs = np.linspace(0.4, 1.0, 50)
    ax.plot(xs, z[0]*xs + z[1], 'k--', alpha=0.5,
            label=f'fit (r={r_ind:+.2f})')
    title_extra = f"\nPearson r = {r_ind:+.3f}  Spearman ρ = {rho_ind:+.3f}\n(InterPLM analog: r=0.11)"
else:
    title_extra = ""

ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax.axhline(overall_median, color='red', linestyle='--', alpha=0.4,
           label=f'overall median r = {overall_median:+.2f}')
ax.set_xlabel('Stage 15 max AUROC (concept-catalog coverage)')
ax.set_ylabel('Stage 17d Pearson r (Claude description quality)')
ax.set_title(f'(c) Independence test: Claude r vs Stage 15 AUROC{title_extra}',
             fontweight='bold')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)

# ===== Panel 4: % well-described per category =====
ax = axes[1, 1]

thresholds = [0.3, 0.5, 0.7]
x = np.arange(len(CAT_ORDER))
width = 0.25

bars = []
for ti, thr in enumerate(thresholds):
    heights = []
    for cat in CAT_ORDER:
        sub = valid[valid['stage15_category'] == cat]
        if len(sub) == 0:
            heights.append(0)
        else:
            heights.append(100 * (sub['pearson_r'] > thr).mean())
    b = ax.bar(x + (ti-1)*width, heights, width,
               label=f'r > {thr}', alpha=0.85,
               edgecolor='black', linewidth=0.5)
    bars.append(b)
    # Annotate
    for xi, h in enumerate(heights):
        if h > 0:
            ax.text(xi + (ti-1)*width, h + 1, f'{h:.0f}%',
                    ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(CAT_ORDER, rotation=15, ha='right')
ax.set_ylabel('% of atoms in category')
ax.set_title('(d) Fraction of atoms with Claude r above threshold,\n'
             'by Stage 15 category', fontweight='bold')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(80, ax.get_ylim()[1]))

plt.suptitle(f'Stage 17d: Claude-Description Validation r Distribution '
             f'(n={len(valid)} random atoms, InterPLM-style prompt)',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(out_dir / 'r_distribution.png')
plt.close()
print(f"\nSaved: {out_dir}/r_distribution.png")

# ============================================================
# Console summary tables
# ============================================================
print(f"\n{'='*70}")
print(f"SUMMARY TABLES")
print(f"{'='*70}")

print(f"\nBy Stage 15 category:")
print(f"{'category':<22} {'n':>4} {'median':>8} {'mean':>8} {'>0.5':>6} {'>0.3':>6} {'<0':>5}")
print("-" * 70)
for cat in CAT_ORDER + ['Dead', 'Unknown']:
    sub = valid[valid['stage15_category'] == cat]
    if len(sub) == 0: continue
    print(f"{cat:<22} {len(sub):>4} {sub['pearson_r'].median():>+8.3f} "
          f"{sub['pearson_r'].mean():>+8.3f} "
          f"{(sub['pearson_r']>0.5).sum():>3}/{len(sub):<3} "
          f"{(sub['pearson_r']>0.3).sum():>3}/{len(sub):<3} "
          f"{(sub['pearson_r']<0).sum():>3}/{len(sub):<3}")

# Mann-Whitney: Separable vs Uninformative
sep = valid[valid['stage15_category'] == 'Separable']['pearson_r']
uninf = valid[valid['stage15_category'] == 'Uninformative']['pearson_r']
if len(sep) >= 5 and len(uninf) >= 5:
    stat, p_mw = mannwhitneyu(sep, uninf, alternative='greater')
    print(f"\nMann-Whitney (Separable r > Uninformative r):")
    print(f"  Sep median = {sep.median():+.3f} (n={len(sep)})")
    print(f"  Uninf median = {uninf.median():+.3f} (n={len(uninf)})")
    print(f"  p = {p_mw:.4f}")
    if p_mw < 0.05:
        print(f"  → Significant: Stage 15 category does affect Claude r")
    else:
        print(f"  → Not significant: Stage 15 category doesn't predict Claude r")

# Independence interpretation
if len(v3) > 10:
    print(f"\n{'='*70}")
    print(f"INDEPENDENCE TEST INTERPRETATION")
    print(f"{'='*70}")
    print(f"\n  Pearson r (Claude r, Stage 15 AUROC) = {r_ind:+.3f}")
    print(f"  Spearman rho                         = {rho_ind:+.3f}")
    print(f"  InterPLM reported                    = 0.11\n")
    if abs(r_ind) < 0.2:
        print(f"  ✓ |r| < 0.2 — Claude description quality is largely INDEPENDENT")
        print(f"    of concept-catalog coverage (matches InterPLM finding).")
        print(f"    Strong paper claim: 'LLM auto-interp captures features")
        print(f"    beyond the Stage 15 concept inventory.'")
    elif abs(r_ind) < 0.4:
        print(f"  | moderate dependence (|r|=0.2-0.4)")
        print(f"    Some overlap, but Claude still finds extra information")
    else:
        print(f"  ⚠ |r| > 0.4 — Claude r partly reflects Stage 15 enrichment.")
        print(f"    Paper should acknowledge this and emphasize cluster-level (Stage 24)")
        print(f"    as the main vehicle for genuinely novel concept discovery.")

print(f"\nDone.")
