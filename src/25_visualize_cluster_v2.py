"""
Stage 25: Visualize Stage 24 cluster interpretation results.

Tells the story:
  1. Real vs Fake r distributions (overlapping histograms)
  2. Effect of strict validation: naive Stage 23 (r~0.85) → strict (r~0.32)
  3. Distribution of cluster sizes vs r quality
  4. Top high-r clusters (potential new concepts) by summary
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import mannwhitneyu

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

mpl.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'figure.dpi': 100, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
v2_dir = sae_dir / "cluster_interp_v2"
v1_dir = sae_dir / "cluster_interp"     # Stage 23 (naive)
fig_dir = v2_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ----- Load -----
real = pd.read_csv(v2_dir / "real_clusters.csv")
fake = pd.read_csv(v2_dir / "fake_clusters.csv")
real_r = real['pearson_r'].dropna().values
fake_r = fake['pearson_r'].dropna().values

# Optional: naive Stage 23 results for the "before" panel
naive = None
if (v1_dir / "cluster_descriptions.csv").exists():
    naive = pd.read_csv(v1_dir / "cluster_descriptions.csv")
    naive_r = naive['pearson_r'].dropna().values
    print(f"Loaded naive results: n={len(naive_r)}, median={np.median(naive_r):.3f}")

print(f"Real: n={len(real_r)}, median={np.median(real_r):.3f}, mean={np.mean(real_r):.3f}")
print(f"Fake: n={len(fake_r)}, median={np.median(fake_r):.3f}, mean={np.mean(fake_r):.3f}")

U, p = mannwhitneyu(real_r, fake_r, alternative='greater')
print(f"Mann-Whitney one-sided p = {p:.4g}")

# ============================================================
# Fig 1: Naive vs Strict validation (the headline story)
# ============================================================
print("\nFig 1: Naive vs strict validation comparison ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: naive (Stage 23)
ax = axes[0]
if naive is not None:
    bins = np.linspace(-0.3, 1.0, 25)
    ax.hist(naive_r, bins=bins, color='#d62728', edgecolor='black',
            alpha=0.85, label=f'Naive validation\n(zero-activation controls)')
    ax.axvline(np.median(naive_r), color='black', linestyle='--', linewidth=2,
               label=f'median = {np.median(naive_r):.2f}')
ax.axvline(0.72, color='purple', linestyle=':', alpha=0.7,
           label='InterPLM (protein): 0.72')
ax.set_xlabel('Held-out Pearson r')
ax.set_ylabel('# clusters')
ax.set_title('BEFORE: Naive validation\n(controls = zero activation only)',
             fontweight='bold', color='#8b0000')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(-0.3, 1.0)
if naive is not None:
    n_high = (naive_r > 0.7).sum()
    ax.text(0.98, 0.97,
            f"{n_high}/{len(naive_r)} clusters r>0.7\n→ implausibly high\n→ suspected leakage",
            transform=ax.transAxes, va='top', ha='right',
            fontsize=8, family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ffe5e5', edgecolor='red'))

# Right: strict (Stage 24)
ax = axes[1]
bins = np.linspace(-0.5, 1.0, 25)
ax.hist(real_r, bins=bins, color='#2ca02c', edgecolor='black',
        alpha=0.8, label=f'Real clusters (n={len(real_r)})')
ax.hist(fake_r, bins=bins, color='#7f7f7f', edgecolor='black',
        alpha=0.5, label=f'Permutation null (n={len(fake_r)})')
ax.axvline(np.median(real_r), color='#2ca02c', linestyle='--', linewidth=2,
           label=f'real median = {np.median(real_r):.2f}')
ax.axvline(np.median(fake_r), color='#7f7f7f', linestyle='--', linewidth=2,
           label=f'fake median = {np.median(fake_r):.2f}')
ax.axvline(0, color='black', alpha=0.3)
ax.set_xlabel('Held-out Pearson r')
ax.set_ylabel('# clusters')
ax.set_title('AFTER: Strict validation\n(activation-matched controls + permutation null)',
             fontweight='bold', color='#006400')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
ax.set_xlim(-0.5, 1.0)

n_r05 = (real_r > 0.5).sum()
n_r07 = (real_r > 0.7).sum()
ax.text(0.98, 0.97,
        f"Δ median = +{np.median(real_r)-np.median(fake_r):.2f}\n"
        f"Mann-Whitney p = {p:.3f}\n"
        f"Real r>0.5: {n_r05}/{len(real_r)}\n"
        f"Real r>0.7: {n_r07}/{len(real_r)}",
        transform=ax.transAxes, va='top', ha='right',
        fontsize=8, family='monospace',
        bbox=dict(boxstyle='round', facecolor='#e5ffe5', edgecolor='green'))

plt.suptitle('Cluster-level interpretation: leakage diagnosis + corrected validation',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(fig_dir / 'fig1_naive_vs_strict.png')
plt.close()
print("  saved")

# ============================================================
# Fig 2: Real vs Fake distribution (focused)
# ============================================================
print("\nFig 2: Real vs Fake detailed comparison ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: violin + scatter
ax = axes[0]
parts = ax.violinplot([fake_r, real_r], positions=[0, 1], widths=0.7,
                      showmeans=False, showmedians=True)
colors_v = ['#7f7f7f', '#2ca02c']
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors_v[i]); pc.set_alpha(0.5)
# Overlay individual points (jittered)
rng = np.random.RandomState(0)
ax.scatter(rng.normal(0, 0.05, len(fake_r)), fake_r, color='#7f7f7f',
           s=40, edgecolor='black', alpha=0.8, zorder=3)
ax.scatter(rng.normal(1, 0.05, len(real_r)), real_r, color='#2ca02c',
           s=20, edgecolor='black', alpha=0.6, zorder=3)
ax.axhline(0, color='black', alpha=0.3)
ax.set_xticks([0, 1])
ax.set_xticklabels([f'Fake clusters\n(random, n={len(fake_r)})',
                     f'Real clusters\n(decoder cosine, n={len(real_r)})'])
ax.set_ylabel('Held-out Pearson r')
ax.set_title(f'Real vs permutation null\nMann-Whitney p = {p:.3f}',
             fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Right: empirical CDF comparison
ax = axes[1]
real_sorted = np.sort(real_r)
fake_sorted = np.sort(fake_r)
ax.plot(real_sorted, np.linspace(0, 1, len(real_sorted)),
        color='#2ca02c', linewidth=2.5, label=f'Real (median={np.median(real_r):.2f})')
ax.plot(fake_sorted, np.linspace(0, 1, len(fake_sorted)),
        color='#7f7f7f', linewidth=2.5, label=f'Fake (median={np.median(fake_r):.2f})')
ax.axvline(0, color='black', alpha=0.3)
ax.axvline(0.5, color='red', linestyle=':', alpha=0.5, label='r=0.5')
ax.set_xlabel('Held-out Pearson r')
ax.set_ylabel('Cumulative fraction')
ax.set_title('Cumulative distribution\n(real curve shifts right of fake)',
             fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 1.0)

plt.tight_layout()
plt.savefig(fig_dir / 'fig2_real_vs_fake_detail.png')
plt.close()
print("  saved")

# ============================================================
# Fig 3: r vs cluster size (does bigger cluster help?)
# ============================================================
print("\nFig 3: r vs cluster size ...")
fig, ax = plt.subplots(figsize=(9, 6))
real_valid = real[real['pearson_r'].notna()]
def bin_color(r):
    if r >= 0.7: return '#006400'
    elif r >= 0.5: return '#2ca02c'
    elif r >= 0.3: return '#fdae61'
    elif r >= 0: return '#d62728'
    else: return '#8b0000'
colors_pts = [bin_color(r) for r in real_valid['pearson_r']]
ax.scatter(real_valid['n_atoms'], real_valid['pearson_r'],
           s=80, c=colors_pts, edgecolor='black', linewidth=0.5, alpha=0.85,
           label='Real clusters')
# Fake overlay
fake_valid = fake[fake['pearson_r'].notna()]
if 'n_atoms' in fake_valid.columns:
    ax.scatter(fake_valid['n_atoms'], fake_valid['pearson_r'],
               s=80, c='#7f7f7f', marker='x', linewidth=2, label='Fake (random)')

ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='r=0.5')
ax.axhline(0, color='black', alpha=0.3)
ax.axhline(np.median(real_r), color='#2ca02c', linestyle=':', alpha=0.6,
           label=f'real median = {np.median(real_r):.2f}')
ax.set_xlabel('# atoms in cluster')
ax.set_ylabel('Held-out Pearson r')
ax.set_title('Cluster size vs description quality\n(does larger cluster help signal?)',
             fontweight='bold')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Annotate top clusters
top = real_valid.nlargest(5, 'pearson_r')
for _, row in top.iterrows():
    ax.annotate(f"#{int(row['cluster_id'])}", 
                xy=(row['n_atoms'], row['pearson_r']),
                xytext=(5, 5), textcoords='offset points', fontsize=7)

plt.tight_layout()
plt.savefig(fig_dir / 'fig3_r_vs_size.png')
plt.close()
print("  saved")

# ============================================================
# Fig 4: Top clusters table (text panel with descriptions)
# ============================================================
print("\nFig 4: Top clusters with descriptions ...")
top10 = real_valid.nlargest(10, 'pearson_r').reset_index(drop=True)
fig, ax = plt.subplots(figsize=(15, max(8, len(top10) * 0.6)))
ax.axis('off')

# Title row + each cluster as a row
y_start = 0.98
row_height = 0.085
ax.text(0.5, y_start, f'Top {len(top10)} clusters by held-out Pearson r (potential new concepts)',
        ha='center', va='top', transform=ax.transAxes,
        fontsize=12, fontweight='bold')

for i, row in top10.iterrows():
    y = y_start - 0.06 - i * row_height
    cid = int(row['cluster_id'])
    n = int(row['n_atoms'])
    r = row['pearson_r']
    summary = str(row.get('summary', ''))[:120]
    color = bin_color(r)
    # Cluster info box
    ax.text(0.02, y, f"Cluster {cid}", transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')
    ax.text(0.10, y, f"n={n} atoms", transform=ax.transAxes,
            fontsize=9, va='top', color='gray')
    ax.text(0.20, y, f"r = {r:.2f}", transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', color=color)
    ax.text(0.32, y, summary, transform=ax.transAxes,
            fontsize=8, va='top', wrap=True)
    # horizontal separator (use plot in axes coordinates, not axhline)
    ax.plot([0.02, 0.98], [y - row_height + 0.005] * 2,
            color='gray', alpha=0.2, linewidth=0.5,
            transform=ax.transAxes, clip_on=False)

plt.tight_layout()
plt.savefig(fig_dir / 'fig4_top_clusters.png')
plt.close()
print("  saved")

# ============================================================
# Fig 5: Summary panel with key numbers + interpretation
# ============================================================
print("\nFig 5: Summary panel ...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: quality bins bar chart
ax = axes[0]
bins_def = [
    ('Strong\n(r ≥ 0.7)',  '#006400', (real_r >= 0.7).sum()),
    ('Moderate\n(0.5–0.7)', '#2ca02c', ((real_r >= 0.5) & (real_r < 0.7)).sum()),
    ('Weak\n(0.3–0.5)',     '#fdae61', ((real_r >= 0.3) & (real_r < 0.5)).sum()),
    ('Poor\n(0–0.3)',       '#d62728', ((real_r >= 0) & (real_r < 0.3)).sum()),
    ('Reverse\n(r < 0)',    '#8b0000', (real_r < 0).sum()),
]
names = [b[0] for b in bins_def]
counts = [b[2] for b in bins_def]
colors_b = [b[1] for b in bins_def]
bars = ax.bar(names, counts, color=colors_b, edgecolor='black', linewidth=0.8)
for bar, c in zip(bars, counts):
    if c > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{c}\n({100*c/len(real_r):.0f}%)',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('# real clusters')
ax.set_title('Real clusters by description quality', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(counts) * 1.25)

# Right: text summary
ax = axes[1]
ax.axis('off')
n_atoms_strong = real_valid[real_valid['pearson_r'] > 0.5]['n_atoms'].sum()
n_atoms_total = real_valid['n_atoms'].sum()
text = f"""STAGE 23–24: CLUSTER-LEVEL INTERPRETATION
of Uninformative Atoms

PIPELINE
  1. Hierarchical clustering (cosine on
     decoder vectors) of 1353 atoms → 80 clusters
  2. Claude generates description from
     activation-matched control comparison
  3. Held-out Pearson r predicts cluster
     activation from description alone
  4. Permutation null (10 random atom groups)
     to detect validation leakage

VALIDATION GAP REVEALED
  Naive validation:  median r = {np.median(naive_r) if naive is not None else float('nan'):.2f}
                     (zero-activation controls)
  Strict validation: median r = {np.median(real_r):.2f}
                     (activation-matched controls)
  → ~{(np.median(naive_r) - np.median(real_r))*100 if naive is not None else 0:.0f} percentage points were leakage

STRICT VALIDATION RESULTS
  Real cluster median r:  {np.median(real_r):.3f}
  Fake cluster median r:  {np.median(fake_r):.3f}
  Effective signal:       +{np.median(real_r) - np.median(fake_r):.3f}
  Mann-Whitney p:         {p:.3f}

CANDIDATE NEW CONCEPTS
  Clusters with r > 0.5:  {(real_r > 0.5).sum()}/{len(real_r)}
  Atoms covered (r > 0.5): {int(n_atoms_strong)}/{int(n_atoms_total)}
                          ({100*n_atoms_strong/n_atoms_total:.0f}% of clustered)

METHODOLOGICAL CONTRIBUTION
  Strict + null validation reveals leakage
  inherent to whole-sequence SAE interpretation
  protocols. Future work should adopt
  activation-matched controls by default.
"""
ax.text(0.02, 0.98, text, transform=ax.transAxes,
        va='top', ha='left', family='monospace', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='#f0f8ff',
                  edgecolor='steelblue', linewidth=1.5))

plt.tight_layout()
plt.savefig(fig_dir / 'fig5_summary_panel.png')
plt.close()
print("  saved")

# ============================================================
print("\n" + "=" * 60)
print(f"All figures in: {fig_dir}")
for f in sorted(fig_dir.glob('*.png')):
    print(f"  {f.name}")
