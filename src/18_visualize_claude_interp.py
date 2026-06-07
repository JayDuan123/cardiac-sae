"""
Stage 18: Visualize Stage 17 Claude interpretation results.

Reads atom_descriptions.csv and produces 4 figures:
  fig1_pearson_distribution : histogram of held-out Pearson r
  fig2_top_atoms_bar        : top-N atoms ranked by r, horizontal bars
  fig3_r_vs_firefreq        : scatter (does fire_pct predict r?)
  fig4_summary_panel        : key statistics + comparison to InterPLM
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

mpl.rcParams.update({
    'font.size': 9, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'figure.dpi': 100, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
interp_dir = cfg.SAE_DIR / SAE_NAME / "claude_interp"
fig_dir = interp_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

INTERPLM_MEDIAN = 0.72   # for comparison

print("Loading Stage 17 outputs ...")
df = pd.read_csv(interp_dir / "atom_descriptions.csv")
df_valid = df[df['pearson_r'].notna()].copy()
print(f"  total atoms: {len(df)}")
print(f"  valid r:     {len(df_valid)}")
print(f"  median r:    {df_valid['pearson_r'].median():.3f}")
print(f"  mean r:      {df_valid['pearson_r'].mean():.3f}")

# ============================================================
# Fig 1: Pearson r distribution histogram
# ============================================================
print("\nFig 1: Pearson r distribution ...")
fig, ax = plt.subplots(figsize=(10, 6))

# Color bins
def bin_color(r):
    if r >= 0.7: return '#2ca02c'    # strong
    elif r >= 0.5: return '#7fbf7b'  # moderate
    elif r >= 0.3: return '#fdae61'  # weak
    else: return '#d62728'           # poor

r_vals = df_valid['pearson_r'].values
bins = np.linspace(-0.2, 1.0, 25)
n, bin_edges, patches = ax.hist(r_vals, bins=bins, edgecolor='black', alpha=0.85)
# Color each bar by its bin's center
for patch, edge in zip(patches, bin_edges[:-1]):
    center = edge + (bin_edges[1] - bin_edges[0]) / 2
    patch.set_facecolor(bin_color(center))

med = df_valid['pearson_r'].median()
mean = df_valid['pearson_r'].mean()

ax.axvline(med, color='black', linestyle='--', linewidth=2,
           label=f'Median = {med:.2f}')
ax.axvline(INTERPLM_MEDIAN, color='purple', linestyle=':', linewidth=2,
           label=f'InterPLM median = {INTERPLM_MEDIAN:.2f}')
ax.axvline(0.5, color='gray', linestyle='-', alpha=0.4, label='r = 0.5 (moderate)')
ax.set_xlabel("Held-out Pearson correlation r\n(Claude's prediction vs true atom activation)")
ax.set_ylabel('# atoms')
ax.set_title(f'Distribution of description quality across {len(df_valid)} Uninformative atoms',
             fontweight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3, axis='y')

# Stats box
n_high = (r_vals >= 0.7).sum()
n_mod = ((r_vals >= 0.5) & (r_vals < 0.7)).sum()
n_weak = ((r_vals >= 0.3) & (r_vals < 0.5)).sum()
n_poor = (r_vals < 0.3).sum()
stats_text = (f"Strong  (r≥0.7):   {n_high} atoms ({100*n_high/len(df_valid):.0f}%)\n"
              f"Moderate(0.5-0.7): {n_mod} atoms ({100*n_mod/len(df_valid):.0f}%)\n"
              f"Weak   (0.3-0.5):  {n_weak} atoms ({100*n_weak/len(df_valid):.0f}%)\n"
              f"Poor   (r<0.3):    {n_poor} atoms ({100*n_poor/len(df_valid):.0f}%)")
ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        va='top', ha='right', family='monospace', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

plt.tight_layout()
plt.savefig(fig_dir / 'fig1_pearson_distribution.png')
plt.close()
print(f"  saved: fig1_pearson_distribution.png")

# ============================================================
# Fig 2: Top-N atoms ranked by r (horizontal bars with descriptions)
# ============================================================
print("\nFig 2: Top atoms bar chart ...")
top_n = min(20, len(df_valid))
top = df_valid.nlargest(top_n, 'pearson_r').reset_index(drop=True)

fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.4)))

colors = [bin_color(r) for r in top['pearson_r']]
y_pos = np.arange(len(top))
ax.barh(y_pos, top['pearson_r'], color=colors, edgecolor='black', linewidth=0.5)

# Y labels = atom_id + truncated summary
ylabels = []
for _, row in top.iterrows():
    s = str(row.get('summary', ''))[:70]
    ylabels.append(f"atom {int(row['atom_id'])}: {s}")
ax.set_yticks(y_pos)
ax.set_yticklabels(ylabels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Held-out Pearson r')
ax.set_xlim(0, max(1.0, top['pearson_r'].max() * 1.05))
ax.set_title(f'Top {top_n} Uninformative atoms by description quality',
             fontweight='bold')
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='r = 0.5')
ax.axvline(INTERPLM_MEDIAN, color='purple', linestyle=':', alpha=0.7,
           label=f'InterPLM median {INTERPLM_MEDIAN:.2f}')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, axis='x')

# Annotate r value at end of each bar
for i, r in enumerate(top['pearson_r']):
    ax.text(r + 0.01, i, f'{r:.2f}', va='center', fontsize=8)

plt.tight_layout()
plt.savefig(fig_dir / 'fig2_top_atoms_bar.png')
plt.close()
print(f"  saved: fig2_top_atoms_bar.png")

# ============================================================
# Fig 3: r vs firing frequency scatter
# ============================================================
print("\nFig 3: r vs firing frequency ...")
fig, ax = plt.subplots(figsize=(9, 6))
colors_pts = [bin_color(r) for r in df_valid['pearson_r']]
ax.scatter(df_valid['fire_pct'], df_valid['pearson_r'],
           s=80, c=colors_pts, edgecolors='black', linewidth=0.5, alpha=0.85)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='r=0.5')
ax.axhline(INTERPLM_MEDIAN, color='purple', linestyle=':', alpha=0.7,
           label=f'InterPLM median')
ax.set_xlabel('Atom firing frequency (% of ECGs)')
ax.set_ylabel('Held-out Pearson r')
ax.set_title('Description quality vs atom firing frequency',
             fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend()

# Annotate a few top atoms
top5 = df_valid.nlargest(5, 'pearson_r')
for _, row in top5.iterrows():
    ax.annotate(f"atom {int(row['atom_id'])}",
                xy=(row['fire_pct'], row['pearson_r']),
                xytext=(5, 5), textcoords='offset points', fontsize=7)

plt.tight_layout()
plt.savefig(fig_dir / 'fig3_r_vs_firefreq.png')
plt.close()
print(f"  saved: fig3_r_vs_firefreq.png")

# ============================================================
# Fig 4: Summary panel (bar chart of bins + key numbers)
# ============================================================
print("\nFig 4: Summary panel ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: stacked bar of r bins
ax = axes[0]
bin_names = ['Poor\n(r<0.3)', 'Weak\n(0.3-0.5)', 'Moderate\n(0.5-0.7)', 'Strong\n(r≥0.7)']
bin_counts = [n_poor, n_weak, n_mod, n_high]
bin_colors_b = ['#d62728', '#fdae61', '#7fbf7b', '#2ca02c']
bars = ax.bar(bin_names, bin_counts, color=bin_colors_b,
              edgecolor='black', linewidth=0.8)
for bar, c in zip(bars, bin_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{c}\n({100*c/len(df_valid):.0f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('# atoms')
ax.set_title('Atoms by description quality category', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(bin_counts) * 1.25)

# Right: textual summary
ax = axes[1]
ax.axis('off')
summary_text = f"""STAGE 17 CLAUDE INTERPRETATION
of Uninformative Atoms

Atoms attempted:    {len(df)}
With valid r:       {len(df_valid)}

PEARSON r STATISTICS
  Median:           {df_valid['pearson_r'].median():.3f}
  Mean:             {df_valid['pearson_r'].mean():.3f}
  Std:              {df_valid['pearson_r'].std():.3f}
  Min:              {df_valid['pearson_r'].min():.3f}
  Max:              {df_valid['pearson_r'].max():.3f}

QUALITY BREAKDOWN
  Strong  (r≥0.7):  {n_high}  ({100*n_high/len(df_valid):.1f}%)
  Moderate(≥0.5):   {n_mod}   ({100*n_mod/len(df_valid):.1f}%)
  Weak   (0.3-0.5): {n_weak}  ({100*n_weak/len(df_valid):.1f}%)
  Poor   (r<0.3):   {n_poor}  ({100*n_poor/len(df_valid):.1f}%)

COMPARISON
  InterPLM (protein,
  Sonnet, 50k ex.):  0.720
  Ours (ECG):       {df_valid['pearson_r'].median():.3f}

INTERPRETATION
  Recovered semantics for
  {n_high + n_mod}/{len(df_valid)} previously
  uncharacterized atoms
  ({100*(n_high+n_mod)/len(df_valid):.1f}% of attempted)
"""
ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
        va='top', ha='left', family='monospace', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='#f0f8ff',
                  edgecolor='steelblue', linewidth=1.5))

plt.tight_layout()
plt.savefig(fig_dir / 'fig4_summary_panel.png')
plt.close()
print(f"  saved: fig4_summary_panel.png")

# ============================================================
# Print summary
# ============================================================
print("\n" + "=" * 60)
print("Visualization complete")
print("=" * 60)
print(f"\nFigures in: {fig_dir}")
for f in sorted(fig_dir.glob('*.png')):
    print(f"  {f.name}")
print(f"\nKey numbers:")
print(f"  median r:        {df_valid['pearson_r'].median():.3f}")
print(f"  InterPLM median: 0.720")
print(f"  strong/moderate: {n_high + n_mod}/{len(df_valid)} "
      f"({100*(n_high+n_mod)/len(df_valid):.1f}%)")
