"""
Grid search results visualization.
Generates a complete set of figures from grid_results.csv:

  Fig 1: 5x4 heatmaps -- 9 metrics in a single figure
  Fig 2: Sparsity-fidelity Pareto (L0 vs val_EV)
  Fig 3: Per-task AUROC curves (one line per K, x=TopK)
  Fig 4: Faithfulness vs PRS (does recon preserve task info?)
  Fig 5: Atom taxonomy stacked bars (Separable/Entangled/Dead)
  Fig 6: Top-1 sparse probe AUROC heatmap (per task)
  Fig 7: Winner-selection composite ranking
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

# ============================================================
# Setup
# ============================================================
mpl.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
})

RESULTS_DIR = cfg.EMBEDDING_DIR.parent / "grid_search"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_DIR / "grid_results.csv")
dense_row = df[df['model'] == 'DENSE'].iloc[0]
df = df[df['model'] != 'DENSE'].copy()
df['topk'] = df['topk'].astype(int)
df['d_sae'] = df['d_sae'].astype(int)

TOPKS = sorted(df['topk'].unique())          # [16, 32, 64, 128, 256]
KS = sorted(df['d_sae'].unique())            # [1536, 3072, 6144, 12288]
EXPANSIONS = sorted(df['expansion'].unique())  # [2, 4, 8, 16]

print(f"Loaded {len(df)} models | TopK={TOPKS} | K={KS}")


def to_matrix(metric):
    """Reshape to TopK (rows) x K (cols) matrix for heatmap."""
    M = np.full((len(TOPKS), len(KS)), np.nan)
    for _, r in df.iterrows():
        i = TOPKS.index(int(r['topk']))
        j = KS.index(int(r['d_sae']))
        M[i, j] = r[metric]
    return M


def heatmap(ax, data, title, cmap='RdYlGn', center=None, fmt='.3f', annot=True,
            vmin=None, vmax=None):
    """Pretty heatmap with annotations."""
    if vmin is None: vmin = np.nanmin(data)
    if vmax is None: vmax = np.nanmax(data)
    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(KS)))
    ax.set_xticklabels([f'{k}\n({2**i+1}x)' if i==0 else f'{k}\n({2**(i+1)}x)' for i,k in enumerate(KS)])
    ax.set_xticklabels([f'K={k}\n({e}x)' for k, e in zip(KS, EXPANSIONS)])
    ax.set_yticks(range(len(TOPKS)))
    ax.set_yticklabels([f'TopK={t}' for t in TOPKS])
    ax.set_title(title, fontweight='bold')
    if annot:
        for i in range(len(TOPKS)):
            for j in range(len(KS)):
                v = data[i, j]
                if np.isnan(v):
                    continue
                # decide text color by background luminance
                norm = (v - vmin) / (vmax - vmin + 1e-9)
                txt_color = 'white' if 0.2 < norm < 0.8 else 'black'
                if isinstance(fmt, str) and fmt.endswith('d'):
                    txt = f'{int(v)}'
                else:
                    txt = f'{v:{fmt}}'
                ax.text(j, i, txt, ha='center', va='center',
                        color=txt_color, fontsize=8, fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


# ============================================================
# Fig 1: 9-panel heatmap of core metrics
# ============================================================
print("Fig 1: Core metrics 3x3 heatmap...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

panels = [
    ('val_ev', 'Reconstruction R² (val_EV)', 'RdYlGn', '.3f'),
    ('dead_pct', 'Dead Atom % (lower better)', 'RdYlGn_r', '.1f'),
    ('actual_l0', 'Actual L0 (≈ TopK)', 'viridis', '.0f'),
    ('top20_purity', 'Top-20 Phenotype Purity', 'RdYlGn', '.3f'),
    ('separable_pct', 'Separable Atom % (AUROC>0.65)', 'RdYlGn', '.2f'),
    ('quality_score', 'Quality (Sep% − Dead%)', 'RdYlGn', '.1f'),
    ('auroc_af', 'AF AUROC (full probe)', 'RdYlGn', '.3f'),
    ('prs_af', 'AF PRS (vs dense)', 'RdYlGn', '.3f'),
    ('faithfulness_af', 'Faithfulness AF', 'RdYlGn', '.3f'),
]

for ax, (metric, title, cmap, fmt) in zip(axes.flat, panels):
    M = to_matrix(metric)
    heatmap(ax, M, title, cmap=cmap, fmt=fmt)
    ax.set_xlabel('Dictionary size')
    ax.set_ylabel('Sparsity (TopK)')

plt.suptitle('Grid Search: Core Metrics across (TopK × Dictionary Size)',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig1_core_heatmaps.png')
plt.savefig(FIG_DIR / 'fig1_core_heatmaps.pdf')
plt.close()
print(f"  saved: {FIG_DIR}/fig1_core_heatmaps.png")


# ============================================================
# Fig 2: Sparsity-Fidelity Pareto (SAEBench main plot)
# ============================================================
print("Fig 2: Sparsity-fidelity Pareto...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: val_EV vs L0
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(KS)))
for k, col in zip(KS, colors):
    sub = df[df['d_sae'] == k].sort_values('actual_l0')
    ax.plot(sub['actual_l0'], sub['val_ev'], 'o-', color=col,
            label=f'K={k}', linewidth=2, markersize=8)
    # annotate each point with TopK
    for _, r in sub.iterrows():
        ax.annotate(f"T={int(r['topk'])}",
                    xy=(r['actual_l0'], r['val_ev']),
                    xytext=(5, -8), textcoords='offset points', fontsize=7)
ax.set_xlabel('L0 (mean active atoms per ECG)')
ax.set_ylabel('Reconstruction R² (val_EV)')
ax.set_title('Sparsity ↔ Fidelity Trade-off')
ax.set_xscale('log')
ax.legend(title='Dictionary size', loc='lower right')
ax.grid(True, alpha=0.3)

# Right: val_EV vs AF PRS (does better recon -> better downstream?)
ax = axes[1]
for k, col in zip(KS, colors):
    sub = df[df['d_sae'] == k].sort_values('actual_l0')
    ax.plot(sub['val_ev'], sub['prs_af'], 'o-', color=col,
            label=f'K={k}', linewidth=2, markersize=8)
    for _, r in sub.iterrows():
        ax.annotate(f"T={int(r['topk'])}",
                    xy=(r['val_ev'], r['prs_af']),
                    xytext=(5, -8), textcoords='offset points', fontsize=7)
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Dense baseline')
ax.set_xlabel('Reconstruction R² (val_EV)')
ax.set_ylabel('AF PRS (SAE AUROC / Dense AUROC)')
ax.set_title('Does better reconstruction = better downstream?')
ax.legend(title='Dictionary size')
ax.grid(True, alpha=0.3)

plt.suptitle('Sparsity-Fidelity-Downstream Analysis (SAEBench-style)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig2_pareto.png')
plt.close()
print(f"  saved: {FIG_DIR}/fig2_pareto.png")


# ============================================================
# Fig 3: Per-task AUROC across TopK (with dense baseline lines)
# ============================================================
print("Fig 3: Per-task AUROC sweeps...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
tasks = ['af', 'hf', 'mi', 'dm', 'htn', 'sex', 'age']
task_titles = {'af':'Atrial Fibrillation', 'hf':'Heart Failure', 'mi':'MI/IHD',
               'dm':'Diabetes', 'htn':'Hypertension', 'sex':'Sex', 'age':'Age (R²)'}
for ax, task in zip(axes.flat[:7], tasks):
    metric = f'auroc_{task}'
    for k, col in zip(KS, colors):
        sub = df[df['d_sae'] == k].sort_values('topk')
        ax.plot(sub['topk'], sub[metric], 'o-', color=col,
                label=f'K={k}', linewidth=2, markersize=7)
    # Dense baseline
    dense_v = dense_row[metric]
    ax.axhline(dense_v, color='red', linestyle='--', alpha=0.6,
               label=f'Dense ({dense_v:.3f})')
    ax.set_xlabel('TopK (sparsity)')
    ax.set_ylabel('AUROC' if task != 'age' else 'R²')
    ax.set_title(task_titles[task])
    ax.set_xscale('log', base=2)
    ax.set_xticks(TOPKS)
    ax.set_xticklabels(TOPKS)
    ax.grid(True, alpha=0.3)
    if task == 'af':
        ax.legend(loc='lower right', fontsize=7)

# Last panel: legend / summary
axes.flat[7].axis('off')
axes.flat[7].text(0.05, 0.5,
    "Each subplot: AUROC (R² for age)\nvs sparsity, one line per\ndictionary size.\n\n"
    "Red dashed line = dense\nembedding baseline.\n\n"
    "Closer to red = better\ndownstream preservation.",
    transform=axes.flat[7].transAxes, fontsize=10, va='center')

plt.suptitle('Per-Task Downstream Performance vs Sparsity',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig3_per_task_auroc.png')
plt.close()
print(f"  saved: {FIG_DIR}/fig3_per_task_auroc.png")


# ============================================================
# Fig 4: Faithfulness vs PRS scatter
# ============================================================
print("Fig 4: Faithfulness analysis...")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: faithfulness mean across tasks, color by TopK, size by K
ax = axes[0]
faith_cols = [c for c in df.columns if c.startswith('faithfulness_')]
df['faith_mean'] = df[faith_cols].mean(axis=1)
prs_cols = [c for c in df.columns if c.startswith('prs_')]
df['prs_mean'] = df[prs_cols].mean(axis=1)

topk_colors = {t: plt.cm.plasma(i/(len(TOPKS)-1)) for i, t in enumerate(TOPKS)}
for _, r in df.iterrows():
    size = 30 + 200 * (np.log2(r['d_sae']) - np.log2(min(KS))) / (np.log2(max(KS)) - np.log2(min(KS)))
    ax.scatter(r['faith_mean'], r['prs_mean'],
               s=size, c=[topk_colors[int(r['topk'])]],
               edgecolors='black', linewidth=0.5, alpha=0.85)
    ax.annotate(r['model'], (r['faith_mean'], r['prs_mean']),
                fontsize=7, xytext=(5, 5), textcoords='offset points')
ax.axhline(1.0, color='red', linestyle='--', alpha=0.5)
ax.axvline(1.0, color='red', linestyle='--', alpha=0.5)
ax.set_xlabel('Mean Faithfulness (recon AUROC / dense AUROC)')
ax.set_ylabel('Mean PRS (SAE AUROC / dense AUROC)')
ax.set_title('Reconstruction Quality ↔ Information Preservation')
ax.grid(True, alpha=0.3)
# Add color legend
for t in TOPKS:
    ax.scatter([], [], c=[topk_colors[t]], s=80, label=f'TopK={t}',
               edgecolors='black', linewidth=0.5)
ax.legend(title='Sparsity', loc='lower right', fontsize=8)

# Right: Faithfulness heatmap (mean across tasks)
ax = axes[1]
M = np.full((len(TOPKS), len(KS)), np.nan)
for _, r in df.iterrows():
    i = TOPKS.index(int(r['topk']))
    j = KS.index(int(r['d_sae']))
    M[i, j] = r['faith_mean']
heatmap(ax, M, 'Mean Faithfulness across 7 tasks', cmap='RdYlGn', fmt='.3f')
ax.set_xlabel('Dictionary size')
ax.set_ylabel('Sparsity (TopK)')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig4_faithfulness.png')
plt.close()
print(f"  saved: {FIG_DIR}/fig4_faithfulness.png")


# ============================================================
# Fig 5: Atom taxonomy stacked bars
# ============================================================
print("Fig 5: Atom taxonomy stacked bars...")
fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)

for ax, topk in zip(axes, TOPKS):
    sub = df[df['topk'] == topk].sort_values('d_sae')
    x = np.arange(len(sub))
    width = 0.7
    sep = sub['separable_pct'].values
    ent = sub['entangled_pct'].values
    dead = sub['dead_atom_pct'].values
    ax.bar(x, sep, width, label='Separable', color='#2ca02c')
    ax.bar(x, ent, width, bottom=sep, label='Entangled', color='#ff7f0e')
    ax.bar(x, dead, width, bottom=sep+ent, label='Dead', color='#d62728')
    ax.set_xticks(x)
    ax.set_xticklabels([f'K={int(k)}' for k in sub['d_sae']], rotation=30)
    ax.set_title(f'TopK={topk}', fontweight='bold')
    ax.set_ylabel('% of atoms' if topk == TOPKS[0] else '')
    ax.set_ylim(0, 100)
    if topk == TOPKS[0]:
        ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    # Annotate sep% on top (it's often tiny)
    for i, s in enumerate(sep):
        if s > 0:
            ax.text(i, s+1, f'{s:.2f}%', ha='center', fontsize=7, color='darkgreen')

plt.suptitle('Atom Taxonomy: Separable / Entangled / Dead (per TopK)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig5_taxonomy_stacked.png')
plt.close()
print(f"  saved: {FIG_DIR}/fig5_taxonomy_stacked.png")


# ============================================================
# Fig 6: Top-1 sparse probe AUROC for binary tasks + Age R² (full probe)
# ============================================================
print("Fig 6: Top-1 sparse probe heatmaps + Age R²...")
binary_tasks = [t for t in tasks if t != 'age']
n_bin = len(binary_tasks)
fig, axes = plt.subplots(2, 4, figsize=(17, 8))

# First 6 panels: binary tasks, top-1 atom AUROC
for ax, task in zip(axes.flat[:n_bin], binary_tasks):
    metric = f'top1_auroc_{task}'
    M = to_matrix(metric)
    if np.all(np.isnan(M)):
        ax.axis('off')
        ax.set_title(f'{task_titles[task]} (no data)')
        continue
    heatmap(ax, M, f'{task_titles[task]} — Top-1 atom AUROC',
            cmap='RdYlGn', fmt='.3f', vmin=0.5, vmax=0.85)
    ax.set_xlabel('Dictionary size')
    ax.set_ylabel('TopK')

# 7th panel: Age R² (from full-dictionary ridge probe — column 'auroc_age' stores R²)
ax = axes.flat[n_bin]
M_age = to_matrix('auroc_age')
age_dense_r2 = dense_row['auroc_age']
if not np.all(np.isnan(M_age)):
    age_vmin = max(0.0, np.nanmin(M_age) - 0.02)
    age_vmax = min(1.0, np.nanmax(M_age) + 0.02)
    heatmap(ax, M_age, f'Age — Full-probe R²\n(dense baseline: {age_dense_r2:.3f})',
            cmap='RdYlGn', fmt='.3f', vmin=age_vmin, vmax=age_vmax)
    ax.set_xlabel('Dictionary size')
    ax.set_ylabel('TopK')
else:
    ax.axis('off')
    ax.set_title('Age R² (no data)')

# 8th panel: summary text
ax = axes.flat[-1]
ax.axis('off')
ax.text(0.05, 0.5,
    "k=1 sparse probe:\n"
    "best single atom's AUROC\nfor each binary task.\n\n"
    "High (>0.70) = there exists\na single atom that captures\nthe concept monosemantically.\n\n"
    "Low (~0.55) = no single atom\ndoes it; concept is distributed\nacross many atoms.\n\n"
    "(Age is regression — see\nFig 3 for its R² metric.)",
    transform=ax.transAxes, fontsize=10, va='center')

plt.suptitle('SAEBench k=1 Sparse Probing per Binary Task',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig6_top1_sparse.png')
plt.close()
print(f"  saved: {FIG_DIR}/fig6_top1_sparse.png")


# ============================================================
# Fig 7: Winner-selection composite ranking
# ============================================================
print("Fig 7: Winner ranking...")

# Normalize each metric to [0,1] (higher=better), then average key metrics
def norm01(x, higher_better=True):
    x = np.asarray(x, dtype=float)
    if higher_better:
        return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)
    else:
        return 1.0 - (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9)

df['_recon_n'] = norm01(df['val_ev'])
df['_dead_n'] = norm01(df['dead_pct'], higher_better=False)
df['_faith_n'] = norm01(df['faith_mean'])
df['_prs_n'] = norm01(df['prs_mean'])
df['_sep_n'] = norm01(df['separable_pct'])
df['_top1_n'] = norm01(df[[f'top1_auroc_{t}' for t in ['af','hf','mi','dm']]].mean(axis=1))

# Composite score with weights (SAEBench-inspired: equal across capabilities)
df['composite'] = (
    0.20 * df['_recon_n']     # reconstruction
    + 0.15 * df['_dead_n']    # dictionary health
    + 0.20 * df['_faith_n']   # faithfulness
    + 0.20 * df['_prs_n']     # downstream
    + 0.10 * df['_sep_n']     # monosemanticity
    + 0.15 * df['_top1_n']    # sparse probing
)

ranked = df.sort_values('composite', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: bar chart of composite scores
ax = axes[0]
ranked_top = ranked.copy()
labels = [f"{r['model']}\nT={int(r['topk'])} K={int(r['d_sae'])}"
          for _, r in ranked_top.iterrows()]
colors_bar = ['#2ca02c' if i < 3 else '#1f77b4' if i < 10 else '#7f7f7f'
              for i in range(len(ranked_top))]
ax.barh(range(len(ranked_top)), ranked_top['composite'], color=colors_bar)
ax.set_yticks(range(len(ranked_top)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Composite Score (0=worst, 1=best)')
ax.set_title('Overall Winner Ranking\n(weighted: recon 20% / health 15% / faith 20% / PRS 20% / sep 10% / top1 15%)')
ax.grid(True, alpha=0.3, axis='x')
ax.axvline(ranked_top['composite'].iloc[2], color='green', linestyle='--', alpha=0.5)

# Right: composite heatmap
ax = axes[1]
M = np.full((len(TOPKS), len(KS)), np.nan)
for _, r in df.iterrows():
    i = TOPKS.index(int(r['topk']))
    j = KS.index(int(r['d_sae']))
    M[i, j] = r['composite']
heatmap(ax, M, 'Composite Score Heatmap', cmap='RdYlGn', fmt='.2f')
ax.set_xlabel('Dictionary size')
ax.set_ylabel('TopK')

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig7_winner_ranking.png')
plt.close()
print(f"  saved: {FIG_DIR}/fig7_winner_ranking.png")


# ============================================================
# Print top-5 winners
# ============================================================
print("\n" + "="*70)
print("TOP-5 MODELS BY COMPOSITE SCORE")
print("="*70)
cols_show = ['model', 'topk', 'd_sae', 'val_ev', 'actual_l0', 'dead_pct',
             'separable_pct', 'faith_mean', 'prs_mean', 'composite']
print(ranked[cols_show].head(5).to_string(index=False))

print("\n" + "="*70)
print("WINNER BY EACH SINGLE CRITERION")
print("="*70)
criteria = [
    ('Reconstruction (val_ev)', 'val_ev', True),
    ('Faithfulness mean', 'faith_mean', True),
    ('PRS mean', 'prs_mean', True),
    ('Separable %', 'separable_pct', True),
    ('Top-1 AF AUROC', 'top1_auroc_af', True),
    ('Quality (sep - dead)', 'quality_score', True),
    ('Lowest dead %', 'dead_pct', False),
]
for name, col, hb in criteria:
    if hb:
        best = df.loc[df[col].idxmax()]
    else:
        best = df.loc[df[col].idxmin()]
    print(f"  {name:25s} -> {best['model']:4s} (TopK={int(best['topk'])}, "
          f"K={int(best['d_sae'])}): {best[col]:.4f}")

# Save ranked CSV
ranked[['model', 'topk', 'd_sae', 'val_ev', 'dead_pct', 'separable_pct',
        'faith_mean', 'prs_mean', 'composite']].to_csv(
    FIG_DIR / 'ranked_models.csv', index=False)

print(f"\n{'='*70}")
print(f"All figures saved to: {FIG_DIR}/")
print(f"{'='*70}")
