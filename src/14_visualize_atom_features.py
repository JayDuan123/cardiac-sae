"""
Visualize Stage 13 results: atom <-> ECG feature correspondence.

Produces:
  Fig 1: Atom-feature correlation heatmap (selected top atoms x features)
  Fig 2: Top atoms per feature (bar charts, 4x4 grid)
  Fig 3: Distribution of |max correlation| across atoms (how many atoms have a feature?)
  Fig 4: Atom case studies -- side-by-side text label + numerical profile
  Fig 5: Feature-feature correlation among atoms (which features cluster?)
  Fig 6: Scatter: per-atom QRS duration vs heart rate, colored by report label
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

mpl.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'figure.dpi': 100, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
feat_dir = sae_dir / "atom_features"
fig_dir = feat_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load all Stage 13 outputs
# ============================================================
print("Loading Stage 13 outputs ...")
corr = pd.read_csv(feat_dir / "atom_feature_correlation.csv", index_col='atom_id')
top_atoms = pd.read_csv(feat_dir / "feature_top_atoms.csv")
profiles = pd.read_csv(feat_dir / "atom_profiles.csv")

# Combined (if exists)
combined_path = feat_dir / "atom_combined_report_and_features.csv"
if combined_path.exists():
    combined = pd.read_csv(combined_path)
    print(f"  loaded combined: {len(combined)} atoms")
else:
    combined = None
    print("  no combined CSV (Stage 12 outputs not found)")

print(f"  correlation matrix: {corr.shape}")
print(f"  top_atoms per feature: {len(top_atoms)} rows")
print(f"  profiles: {len(profiles)} atoms")

FEATURES_NUM = ['heart_rate', 'pr_interval', 'qrs_duration', 'qt_interval', 'qtc',
                'p_axis', 'qrs_axis', 't_axis']
FEATURES_BIN = ['tachycardia', 'bradycardia', 'wide_qrs', 'long_qt',
                'left_axis', 'right_axis', 'st_elevation', 'st_depression']
ALL_FEATURES = FEATURES_NUM + FEATURES_BIN

FEAT_LABELS = {
    'heart_rate': 'Heart rate', 'pr_interval': 'PR interval',
    'qrs_duration': 'QRS duration', 'qt_interval': 'QT interval',
    'qtc': 'QTc', 'p_axis': 'P axis', 'qrs_axis': 'QRS axis', 't_axis': 'T axis',
    'tachycardia': 'Tachycardia', 'bradycardia': 'Bradycardia',
    'wide_qrs': 'Wide QRS (>120ms)', 'long_qt': 'Long QT',
    'left_axis': 'Left axis', 'right_axis': 'Right axis',
    'st_elevation': 'ST elevation', 'st_depression': 'ST depression',
}


# ============================================================
# Fig 1: Top-50 atoms x features correlation heatmap
# ============================================================
print("\nFig 1: Top-50 atom-feature correlation heatmap ...")

# Select atoms that have at least one strong correlation (|r| > 0.3) with any feature
abs_corr = corr.abs()
strong_atoms = abs_corr[abs_corr.max(axis=1) > 0.3].index.tolist()
print(f"  {len(strong_atoms)} atoms with |r|>0.3 for some feature")

if len(strong_atoms) > 50:
    # Take top 50 by max abs correlation
    top50 = abs_corr.loc[strong_atoms].max(axis=1).nlargest(50).index.tolist()
else:
    top50 = strong_atoms

# Order atoms by hierarchical clustering of correlation patterns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist

heat = corr.loc[top50, ALL_FEATURES].values
# Fill NaN with 0 for clustering (NaN = atom didn't fire enough; treat as "no correlation")
heat_for_cluster = np.nan_to_num(heat, nan=0.0)
if len(top50) >= 2:
    dist = pdist(heat_for_cluster, metric='euclidean')
    if np.all(np.isfinite(dist)) and dist.sum() > 0:
        Z = linkage(dist, method='average')
        order = leaves_list(Z)
        heat = heat[order]
        atom_labels = [str(a) for a in np.array(top50)[order]]
    else:
        atom_labels = [str(a) for a in top50]
else:
    atom_labels = [str(a) for a in top50]
# Keep NaN visible in the heatmap (will show as a masked cell)
heat_display = np.ma.masked_invalid(heat)

fig, ax = plt.subplots(figsize=(13, max(10, len(top50) * 0.22)))
vmax = max(0.5, np.nanmax(np.abs(heat)))
cmap = plt.cm.RdBu_r.copy()
cmap.set_bad(color='lightgray')   # NaN cells -> gray
im = ax.imshow(heat_display, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(ALL_FEATURES)))
ax.set_xticklabels([FEAT_LABELS[f] for f in ALL_FEATURES], rotation=45, ha='right')
ax.set_yticks(range(len(atom_labels)))
ax.set_yticklabels([f'atom {a}' for a in atom_labels])
ax.set_title(f'Atom × ECG Feature Spearman Correlation\n(top {len(top50)} atoms by |max r|, hierarchically clustered)',
             fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label='Spearman r')
# Add value annotations
for i in range(len(atom_labels)):
    for j in range(len(ALL_FEATURES)):
        v = heat[i, j]
        if not np.isnan(v) and abs(v) > 0.3:
            ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                    fontsize=6, color='white' if abs(v) > 0.5 else 'black')
plt.tight_layout()
plt.savefig(fig_dir / 'fig1_correlation_heatmap.png')
plt.close()
print(f"  saved: {fig_dir}/fig1_correlation_heatmap.png")


# ============================================================
# Fig 2: Top atoms per feature (bar charts)
# ============================================================
print("\nFig 2: Top atoms per feature bar charts ...")
n_features = len(ALL_FEATURES)
ncols = 4
nrows = (n_features + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3 * nrows))
axes = axes.flatten()

for ax, feat in zip(axes, ALL_FEATURES):
    sub = top_atoms[top_atoms['feature'] == feat].head(10).copy()
    if len(sub) == 0:
        ax.axis('off')
        ax.set_title(f'{FEAT_LABELS[feat]} (no atoms)')
        continue
    sub = sub.sort_values('spearman_r')
    colors = ['#2ca02c' if r > 0 else '#d62728' for r in sub['spearman_r']]
    ax.barh(range(len(sub)), sub['spearman_r'], color=colors)
    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels([f'#{a}' for a in sub['atom_id']], fontsize=7)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Spearman r', fontsize=8)
    ax.set_title(FEAT_LABELS[feat], fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    # Annotate the bar values
    for i, r in enumerate(sub['spearman_r']):
        ha = 'left' if r > 0 else 'right'
        offset = 0.005 if r > 0 else -0.005
        ax.text(r + offset, i, f'{r:.2f}', va='center', ha=ha, fontsize=7)

for ax in axes[n_features:]:
    ax.axis('off')

plt.suptitle('Top-10 Atoms per ECG Feature (Spearman r, sorted)',
             fontsize=13, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(fig_dir / 'fig2_top_atoms_per_feature.png')
plt.close()
print(f"  saved: {fig_dir}/fig2_top_atoms_per_feature.png")


# ============================================================
# Fig 3: Distribution of |max r| across all atoms
# ============================================================
print("\nFig 3: Distribution of atom-feature correlations ...")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: max |r| per atom (how strongly each atom encodes ANY feature)
ax = axes[0]
max_abs = corr.abs().max(axis=1).dropna()
ax.hist(max_abs, bins=40, color='steelblue', edgecolor='black', alpha=0.8)
ax.axvline(0.3, color='orange', linestyle='--', label='|r|=0.3')
ax.axvline(0.5, color='red', linestyle='--', label='|r|=0.5')
ax.set_xlabel('max(|Spearman r|) across all features')
ax.set_ylabel('# atoms')
ax.set_title(f'How well-grounded is each atom?\n(n={len(max_abs)} atoms with computable correlations)')
ax.legend()
ax.grid(True, alpha=0.3)
# Annotate counts
n_03 = (max_abs > 0.3).sum()
n_05 = (max_abs > 0.5).sum()
ax.text(0.98, 0.98, f'|r|>0.3: {n_03} atoms ({100*n_03/len(max_abs):.1f}%)\n'
                     f'|r|>0.5: {n_05} atoms ({100*n_05/len(max_abs):.1f}%)',
        transform=ax.transAxes, va='top', ha='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Right: # atoms strongly correlated (|r|>0.3) with each feature
ax = axes[1]
feat_counts = (corr.abs() > 0.3).sum(axis=0).sort_values(ascending=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(feat_counts)))
ax.barh(range(len(feat_counts)), feat_counts.values, color=colors)
ax.set_yticks(range(len(feat_counts)))
ax.set_yticklabels([FEAT_LABELS.get(f, f) for f in feat_counts.index])
ax.set_xlabel('# atoms with |r| > 0.3')
ax.set_title('How many atoms strongly encode each feature?')
ax.grid(True, alpha=0.3, axis='x')
for i, v in enumerate(feat_counts.values):
    ax.text(v + 1, i, str(v), va='center', fontsize=8)

plt.tight_layout()
plt.savefig(fig_dir / 'fig3_correlation_distribution.png')
plt.close()
print(f"  saved: {fig_dir}/fig3_correlation_distribution.png")


# ============================================================
# Fig 4: Case studies -- top atoms, text label + numerical profile
# ============================================================
print("\nFig 4: Atom case studies ...")
if combined is not None and len(combined) > 0:
    # Pick atoms that have BOTH strong text lift AND strong numerical correlation
    # Strong text: lift > 100 ; Strong numerical: any |r| > 0.3
    
    # Merge: for each atom in combined, attach its max abs correlation
    combined = combined.copy()
    combined['max_abs_r'] = combined['atom_id'].map(corr.abs().max(axis=1))
    combined['max_r_feature'] = combined['atom_id'].map(
        lambda a: corr.loc[a].abs().idxmax() if a in corr.index else None
    )
    
    # Pick top atoms that have both signals
    case_pool = combined[
        (combined['lift'] > 50) & (combined['max_abs_r'] > 0.2)
    ].nlargest(12, 'lift')
    
    if len(case_pool) > 0:
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        axes = axes.flatten()
        for ax, (_, r) in zip(axes, case_pool.iterrows()):
            atom_id = int(r['atom_id'])
            ax.axis('off')
            # Get profile
            prof = profiles[profiles['atom_id'] == atom_id]
            if len(prof) == 0:
                continue
            prof = prof.iloc[0]
            
            # Build text block
            text_lines = [
                f"ATOM {atom_id}",
                f"freq={r.get('freq_pct', 0):.2f}%",
                "",
                "REPORT LABEL (Stage 12):",
                f"  '{r['label'][:55]}'",
                f"  lift={r['lift']:.0f}",
                "",
                "NUMERICAL PROFILE (top-50):",
            ]
            # Add measurements that exist
            for feat, label, unit in [
                ('heart_rate_median', 'HR', 'bpm'),
                ('pr_interval_median', 'PR', 'ms'),
                ('qrs_duration_median', 'QRS', 'ms'),
                ('qt_interval_median', 'QT', 'ms'),
                ('qtc_median', 'QTc', 'ms'),
                ('qrs_axis_median', 'QRS axis', '°'),
            ]:
                if feat in prof and pd.notna(prof[feat]):
                    text_lines.append(f"  {label}: {prof[feat]:.0f} {unit}")
            
            text_lines.append("")
            text_lines.append("STRONG CORRELATIONS:")
            atom_corrs = corr.loc[atom_id].dropna()
            strong = atom_corrs[atom_corrs.abs() > 0.3].sort_values(key=abs, ascending=False).head(4)
            for feat_name, val in strong.items():
                sign = '+' if val > 0 else ''
                text_lines.append(f"  {FEAT_LABELS.get(feat_name, feat_name)}: {sign}{val:.2f}")
            if len(strong) == 0:
                text_lines.append("  (none with |r| > 0.3)")
            
            text = '\n'.join(text_lines)
            ax.text(0.05, 0.95, text, transform=ax.transAxes,
                    va='top', ha='left', family='monospace', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.9,
                              edgecolor='steelblue', linewidth=1.5))
        
        for ax in axes[len(case_pool):]:
            ax.axis('off')
        
        plt.suptitle('Atom Case Studies: Text Label + Numerical Profile',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(fig_dir / 'fig4_case_studies.png')
        plt.close()
        print(f"  saved: {fig_dir}/fig4_case_studies.png")
    else:
        print("  no atoms with both strong lift and strong correlation; skipping")
else:
    print("  no combined CSV; skipping case studies")


# ============================================================
# Fig 5: Feature-feature clustering (which features are encoded together?)
# ============================================================
print("\nFig 5: Feature-feature clustering across atoms ...")
# For each pair of features, count how many atoms have |r|>0.3 with both
strong = (corr.abs() > 0.3).astype(int)
co_occur = strong.T @ strong   # (n_feat, n_feat) co-occurrence matrix
co_occur = co_occur.values

# Normalize: jaccard-like
norm = co_occur / (np.diag(co_occur)[:, None] + np.diag(co_occur)[None, :] - co_occur + 1e-9)
np.fill_diagonal(norm, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

ax = axes[0]
im = ax.imshow(co_occur, cmap='YlOrRd', aspect='auto')
ax.set_xticks(range(len(ALL_FEATURES)))
ax.set_xticklabels([FEAT_LABELS[f] for f in ALL_FEATURES], rotation=45, ha='right')
ax.set_yticks(range(len(ALL_FEATURES)))
ax.set_yticklabels([FEAT_LABELS[f] for f in ALL_FEATURES])
ax.set_title('# atoms strongly correlated with both features')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for i in range(len(ALL_FEATURES)):
    for j in range(len(ALL_FEATURES)):
        ax.text(j, i, str(co_occur[i, j]), ha='center', va='center', fontsize=6)

ax = axes[1]
im = ax.imshow(norm, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(len(ALL_FEATURES)))
ax.set_xticklabels([FEAT_LABELS[f] for f in ALL_FEATURES], rotation=45, ha='right')
ax.set_yticks(range(len(ALL_FEATURES)))
ax.set_yticklabels([FEAT_LABELS[f] for f in ALL_FEATURES])
ax.set_title('Jaccard overlap (shared atoms / total atoms)')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for i in range(len(ALL_FEATURES)):
    for j in range(len(ALL_FEATURES)):
        ax.text(j, i, f'{norm[i,j]:.2f}', ha='center', va='center', fontsize=6)

plt.suptitle('Which ECG features cluster together in the SAE dictionary?',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(fig_dir / 'fig5_feature_clustering.png')
plt.close()
print(f"  saved: {fig_dir}/fig5_feature_clustering.png")


# ============================================================
# Fig 6: HR vs QRS scatter per atom, colored by report label
# ============================================================
print("\nFig 6: HR vs QRS atom map ...")
if 'heart_rate_median' in profiles.columns and 'qrs_duration_median' in profiles.columns:
    # Atom-level scatter
    sub = profiles.dropna(subset=['heart_rate_median', 'qrs_duration_median'])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: density of all atoms in (HR, QRS) space
    ax = axes[0]
    ax.scatter(sub['heart_rate_median'], sub['qrs_duration_median'],
               s=20, alpha=0.3, c='steelblue', edgecolors='none')
    ax.axvline(60, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axvline(100, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.axhline(120, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.set_xlabel('Median HR of top-50 ECGs (bpm)')
    ax.set_ylabel('Median QRS duration of top-50 ECGs (ms)')
    ax.set_title(f'All atoms (n={len(sub)}) in HR-QRS space')
    ax.text(40, 110, 'Brady\nNarrow', ha='center', fontsize=8, color='gray')
    ax.text(130, 110, 'Tachy\nNarrow', ha='center', fontsize=8, color='gray')
    ax.text(130, 160, 'Tachy\nWide (VT)', ha='center', fontsize=8, color='gray')
    ax.text(40, 160, 'Brady\nWide', ha='center', fontsize=8, color='gray')
    ax.grid(True, alpha=0.3)
    
    # Right: atoms with strong labels, colored by report
    ax = axes[1]
    if combined is not None:
        merged = sub.merge(combined[['atom_id', 'label', 'lift']], on='atom_id')
        # Pick canonical labels
        label_groups = {
            'Tachycardia': ['Sinus tachycardia', 'Sinus tachycardia.'],
            'Bradycardia': ['Sinus bradycardia', 'Sinus bradycardia.',
                            'Marked sinus bradycardia.'],
            'LBBB': ['Left bundle branch block'],
            'RBBB': ['Right bundle branch block'],
            'AF/Aflu': ['Atrial fibrillation with rapid ventricular response',
                        'Atrial fibrillation with rapid ventricular response.',
                        'Atrial flutter with uncontrolled ventricular response with 2:1 A-V block',
                        'Atrial flutter with 4:1 A-V block'],
            'VT': ['Probable ventricular tachycardia', 'Ventricular tachycardia, unsustained'],
            'Pacing': ['Ventricular pacing', 'Ventricular pacing.',
                       'Atrial pacing', 'A-V sequential pacemaker',
                       'Atrial-sensed ventricular-paced complexes',
                       'Atrial-ventricular dual-paced rhythm',
                       'Atrial-sensed ventricular-paced rhythm'],
        }
        color_map = {
            'Tachycardia': '#e377c2', 'Bradycardia': '#17becf',
            'LBBB': '#d62728', 'RBBB': '#ff7f0e',
            'AF/Aflu': '#9467bd', 'VT': '#8c564b',
            'Pacing': '#2ca02c',
        }
        # plot the background
        ax.scatter(sub['heart_rate_median'], sub['qrs_duration_median'],
                   s=15, alpha=0.15, c='gray', edgecolors='none')
        # plot the labelled groups
        for group, labels in label_groups.items():
            sel = merged[merged['label'].isin(labels)]
            if len(sel) > 0:
                ax.scatter(sel['heart_rate_median'], sel['qrs_duration_median'],
                           s=60, color=color_map[group], label=f'{group} (n={len(sel)})',
                           edgecolors='black', linewidth=0.5, alpha=0.85)
        ax.axvline(60, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.axvline(100, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(120, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax.set_xlabel('Median HR of top-50 ECGs (bpm)')
        ax.set_ylabel('Median QRS duration of top-50 ECGs (ms)')
        ax.set_title('Atoms by report label')
        ax.legend(loc='upper left', fontsize=7)
        ax.grid(True, alpha=0.3)
    else:
        ax.axis('off')
    
    plt.suptitle('Atom Localization in HR × QRS Space',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(fig_dir / 'fig6_hr_qrs_atom_map.png')
    plt.close()
    print(f"  saved: {fig_dir}/fig6_hr_qrs_atom_map.png")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print(f"All figures saved to: {fig_dir}/")
print(f"{'='*60}")
print("\nGenerated:")
for f in sorted(fig_dir.glob('*.png')):
    print(f"  {f.name}")
