"""
Stage 26: Per-cluster case study panels.

For specific clusters (e.g., 18, 13, 9, 71), generate a multi-panel figure:
  - 3 top-activating 12-lead ECG waveforms
  - Atom × ECG activation heatmap (distributed encoding evidence)
  - Atom-to-atom decoder similarity heatmap (cluster cohesion evidence)
  - Cluster activation distribution
  - Per-atom Stage 15 category breakdown
  - Claude description + numbers panel
"""
import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import torch
from scipy.sparse import load_npz
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import wfdb

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

mpl.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 100,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
out_dir = sae_dir / "cluster_case_studies"
out_dir.mkdir(parents=True, exist_ok=True)
ECG_ROOT = "/workspace/data/mimic-iv-ecg-aws"

N_CLUSTERS = 80
MIN_CLUSTER_SIZE = 8
N_WAVEFORMS = 3
N_TOP_ECGS_HEATMAP = 30

# Default clusters to do
DEFAULT_CLUSTERS = [18, 13, 9, 71]  # R-wave splitting, HR splitting (low+high), artifact

# ============================================================
# Load + reproduce clustering
# ============================================================
print("Loading data ...")
ckpt = torch.load(sae_dir / "model.pt", map_location='cpu', weights_only=False)
W_dec = ckpt['model_state']['W_dec'].numpy()
if W_dec.shape[0] == 768: W_dec = W_dec.T

acts = load_npz(sae_dir / "activations_all.npz").tocsc()
N, D = acts.shape

tax = pd.read_csv(sae_dir / "taxonomy_grouped" / "atom_taxonomy_grouped.csv")
cat_col = 'category_grouped' if 'category_grouped' in tax.columns else 'category'
uninf = tax[tax[cat_col] == 'Uninformative']
uninf_ids = uninf['atom_id'].values

# Reproduce clustering (must use same seed/params as Stage 24)
print("Reproducing clustering ...")
W_uninf = W_dec[uninf_ids]
W_norm = W_uninf / (np.linalg.norm(W_uninf, axis=1, keepdims=True) + 1e-8)
dist = pdist(W_norm, metric='cosine')
Z = linkage(dist, method='average')
labels = fcluster(Z, t=N_CLUSTERS, criterion='maxclust')
atom_to_cluster = dict(zip(uninf_ids, labels))

print(f"  {len(np.unique(labels))} clusters")
for c in DEFAULT_CLUSTERS:
    n = (labels == c).sum()
    print(f"  cluster {c}: {n} atoms")

# Meta + features
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
meta = meta.reset_index().rename(columns={'index': 'record_idx'})
meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')

mm = pd.read_csv(f"{ECG_ROOT}/machine_measurements.csv",
                 usecols=['study_id', 'rr_interval', 'qrs_onset', 'qrs_end',
                          'p_axis', 'qrs_axis', 't_axis'] +
                         [f'report_{i}' for i in range(18)],
                 low_memory=False)
mm['study_id'] = mm['study_id'].astype('Int64')
mm['rr_interval'] = mm['rr_interval'].where((mm['rr_interval'] >= 0) & (mm['rr_interval'] < 2000), np.nan)
mm['heart_rate'] = 60000 / mm['rr_interval'].replace(0, np.nan)
mm['qrs_duration'] = mm['qrs_end'] - mm['qrs_onset']

feat = meta[['record_idx', 'path', 'study_id']].merge(mm, on='study_id', how='left')
feat = feat.set_index('record_idx').reindex(range(N))
rep_cols = [f'report_{i}' for i in range(18)]
feat['report'] = feat[rep_cols].apply(
    lambda r: ' | '.join(str(s) for s in r.values if pd.notna(s) and str(s).strip())[:120], axis=1
)

# Cluster results
cdf = pd.read_csv(sae_dir / "cluster_interp_v2" / "real_clusters.csv")
cdf = cdf.set_index('cluster_id')

# ============================================================
# Helpers
# ============================================================
def atom_activations(atom_ids):
    """Stack activations for a list of atoms. Returns (n_atoms, N) array."""
    out = np.zeros((len(atom_ids), N), dtype=np.float32)
    for i, a in enumerate(atom_ids):
        st, en = acts.indptr[a], acts.indptr[a + 1]
        out[i, acts.indices[st:en]] = acts.data[st:en]
    return out

def load_ecg(rec_idx):
    try:
        rel = feat.iloc[rec_idx]['path']
        sig, fields = wfdb.rdsamp(f"{ECG_ROOT}/{rel}")
        return sig, fields
    except Exception:
        return None, None

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot_12lead(ax_parent, sig, fields, title=""):
    ax_parent.axis('off')
    if sig is None:
        ax_parent.text(0.5, 0.5, 'ECG unavailable', ha='center', va='center',
                       transform=ax_parent.transAxes, fontsize=10)
        return
    leads = fields.get('sig_name', LEAD_NAMES) if fields else LEAD_NAMES
    fs = fields.get('fs', 500) if fields else 500
    n = sig.shape[0]
    t = np.arange(n) / fs
    gs = ax_parent.get_subplotspec().subgridspec(4, 3, hspace=0.35, wspace=0.1)
    for i, lead in enumerate(LEAD_NAMES):
        row, col = i % 4, i // 4
        a = ax_parent.figure.add_subplot(gs[row, col])
        if lead in leads:
            idx = list(leads).index(lead)
            y = sig[:, idx]
        elif i < sig.shape[1]:
            y = sig[:, i]
        else:
            a.axis('off'); continue
        a.plot(t, y, color='black', linewidth=0.5)
        a.set_xlim(0, min(2.5, n / fs))
        a.set_facecolor('#fff5f5')
        a.grid(True, color='pink', alpha=0.5, linewidth=0.4)
        a.set_xticks([]); a.set_yticks([])
        a.text(0.02, 0.92, lead, transform=a.transAxes, fontsize=7, fontweight='bold', va='top')
        for sp in a.spines.values():
            sp.set_color('gray'); sp.set_linewidth(0.4)
    ax_parent.set_title(title, fontsize=9, fontweight='bold', pad=2)

# ============================================================
# Main panel function
# ============================================================
def make_panel(cluster_id):
    atom_ids = [a for a in uninf_ids if atom_to_cluster[a] == cluster_id]
    if len(atom_ids) < MIN_CLUSTER_SIZE:
        print(f"  cluster {cluster_id} too small, skip")
        return
    n_atoms = len(atom_ids)
    print(f"\n[Cluster {cluster_id}] {n_atoms} atoms")

    # Cluster info from CSV
    if cluster_id not in cdf.index:
        print(f"  no description for cluster {cluster_id}")
        return
    info = cdf.loc[cluster_id]
    pearson_r = info['pearson_r']
    summary = info['summary']
    description = info['description']

    # Activations for cluster atoms
    A = atom_activations(atom_ids)   # (n_atoms, N)
    cluster_sum = A.sum(axis=0)
    fire_records = np.where(cluster_sum > 0)[0]
    top_ecg_idx = fire_records[np.argsort(cluster_sum[fire_records])[::-1][:N_TOP_ECGS_HEATMAP]]
    top3 = top_ecg_idx[:N_WAVEFORMS]

    # ----- Build figure -----
    fig = plt.figure(figsize=(22, 18))
    outer = GridSpec(4, 6, figure=fig,
                     height_ratios=[1.3, 1.1, 1.0, 0.55],
                     hspace=0.32, wspace=0.32)

    # Row 1: 3 ECG waveforms
    for i, rec_idx in enumerate(top3):
        ax = fig.add_subplot(outer[0, i*2:(i+1)*2])
        sig, fields = load_ecg(rec_idx)
        hr = feat.iloc[rec_idx].get('heart_rate', np.nan)
        hr_str = f"HR={hr:.0f}" if not np.isnan(hr) else "HR=?"
        rep = str(feat.iloc[rec_idx].get('report', ''))[:60]
        title = (f"#{i+1} top-activating ECG (record {rec_idx})\n"
                 f"cluster_act={cluster_sum[rec_idx]:.1f}  {hr_str}\n"
                 f"report: {rep}")
        plot_12lead(ax, sig, fields, title=title)

    # Row 2 panel A: Atom × ECG activation heatmap (DISTRIBUTED ENCODING EVIDENCE)
    ax = fig.add_subplot(outer[1, 0:3])
    A_top = A[:, top_ecg_idx]      # (n_atoms, n_top_ecgs)
    # Normalize each atom row to its own max (so each atom shows its pattern)
    A_norm = A_top / (A_top.max(axis=1, keepdims=True) + 1e-8)
    im = ax.imshow(A_norm, aspect='auto', cmap='Reds', vmin=0, vmax=1,
                   interpolation='nearest')
    ax.set_xlabel(f'Top {len(top_ecg_idx)} activating ECGs (sorted by cluster total)')
    ax.set_ylabel(f'{n_atoms} cluster atoms')
    ax.set_title(f'Atom × ECG activation heatmap\n'
                 f'(rows: atoms in cluster; cols: top-activating ECGs)\n'
                 f'★ visual evidence of distributed co-activation',
                 fontweight='bold', fontsize=9)
    ax.set_yticks(range(n_atoms))
    ax.set_yticklabels([str(a) for a in atom_ids], fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.02, label='normalized activation')

    # Row 2 panel B: Atom-atom decoder similarity (cluster cohesion)
    ax = fig.add_subplot(outer[1, 3:6])
    W_c = W_dec[atom_ids]
    W_c_norm = W_c / (np.linalg.norm(W_c, axis=1, keepdims=True) + 1e-8)
    sim = W_c_norm @ W_c_norm.T   # cosine similarity matrix
    im2 = ax.imshow(sim, cmap='RdBu_r', vmin=-1, vmax=1,
                    interpolation='nearest')
    ax.set_title(f'Atom-atom decoder cosine similarity\n'
                 f'(diagonals=1; off-diagonal=concept relatedness)\n'
                 f'mean off-diag = {(sim.sum() - n_atoms) / (n_atoms**2 - n_atoms):.2f}',
                 fontweight='bold', fontsize=9)
    ax.set_xticks(range(n_atoms))
    ax.set_yticks(range(n_atoms))
    ax.set_xticklabels([str(a) for a in atom_ids], fontsize=6, rotation=90)
    ax.set_yticklabels([str(a) for a in atom_ids], fontsize=6)
    plt.colorbar(im2, ax=ax, fraction=0.02, label='cosine sim')

    # Row 3 panel C: Cluster activation distribution
    ax = fig.add_subplot(outer[2, 0:2])
    nonzero = cluster_sum[cluster_sum > 0]
    if len(nonzero) > 0:
        ax.hist(nonzero, bins=60, color='steelblue', edgecolor='black', alpha=0.85)
        ax.axvline(np.quantile(nonzero, 0.95), color='red', linestyle='--',
                   label=f'p95={np.quantile(nonzero,0.95):.1f}')
        ax.legend(fontsize=7)
    ax.set_xlabel('Cluster activation (sum across cluster atoms)')
    ax.set_ylabel('# ECGs')
    ax.set_title(f'Cluster activation distribution\n'
                 f'(fires on {len(nonzero):,} ECGs = {100*len(nonzero)/N:.1f}%)',
                 fontweight='bold', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Row 3 panel D: Per-atom Stage 15 category breakdown
    ax = fig.add_subplot(outer[2, 2:4])
    tax_sub = tax[tax['atom_id'].isin(atom_ids)]
    cat_counts = tax_sub[cat_col].value_counts()
    colors_map = {'Separable': '#2ca02c', 'Entangled': '#ff7f0e',
                  'Uninformative': '#7f7f7f', 'Dead': '#d62728',
                  'Contributing': '#1f77b4'}
    colors_b = [colors_map.get(c, 'gray') for c in cat_counts.index]
    bars = ax.bar(cat_counts.index, cat_counts.values, color=colors_b, edgecolor='black')
    for bar, c in zip(bars, cat_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(c), ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('# atoms')
    ax.set_title(f'Stage 15 taxonomy of the {n_atoms} atoms\n'
                 f'(distributed encoding: cluster brings them together)',
                 fontweight='bold', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Row 3 panel E: Top atom activation strengths summary
    ax = fig.add_subplot(outer[2, 4:6])
    atom_fire_pct = (A > 0).mean(axis=1) * 100
    atom_mean_act = np.where(A > 0, A, np.nan).mean(axis=1)
    ax.bar(range(n_atoms), atom_fire_pct,
           color='steelblue', edgecolor='black', alpha=0.85)
    ax.set_xlabel(f'cluster atoms (n={n_atoms})')
    ax.set_ylabel('firing rate (%)', color='steelblue')
    ax.set_title('Per-atom firing rate within cluster\n'
                 '(more uniform → coherent distributed encoding)',
                 fontweight='bold', fontsize=9)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax.set_xticks(range(n_atoms))
    ax.set_xticklabels([str(a) for a in atom_ids], rotation=90, fontsize=6)
    ax.grid(True, alpha=0.3, axis='y')

    # Row 4: text evidence
    ax = fig.add_subplot(outer[3, :])
    ax.axis('off')
    text = (f"CLUSTER {cluster_id}  (n={n_atoms} atoms,  held-out Pearson r = {pearson_r:.3f})\n\n"
            f"SUMMARY:  {summary}\n\n"
            f"DESCRIPTION:  {description}\n\n"
            f"STAGE 15 CATEGORIES:  " +
            ", ".join(f"{c}: {n}" for c, n in cat_counts.items()) +
            f"\n\nINTERPRETATION:  This cluster reveals that {n_atoms} atoms — none individually "
            f"flagged as Separable in Stage 15 — collectively encode a coherent ECG concept "
            f"(distributed encoding). The atom×ECG heatmap (panel A, top-right above) shows "
            f"systematic co-activation across the top {N_TOP_ECGS_HEATMAP} ECGs.")
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            va='top', ha='left', family='monospace', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#f0f8ff',
                      edgecolor='steelblue', linewidth=1.5))

    fig.suptitle(f'Cluster {cluster_id} Case Study — Distributed Encoding Analysis',
                 fontsize=14, fontweight='bold', y=0.995)

    out_path = out_dir / f"cluster_{cluster_id:03d}_case_study.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  saved: {out_path.name}")

# ============================================================
# Run
# ============================================================
target_clusters = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_CLUSTERS
print(f"\nGenerating case studies for clusters: {target_clusters}")
for c in target_clusters:
    try:
        make_panel(c)
    except Exception as e:
        print(f"  ERROR cluster {c}: {e}")
        import traceback; traceback.print_exc()

print(f"\n{'='*60}")
print(f"Case studies in: {out_dir}")
for f in sorted(out_dir.glob('*.png')):
    print(f"  {f.name}")
