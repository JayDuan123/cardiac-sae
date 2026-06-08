"""
Stage 19: Per-atom case study panels.

For a list of atoms, generate one PNG each combining:
  - 12-lead ECG waveform from top-activating record
  - Numerical feature radar (atom's signature vs population)
  - Activation distribution histogram
  - Text evidence panel (Stage 12 labels + Stage 15 enrichment + Stage 17 Claude desc)

Usage:
  python 19_per_atom_case_study.py             # default: top atoms by category
  python 19_per_atom_case_study.py 1204 69 8   # specific atom IDs
"""
import sys, os, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from scipy.sparse import load_npz

try:
    import wfdb
except ImportError:
    os.system("pip install wfdb --quiet")
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
out_dir = sae_dir / "case_studies"
out_dir.mkdir(parents=True, exist_ok=True)

ECG_ROOT = "/workspace/data/mimic-iv-ecg-aws"
N_WAVEFORM_EXAMPLES = 3       # how many top ECG waveforms to show per atom

# ============================================================
# Load all data
# ============================================================
print("Loading data ...")
acts = load_npz(sae_dir / "activations_all.npz").tocsc()
N, D = acts.shape

# Meta
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
meta = meta.reset_index().rename(columns={'index': 'record_idx'})

# Machine measurements (numerical features)
mm = pd.read_csv(f"{ECG_ROOT}/machine_measurements.csv",
                 usecols=['study_id', 'rr_interval', 'p_onset', 'qrs_onset',
                          'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis'] +
                         [f'report_{i}' for i in range(18)],
                 low_memory=False)
for col in ['rr_interval', 'p_onset', 'qrs_onset', 'qrs_end', 't_end']:
    mm[col] = mm[col].where((mm[col] >= 0) & (mm[col] < 2000), np.nan)
for col in ['p_axis', 'qrs_axis', 't_axis']:
    mm[col] = mm[col].where((mm[col] >= -180) & (mm[col] <= 180), np.nan)
mm['heart_rate'] = 60000 / mm['rr_interval'].replace(0, np.nan)
mm['pr_interval'] = mm['qrs_onset'] - mm['p_onset']
mm['qrs_duration'] = mm['qrs_end'] - mm['qrs_onset']
mm['qt_interval'] = mm['t_end'] - mm['qrs_onset']
mm['study_id'] = mm['study_id'].astype('Int64')

meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')
feat_cols = ['heart_rate', 'pr_interval', 'qrs_duration', 'qt_interval',
             'p_axis', 'qrs_axis', 't_axis']
rep_cols = [f'report_{i}' for i in range(18)]
feat = meta[['record_idx', 'path', 'study_id']].merge(
    mm[['study_id'] + feat_cols + rep_cols], on='study_id', how='left'
).set_index('record_idx').reindex(range(N))

# Population medians for radar
POP_MEDIAN = {c: feat[c].median() for c in feat_cols}
print(f"  population medians: " +
      ", ".join(f"{c}={POP_MEDIAN[c]:.0f}" for c in feat_cols))

# Stage 12 labels
try:
    s12 = pd.read_csv(sae_dir / "atom_reports" / "atom_report_labels_v2.csv")
    s12 = s12.set_index('atom_id')
    print(f"  Stage 12 labels: {len(s12)} atoms")
except Exception:
    s12 = None
    print("  no Stage 12 labels")

# Stage 15 taxonomy
try:
    s15_tax = pd.read_csv(sae_dir / "taxonomy" / "atom_taxonomy.csv").set_index('atom_id')
    print(f"  Stage 15 taxonomy: {len(s15_tax)} atoms")
except Exception:
    s15_tax = None
    print("  no Stage 15 taxonomy")

# Stage 17 Claude
try:
    s17 = pd.read_csv(sae_dir / "claude_interp" / "atom_descriptions.csv")
    s17 = s17.set_index('atom_id')
    print(f"  Stage 17 Claude descriptions: {len(s17)} atoms")
except Exception:
    s17 = None
    print("  no Stage 17 descriptions")


# ============================================================
# Waveform loading
# ============================================================
def load_ecg(record_idx):
    """Load 12-lead ECG for a record_idx. Returns (5000, 12) array or None."""
    try:
        rel = meta.iloc[record_idx]['path']
        full = f"{ECG_ROOT}/{rel}"
        # wfdb expects no extension
        sig, fields = wfdb.rdsamp(full)
        return sig, fields
    except Exception as e:
        print(f"    warn: cannot load record {record_idx}: {e}")
        return None, None


# ============================================================
# Sub-panel: 12-lead ECG plot
# ============================================================
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot_ecg_12lead(ax, sig, fields, title=""):
    """Plot 12-lead ECG in a 4x3 clinical layout inside the given axis."""
    ax.axis('off')
    if sig is None:
        ax.text(0.5, 0.5, 'ECG unavailable', ha='center', va='center',
                transform=ax.transAxes, fontsize=10)
        return
    # Get lead names if available
    leads_in_file = fields.get('sig_name', LEAD_NAMES) if fields else LEAD_NAMES
    fs = fields.get('fs', 500) if fields else 500
    n = sig.shape[0]
    t = np.arange(n) / fs   # time in seconds

    # Inset 12 small axes in 4x3 grid inside this ax
    gs = ax.get_subplotspec().subgridspec(4, 3, hspace=0.4, wspace=0.15)
    for i, lead in enumerate(LEAD_NAMES):
        row, col = i % 4, i // 4
        a = ax.figure.add_subplot(gs[row, col])
        # Find this lead in the file
        if lead in leads_in_file:
            idx = list(leads_in_file).index(lead)
            y = sig[:, idx]
        elif i < sig.shape[1]:
            y = sig[:, i]
        else:
            a.axis('off'); continue
        a.plot(t, y, color='black', linewidth=0.5)
        a.set_xlim(0, min(2.5, n / fs))   # show first 2.5 sec
        a.set_facecolor('#fff5f5')        # ECG paper feel
        # gridlines like ECG paper
        a.grid(True, which='major', color='pink', linewidth=0.6, alpha=0.7)
        a.set_xticks([]); a.set_yticks([])
        a.text(0.02, 0.92, lead, transform=a.transAxes,
               fontsize=7, fontweight='bold', va='top')
        for spine in a.spines.values():
            spine.set_color('gray'); spine.set_linewidth(0.5)
    # Title above the grid
    ax.set_title(title, fontsize=9, fontweight='bold', pad=2)


# ============================================================
# Sub-panel: feature radar
# ============================================================
def plot_radar(ax, atom_features, pop_medians):
    """Polar radar showing atom's median vs population median for 7 features."""
    feats = ['heart_rate', 'pr_interval', 'qrs_duration', 'qt_interval',
             'qrs_axis', 'p_axis', 't_axis']
    labels = ['HR', 'PR', 'QRS', 'QT', 'QRSax', 'Pax', 'Tax']

    # Normalize each feature to [0, 1] using population P5..P95
    norm = []
    norm_pop = []
    for f in feats:
        v = atom_features.get(f, np.nan)
        if np.isnan(v):
            norm.append(0); norm_pop.append(0); continue
        p5 = feat[f].quantile(0.05)
        p95 = feat[f].quantile(0.95)
        if p95 - p5 < 1e-6:
            norm.append(0.5); norm_pop.append(0.5); continue
        norm.append(np.clip((v - p5) / (p95 - p5), 0, 1))
        norm_pop.append(np.clip((pop_medians[f] - p5) / (p95 - p5), 0, 1))

    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False).tolist()
    norm += norm[:1]; norm_pop += norm_pop[:1]; angles += angles[:1]

    ax.plot(angles, norm_pop, 'o-', color='gray', linewidth=1.5,
            alpha=0.6, label='Population median')
    ax.fill(angles, norm_pop, alpha=0.1, color='gray')
    ax.plot(angles, norm, 'o-', color='#d62728', linewidth=2,
            label='Atom top-50 median')
    ax.fill(angles, norm, alpha=0.25, color='#d62728')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(['25%', '50%', '75%'], fontsize=6, color='gray')
    ax.set_ylim(0, 1)
    ax.set_title('Numerical signature\n(top-50 vs population)',
                 fontsize=9, fontweight='bold', pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=7)


# ============================================================
# Sub-panel: activation distribution
# ============================================================
def plot_activation_dist(ax, atom_id):
    st, en = acts.indptr[atom_id], acts.indptr[atom_id + 1]
    vals = acts.data[st:en]
    if len(vals) == 0:
        ax.text(0.5, 0.5, 'No activations', ha='center', va='center',
                transform=ax.transAxes); return
    ax.hist(vals, bins=50, color='steelblue', edgecolor='black', alpha=0.85)
    top50_threshold = np.sort(vals)[-50] if len(vals) >= 50 else vals.min()
    ax.axvline(top50_threshold, color='red', linestyle='--', linewidth=2,
               label=f'top-50 threshold ({top50_threshold:.2f})')
    ax.set_xlabel('Activation value')
    ax.set_ylabel('# ECGs (firing only)')
    ax.set_title(f'Activation distribution\n(fires on {len(vals):,} ECGs = '
                 f'{100*len(vals)/N:.2f}%)',
                 fontsize=9, fontweight='bold')
    ax.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)


# ============================================================
# Sub-panel: text evidence
# ============================================================
def plot_text_evidence(ax, atom_id):
    ax.axis('off')
    lines = [f"ATOM {atom_id}", ""]

    if s15_tax is not None and atom_id in s15_tax.index:
        row = s15_tax.loc[atom_id]
        cat = row['category']
        n_conc = int(row.get('n_enriched_concepts', 0))
        lines.append(f"TAXONOMY (Stage 15)")
        lines.append(f"  Category:  {cat}")
        lines.append(f"  Enriched:  {n_conc} concept(s)")
        ecs = row.get('enriched_concepts', '')
        if isinstance(ecs, str) and ecs:
            for c in ecs.split('; ')[:5]:
                lines.append(f"     · {c[:55]}")
        lines.append("")

    if s12 is not None and atom_id in s12.index:
        row = s12.loc[atom_id]
        lines.append(f"REPORT LABEL (Stage 12)")
        lines.append(f"  '{str(row.get('label',''))[:55]}'")
        if 'lift' in row:
            lines.append(f"  lift = {row['lift']:.0f}")
        lines.append("")

    if s17 is not None and atom_id in s17.index:
        row = s17.loc[atom_id]
        lines.append(f"CLAUDE DESCRIPTION (Stage 17)")
        summ = str(row.get('summary', ''))
        # Wrap long summary
        for chunk in [summ[i:i+55] for i in range(0, len(summ), 55)][:4]:
            lines.append(f"  {chunk}")
        r = row.get('pearson_r', np.nan)
        if not np.isnan(r):
            lines.append(f"  held-out r = {r:.3f}")

    text = '\n'.join(lines)
    ax.text(0.03, 0.97, text, transform=ax.transAxes,
            va='top', ha='left', family='monospace', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='#f0f8ff',
                      edgecolor='steelblue', linewidth=1.5,
                      pad=0.6))


# ============================================================
# Atom top examples
# ============================================================
def get_top_ecgs(atom_id, n=N_WAVEFORM_EXAMPLES):
    st, en = acts.indptr[atom_id], acts.indptr[atom_id + 1]
    if en - st < n:
        return []
    rec = acts.indices[st:en]
    val = acts.data[st:en]
    order = np.argsort(val)[::-1][:n]
    return [(int(rec[i]), float(val[i])) for i in order]


def get_atom_feature_medians(atom_id, top_n=50):
    """Get median of each numerical feature across top-N activating ECGs."""
    st, en = acts.indptr[atom_id], acts.indptr[atom_id + 1]
    if en - st == 0:
        return {f: np.nan for f in feat_cols}
    rec = acts.indices[st:en]
    val = acts.data[st:en]
    top_idx = rec[np.argsort(val)[::-1][:top_n]]
    sub = feat.loc[top_idx]
    return {c: sub[c].median() for c in feat_cols}


# ============================================================
# Main panel function
# ============================================================
def make_panel(atom_id):
    print(f"\n[atom {atom_id}]")
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.3, 1.0],
                  hspace=0.3, wspace=0.25)

    # Top row: 3 ECG waveform panels (wider)
    top_ecgs = get_top_ecgs(atom_id, n=3)
    for i, (rec_idx, act_val) in enumerate(top_ecgs):
        ax = fig.add_subplot(gs[0, i])
        sig, fields = load_ecg(rec_idx)
        rep = str(feat.iloc[rec_idx].get('report_0', ''))[:50]
        hr = feat.iloc[rec_idx].get('heart_rate', np.nan)
        hr_str = f"{hr:.0f}" if not np.isnan(hr) else "?"
        title = (f"#{i+1} top-activating ECG (record {rec_idx})\n"
                 f"activation={act_val:.2f}, HR={hr_str} bpm\n"
                 f"report: {rep}")
        plot_ecg_12lead(ax, sig, fields, title=title)

    # Bottom row: 3 evidence panels
    # Bottom-left: activation distribution
    ax = fig.add_subplot(gs[1, 0])
    plot_activation_dist(ax, atom_id)

    # Bottom-middle: numerical radar
    ax = fig.add_subplot(gs[1, 1], polar=True)
    atom_features = get_atom_feature_medians(atom_id)
    plot_radar(ax, atom_features, POP_MEDIAN)

    # Bottom-right: text evidence
    ax = fig.add_subplot(gs[1, 2])
    plot_text_evidence(ax, atom_id)

    cat = s15_tax.loc[atom_id]['category'] if s15_tax is not None and atom_id in s15_tax.index else 'unknown'
    fig.suptitle(f'Atom {atom_id} — Case Study ({cat})',
                 fontsize=14, fontweight='bold', y=0.995)

    out_path = out_dir / f"atom_{atom_id:04d}_case_study.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  saved: {out_path.name}")


# ============================================================
# Select atoms to do
# ============================================================
if len(sys.argv) > 1:
    atom_ids = [int(a) for a in sys.argv[1:]]
    print(f"\nUser-specified atoms: {atom_ids}")
else:
    # Pick a representative mix:
    #   3 best Separable (Stage 15)
    #   3 best Claude-described Uninformative (Stage 17, highest r)
    atom_ids = []
    if s15_tax is not None:
        sep = s15_tax[s15_tax['category'] == 'Separable']
        atom_ids.extend(sep.head(3).index.tolist())
    if s17 is not None:
        top_claude = s17.sort_values('pearson_r', ascending=False).head(3)
        atom_ids.extend(top_claude.index.tolist())
    atom_ids = list(dict.fromkeys(atom_ids))   # dedup, keep order
    print(f"\nDefault selection: {atom_ids}")

# ============================================================
# Run
# ============================================================
for aid in atom_ids:
    try:
        make_panel(int(aid))
    except Exception as e:
        print(f"  ERROR on atom {aid}: {e}")
        import traceback; traceback.print_exc()

print(f"\n{'='*60}")
print(f"Case studies in: {out_dir}")
print(f"{'='*60}")
for f in sorted(out_dir.glob('*.png')):
    print(f"  {f.name}")
