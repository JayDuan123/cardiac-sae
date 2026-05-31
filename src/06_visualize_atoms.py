"""
Visualize SAE atoms: for selected atoms, plot top-N ECG signals that activate them most.

For each atom:
  - Read the top-N record paths (from top_records.npz)
  - Load the actual ECG waveforms (Lead II by default for clarity)
  - Overlay or grid plot to see what waveform pattern this atom fires for

Outputs:  figures/atom_XXXX_top.png  per selected atom
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
fig_dir = sae_dir / "figures" / "atom_ecgs"
fig_dir.mkdir(parents=True, exist_ok=True)

# How many records to show per atom (grid)
N_SHOW = 10
LEAD = "II"   # which lead to plot (II is most diagnostic)

# === Load ===
print("Loading metadata and top records ...")
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
top = np.load(sae_dir / "top_records.npz")
top_indices = top['indices']   # (D, TOP_N)
top_values = top['values']
atom_stats = pd.read_csv(sae_dir / "atom_stats.csv")

# === Pick atoms to visualize ===
# Strategy: mix of high-freq, mid-freq, and low-freq atoms
high_freq = atom_stats.nlargest(5, "freq_pct")["atom_id"].tolist()
mid_freq = atom_stats[(atom_stats["freq_pct"] > 0.5) & (atom_stats["freq_pct"] < 5)]\
    .nlargest(5, "mean_strength")["atom_id"].tolist()
low_freq = atom_stats[atom_stats["n_active"].between(20, 100)]\
    .sample(5, random_state=42)["atom_id"].tolist()

atoms_to_plot = sorted(set(high_freq + mid_freq + low_freq))
print(f"Atoms to visualize: {len(atoms_to_plot)}")
print(f"  high-freq: {high_freq}")
print(f"  mid-freq:  {mid_freq}")
print(f"  low-freq:  {low_freq}")

# === Plot ===
def load_lead(record_path, lead_name="II"):
    full = cfg.DATA_ROOT / record_path
    try:
        rec = wfdb.rdrecord(str(full))
        idx = rec.sig_name.index(lead_name)
        sig = rec.p_signal[:, idx]
        if np.isnan(sig).any():
            sig = np.nan_to_num(sig, nan=0.0)
        return sig, rec.fs
    except Exception as e:
        return None, None

for atom_id in atoms_to_plot:
    stats_row = atom_stats.iloc[atom_id]
    print(f"\nAtom {atom_id}: freq={stats_row['freq_pct']:.3f}%, "
          f"mean_strength={stats_row['mean_strength']:.2f}")

    rec_indices = top_indices[atom_id]
    rec_values = top_values[atom_id]

    # Filter valid (n_active < TOP_N may mean some are -1)
    valid = rec_indices >= 0
    rec_indices = rec_indices[valid][:N_SHOW]
    rec_values = rec_values[valid][:N_SHOW]

    if len(rec_indices) == 0:
        print(f"  No active records, skip.")
        continue

    fig, axes = plt.subplots(N_SHOW, 1, figsize=(12, 1.5 * N_SHOW),
                             sharex=True, sharey=False)
    if N_SHOW == 1:
        axes = [axes]

    for i, (rec_idx, val) in enumerate(zip(rec_indices, rec_values)):
        rec_path = meta.iloc[rec_idx]["path"]
        sig, fs = load_lead(rec_path, lead_name=LEAD)
        if sig is None:
            axes[i].text(0.5, 0.5, f"Failed to read {rec_path}",
                         transform=axes[i].transAxes, ha='center')
            continue
        t = np.arange(len(sig)) / fs
        axes[i].plot(t, sig, lw=0.7, color='steelblue')
        axes[i].set_ylabel(f"#{i+1}\nact={val:.2f}", fontsize=8)
        axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"Atom {atom_id}: top-{N_SHOW} ECGs (Lead {LEAD})  "
                 f"| freq={stats_row['freq_pct']:.2f}%, "
                 f"mean_str={stats_row['mean_strength']:.2f}",
                 fontsize=11)
    plt.tight_layout()
    out = fig_dir / f"atom_{atom_id:05d}_top{N_SHOW}.png"
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")

print(f"\nDone. Figures in {fig_dir}/")
