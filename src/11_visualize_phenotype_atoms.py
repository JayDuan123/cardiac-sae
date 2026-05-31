"""
Visualize ECGs of specific phenotype-enriched atoms.
For each (phenotype, top_atom_id), plot the top-10 ECGs.
"""
import sys
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

CLINICAL_DIR = cfg.EMBEDDING_DIR.parent / "clinical"
out_dir = sae_dir / "atom_phenotype" / "phenotype_atom_ecgs"
out_dir.mkdir(parents=True, exist_ok=True)

# Specific atoms to visualize (from freq_lift analysis)
TARGET_ATOMS = [
    ('atrial_fibrillation', 341),    # 14.6x lift
    ('atrial_fibrillation', 823),    # 12.2x lift
    ('atrial_fibrillation', 757),    # 10.6x lift
    ('atrial_fibrillation', 1419),   # 8.2x lift
    ('sepsis', 165),                  # 2.9x lift
    ('heart_failure', 1252),          # 6.3x lift
    ('ckd', 1252),
]

# Load
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
top = np.load(sae_dir / "top_records.npz")
top_indices = top['indices']
top_values = top['values']
clin = pd.read_csv(CLINICAL_DIR / "record_with_clinical.csv")
flags = pd.read_csv(CLINICAL_DIR / "phenotype_flags.csv")
df = clin.merge(flags, on='record_idx')

N_SHOW = 10
LEAD = "II"


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


for phenotype, atom_id in TARGET_ATOMS:
    print(f"\nAtom {atom_id} (top for {phenotype}):")

    rec_indices = top_indices[atom_id]
    rec_values = top_values[atom_id]
    valid = rec_indices >= 0
    rec_indices = rec_indices[valid][:N_SHOW]
    rec_values = rec_values[valid][:N_SHOW]

    # Check phenotype status for each
    pheno_labels = [df.iloc[i][phenotype] for i in rec_indices]

    fig, axes = plt.subplots(N_SHOW, 1, figsize=(13, 1.5 * N_SHOW),
                             sharex=True, sharey=False)
    for i, (rec_idx, val, has_pheno) in enumerate(
            zip(rec_indices, rec_values, pheno_labels)):
        rec_path = meta.iloc[rec_idx]["path"]
        sig, fs = load_lead(rec_path, LEAD)
        if sig is None:
            axes[i].text(0.5, 0.5, f"Failed: {rec_path}",
                         transform=axes[i].transAxes, ha='center')
            continue
        t = np.arange(len(sig)) / fs
        color = 'crimson' if has_pheno else 'steelblue'
        axes[i].plot(t, sig, lw=0.7, color=color)
        label = f"+{phenotype[:3]}" if has_pheno else "no"
        axes[i].set_ylabel(f"#{i+1}\nact={val:.2f}\n[{label}]", fontsize=8)
        axes[i].grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Atom {atom_id} (top-{N_SHOW} ECGs, Lead {LEAD})  | "
        f"top atom for: {phenotype.replace('_', ' ')}",
        fontsize=11)
    plt.tight_layout()
    out = out_dir / f"atom_{atom_id:05d}_{phenotype}.png"
    plt.savefig(out, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out.name}")

print(f"\nDone. Figures in {out_dir}/")
