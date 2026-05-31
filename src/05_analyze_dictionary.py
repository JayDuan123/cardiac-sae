"""
SAE Dictionary Analysis - Stage A & C.

A) Atom activation statistics: frequency, mean strength distribution
C) Top-N representative records per atom

Outputs:
  - atom_stats.csv:     per-atom freq, mean_strength, max_strength
  - top_records.npz:    for each atom, top-N record indices and activations
  - figures/*.png:      histograms, distributions
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, csr_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

# === Config ===
SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
TOP_N = 20   # top-N records per atom

sae_dir = cfg.SAE_DIR / SAE_NAME
fig_dir = sae_dir / "figures"
fig_dir.mkdir(exist_ok=True)

# === Load ===
print("Loading activations ...")
t0 = time.time()
acts = load_npz(sae_dir / "activations_all.npz")  # CSR (800k, 12288)
N, D = acts.shape
print(f"  shape: {acts.shape}  nnz: {acts.nnz:,}  ({time.time()-t0:.1f}s)")

meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
print(f"  meta: {len(meta):,} rows")
assert len(meta) == N

# ============================================================
# Module A: Per-atom statistics
# ============================================================
print("\n=== Module A: Atom statistics ===")
t0 = time.time()

# Convert to CSC for efficient column operations
acts_csc = acts.tocsc()

# For each atom (column), compute:
# - n_active:  how many records activate this atom
# - mean_strength:  average activation value (over active records)
# - max_strength:   max activation value
freq = np.diff(acts_csc.indptr)   # nnz per column = activation count

# Mean / max strength per atom (only over active records)
mean_strength = np.zeros(D, dtype=np.float32)
max_strength = np.zeros(D, dtype=np.float32)
for atom_id in range(D):
    start, end = acts_csc.indptr[atom_id], acts_csc.indptr[atom_id+1]
    if end > start:
        vals = acts_csc.data[start:end]
        mean_strength[atom_id] = vals.mean()
        max_strength[atom_id] = vals.max()

freq_pct = 100.0 * freq / N
atom_stats = pd.DataFrame({
    "atom_id": np.arange(D),
    "n_active": freq,
    "freq_pct": freq_pct,
    "mean_strength": mean_strength,
    "max_strength": max_strength,
})
atom_stats.to_csv(sae_dir / "atom_stats.csv", index=False)
print(f"  Saved atom_stats.csv  ({time.time()-t0:.1f}s)")

# Summary
print(f"\n  Activation frequency distribution:")
print(f"    median: {np.median(freq_pct):.3f}%")
print(f"    p1:     {np.percentile(freq_pct, 1):.4f}%")
print(f"    p99:    {np.percentile(freq_pct, 99):.2f}%")
print(f"    min:    {freq_pct.min():.4f}%  (rarest atom)")
print(f"    max:    {freq_pct.max():.2f}%  (most common atom)")
print(f"\n  Atoms by frequency tier:")
print(f"    Ultra-rare (<0.01%):  {((freq_pct < 0.01) & (freq_pct > 0)).sum()}")
print(f"    Dead (0%):            {(freq == 0).sum()}")
print(f"    Rare (0.01-1%):       {((freq_pct >= 0.01) & (freq_pct < 1)).sum()}")
print(f"    Common (1-10%):       {((freq_pct >= 1) & (freq_pct < 10)).sum()}")
print(f"    Very common (>10%):   {(freq_pct >= 10).sum()}")

# ---- Plot: frequency histogram (log scale) ----
plt.figure(figsize=(10, 5))
plt.hist(freq_pct[freq_pct > 0], bins=100, log=True, edgecolor='k', alpha=0.7)
plt.axvline(np.median(freq_pct[freq_pct > 0]), color='r', linestyle='--',
            label=f'median={np.median(freq_pct[freq_pct > 0]):.3f}%')
plt.xlabel('Activation frequency (%)')
plt.ylabel('Number of atoms (log)')
plt.title(f'Atom activation frequency distribution ({D} atoms)')
plt.legend()
plt.tight_layout()
plt.savefig(fig_dir / "atom_freq_dist.png", dpi=120)
plt.close()

# ---- Plot: strength vs frequency scatter ----
plt.figure(figsize=(10, 5))
plt.scatter(freq_pct[freq > 0], mean_strength[freq > 0],
            s=2, alpha=0.4, c='steelblue')
plt.xscale('log')
plt.xlabel('Activation frequency (%, log)')
plt.ylabel('Mean activation strength')
plt.title('Strength vs frequency: each dot = one atom')
plt.tight_layout()
plt.savefig(fig_dir / "atom_strength_vs_freq.png", dpi=120)
plt.close()

# ============================================================
# Module C: Top-N records per atom
# ============================================================
print(f"\n=== Module C: Top-{TOP_N} records per atom ===")
t0 = time.time()

# For each atom, find indices of records with highest activation.
# Using CSC: for each column, look at top-N values
top_record_idx = np.full((D, TOP_N), -1, dtype=np.int32)
top_record_val = np.zeros((D, TOP_N), dtype=np.float32)

# Vectorized would be faster but more memory; loop is OK at D=12288
for atom_id in range(D):
    start, end = acts_csc.indptr[atom_id], acts_csc.indptr[atom_id+1]
    if end == start:
        continue
    rec_indices = acts_csc.indices[start:end]
    vals = acts_csc.data[start:end]
    n_take = min(TOP_N, len(vals))
    # argpartition for top-n (faster than sort)
    top_local = np.argpartition(vals, -n_take)[-n_take:]
    # Sort just these
    top_local = top_local[np.argsort(-vals[top_local])]
    top_record_idx[atom_id, :n_take] = rec_indices[top_local]
    top_record_val[atom_id, :n_take] = vals[top_local]

np.savez_compressed(
    sae_dir / "top_records.npz",
    indices=top_record_idx,
    values=top_record_val,
)
print(f"  Saved top_records.npz  ({time.time()-t0:.1f}s)")
print(f"  shape: indices={top_record_idx.shape}, values={top_record_val.shape}")

# ============================================================
# Module summary
# ============================================================
print("\n=== Summary ===")
print(f"Output files in {sae_dir}:")
print(f"  atom_stats.csv         {(sae_dir / 'atom_stats.csv').stat().st_size / 1e6:.1f} MB")
print(f"  top_records.npz        {(sae_dir / 'top_records.npz').stat().st_size / 1e6:.1f} MB")
print(f"  figures/")
for f in sorted(fig_dir.iterdir()):
    print(f"    {f.name}  ({f.stat().st_size / 1e3:.0f} KB)")

# ============================================================
# Print top-10 most/least common atoms (text summary)
# ============================================================
print("\n=== Top 10 most common atoms ===")
top_common = atom_stats.sort_values("freq_pct", ascending=False).head(10)
print(top_common.to_string(index=False))

print("\n=== Top 10 least common atoms (with at least 5 activations) ===")
rare = atom_stats[atom_stats["n_active"] >= 5].sort_values("freq_pct").head(10)
print(rare.to_string(index=False))

print("\nDone.")
