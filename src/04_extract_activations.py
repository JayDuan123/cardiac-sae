"""
Extract sparse SAE activations for all 800k records.
Stores as scipy.sparse CSR matrix (efficient for ~32 non-zeros per row).
"""
import sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

# ============ Load SAE ============
SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
ckpt_path = sae_dir / "model.pt"
mean_path = sae_dir / "norm_mean.npy"
std_path = sae_dir / "norm_std.npy"

print(f"Loading SAE from {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sae_config = ckpt['config']
theta = ckpt['theta']
print(f"  d_sae={sae_config['d_sae']}, theta={theta:.5f}")

# Rebuild model (same architecture as in 03)
from importlib import import_module
sys.path.insert(0, str(Path(__file__).parent))
spec = __import__('importlib.util', fromlist=['']).spec_from_file_location(
    "m03", Path(__file__).parent / "03_train_sae.py")
m03 = __import__('importlib.util', fromlist=['']).module_from_spec(spec)
spec.loader.exec_module(m03)

model = m03.BatchTopKSAE(
    d_in=sae_config['d_in'],
    d_sae=sae_config['d_sae'],
    k=sae_config['topk'],
    aux_k=sae_config['aux_k'],
).cuda().eval()
model.load_state_dict(ckpt['model_state'])

norm_mean = torch.from_numpy(np.load(mean_path)).cuda()
norm_std = torch.from_numpy(np.load(std_path)).cuda()

# ============ Load embeddings ============
variant = cfg.CSFM_VARIANT.lower()
tag = cfg.RUN_TAG
emb_path = cfg.EMBEDDING_DIR / f"csfm_{variant}_{tag}_embeddings.npy"
done_path = cfg.EMBEDDING_DIR / f"csfm_{variant}_{tag}_done_idx.npy"
done = np.load(done_path)
n_total = len(done)
print(f"Loading embeddings: {n_total:,} records")
emb_mmap = np.memmap(emb_path, dtype=np.float16, mode='r',
                     shape=(n_total, cfg.CSFM_DIM))

# ============ Extract activations ============
BATCH = 8192
indices_list = []
indptr = [0]
values_list = []

print(f"Extracting sparse activations (JumpReLU mode, theta={theta:.5f}) ...")
t0 = time.time()
with torch.no_grad():
    for i in tqdm(range(0, n_total, BATCH)):
        batch = np.asarray(emb_mmap[i:i+BATCH], dtype=np.float32)
        x = torch.from_numpy(batch).cuda()
        x = (x - norm_mean) / norm_std  # normalize same as training

        _, z, _, _ = model(x, mode="jumprelu")  # (B, d_sae)
        z = z.cpu().numpy()  # float32

        # Convert to sparse: per-row nonzero indices and values
        for row in z:
            nz = np.nonzero(row)[0]
            indices_list.append(nz.astype(np.int32))
            values_list.append(row[nz].astype(np.float32))
            indptr.append(indptr[-1] + len(nz))

# ============ Build CSR matrix ============
indices = np.concatenate(indices_list)
values = np.concatenate(values_list)
indptr = np.array(indptr, dtype=np.int64)

print(f"\nBuilding CSR matrix ...")
acts = csr_matrix(
    (values, indices, indptr),
    shape=(n_total, sae_config['d_sae']),
    dtype=np.float32,
)
print(f"  shape: {acts.shape}")
print(f"  nnz:   {acts.nnz:,}")
print(f"  avg per row: {acts.nnz / n_total:.1f}")
print(f"  density: {100 * acts.nnz / (acts.shape[0]*acts.shape[1]):.3f}%")

# Save
out_path = sae_dir / "activations_all.npz"
save_npz(out_path, acts)
size_gb = out_path.stat().st_size / 1e9
print(f"\nSaved to {out_path}")
print(f"  size: {size_gb:.2f} GB")
print(f"  elapsed: {(time.time()-t0)/60:.1f} min")
