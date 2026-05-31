"""
BatchTopK Sparse Autoencoder (Bussmann et al. 2024) on CSFM embeddings.

Key differences from standard TopK:
- TopK applied across (B * k) global, not per-sample
- After training, theta is estimated as mean of min-active-values for inference
- Inference uses JumpReLU(z > theta) instead of BatchTopK
"""
import sys, os, time, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg


# ============================================================
# Hyperparameters
# ============================================================
D_IN          = 768
EXPANSION     = 2
D_SAE         = D_IN * EXPANSION       # 12288
TOPK          = 32                     # avg sparsity: 32/12288 ≈ 0.26%
LR            = 3e-4                   # paper uses 3e-4
BATCH_SIZE    = 4096
EPOCHS        = 5
VAL_FRACTION  = 0.05
SEED          = 42

# AuxK (recycle dead latents)
AUX_K         = 256
AUX_COEFF     = 1.0 / 32
DEAD_STEPS    = 1000

# Theta estimation (last fraction of training)
THETA_ESTIMATE_FRACTION = 0.1   # use last 10% of steps to estimate theta


# ============================================================
# BatchTopK SAE
# ============================================================
class BatchTopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k, aux_k=256):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.aux_k = aux_k

        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_in, d_sae))

        # Learned threshold for inference (estimated from BatchTopK statistics)
        self.register_buffer("theta", torch.tensor(0.0))

        self._init_weights()
        self._normalize_decoder()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    def encode_pre(self, x):
        """Pre-activation: (B, d_in) -> (B, d_sae)"""
        return (x - self.pre_bias) @ self.W_enc.t() + self.b_enc

    def encode_batchtopk(self, x):
        """Training: BatchTopK across the whole batch."""
        z_pre = self.encode_pre(x)
        B = z_pre.shape[0]
        n_keep = B * self.k

        flat = z_pre.flatten()
        threshold = flat.topk(n_keep, sorted=False).values.min()
        # Apply: keep values >= threshold, otherwise 0
        z = torch.where(z_pre >= threshold, z_pre, torch.zeros_like(z_pre))
        z = F.relu(z)  # in case threshold is negative
        return z, z_pre, threshold

    def encode_jumprelu(self, x):
        """Inference: JumpReLU with learned theta."""
        z_pre = self.encode_pre(x)
        z = torch.where(z_pre > self.theta, z_pre, torch.zeros_like(z_pre))
        return z, z_pre

    def decode(self, z):
        return z @ self.W_dec.t() + self.pre_bias

    def forward(self, x, mode="batchtopk"):
        if mode == "batchtopk":
            z, z_pre, threshold = self.encode_batchtopk(x)
        else:
            z, z_pre = self.encode_jumprelu(x)
            threshold = self.theta
        x_hat = self.decode(z)
        return x_hat, z, z_pre, threshold

    def aux_loss(self, residual, z_pre, dead_mask):
        """AuxK: reconstruct residual using only dead features."""
        if dead_mask.sum() < self.aux_k:
            return torch.tensor(0.0, device=residual.device)
        z_dead = z_pre.clone()
        z_dead[:, ~dead_mask] = -float('inf')
        topk_vals, topk_idx = z_dead.topk(self.aux_k, dim=-1)
        z_aux = torch.zeros_like(z_dead)
        z_aux.scatter_(-1, topk_idx, topk_vals)
        z_aux = F.relu(z_aux)
        residual_hat = z_aux @ self.W_dec.t()
        return (residual_hat - residual).pow(2).mean()


# ============================================================
# Data
# ============================================================
def load_embeddings():
    variant = cfg.CSFM_VARIANT.lower()
    tag = cfg.RUN_TAG
    emb_path = cfg.EMBEDDING_DIR / f"csfm_{variant}_{tag}_embeddings.npy"
    done_path = cfg.EMBEDDING_DIR / f"csfm_{variant}_{tag}_done_idx.npy"

    print(f"Loading embeddings from {emb_path}")
    done = np.load(done_path)
    n_total = len(done)
    emb = np.memmap(emb_path, dtype=np.float16, mode='r',
                    shape=(n_total, cfg.CSFM_DIM))
    valid_idx = np.where(done)[0]
    print(f"  loading {len(valid_idx):,} rows as float32 ...")
    emb_f32 = np.asarray(emb[valid_idx], dtype=np.float32)
    print(f"  loaded: shape={emb_f32.shape}, size={emb_f32.nbytes/1e9:.2f} GB")

    # Normalize globally
    mean = emb_f32.mean(axis=0)
    std = emb_f32.std(axis=0).clip(min=1e-6)
    emb_n = (emb_f32 - mean) / std

    return torch.from_numpy(emb_n), mean, std


# ============================================================
# Train
# ============================================================
def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    variant = cfg.CSFM_VARIANT.lower()
    tag = cfg.RUN_TAG
    ckpt_dir = cfg.SAE_DIR / f"batchtopk_{variant}_{tag}_k{TOPK}_d{D_SAE}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train.log"
    log_file = open(log_path, 'w')

    def log(msg):
        print(msg); log_file.write(msg + "\n"); log_file.flush()

    log(f"=== BatchTopK SAE Training ===")
    log(f"d_in={D_IN}  d_sae={D_SAE}  topk={TOPK}")
    log(f"epochs={EPOCHS}  batch={BATCH_SIZE}  lr={LR}")
    log(f"ckpt={ckpt_dir}")

    # Load data
    data, mean, std = load_embeddings()
    N = len(data)
    n_val = int(N * VAL_FRACTION)
    n_train = N - n_val
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))
    train_data = data[perm[:n_train]].contiguous().cuda()
    val_data = data[perm[n_train:]].contiguous().cuda()
    log(f"train={n_train:,}  val={n_val:,}")
    log(f"GPU mem used by data: {(train_data.element_size()*train_data.numel() + val_data.element_size()*val_data.numel())/1e9:.2f} GB")

    np.save(ckpt_dir / "norm_mean.npy", mean)
    np.save(ckpt_dir / "norm_std.npy", std)

    # Model
    model = BatchTopKSAE(D_IN, D_SAE, TOPK, AUX_K).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"params: {n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99))

    steps_since_active = torch.zeros(D_SAE, dtype=torch.long, device='cuda')

    steps_per_epoch = n_train // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    theta_collect_start = int(total_steps * (1 - THETA_ESTIMATE_FRACTION))
    log(f"steps_per_epoch={steps_per_epoch}  total_steps={total_steps}")
    log(f"theta estimation starts at step {theta_collect_start}")
    log("")

    # Theta estimation: collect min positive activation per batch
    theta_samples = []

    global_step = 0
    t0 = time.time()

    for epoch in range(EPOCHS):
        epoch_perm = torch.randperm(n_train, device='cuda')

        recon_losses, l0_values = [], []
        thresholds_seen = []
        per_sample_l0_dist = []  # for histogram

        pbar = tqdm(range(steps_per_epoch), desc=f"Ep {epoch+1}/{EPOCHS}")
        for step in pbar:
            idx = epoch_perm[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
            x = train_data[idx]

            x_hat, z, z_pre, threshold = model(x, mode="batchtopk")
            recon_loss = (x_hat - x).pow(2).mean()

            active = (z > 0).any(dim=0)
            steps_since_active = torch.where(active, torch.zeros_like(steps_since_active),
                                             steps_since_active + 1)
            dead_mask = steps_since_active > DEAD_STEPS

            if dead_mask.any():
                residual = x - x_hat.detach()
                aux = model.aux_loss(residual, z_pre, dead_mask)
                total_loss = recon_loss + AUX_COEFF * aux
            else:
                total_loss = recon_loss

            opt.zero_grad()
            total_loss.backward()
            opt.step()
            model._normalize_decoder()

            # Theta collection (paper: avg of min-positive activations over batches)
            if global_step >= theta_collect_start:
                pos_z = z[z > 0]
                if pos_z.numel() > 0:
                    theta_samples.append(pos_z.min().item())

            recon_losses.append(recon_loss.item())
            l0 = (z > 0).float().sum(dim=-1).mean().item()
            l0_values.append(l0)
            thresholds_seen.append(threshold.item())

            global_step += 1
            if step % 50 == 0:
                pbar.set_postfix({
                    "recon": f"{recon_loss.item():.5f}",
                    "L0":    f"{l0:.1f}",
                    "thr":   f"{threshold.item():.3f}",
                    "dead":  int(dead_mask.sum().item()),
                })

        # ---- Validation with BatchTopK still (theta not finalized yet) ----
        model.eval()
        with torch.no_grad():
            val_chunks = val_data.split(BATCH_SIZE)
            val_recon_b = 0
            val_l0_b = 0
            sample_count = 0
            for vx in val_chunks:
                vh, vz, _, _ = model(vx, mode="batchtopk")
                val_recon_b += (vh - vx).pow(2).mean().item() * len(vx)
                val_l0_b += (vz > 0).float().sum(dim=-1).mean().item() * len(vx)
                sample_count += len(vx)
                per_sample_l0_dist.extend((vz > 0).sum(dim=-1).cpu().tolist())
            val_recon_b /= sample_count
            val_l0_b /= sample_count
            val_var = val_data.var().item()
            val_ev = 1.0 - val_recon_b / val_var
        model.train()

        elapsed = time.time() - t0
        log(f"[Ep {epoch+1}] train_recon={np.mean(recon_losses):.5f} "
            f"val_recon={val_recon_b:.5f} val_EV={val_ev:.4f} "
            f"L0_avg={np.mean(l0_values):.1f} L0_val={val_l0_b:.1f} "
            f"L0_p99={np.percentile(per_sample_l0_dist,99):.0f} "
            f"L0_min={int(np.min(per_sample_l0_dist))} "
            f"dead={int(dead_mask.sum())} elapsed={elapsed/60:.1f}min")

        ckpt = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "config": {"d_in": D_IN, "d_sae": D_SAE, "topk": TOPK,
                       "expansion": EXPANSION, "aux_k": AUX_K},
            "train_recon": float(np.mean(recon_losses)),
            "val_recon_batchtopk": float(val_recon_b),
            "val_ev": float(val_ev),
            "val_l0": float(val_l0_b),
            "n_dead": int(dead_mask.sum().item()),
        }
        torch.save(ckpt, ckpt_dir / f"epoch_{epoch+1}.pt")

    # ============================================================
    # Finalize theta (mean of min-active per batch over last 10% of steps)
    # ============================================================
    if len(theta_samples) > 0:
        theta_value = float(np.mean(theta_samples))
        log(f"\nTheta estimated from {len(theta_samples)} batches: {theta_value:.5f}")
        log(f"  range: [{np.min(theta_samples):.5f}, {np.max(theta_samples):.5f}]")
        log(f"  std:   {np.std(theta_samples):.5f}")
        model.theta.fill_(theta_value)
    else:
        log("WARNING: no theta samples collected!")

    # ============================================================
    # Validation with JumpReLU (using estimated theta)
    # ============================================================
    log("\n=== Final validation: JumpReLU with theta ===")
    model.eval()
    with torch.no_grad():
        val_chunks = val_data.split(BATCH_SIZE)
        val_recon_j = 0
        val_l0_j = 0
        l0_list = []
        sample_count = 0
        for vx in val_chunks:
            vh, vz, _, _ = model(vx, mode="jumprelu")
            val_recon_j += (vh - vx).pow(2).mean().item() * len(vx)
            val_l0_j += (vz > 0).float().sum(dim=-1).mean().item() * len(vx)
            sample_count += len(vx)
            l0_list.extend((vz > 0).sum(dim=-1).cpu().tolist())
        val_recon_j /= sample_count
        val_l0_j /= sample_count
        val_var = val_data.var().item()
        val_ev_j = 1.0 - val_recon_j / val_var

    log(f"JumpReLU val_recon={val_recon_j:.5f}  val_EV={val_ev_j:.4f}")
    log(f"L0 distribution: avg={np.mean(l0_list):.1f} median={np.median(l0_list):.0f} "
        f"p1={np.percentile(l0_list,1):.0f} p99={np.percentile(l0_list,99):.0f} "
        f"min={int(np.min(l0_list))} max={int(np.max(l0_list))}")

    # Save final
    final_ckpt = {
        "model_state": model.state_dict(),
        "config": {"d_in": D_IN, "d_sae": D_SAE, "topk": TOPK,
                   "expansion": EXPANSION, "aux_k": AUX_K},
        "theta": float(model.theta.item()),
        "val_recon_jumprelu": float(val_recon_j),
        "val_ev_jumprelu": float(val_ev_j),
        "val_l0_jumprelu": float(val_l0_j),
        "l0_distribution_stats": {
            "mean": float(np.mean(l0_list)),
            "median": float(np.median(l0_list)),
            "p1": float(np.percentile(l0_list, 1)),
            "p99": float(np.percentile(l0_list, 99)),
            "min": int(np.min(l0_list)),
            "max": int(np.max(l0_list)),
        },
    }
    torch.save(final_ckpt, ckpt_dir / "model.pt")
    log(f"\nDone. Final model: {ckpt_dir}/model.pt")
    log_file.close()


if __name__ == "__main__":
    train()
