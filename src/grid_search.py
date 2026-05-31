"""
Grid search over (TopK, dictionary_size) for BatchTopK SAE.
20 models = 5 TopK x 4 dict_size.

Metrics (combines SAEBench + EEG SAE methodology):

  RECONSTRUCTION:
    val_ev, val_recon, actual_l0, l0_min, l0_max, converged

  DICTIONARY HEALTH:
    dead_pct, ultra_rare_pct, effective_atoms

  MONOSEMANTICITY:
    top20_purity (top-20 phenotype purity, our existing metric)

  FAITHFULNESS (NEW, EEG SAE Section 4.1):
    auroc_recon_{task}  -- probe on SAE-reconstructed embedding x_hat
    faithfulness_{task} = auroc_recon / auroc_dense

  SPARSE PROBING (NEW, SAEBench 3.2.3):
    top1_auroc_{task}   -- best single atom AUROC per task
    top5_auroc_{task}   -- best 5-atom linear probe AUROC

  FULL PROBE:
    auroc_{task}        -- full-dict logistic probe on SAE activations
    prs_{task} = auroc / auroc_dense

  ATOM TAXONOMY (NEW, EEG SAE 3.7):
    separable_pct  -- atoms with max single-atom AUROC > 0.65 (across phenotypes)
    entangled_pct  -- atoms that fire but no clean concept
    quality_score = separable_pct - dead_pct  (winner-selection metric)

Configuration:
  - Fixed LR=3e-4 (BatchTopK paper, AdamW handles per-param scaling)
  - GPU probes (PyTorch L-BFGS for logistic, closed-form for ridge)
  - Full samples, C grid, 7 tasks, resumable
"""
import sys, time, gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

# ============================================================
# Config
# ============================================================
TOPK_GRID = [16, 32, 64, 128, 256]
EXPANSION_GRID = [2, 4, 8, 16]            # K = 1536, 3072, 6144, 12288

PROBE_TASKS = ['af', 'hf', 'mi', 'dm', 'htn', 'sex', 'age']
PROBE_C_GRID = [0.1, 1.0, 10.0]
PROBE_LBFGS_ITERS = 100
SEPARABLE_THRESHOLD = 0.65   # single-atom AUROC > this -> separable
TOP_K_SPARSE_PROBE = [1, 5]   # SAEBench: k=1 and k=5

# For huge dictionaries, subsample probe training to fit in GPU memory.
PROBE_LARGE_K_THRESHOLD = 6144   # if d_sae > this, subsample training set
PROBE_LARGE_K_NTRAIN = 200_000   # subsample size for large K

LR = 3e-4
BATCH_SIZE = 4096
EPOCHS = 5
VAL_FRACTION = 0.05
SEED = 42
AUX_K = 256
AUX_COEFF = 1.0 / 32
DEAD_STEPS = 1000
THETA_FRAC = 0.1
CONVERGE_EPS = 0.003

OUT_DIR = cfg.EMBEDDING_DIR.parent / "grid_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUT_DIR / "grid_results.csv"
DEVICE = 'cuda'


# ============================================================
# BatchTopK SAE
# ============================================================
class BatchTopKSAE(nn.Module):
    def __init__(self, d_in, d_sae, k, aux_k=256):
        super().__init__()
        self.d_in, self.d_sae, self.k, self.aux_k = d_in, d_sae, k, aux_k
        self.pre_bias = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.zeros(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.zeros(d_in, d_sae))
        self.register_buffer("theta", torch.tensor(0.0))
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    def encode_pre(self, x):
        return (x - self.pre_bias) @ self.W_enc.t() + self.b_enc

    def encode_batchtopk(self, x):
        z_pre = self.encode_pre(x)
        B = z_pre.shape[0]
        flat = z_pre.flatten()
        threshold = flat.topk(B * self.k, sorted=False).values.min()
        z = torch.where(z_pre >= threshold, z_pre, torch.zeros_like(z_pre)).relu()
        return z, z_pre, threshold

    def encode_jumprelu(self, x):
        z_pre = self.encode_pre(x)
        z = torch.where(z_pre > self.theta, z_pre, torch.zeros_like(z_pre))
        return z, z_pre

    def decode(self, z):
        return z @ self.W_dec.t() + self.pre_bias

    def forward(self, x, mode="batchtopk"):
        if mode == "batchtopk":
            z, z_pre, thr = self.encode_batchtopk(x)
        else:
            z, z_pre = self.encode_jumprelu(x); thr = self.theta
        return self.decode(z), z, z_pre, thr

    def aux_loss(self, residual, z_pre, dead_mask):
        if dead_mask.sum() < self.aux_k:
            return torch.tensor(0.0, device=residual.device)
        z_dead = z_pre.clone()
        z_dead[:, ~dead_mask] = -float('inf')
        v, idx = z_dead.topk(self.aux_k, dim=-1)
        z_aux = torch.zeros_like(z_dead).scatter_(-1, idx, v).relu()
        return (z_aux @ self.W_dec.t() - residual).pow(2).mean()


# ============================================================
# Load data
# ============================================================
print("=" * 60)
print("Loading shared data ...")
print("=" * 60)

variant, tag = cfg.CSFM_VARIANT.lower(), cfg.RUN_TAG
emb_path = cfg.EMBEDDING_DIR / f"csfm_{variant}_{tag}_embeddings.npy"
done = np.load(cfg.EMBEDDING_DIR / f"csfm_{variant}_{tag}_done_idx.npy")
N = len(done)
emb_mmap = np.memmap(emb_path, dtype=np.float16, mode='r', shape=(N, cfg.CSFM_DIM))
emb_f32 = np.asarray(emb_mmap, dtype=np.float32)

norm_mean = emb_f32.mean(axis=0)
norm_std = emb_f32.std(axis=0).clip(min=1e-6)
emb_n = (emb_f32 - norm_mean) / norm_std
print(f"  embeddings: {emb_n.shape}")

torch.manual_seed(SEED); np.random.seed(SEED)
perm = np.random.permutation(N)
n_val = int(N * VAL_FRACTION)
sae_val_idx = perm[:n_val]
sae_train_idx = perm[n_val:]

data_gpu = torch.from_numpy(emb_n).to(DEVICE)
print(f"  SAE data on GPU (normalized): {data_gpu.element_size()*data_gpu.numel()/1e9:.2f} GB")

# Standardized dense for probes (different from SAE normalization)
dense_mean = emb_f32.mean(axis=0)
dense_std = emb_f32.std(axis=0).clip(min=1e-6)
dense_std_gpu = torch.from_numpy((emb_f32 - dense_mean) / dense_std).to(DEVICE)
print(f"  dense probe data on GPU: {dense_std_gpu.element_size()*dense_std_gpu.numel()/1e9:.2f} GB")

# Clinical
CLINICAL_DIR = cfg.EMBEDDING_DIR.parent / "clinical"
clin = pd.read_csv(CLINICAL_DIR / "record_with_clinical.csv")
flags = pd.read_csv(CLINICAL_DIR / "phenotype_flags.csv")
df_clin = clin.merge(flags, on='record_idx')

PHENO_COL = {
    'af': 'atrial_fibrillation', 'hf': 'heart_failure',
    'mi': 'mi___ischemic_heart', 'dm': 'diabetes_mellitus',
    'htn': 'hypertension_primary',
}
MONO_PHENOS = ['atrial_fibrillation', 'heart_failure',
               'mi___ischemic_heart', 'diabetes_mellitus']

# Atom taxonomy uses the 5 disease phenotypes (no sex/age, which is fine)
TAXONOMY_PHENOS = list(PHENO_COL.keys())

np.random.seed(123)
subjects = df_clin['subject_id'].unique()
np.random.shuffle(subjects)
n_te = int(0.1 * len(subjects)); n_va = int(0.1 * len(subjects))
test_subj = set(subjects[:n_te]); val_subj = set(subjects[n_te:n_te+n_va])
df_clin['split'] = 'train'
df_clin.loc[df_clin['subject_id'].isin(val_subj), 'split'] = 'val'
df_clin.loc[df_clin['subject_id'].isin(test_subj), 'split'] = 'test'
train_mask = (df_clin['split'] == 'train').values
val_mask = (df_clin['split'] == 'val').values
test_mask = (df_clin['split'] == 'test').values
has_dx = df_clin['n_diagnoses'].values > 0
print(f"  probe split: train={train_mask.sum():,} val={val_mask.sum():,} test={test_mask.sum():,}")


# ============================================================
# Labels + valid mask per task
# ============================================================
def task_labels_and_valid(task):
    if task == 'age':
        y = df_clin['age_at_ecg'].values.astype(np.float32)
        valid = ~np.isnan(y)
        return y, valid, 'regression'
    elif task == 'sex':
        y = (df_clin['gender'] == 'F').astype(np.float32).values
        valid = df_clin['gender'].isin(['F', 'M']).values
        return y, valid, 'binary'
    else:
        y = df_clin[PHENO_COL[task]].astype(np.float32).values
        valid = has_dx
        return y, valid, 'binary'


# ============================================================
# GPU probe primitives
# ============================================================
def gpu_auroc(scores, labels):
    """AUROC via rank formula."""
    n_pos = labels.sum()
    n_neg = (1 - labels).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, device=scores.device, dtype=torch.float32)
    sum_rank_pos = ranks[labels == 1].sum()
    return ((sum_rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def gpu_logistic_fit(X, y, C, iters=PROBE_LBFGS_ITERS):
    """L-BFGS logistic regression on GPU."""
    d = X.shape[1]
    w = torch.zeros(d, device=DEVICE, requires_grad=True)
    b = torch.zeros(1, device=DEVICE, requires_grad=True)
    l2 = 1.0 / C
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters,
                            line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        logits = X @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y) + l2 * w.pow(2).sum() / X.shape[0]
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def gpu_ridge_fit(X, y, C):
    """Closed-form ridge w/o materializing [X|1]. Centered formulation.
    Returns (d+1,) vector: [w (d,), b (1,)]."""
    d = X.shape[1]
    lam = 1.0 / C
    x_mean = X.mean(dim=0, keepdim=True)
    y_mean = y.mean()
    Xc = X - x_mean
    yc = y - y_mean
    XtX = Xc.t() @ Xc
    XtX.add_(torch.eye(d, device=DEVICE), alpha=lam)
    Xty = Xc.t() @ yc
    w = torch.linalg.solve(XtX, Xty)
    b = y_mean - (x_mean @ w).squeeze()
    return torch.cat([w, b.unsqueeze(0)])

def probe_dense_or_features(X_all_gpu, task, is_sparse_input=False):
    """
    Generic GPU probe. X_all_gpu: (N, d) GPU tensor. C grid -> best on val -> test.
    Returns metric (AUROC binary / R2 regression).
    """
    y_np, valid, ttype = task_labels_and_valid(task)
    tr = np.where(train_mask & valid)[0]
    # Subsample training set for huge dictionaries (OOM protection)
    if is_sparse_input and X_all_gpu.shape[1] > PROBE_LARGE_K_THRESHOLD:
        if len(tr) > PROBE_LARGE_K_NTRAIN:
            rng = np.random.RandomState(42)
            tr = rng.choice(tr, PROBE_LARGE_K_NTRAIN, replace=False)
    va = np.where(val_mask & valid)[0]
    te = np.where(test_mask & valid)[0]
    tr_t = torch.from_numpy(tr).to(DEVICE)
    va_t = torch.from_numpy(va).to(DEVICE)
    te_t = torch.from_numpy(te).to(DEVICE)
    Xtr, Xva, Xte = X_all_gpu[tr_t], X_all_gpu[va_t], X_all_gpu[te_t]
    ytr = torch.from_numpy(y_np[tr]).to(DEVICE)
    yva = torch.from_numpy(y_np[va]).to(DEVICE)
    yte = torch.from_numpy(y_np[te]).to(DEVICE)

    if ttype == 'binary':
        best_val, best_wb = -1, None
        for C in PROBE_C_GRID:
            w, b = gpu_logistic_fit(Xtr, ytr, C)
            val_auc = gpu_auroc(Xva @ w + b, yva)
            if val_auc > best_val:
                best_val, best_wb = val_auc, (w, b)
        w, b = best_wb
        return gpu_auroc(Xte @ w + b, yte)
    else:
        best_val, best_w = -1e9, None
        for C in PROBE_C_GRID:
            w = gpu_ridge_fit(Xtr, ytr, C)
            pred = Xva @ w[:-1] + w[-1]
            r2 = (1 - ((yva-pred)**2).sum() / ((yva-yva.mean())**2).sum()).item()
            if r2 > best_val:
                best_val, best_w = r2, w
        pred = Xte @ best_w[:-1] + best_w[-1]
        return (1 - ((yte-pred)**2).sum() / ((yte-yte.mean())**2).sum()).item()


# ============================================================
# Sparse probing: per-atom AUROC then top-k (SAEBench 3.2.3)
# ============================================================
def compute_per_atom_auroc(acts_gpu, task):
    """
    For each atom, compute AUROC for this task on TRAIN (used to rank atoms).
    Returns: per_atom_auroc (D,) on GPU.
    """
    y_np, valid, ttype = task_labels_and_valid(task)
    tr = np.where(train_mask & valid)[0]
    if ttype == 'regression':
        return None  # skip per-atom AUROC for regression
    y = torch.from_numpy(y_np[tr]).to(DEVICE)
    X = acts_gpu[torch.from_numpy(tr).to(DEVICE)]   # (n_tr, D)
    # Compute per-atom AUROC: rank-based, vectorized via mean difference proxy
    # For speed, use mean-difference heuristic: |mean_pos - mean_neg| / std (Gurnee et al. method used in SAEBench)
    pos_mask = (y == 1)
    neg_mask = (y == 0)
    if pos_mask.sum() < 5 or neg_mask.sum() < 5:
        return None
    # Per-atom: AUROC via Mann-Whitney U on GPU
    # Compute ranks per atom (this is the costly part); use a fast Wilcoxon proxy:
    # AUROC ~ 0.5 + (mean_diff / max_abs) approximation; but we'll do real ranks per task
    # For computational tractability, use mean activation comparison (proxy for AUROC):
    # |E[z|y=1] - E[z|y=0]| > 0 ranked, then convert to AUROC via Mann-Whitney
    n_pos = pos_mask.sum().item()
    n_neg = neg_mask.sum().item()
    aurocs = torch.zeros(X.shape[1], device=DEVICE)
    BATCH_ATOMS = 64   # smaller chunks: argsort().argsort() uses 4x chunk memory
    for s in range(0, X.shape[1], BATCH_ATOMS):
        e = min(s + BATCH_ATOMS, X.shape[1])
        Xc = X[:, s:e].contiguous()   # (n_tr, batch_atoms)
        # rank within each column (Mann-Whitney rank-AUROC)
        ranks = Xc.argsort(dim=0).argsort(dim=0).float() + 1
        sum_rank_pos = ranks[pos_mask].sum(dim=0)
        auc = (sum_rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        auc = torch.maximum(auc, 1 - auc)
        aurocs[s:e] = auc
        del Xc, ranks
    return aurocs


def sparse_probe_topk(acts_gpu, task, k_values=TOP_K_SPARSE_PROBE):
    """
    SAEBench k-sparse probing: select top-k atoms by mean-diff, train logistic on them.
    Returns: dict {k: test_auroc}
    Memory-safe: subsamples train set for large K, no boolean-index copies of huge tensors.
    """
    y_np, valid, ttype = task_labels_and_valid(task)
    if ttype != 'binary':
        return {k: float('nan') for k in k_values}

    tr = np.where(train_mask & valid)[0]
    # Subsample for large dictionaries
    if acts_gpu.shape[1] > PROBE_LARGE_K_THRESHOLD:
        if len(tr) > PROBE_LARGE_K_NTRAIN:
            rng = np.random.RandomState(42)
            tr = rng.choice(tr, PROBE_LARGE_K_NTRAIN, replace=False)
    te = np.where(test_mask & valid)[0]
    va = np.where(val_mask & valid)[0]
    tr_t = torch.from_numpy(tr).to(DEVICE)
    va_t = torch.from_numpy(va).to(DEVICE)
    te_t = torch.from_numpy(te).to(DEVICE)

    X_tr = acts_gpu[tr_t]
    y_tr = torch.from_numpy(y_np[tr]).to(DEVICE)
    pos = y_tr == 1
    n_pos = pos.sum().item()
    n_neg = (~pos).sum().item()
    if n_pos < 5 or n_neg < 5:
        del X_tr; torch.cuda.empty_cache()
        return {k: float('nan') for k in k_values}

    # Mean-diff per atom WITHOUT boolean-indexing copy: use masked sums
    # mean_pos = (X_tr * pos_mask).sum(0) / n_pos
    pos_f = pos.float().unsqueeze(1)   # (n_tr, 1)
    neg_f = 1.0 - pos_f
    # Chunk over atoms to avoid creating (n_tr, D) intermediate * (n_tr, 1)
    D = X_tr.shape[1]
    mean_diff = torch.zeros(D, device=DEVICE)
    CHUNK = 2048
    for s in range(0, D, CHUNK):
        e = min(s + CHUNK, D)
        Xc = X_tr[:, s:e]   # (n_tr, chunk)
        sum_pos = (Xc * pos_f).sum(dim=0) / n_pos
        sum_neg = (Xc * neg_f).sum(dim=0) / n_neg
        mean_diff[s:e] = (sum_pos - sum_neg).abs()
        del Xc, sum_pos, sum_neg
    del pos_f, neg_f
    torch.cuda.empty_cache()

    top_idx = mean_diff.topk(max(k_values)).indices

    out = {}
    for k in k_values:
        sel = top_idx[:k]
        if k == 1:
            # Single-atom AUROC: just use the activation column
            atom = sel[0].item()
            scores_tr = X_tr[:, atom]
            scores_te = acts_gpu[te_t][:, atom]
            yte = torch.from_numpy(y_np[te]).to(DEVICE)
            test_auc = gpu_auroc(scores_te, yte)
            out[k] = max(test_auc, 1 - test_auc)
        else:
            # Top-k linear probe (small features, safe)
            X_tr_k = X_tr[:, sel]
            X_va_k = acts_gpu[va_t][:, sel]
            X_te_k = acts_gpu[te_t][:, sel]
            yva = torch.from_numpy(y_np[va]).to(DEVICE)
            yte = torch.from_numpy(y_np[te]).to(DEVICE)
            best_val, best_wb = -1, None
            for C in PROBE_C_GRID:
                w, b = gpu_logistic_fit(X_tr_k, y_tr, C)
                va_auc = gpu_auroc(X_va_k @ w + b, yva)
                if va_auc > best_val:
                    best_val, best_wb = va_auc, (w, b)
            w, b = best_wb
            out[k] = gpu_auroc(X_te_k @ w + b, yte)
            del X_tr_k, X_va_k, X_te_k
    del X_tr, y_tr, pos, mean_diff, top_idx
    torch.cuda.empty_cache()
    return out

# ============================================================
# Atom taxonomy (EEG SAE Section 3.7)
# ============================================================
def atom_taxonomy(acts_gpu):
    """
    Classify each atom as Separable / Entangled / Dead.
      Dead: never fires
      Separable: max single-atom AUROC across phenotypes > SEPARABLE_THRESHOLD
      Entangled: fires but no phenotype reaches threshold
    Returns: dict with percentages.
    """
    D = acts_gpu.shape[1]
    fires = (acts_gpu > 0).any(dim=0).cpu().numpy()
    dead_mask = ~fires

    # For each phenotype, get per-atom AUROC (mean-diff proxy via rank)
    max_auroc = np.zeros(D)
    for task in TAXONOMY_PHENOS:
        auc = compute_per_atom_auroc(acts_gpu, task)
        if auc is not None:
            max_auroc = np.maximum(max_auroc, auc.cpu().numpy())

    separable_mask = (~dead_mask) & (max_auroc > SEPARABLE_THRESHOLD)
    entangled_mask = (~dead_mask) & (~separable_mask)

    sep_pct = 100.0 * separable_mask.mean()
    ent_pct = 100.0 * entangled_mask.mean()
    dead_pct = 100.0 * dead_mask.mean()
    return {
        'separable_pct': float(sep_pct),
        'entangled_pct': float(ent_pct),
        'dead_atom_pct': float(dead_pct),
        'quality_score': float(sep_pct - dead_pct),
    }


# ============================================================
# Probe wrappers
# ============================================================
def probe_dense(task):
    return probe_dense_or_features(dense_std_gpu, task, is_sparse_input=False)


def probe_full_sparse(acts_gpu, task):
    """Full-dictionary logistic probe on SAE activations."""
    return probe_dense_or_features(acts_gpu, task, is_sparse_input=True)


def probe_reconstruction(model, task):
    """
    Faithfulness: probe on the SAE-reconstructed dense embedding (EEG SAE 4.1).
    Reconstruct x_hat for ALL records, standardize the same way as dense_std_gpu, probe.
    """
    # Compute x_hat in batches
    model.eval()
    x_hat = torch.empty((N, cfg.CSFM_DIM), dtype=torch.float32, device=DEVICE)
    BATCH = 8192
    with torch.no_grad():
        for i in range(0, N, BATCH):
            xh, _, _, _ = model(data_gpu[i:i+BATCH], "jumprelu")
            x_hat[i:i+xh.shape[0]] = xh
    model.train()
    # x_hat is in SAE-normalized space; un-normalize back to original embedding space
    # data_gpu = (emb_f32 - norm_mean) / norm_std
    # So x_hat (which reconstructs data_gpu) is already in normalized space.
    # Standardize again with dense scaling? No -- use as-is since same scaling.
    # For probe stability, restandardize per-dim of x_hat:
    xh_mean = x_hat.mean(dim=0, keepdim=True)
    xh_std = x_hat.std(dim=0, keepdim=True).clamp(min=1e-6)
    x_hat_std = (x_hat - xh_mean) / xh_std
    result = probe_dense_or_features(x_hat_std, task, is_sparse_input=False)
    del x_hat, x_hat_std
    torch.cuda.empty_cache()
    return result


# Dense baselines (GPU) -- computed once
print("\nComputing dense probe baselines (GPU) ...")
dense_baseline = {}
for task in PROBE_TASKS:
    t0 = time.time()
    dense_baseline[task] = probe_dense(task)
    print(f"  dense {task}: {dense_baseline[task]:.4f} ({time.time()-t0:.1f}s)")


# ============================================================
# SAE training
# ============================================================
def _val_ev(model, val_t, mode="batchtopk"):
    model.eval()
    with torch.no_grad():
        vr, cnt = 0, 0
        for vx in val_t.split(BATCH_SIZE):
            vh, _, _, _ = model(vx, mode)
            vr += (vh - vx).pow(2).mean().item() * len(vx)
            cnt += len(vx)
        vr /= cnt
        ev = 1.0 - vr / val_t.var().item()
    model.train()
    return ev, vr


def train_one_sae(topk, d_sae):
    model = BatchTopKSAE(cfg.CSFM_DIM, d_sae, topk, AUX_K).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99))
    steps_since = torch.zeros(d_sae, dtype=torch.long, device=DEVICE)
    train_t = data_gpu[torch.from_numpy(sae_train_idx).to(DEVICE)]
    val_t = data_gpu[torch.from_numpy(sae_val_idx).to(DEVICE)]
    n_train = len(train_t)
    spe = n_train // BATCH_SIZE
    total_steps = spe * EPOCHS
    theta_start = int(total_steps * (1 - THETA_FRAC))
    theta_samples = []; gstep = 0; ev_hist = []

    for ep in range(EPOCHS):
        ep_perm = torch.randperm(n_train, device=DEVICE)
        for s in range(spe):
            idx = ep_perm[s*BATCH_SIZE:(s+1)*BATCH_SIZE]
            x = train_t[idx]
            xh, z, z_pre, thr = model(x, "batchtopk")
            loss = (xh - x).pow(2).mean()
            active = (z > 0).any(dim=0)
            steps_since = torch.where(active, torch.zeros_like(steps_since), steps_since+1)
            dead = steps_since > DEAD_STEPS
            if dead.any():
                loss = loss + AUX_COEFF * model.aux_loss(x - xh.detach(), z_pre, dead)
            opt.zero_grad(); loss.backward(); opt.step()
            model._normalize_decoder()
            if gstep >= theta_start:
                pz = z[z > 0]
                if pz.numel() > 0:
                    theta_samples.append(pz.min().item())
            gstep += 1
        ev_ep, _ = _val_ev(model, val_t, "batchtopk")
        ev_hist.append(ev_ep)

    if theta_samples:
        model.theta.fill_(float(np.mean(theta_samples)))
    converged = (len(ev_hist) >= 2 and
                 abs(ev_hist[-1] - ev_hist[-2]) < CONVERGE_EPS and
                 ev_hist[-1] >= ev_hist[-2] - 0.001)

    model.eval()
    with torch.no_grad():
        vr, vl0, cnt = 0, 0, 0; l0_all = []
        for vx in val_t.split(BATCH_SIZE):
            vh, vz, _, _ = model(vx, "jumprelu")
            vr += (vh - vx).pow(2).mean().item() * len(vx)
            l0b = (vz > 0).sum(dim=-1)
            vl0 += l0b.float().mean().item() * len(vx)
            l0_all.extend(l0b.cpu().tolist()); cnt += len(vx)
        vr /= cnt; vl0 /= cnt
        ev = 1.0 - vr / val_t.var().item()
    model.train()
    return model, {
        'val_ev': ev, 'val_recon': vr, 'actual_l0': vl0,
        'l0_min': int(np.min(l0_all)), 'l0_max': int(np.max(l0_all)),
        'l0_p99': float(np.percentile(l0_all, 99)),
        'theta': float(model.theta.item()),
        'converged': bool(converged),
        'ev_history': str([round(e, 4) for e in ev_hist]),
    }


def extract_acts_gpu_dense(model, d_sae):
    """SAE activations as GPU dense tensor (N, d_sae)."""
    model.eval()
    out = torch.empty((N, d_sae), dtype=torch.float32, device=DEVICE)
    BATCH = 8192
    with torch.no_grad():
        for i in range(0, N, BATCH):
            _, z, _, _ = model(data_gpu[i:i+BATCH], "jumprelu")
            out[i:i+z.shape[0]] = z
    model.train()
    return out


def dict_stats_gpu(acts_gpu):
    """Compute per-atom firing count, chunked over atoms to avoid OOM."""
    D = acts_gpu.shape[1]
    freq = torch.zeros(D, dtype=torch.long, device=DEVICE)
    CHUNK = 1024   # atoms per chunk
    for s in range(0, D, CHUNK):
        e = min(s + CHUNK, D)
        # (N, chunk) > 0 -> bool -> sum along N
        freq[s:e] = (acts_gpu[:, s:e] > 0).sum(dim=0)
    freq = freq.cpu().numpy()
    freq_pct = 100.0 * freq / N
    return {
        'dead_pct': float(100.0 * (freq == 0).mean()),
        'ultra_rare_pct': float(100.0 * ((freq_pct > 0) & (freq_pct < 0.01)).mean()),
        'effective_atoms': int((freq_pct >= 0.01).sum()),
        'common_atoms': int(((freq_pct >= 1) & (freq_pct < 10)).sum()),
    }


def monosemanticity_gpu(acts_gpu, top_n=20):
    """Top-20 phenotype purity (our existing metric)."""
    D = acts_gpu.shape[1]
    pheno_mat = torch.from_numpy(
        np.stack([df_clin[c].values.astype(np.float32) for c in MONO_PHENOS], axis=1)
    ).to(DEVICE)
    purities = []
    for atom in range(D):
        col = acts_gpu[:, atom]
        nz = (col > 0).sum().item()
        if nz < top_n:
            continue
        top_idx = torch.topk(col, top_n).indices
        frac = pheno_mat[top_idx].mean(dim=0)
        purities.append(frac.max().item())
    return float(np.mean(purities)) if purities else 0.0


# ============================================================
# Main grid loop
# ============================================================
if RESULTS_CSV.exists():
    done_df = pd.read_csv(RESULTS_CSV)
    done_keys = set(zip(done_df['topk'], done_df['d_sae']))
    print(f"\nResuming: {len(done_keys)} models done")
else:
    done_df = pd.DataFrame(); done_keys = set()

model_num = 0
for topk in TOPK_GRID:
    for exp in EXPANSION_GRID:
        model_num += 1
        d_sae = cfg.CSFM_DIM * exp
        if (topk, d_sae) in done_keys:
            print(f"\n[M{model_num}] TopK={topk} K={d_sae} -- SKIP"); continue

        print(f"\n{'='*60}\n[M{model_num}/20] TopK={topk} K={d_sae} ({exp}x)\n{'='*60}")
        t_total = time.time()

        # ---- Train SAE ----
        t = time.time()
        model, recon = train_one_sae(topk, d_sae)
        conv = "OK" if recon['converged'] else "*** NOT CONVERGED ***"
        print(f"  [train] EV={recon['val_ev']:.4f} L0={recon['actual_l0']:.1f} "
              f"range=[{recon['l0_min']},{recon['l0_max']}] [{conv}] ({time.time()-t:.0f}s)")
        print(f"  EV history: {recon['ev_history']}")

        # ---- Extract activations ----
        t = time.time()
        acts_gpu = extract_acts_gpu_dense(model, d_sae)
        mem = acts_gpu.element_size() * acts_gpu.numel() / 1e9
        print(f"  [acts] shape={acts_gpu.shape} mem={mem:.1f}GB ({time.time()-t:.0f}s)")

        # ---- Dict stats ----
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        t = time.time()
        ds = dict_stats_gpu(acts_gpu)
        torch.cuda.empty_cache()
        print(f"  [dict] dead={ds['dead_pct']:.1f}% effective={ds['effective_atoms']} "
              f"({time.time()-t:.0f}s)")

        # ---- Monosemanticity (top-20 purity) ----
        t = time.time()
        mono = monosemanticity_gpu(acts_gpu)
        print(f"  [monosem] top20_purity={mono:.4f} ({time.time()-t:.0f}s)")

        # ---- Atom taxonomy (NEW) ----
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        t = time.time()
        tax = atom_taxonomy(acts_gpu)
        torch.cuda.empty_cache()
        print(f"  [taxonomy] sep={tax['separable_pct']:.1f}% ent={tax['entangled_pct']:.1f}% "
              f"dead={tax['dead_atom_pct']:.1f}% quality={tax['quality_score']:.1f} "
              f"({time.time()-t:.0f}s)")

        # ---- Full sparse probe (existing) ----
        torch.cuda.empty_cache()
        t = time.time()
        probes = {task: probe_full_sparse(acts_gpu, task) for task in PROBE_TASKS}
        print(f"  [full probe] AF={probes['af']:.4f} HF={probes['hf']:.4f} "
              f"MI={probes['mi']:.4f} age_R2={probes['age']:.4f} ({time.time()-t:.0f}s)")

        # ---- Faithfulness probe (NEW): probe on x_hat ----
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        t = time.time()
        recon_probes = {task: probe_reconstruction(model, task) for task in PROBE_TASKS}
        faithfulness = {task: (recon_probes[task] / dense_baseline[task]
                               if dense_baseline[task] > 1e-6 else 0.0)
                        for task in PROBE_TASKS}
        print(f"  [faith] AF={recon_probes['af']:.4f}(f={faithfulness['af']:.3f}) "
              f"HF={recon_probes['hf']:.4f}(f={faithfulness['hf']:.3f}) "
              f"age_R2={recon_probes['age']:.4f} ({time.time()-t:.0f}s)")

        # ---- Sparse probing k=1, k=5 (NEW, SAEBench) ----
        torch.cuda.empty_cache()
        t = time.time()
        sp_results = {task: sparse_probe_topk(acts_gpu, task) for task in PROBE_TASKS}
        # Flatten for CSV: top1_auroc_af, top5_auroc_af, ...
        sp_flat = {}
        for task in PROBE_TASKS:
            for k in TOP_K_SPARSE_PROBE:
                sp_flat[f'top{k}_auroc_{task}'] = sp_results[task].get(k, float('nan'))
        nan_mean_1 = np.nanmean([sp_flat[f'top1_auroc_{t}'] for t in PROBE_TASKS])
        nan_mean_5 = np.nanmean([sp_flat[f'top5_auroc_{t}'] for t in PROBE_TASKS])
        print(f"  [sparse probe] mean top1={nan_mean_1:.4f} top5={nan_mean_5:.4f} "
              f"({time.time()-t:.0f}s)")

        # ---- PRS ----
        prs = {f'prs_{t}': probes[t]/dense_baseline[t]
               for t in PROBE_TASKS if dense_baseline[t] != 0}

        # ---- Record ----
        row = {
            'model': f'M{model_num}', 'topk': topk, 'd_sae': d_sae, 'expansion': exp,
            **recon,
            **ds,
            'top20_purity': mono,
            **tax,
            **{f'auroc_{t}': probes[t] for t in PROBE_TASKS},
            **prs,
            **{f'auroc_recon_{t}': recon_probes[t] for t in PROBE_TASKS},
            **{f'faithfulness_{t}': faithfulness[t] for t in PROBE_TASKS},
            **sp_flat,
            'time_s': time.time() - t_total,
        }
        done_df = pd.concat([done_df, pd.DataFrame([row])], ignore_index=True)
        done_df.to_csv(RESULTS_CSV, index=False)
        print(f"  -> saved. total elapsed {time.time()-t_total:.0f}s")

        del model, acts_gpu
        gc.collect(); torch.cuda.empty_cache()

# ---- Dense baselines also in the CSV for reference ----
dense_row = {'model': 'DENSE', 'topk': -1, 'd_sae': cfg.CSFM_DIM, 'expansion': -1}
for t in PROBE_TASKS:
    dense_row[f'auroc_{t}'] = dense_baseline[t]
done_df = pd.concat([done_df, pd.DataFrame([dense_row])], ignore_index=True)
done_df.to_csv(RESULTS_CSV, index=False)

print(f"\n{'='*60}\nGRID COMPLETE: {RESULTS_CSV}\n{'='*60}")
# Print summary table of key columns
key_cols = ['model', 'topk', 'd_sae', 'val_ev', 'actual_l0', 'dead_pct',
            'separable_pct', 'quality_score', 'auroc_af', 'prs_af',
            'faithfulness_af', 'top1_auroc_af', 'top5_auroc_af']
key_cols = [c for c in key_cols if c in done_df.columns]
print(done_df[key_cols].to_string())
