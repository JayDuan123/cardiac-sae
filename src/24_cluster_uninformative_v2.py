"""
Stage 24: Re-run cluster interpretation with activation-matched controls
+ permutation null test.

The original Stage 23 design suffered from validation leakage:
  - controls were "cluster activation == 0"
  - this implicitly selected ECGs the entire SAE ignored
  - Claude could predict the binary [active vs inactive] using HR or
    any global SAE-firing covariate, not the cluster-specific concept

Fix A: Activation-matched controls
  Controls are sampled to match the TOTAL SAE activation level of
  high-cluster-activation ECGs, but with LOW activation on THIS cluster.
  → Forces Claude to describe cluster-specific features, not global activity.

Fix B: Permutation null
  Run the same pipeline on 10 random "fake clusters" (random atom groups
  of similar size). Compare true-cluster r distribution vs fake-cluster r.
  If true >> fake, the methodology has signal. If similar, leakage dominates.
"""
import sys, os, json, time, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.sparse import load_npz
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import anthropic

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
out_dir = sae_dir / "cluster_interp_v2"
out_dir.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 80
MIN_CLUSTER_SIZE = 8
N_HIGH = 25         # high-activation ECGs (description)
N_CONTROL = 8       # matched controls (description)
N_HELDOUT = 24      # held-out test size
N_PERMUTATIONS = 10  # fake clusters for null
MODEL = "claude-sonnet-4-5-20250929"
HELDOUT_FRAC = 0.2

client = anthropic.Anthropic()

# ============================================================
# Load (same as Stage 23)
# ============================================================
print("=" * 70)
print("Stage 24: v2 cluster interpretation (activation-matched + permutation)")
print("=" * 70)

ckpt = torch.load(sae_dir / "model.pt", map_location='cpu', weights_only=False)
W_dec = ckpt['model_state']['W_dec'].numpy()
if W_dec.shape[0] == 768: W_dec = W_dec.T
assert W_dec.shape == (1536, 768)

acts = load_npz(sae_dir / "activations_all.npz").tocsc()
N, D = acts.shape

tax = pd.read_csv(sae_dir / "taxonomy_grouped" / "atom_taxonomy_grouped.csv")
cat_col = 'category_grouped' if 'category_grouped' in tax.columns else 'category'
uninf = tax[tax[cat_col] == 'Uninformative']
uninf_ids = uninf['atom_id'].values
print(f"  Uninformative atoms: {len(uninf_ids)}")

# Meta + features
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
meta = meta.reset_index().rename(columns={'index': 'record_idx'})
meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')

CLINICAL = cfg.EMBEDDING_DIR.parent / "clinical"
clin = pd.read_csv(CLINICAL / "record_with_clinical.csv").set_index('record_idx').reindex(range(N))

mm = pd.read_csv("/workspace/data/mimic-iv-ecg-aws/machine_measurements.csv",
                 usecols=['study_id', 'rr_interval', 'p_onset', 'qrs_onset', 'qrs_end',
                          't_end', 'qrs_axis'] + [f'report_{i}' for i in range(18)],
                 low_memory=False)
mm['study_id'] = mm['study_id'].astype('Int64')
for c in ['rr_interval', 'p_onset', 'qrs_onset', 'qrs_end', 't_end']:
    mm[c] = mm[c].where((mm[c] >= 0) & (mm[c] < 2000), np.nan)
mm['heart_rate'] = 60000 / mm['rr_interval'].replace(0, np.nan)
mm['pr_interval'] = mm['qrs_onset'] - mm['p_onset']
mm['qrs_duration'] = mm['qrs_end'] - mm['qrs_onset']
mm['qt_interval'] = mm['t_end'] - mm['qrs_onset']

feat = meta[['record_idx', 'study_id']].merge(mm, on='study_id', how='left')
feat = feat.set_index('record_idx').reindex(range(N))
rep_cols = [f'report_{i}' for i in range(18)]
feat['report'] = feat[rep_cols].apply(
    lambda r: ' | '.join(str(s) for s in r.values if pd.notna(s) and str(s).strip())[:250], axis=1
)

# ============================================================
# Compute TOTAL SAE activation per ECG (for matching)
# ============================================================
print("\nComputing total SAE activation per ECG ...")
# Sum across all atoms (gives a scalar per ECG: how "active" is this ECG overall)
total_act = np.zeros(N, dtype=np.float32)
acts_csr = acts.tocsr()
for i in range(N):
    if i % 100000 == 0: print(f"    {i}/{N}")
    st, en = acts_csr.indptr[i], acts_csr.indptr[i+1]
    total_act[i] = acts_csr.data[st:en].sum()
print(f"  total_act: median={np.median(total_act):.2f}, "
      f"p25={np.quantile(total_act,0.25):.2f}, p75={np.quantile(total_act,0.75):.2f}")

# ============================================================
# Cluster (same as before)
# ============================================================
print("\nClustering ...")
W_uninf = W_dec[uninf_ids]
W_norm = W_uninf / (np.linalg.norm(W_uninf, axis=1, keepdims=True) + 1e-8)
dist = pdist(W_norm, metric='cosine')
Z = linkage(dist, method='average')
labels = fcluster(Z, t=N_CLUSTERS, criterion='maxclust')

from collections import Counter
cluster_sizes = Counter(labels)
valid_clusters = [c for c, n in cluster_sizes.items() if n >= MIN_CLUSTER_SIZE]
print(f"  {len(valid_clusters)} clusters with >= {MIN_CLUSTER_SIZE} atoms")

atom_to_cluster = dict(zip(uninf_ids, labels))
def atoms_of(c): return [a for a in uninf_ids if atom_to_cluster[a] == c]

def cluster_act_vec(atom_list):
    """Sum activations of a list of atoms across all ECGs."""
    out = np.zeros(N, dtype=np.float32)
    for a in atom_list:
        st, en = acts.indptr[a], acts.indptr[a+1]
        out[acts.indices[st:en]] += acts.data[st:en]
    return out

# Subject split
subjects = clin['subject_id'].values
unique_subj = np.unique(subjects[~pd.isna(subjects)])
rng_g = np.random.RandomState(42); rng_g.shuffle(unique_subj)
n_test_subj = max(1, int(len(unique_subj) * HELDOUT_FRAC))
heldout_subj = set(unique_subj[:n_test_subj].tolist())
is_heldout = np.array([s in heldout_subj if pd.notna(s) else False for s in subjects])
print(f"  heldout: {is_heldout.sum():,} records / {n_test_subj} subjects")

# ============================================================
# Activation-matched control sampling
# ============================================================
def sample_matched_controls(cluster_act, train_mask, n_high, n_control, seed):
    """
    high: top n_high ECGs by cluster_act (within train)
    control: ECGs with low cluster_act but matched TOTAL SAE activation to high group
    """
    rng = np.random.RandomState(seed)
    # High = top N
    train_cact = np.where(train_mask, cluster_act, -1)
    high_idx = np.argsort(train_cact)[::-1][:n_high]

    # What's the total_act distribution of the high group?
    high_total = total_act[high_idx]
    p_lo, p_hi = np.quantile(high_total, [0.1, 0.9])

    # Control candidates: train ECGs with cluster_act LOW (< 10th percentile of high group's cluster_act)
    # AND total_act in the high group's range
    low_cluster_threshold = np.quantile(cluster_act[high_idx], 0.1) * 0.3
    eligible = (train_mask &
                (cluster_act < low_cluster_threshold) &
                (total_act >= p_lo) &
                (total_act <= p_hi))
    eligible_idx = np.where(eligible)[0]
    if len(eligible_idx) < n_control:
        return None, None  # cannot match
    control_idx = rng.choice(eligible_idx, size=n_control, replace=False)
    return high_idx, control_idx


# ============================================================
# Claude wrapper
# ============================================================
SYSTEM = """You are an expert cardiac electrophysiologist. You compare two groups of ECGs
that are matched on overall complexity (both have similar levels of SAE-detected patterns)
but differ in ONE specific feature. Identify that specific feature."""

def prompt_describe(high, ctrl):
    def fmt(ex):
        return (f"HR={ex['heart_rate']:.0f}" if not pd.isna(ex['heart_rate']) else "HR=?") + \
               (f", QRS={ex['qrs_duration']:.0f}ms" if not pd.isna(ex['qrs_duration']) else "") + \
               (f", QT={ex['qt_interval']:.0f}ms" if not pd.isna(ex['qt_interval']) else "") + \
               (f", QRSaxis={ex['qrs_axis']:.0f}°" if not pd.isna(ex['qrs_axis']) else "") + \
               f" | {ex['report']}"
    h = "\n".join(f"  [HIGH]    {fmt(ex)}" for ex in high)
    c = "\n".join(f"  [CONTROL] {fmt(ex)}" for ex in ctrl)
    return f"""HIGH-activation ECGs (n={len(high)}) and matched CONTROL ECGs (n={len(ctrl)}):

{h}

{c}

Both groups are matched on overall ECG complexity. Find the SPECIFIC feature
present in HIGH but absent in CONTROL.

Output STRICTLY:
{{
  "summary": "The detector activates on ...",
  "description": "<2-4 sentences with specific features>",
  "key_evidence": ["<f1>", "<f2>"]
}}"""

def prompt_predict(desc, ecgs):
    lines = []
    for i, ex in enumerate(ecgs):
        s = (f"HR={ex['heart_rate']:.0f}" if not pd.isna(ex['heart_rate']) else "HR=?") + \
            (f", QRS={ex['qrs_duration']:.0f}ms" if not pd.isna(ex['qrs_duration']) else "") + \
            (f", QT={ex['qt_interval']:.0f}ms" if not pd.isna(ex['qt_interval']) else "") + \
            f" | {ex['report']}"
        lines.append(f"  ECG_{i}: {s}")
    return f"""DESCRIPTION: {desc}

Rate each ECG 0-10 by match strength.

{chr(10).join(lines)}

Output STRICTLY:
[score_0, ..., score_{len(ecgs)-1}]"""

def call_claude(prompt, max_tok=1024):
    for attempt in range(3):
        try:
            r = client.messages.create(model=MODEL, max_tokens=max_tok, system=SYSTEM,
                                        messages=[{"role":"user","content":prompt}])
            return r.content[0].text
        except Exception as e:
            print(f"    err {attempt}: {e}"); time.sleep(2**attempt)
    return None

def parse_json(t):
    try:
        m = re.search(r'\{.*\}', t, re.DOTALL)
        return json.loads(m.group(0)) if m else None
    except: return None

def parse_scores(t, n):
    try:
        m = re.search(r'\[[\d,\s]+\]', t)
        if m:
            a = json.loads(m.group(0))
            return a if len(a) == n else None
    except: return None
    return None

# ============================================================
# Process one cluster (used for both real + fake)
# ============================================================
def process_cluster(atom_list, label, seed):
    cact = cluster_act_vec(atom_list)
    train_mask = ~is_heldout

    # Matched controls
    res = sample_matched_controls(cact, train_mask, N_HIGH, N_CONTROL, seed=seed)
    if res[0] is None:
        return None
    high_idx, ctrl_idx = res

    high_ex = [feat.iloc[i].to_dict() for i in high_idx]
    ctrl_ex = [feat.iloc[i].to_dict() for i in ctrl_idx]

    # Describe
    d_text = call_claude(prompt_describe(high_ex, ctrl_ex))
    if d_text is None: return None
    desc = parse_json(d_text)
    if desc is None: return None

    # Held-out test
    ho_high_idx = np.where(is_heldout & (cact > np.quantile(cact[cact>0], 0.7) if (cact>0).sum()>10 else (cact>0)))[0]
    ho_zero_idx = np.where(is_heldout & (cact < np.quantile(cact[cact>0], 0.1) if (cact>0).sum()>10 else (cact==0)))[0]
    
    # Match held-out controls on total_act too
    if len(ho_high_idx) < 6 or len(ho_zero_idx) < 6: return None
    n_each = N_HELDOUT // 2
    rng = np.random.RandomState(seed + 999)
    ho_high_pick = rng.choice(ho_high_idx, size=min(n_each, len(ho_high_idx)), replace=False)
    
    # Match ho_zero to ho_high's total_act distribution
    high_total_range = (np.quantile(total_act[ho_high_pick], 0.1),
                        np.quantile(total_act[ho_high_pick], 0.9))
    ho_zero_matched = ho_zero_idx[(total_act[ho_zero_idx] >= high_total_range[0]) &
                                    (total_act[ho_zero_idx] <= high_total_range[1])]
    if len(ho_zero_matched) < n_each:
        ho_zero_matched = ho_zero_idx  # fallback
    ho_zero_pick = rng.choice(ho_zero_matched, size=min(n_each, len(ho_zero_matched)), replace=False)
    
    ho_idx = np.concatenate([ho_high_pick, ho_zero_pick])
    ho_ecgs = [feat.iloc[i].to_dict() for i in ho_idx]
    true_acts = cact[ho_idx]

    p_text = call_claude(prompt_predict(desc.get('description',''), ho_ecgs))
    if p_text is None: return None
    scores = parse_scores(p_text, len(ho_ecgs))
    if scores is None: return None
    if np.std(scores) < 1e-6 or np.std(true_acts) < 1e-6:
        r = np.nan; p = np.nan
    else:
        r, p = pearsonr(scores, true_acts)

    return {
        'label': label,
        'n_atoms': len(atom_list),
        'summary': desc.get('summary',''),
        'description': desc.get('description',''),
        'pearson_r': r,
        'p_value': p,
        'n_heldout': len(ho_ecgs),
    }

# ============================================================
# Run on real clusters
# ============================================================
print("\n[Real clusters]")
real_results = []
for ci, c in enumerate(valid_clusters):
    print(f"  [{ci+1}/{len(valid_clusters)}] cluster {c} ...", end=' ', flush=True)
    res = process_cluster(atoms_of(c), label=f"cluster_{c}", seed=int(c))
    if res:
        res['cluster_id'] = int(c); res['is_real'] = True
        real_results.append(res)
        print(f"r={res['pearson_r']:.3f}")
    else:
        print("skipped")
    time.sleep(0.5)
    pd.DataFrame(real_results).to_csv(out_dir / "real_clusters.csv", index=False)

# ============================================================
# Run on PERMUTATION null (random atom groups)
# ============================================================
print(f"\n[Permutation null: {N_PERMUTATIONS} random clusters]")
fake_results = []
# Match fake cluster sizes to real cluster size distribution
real_sizes = [len(atoms_of(c)) for c in valid_clusters]
median_size = int(np.median(real_sizes))

rng_perm = np.random.RandomState(2024)
for pi in range(N_PERMUTATIONS):
    fake_atoms = rng_perm.choice(uninf_ids, size=median_size, replace=False).tolist()
    print(f"  [{pi+1}/{N_PERMUTATIONS}] fake cluster (n={median_size}) ...", end=' ', flush=True)
    res = process_cluster(fake_atoms, label=f"fake_{pi}", seed=pi + 10000)
    if res:
        res['cluster_id'] = -pi - 1; res['is_real'] = False
        fake_results.append(res)
        print(f"r={res['pearson_r']:.3f}")
    else:
        print("skipped")
    time.sleep(0.5)
    pd.DataFrame(fake_results).to_csv(out_dir / "fake_clusters.csv", index=False)

# ============================================================
# Final analysis
# ============================================================
real_df = pd.DataFrame(real_results)
fake_df = pd.DataFrame(fake_results)
all_df = pd.concat([real_df, fake_df], ignore_index=True)
all_df.to_csv(out_dir / "all_results.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nReal clusters:    n={len(real_df)}, median r={real_df['pearson_r'].median():.3f}, mean={real_df['pearson_r'].mean():.3f}")
print(f"Fake clusters:    n={len(fake_df)}, median r={fake_df['pearson_r'].median():.3f}, mean={fake_df['pearson_r'].mean():.3f}")

real_r = real_df['pearson_r'].dropna().values
fake_r = fake_df['pearson_r'].dropna().values
if len(real_r) > 5 and len(fake_r) > 3:
    from scipy.stats import mannwhitneyu
    U, p = mannwhitneyu(real_r, fake_r, alternative='greater')
    print(f"\nReal > Fake test (Mann-Whitney): U={U:.0f}, p={p:.3g}")
    if p < 0.05:
        print("  ✓ Real clusters have significantly higher r than random — cluster signal is real")
    else:
        print("  ⚠ Real clusters NOT significantly higher than random — leakage dominates")
    
    print(f"\nEffective signal: real median - fake median = {real_df['pearson_r'].median() - fake_df['pearson_r'].median():+.3f}")

print(f"\nResults in: {out_dir}")
