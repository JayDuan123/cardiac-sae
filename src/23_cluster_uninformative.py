"""
Stage 23: Cluster-level interpretation of Uninformative atoms.

Single-atom Claude interpretation (Stage 17) achieved median Pearson r = 0.30,
limited by:
  (a) sparse per-atom signal (each atom's top-N is small)
  (b) report+numerical feature space is narrow for fine ECG morphology
  (c) Uninformative atoms are the residual after Stage 12/13/15 filtered
      strong signals — what's left is hard to describe

Cluster-level analysis addresses (a) by aggregating across decoder-similar
atoms, raising signal-to-noise for Claude pattern detection.

Pipeline:
  1. Extract decoder vectors for Uninformative atoms (1353 × 768)
  2. Hierarchical clustering (cosine, average linkage) into 60-80 clusters
  3. Validate clusters: intra-cluster activation correlation, top-ECG overlap
  4. For each cluster:
     - Find "high-consistency" ECGs (activated by many cluster members)
     - Find "zero" ECGs (activated by no cluster members)
     - Subject-level train/test split
     - Claude generates description from contrast
     - Claude predicts cluster mean activation on held-out → Pearson r
  5. Output: cluster_descriptions.csv + diagnostic figures
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
from sklearn.metrics.pairwise import cosine_similarity
import anthropic

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

# ============================================================
# Config
# ============================================================
SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
out_dir = sae_dir / "cluster_interp"
out_dir.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 80              # target cluster count
MIN_CLUSTER_SIZE = 5         # skip very small clusters
N_HIGH_CONSISTENCY = 30      # ECGs per cluster for description
N_ZERO_CONTROL = 8           # zero-activation controls
N_HELDOUT_TEST = 25
HELDOUT_SUBJ_FRAC = 0.2
MODEL = "claude-sonnet-4-5-20250929"
MAX_RETRIES = 3
SLEEP = 1.0

client = anthropic.Anthropic()

# ============================================================
# Load
# ============================================================
print("=" * 70)
print("Stage 23: Cluster-level interpretation")
print("=" * 70)

# SAE decoder weights (your checkpoint: model_state['W_dec'] shape (768, 1536))
sae_path = sae_dir / "model.pt"
ckpt = torch.load(sae_path, map_location='cpu', weights_only=False)
sd = ckpt['model_state']
W_dec = sd['W_dec'].numpy()   # shape (768, 1536)
# Make it (n_atoms, d_model) = (1536, 768): each row = one atom's concept vector
if W_dec.shape[0] == 768 and W_dec.shape[1] == 1536:
    W_dec = W_dec.T
print(f"  decoder shape: {W_dec.shape} (n_atoms, d_in)")
assert W_dec.shape == (1536, 768), f"unexpected shape {W_dec.shape}"
print(f"  SAE config: {ckpt.get('config')}")

# Activations and taxonomy
acts = load_npz(sae_dir / "activations_all.npz").tocsc()
N, D = acts.shape

# Use the latest taxonomy
TAX_FILES = [
    sae_dir / "taxonomy_grouped" / "atom_taxonomy_grouped.csv",
    sae_dir / "taxonomy_distributed" / "atom_taxonomy_v3.csv",
    sae_dir / "taxonomy" / "atom_taxonomy.csv",
]
tax = None
for p in TAX_FILES:
    if p.exists():
        tax = pd.read_csv(p); print(f"  using taxonomy: {p.name}"); break
cat_col = 'category_grouped' if 'category_grouped' in tax.columns else 'category'
uninf = tax[tax[cat_col] == 'Uninformative']
print(f"  Uninformative atoms: {len(uninf)}")

# Meta + features (for Claude prompts)
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
meta = meta.reset_index().rename(columns={'index': 'record_idx'})
meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')

CLINICAL_DIR = cfg.EMBEDDING_DIR.parent / "clinical"
clin = pd.read_csv(CLINICAL_DIR / "record_with_clinical.csv").set_index('record_idx').reindex(range(N))

mm = pd.read_csv("/workspace/data/mimic-iv-ecg-aws/machine_measurements.csv",
                 usecols=['study_id', 'rr_interval', 'p_onset', 'qrs_onset', 'qrs_end',
                          't_end', 'p_axis', 'qrs_axis', 't_axis'] +
                         [f'report_{i}' for i in range(18)],
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
    lambda r: ' | '.join(str(s) for s in r.values if pd.notna(s) and str(s).strip())[:300], axis=1
)

# ============================================================
# STEP 1: Hierarchical clustering on decoder vectors
# ============================================================
print(f"\n[Step 1] Clustering {len(uninf)} Uninformative atoms ...")
uninf_ids = uninf['atom_id'].values
W_uninf = W_dec[uninf_ids]   # (n_uninf, 768)
# Normalize for cosine
W_norm = W_uninf / (np.linalg.norm(W_uninf, axis=1, keepdims=True) + 1e-8)

# Cosine distance
print("  computing pairwise cosine distances ...")
dist = pdist(W_norm, metric='cosine')
print("  hierarchical linkage ...")
Z = linkage(dist, method='average')
labels = fcluster(Z, t=N_CLUSTERS, criterion='maxclust')

# Cluster sizes
from collections import Counter
cluster_sizes = Counter(labels)
valid_clusters = [c for c, n in cluster_sizes.items() if n >= MIN_CLUSTER_SIZE]
print(f"  total clusters: {len(cluster_sizes)}")
print(f"  clusters with >= {MIN_CLUSTER_SIZE} atoms: {len(valid_clusters)}")
print(f"  size distribution: min={min(cluster_sizes.values())}, "
      f"max={max(cluster_sizes.values())}, "
      f"median={np.median(list(cluster_sizes.values())):.0f}")

# Map atom_id → cluster_id
atom_to_cluster = dict(zip(uninf_ids, labels))

# ============================================================
# STEP 2: Validate clusters
# ============================================================
print("\n[Step 2] Validating cluster coherence ...")
# Sample a few clusters: compute intra vs inter activation correlation
def atom_act_vector(aid):
    a = np.zeros(N, dtype=np.float32)
    st, en = acts.indptr[aid], acts.indptr[aid + 1]
    a[acts.indices[st:en]] = acts.data[st:en]
    return a

# Sample validation: 5 random valid clusters
sample_clusters = np.random.RandomState(0).choice(valid_clusters,
                                                  size=min(5, len(valid_clusters)),
                                                  replace=False)
print("  sampling 5 clusters for coherence check:")
for c in sample_clusters:
    atoms_c = [a for a in uninf_ids if atom_to_cluster[a] == c]
    if len(atoms_c) < 2: continue
    # Pick 5 random atom pairs intra
    rng = np.random.RandomState(c)
    pairs = [(atoms_c[i], atoms_c[j]) for i in range(min(3, len(atoms_c)))
                                       for j in range(i+1, min(3, len(atoms_c)))]
    if len(pairs) == 0: continue
    intra_corrs = []
    for a1, a2 in pairs[:5]:
        v1, v2 = atom_act_vector(a1), atom_act_vector(a2)
        r, _ = pearsonr(v1, v2)
        intra_corrs.append(r)
    print(f"    cluster {c} (n={len(atoms_c)}): intra activation r = {np.mean(intra_corrs):.3f}")

# ============================================================
# STEP 3: Build cluster activation vectors (sum across cluster atoms)
# ============================================================
print("\n[Step 3] Building cluster-level activation vectors ...")
cluster_acts = {}   # cluster_id → np.array of shape (N,)
for c in valid_clusters:
    atoms_c = [a for a in uninf_ids if atom_to_cluster[a] == c]
    cluster_sum = np.zeros(N, dtype=np.float32)
    for a in atoms_c:
        cluster_sum += atom_act_vector(a)
    cluster_acts[c] = cluster_sum
print(f"  built activation vectors for {len(cluster_acts)} clusters")

# ============================================================
# STEP 4: For each cluster, sample ECGs for Claude
# ============================================================
print("\n[Step 4] Running Claude descriptions per cluster ...")

SYSTEM = """You are an expert cardiac electrophysiologist analyzing patterns in ECG data.
You will compare a group of ECGs that strongly activate a specific detector pattern against
ECGs that do not activate it at all. Your task is to identify the precise ECG feature this
detector is capturing — what is COMMON in the activated group but ABSENT in the control group.
Be specific and clinical."""

def prompt_describe(activated, controls):
    a_lines = []
    for ex in activated:
        nums = ", ".join([
            f"HR={ex['heart_rate']:.0f}" if not pd.isna(ex['heart_rate']) else "HR=?",
            f"QRS={ex['qrs_duration']:.0f}ms" if not pd.isna(ex['qrs_duration']) else "",
            f"QT={ex['qt_interval']:.0f}ms" if not pd.isna(ex['qt_interval']) else "",
            f"QRSaxis={ex['qrs_axis']:.0f}°" if not pd.isna(ex['qrs_axis']) else "",
        ])
        a_lines.append(f"  [ACTIVATED] {nums} | {ex['report']}")
    c_lines = []
    for ex in controls:
        nums = ", ".join([
            f"HR={ex['heart_rate']:.0f}" if not pd.isna(ex['heart_rate']) else "HR=?",
            f"QRS={ex['qrs_duration']:.0f}ms" if not pd.isna(ex['qrs_duration']) else "",
            f"QT={ex['qt_interval']:.0f}ms" if not pd.isna(ex['qt_interval']) else "",
        ])
        c_lines.append(f"  [CONTROL]   {nums} | {ex['report']}")
    return f"""Below are {len(activated)} ECGs that STRONGLY activate a detector pattern, and
{len(controls)} ECGs that do NOT activate it at all. Find what is COMMON in [ACTIVATED] but
ABSENT in [CONTROL].

{chr(10).join(a_lines)}

{chr(10).join(c_lines)}

Output STRICTLY in this JSON format:
{{
  "summary": "The detector activates on ...",
  "description": "<2-4 sentences with specific clinical/numerical features that distinguish activated from control>",
  "key_evidence": ["<feature1>", "<feature2>", "<feature3>"]
}}"""

def prompt_predict(description, ecgs):
    lines = []
    for i, ex in enumerate(ecgs):
        nums = ", ".join([
            f"HR={ex['heart_rate']:.0f}" if not pd.isna(ex['heart_rate']) else "HR=?",
            f"QRS={ex['qrs_duration']:.0f}ms" if not pd.isna(ex['qrs_duration']) else "",
            f"QT={ex['qt_interval']:.0f}ms" if not pd.isna(ex['qt_interval']) else "",
            f"QRSaxis={ex['qrs_axis']:.0f}°" if not pd.isna(ex['qrs_axis']) else "",
        ])
        lines.append(f"  ECG_{i}: {nums} | {ex['report']}")
    return f"""DETECTOR DESCRIPTION: {description}

For each ECG below, rate from 0 (does not match) to 10 (strong match) how likely this detector
would activate, using the description above as the rubric.

{chr(10).join(lines)}

Output STRICTLY a JSON array of {len(ecgs)} integers 0-10:
[score_for_ECG_0, ..., score_for_ECG_{len(ecgs)-1}]"""

def call_claude(prompt, system=SYSTEM, max_tok=1024):
    for attempt in range(MAX_RETRIES):
        try:
            r = client.messages.create(
                model=MODEL, max_tokens=max_tok, system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            return r.content[0].text
        except Exception as e:
            print(f"    API error attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return None

def parse_json_obj(text):
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m: return json.loads(m.group(0))
    except Exception: pass
    return None

def parse_scores(text, n):
    try:
        m = re.search(r'\[[\d,\s]+\]', text)
        if m:
            arr = json.loads(m.group(0))
            if len(arr) == n: return arr
    except Exception: pass
    return None

# Subject-level split for held-out
subjects = clin['subject_id'].values
unique_subj = np.unique(subjects[~pd.isna(subjects)])
rng_global = np.random.RandomState(42)
rng_global.shuffle(unique_subj)
n_test = max(1, int(len(unique_subj) * HELDOUT_SUBJ_FRAC))
heldout_subj_set = set(unique_subj[:n_test].tolist())
print(f"  held-out subjects: {len(heldout_subj_set)} / {len(unique_subj)}")

is_heldout = np.array([s in heldout_subj_set if pd.notna(s) else False for s in subjects])

# Main loop over clusters
results = []
t0 = time.time()
for ci, c in enumerate(valid_clusters):
    atoms_c = [a for a in uninf_ids if atom_to_cluster[a] == c]
    cact = cluster_acts[c]

    # ---- Find ECGs ----
    # ACTIVATED: high cluster_act, train subjects
    train_mask = ~is_heldout
    train_act = np.where(train_mask, cact, 0)
    if (train_act > 0).sum() < N_HIGH_CONSISTENCY + 10:
        continue
    top_idx = np.argsort(train_act)[::-1][:N_HIGH_CONSISTENCY]

    # CONTROLS: cluster_act == 0 in train, random sample
    zero_mask = train_mask & (cact == 0)
    zero_pool = np.where(zero_mask)[0]
    if len(zero_pool) < N_ZERO_CONTROL: continue
    rng_c = np.random.RandomState(c)
    ctrl_idx = rng_c.choice(zero_pool, size=N_ZERO_CONTROL, replace=False)

    activated = [feat.iloc[i].to_dict() for i in top_idx]
    controls = [feat.iloc[i].to_dict() for i in ctrl_idx]

    # ---- Claude describe ----
    print(f"\n[{ci+1}/{len(valid_clusters)}] cluster {c} (n_atoms={len(atoms_c)}) "
          f"-- {time.time()-t0:.0f}s")
    desc_text = call_claude(prompt_describe(activated, controls))
    if desc_text is None: continue
    desc = parse_json_obj(desc_text)
    if desc is None: continue
    print(f"  summary: {desc.get('summary','')[:80]}")
    time.sleep(SLEEP)

    # ---- Held-out evaluation ----
    # Mix held-out ECGs: some activated, some zero
    ho_act = np.where(is_heldout, cact, 0)
    ho_act_idx = np.where(is_heldout & (cact > 0))[0]
    ho_zero_idx = np.where(is_heldout & (cact == 0))[0]
    if len(ho_act_idx) < 5 or len(ho_zero_idx) < 5: continue
    n_each = N_HELDOUT_TEST // 2
    rng_h = np.random.RandomState(c + 1000)
    if len(ho_act_idx) > n_each:
        ho_act_pick = ho_act_idx[np.argsort(cact[ho_act_idx])[::-1][:n_each]]
    else:
        ho_act_pick = ho_act_idx
    ho_zero_pick = rng_h.choice(ho_zero_idx, size=min(n_each, len(ho_zero_idx)), replace=False)
    ho_idx = np.concatenate([ho_act_pick, ho_zero_pick])
    ho_ecgs = [feat.iloc[i].to_dict() for i in ho_idx]
    true_acts = cact[ho_idx]

    pred_text = call_claude(prompt_predict(desc.get('description', ''), ho_ecgs))
    if pred_text is None: continue
    scores = parse_scores(pred_text, len(ho_ecgs))
    if scores is None: continue
    if np.std(scores) < 1e-6 or np.std(true_acts) < 1e-6:
        r = np.nan; p = np.nan
    else:
        r, p = pearsonr(scores, true_acts)
    print(f"  held-out r = {r:.3f}  (n_test = {len(ho_ecgs)})")

    results.append({
        'cluster_id': int(c),
        'n_atoms': len(atoms_c),
        'example_atoms': ';'.join(str(a) for a in atoms_c[:10]),
        'summary': desc.get('summary', ''),
        'description': desc.get('description', ''),
        'key_evidence': '; '.join(desc.get('key_evidence', [])),
        'pearson_r': r,
        'p_value': p,
        'n_heldout': len(ho_ecgs),
    })
    pd.DataFrame(results).to_csv(out_dir / "cluster_descriptions.csv", index=False)
    time.sleep(SLEEP)

# ============================================================
# Summary
# ============================================================
df = pd.DataFrame(results)
df.to_csv(out_dir / "cluster_descriptions.csv", index=False)

# Save atom→cluster map
am = pd.DataFrame({'atom_id': uninf_ids, 'cluster_id': labels})
am.to_csv(out_dir / "atom_to_cluster.csv", index=False)

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
valid = df[df['pearson_r'].notna()]
print(f"  clusters attempted:    {len(df)}")
print(f"  with valid r:          {len(valid)}")
if len(valid) > 0:
    print(f"  median cluster r:      {valid['pearson_r'].median():.3f}")
    print(f"  mean cluster r:        {valid['pearson_r'].mean():.3f}")
    print(f"  r > 0.5:               {(valid['pearson_r'] > 0.5).sum()} clusters")
    print(f"  r > 0.7:               {(valid['pearson_r'] > 0.7).sum()} clusters")
    n_atoms_covered = sum(int(r['n_atoms']) for _, r in valid[valid['pearson_r'] > 0.5].iterrows())
    print(f"  atoms covered by r>0.5: {n_atoms_covered} / {len(uninf)} "
          f"({100*n_atoms_covered/len(uninf):.1f}% of Uninformative)")

print("\nTop 5 clusters by r:")
for _, r in valid.nlargest(5, 'pearson_r').iterrows():
    print(f"  cluster {int(r['cluster_id']):3d} (n={int(r['n_atoms'])}) "
          f"r={r['pearson_r']:.2f}  {r['summary'][:60]}")

print(f"\nOutputs in: {out_dir}")
