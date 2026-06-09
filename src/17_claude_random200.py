"""
Stage 17d: Claude interpretation on 200 random atoms (InterPLM-style, parallel).

Same prompt design as 17b/17c. Differences:
  - 200 random atoms (vs 50)
  - Concurrent API calls (4 workers) → ~4x speedup
  - Resume from checkpoint (skip already-done atoms)
  - Saves after each completed atom (not every 10)
"""
import sys, json, time, re, warnings, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import pearsonr
import anthropic

sys.path.insert(0, '/workspace/jay/stsae_project/Cardiac-Sensing-FM')
import config as cfg

SAE = cfg.SAE_DIR / "batchtopk_tiny_aws_k32_d1536"
out_dir = SAE / "claude_interp_random200"
out_dir.mkdir(parents=True, exist_ok=True)

N_TARGET = 200
SEED = 2026
N_WORKERS = 4   # parallel API calls

N_HIGH = 8; N_MED = 8; N_ZERO = 8
N_HELDOUT_HIGH = 10; N_HELDOUT_ZERO = 10
MODEL = "claude-sonnet-4-5-20250929"
HELDOUT_FRAC = 0.2

# ============================================================
# Sample 200 atoms
# ============================================================
tax = pd.read_csv(SAE / "taxonomy_grouped" / "atom_taxonomy_grouped.csv")
cat_col = 'category_grouped' if 'category_grouped' in tax.columns else 'category'
non_dead = tax[tax[cat_col] != 'Dead']['atom_id'].astype(int).values
print(f"Non-dead atom pool: {len(non_dead)}")

rng_g = np.random.RandomState(SEED)
TARGET_ATOMS = sorted(rng_g.choice(non_dead, N_TARGET, replace=False).tolist())

breakdown = tax[tax['atom_id'].isin(TARGET_ATOMS)][cat_col].value_counts()
print(f"\nRandom 200 atoms (seed={SEED}):")
for cat, n in breakdown.items():
    print(f"  {cat:<25} {n}")

pd.DataFrame({'atom_id': TARGET_ATOMS}).to_csv(out_dir / "selected_atoms.csv", index=False)

# ============================================================
# Resume: load existing results
# ============================================================
results_path = out_dir / "atom_descriptions_random200.csv"
done_atoms = set()
results = []
if results_path.exists():
    prev_df = pd.read_csv(results_path)
    done_atoms = set(prev_df['atom_id'].astype(int).tolist())
    results = prev_df.to_dict('records')
    print(f"\n✓ Resuming: {len(done_atoms)} atoms already done, "
          f"{len(TARGET_ATOMS) - len(done_atoms)} to go")

todo_atoms = [a for a in TARGET_ATOMS if a not in done_atoms]
print(f"Atoms to process: {len(todo_atoms)}\n")

# ============================================================
# Auth check (fast fail)
# ============================================================
client = anthropic.Anthropic()
try:
    client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=5,
                            messages=[{"role":"user","content":"hi"}])
    print("✓ API auth OK\n")
except Exception as e:
    print(f"✗ API auth FAILED: {e}"); sys.exit(1)

# ============================================================
# Load data
# ============================================================
print("Loading data ...")
acts = load_npz(SAE / "activations_all.npz").tocsc()
N, D = acts.shape

clin = pd.read_csv(cfg.EMBEDDING_DIR.parent / "clinical" / "record_with_clinical.csv")
clin = clin.set_index('record_idx').reindex(range(N))

meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
meta = meta.reset_index().rename(columns={'index': 'record_idx'})
meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')

mm = pd.read_csv("/workspace/data/mimic-iv-ecg-aws/machine_measurements.csv",
                 usecols=['study_id', 'rr_interval', 'p_onset', 'qrs_onset',
                          'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis'] +
                         [f'report_{i}' for i in range(18)], low_memory=False)
mm['study_id'] = mm['study_id'].astype('Int64')

for c in ['rr_interval','p_onset','qrs_onset','qrs_end','t_end']:
    mm[c] = mm[c].where((mm[c]>=0)&(mm[c]<2000), np.nan)
for c in ['p_axis','qrs_axis','t_axis']:
    mm[c] = mm[c].where((mm[c]>=-180)&(mm[c]<=180), np.nan)

mm['heart_rate'] = 60000 / mm['rr_interval'].replace(0, np.nan)
mm['pr_interval'] = mm['qrs_onset'] - mm['p_onset']
mm['qrs_duration'] = mm['qrs_end'] - mm['qrs_onset']
mm['qt_interval'] = mm['t_end'] - mm['qrs_onset']

feat = meta[['record_idx','study_id']].merge(mm, on='study_id', how='left').set_index('record_idx').reindex(range(N))
feat['age'] = clin['age_at_ecg'].values
feat['sex'] = clin['gender'].values

subj = clin['subject_id'].values
unique_subj = np.unique(subj[~pd.isna(subj)])
rng_split = np.random.RandomState(42); rng_split.shuffle(unique_subj)
n_test = max(1, int(len(unique_subj) * HELDOUT_FRAC))
heldout_subj = set(unique_subj[:n_test].tolist())
is_heldout = np.array([s in heldout_subj if pd.notna(s) else False for s in subj])

# ============================================================
# Format + prompts (same as 17b/17c)
# ============================================================
def format_ecg_full(rec_idx, activation_level=None, activation_value=None):
    r = feat.iloc[rec_idx]
    lines = []
    if activation_level is not None:
        if activation_value is not None:
            lines.append(f"  Activation: {activation_level} (value={activation_value:.2f})")
        else:
            lines.append(f"  Activation: {activation_level}")
    age = f"{r['age']:.0f}" if not pd.isna(r['age']) else "?"
    sex = str(r['sex'])[:1] if not pd.isna(r['sex']) else "?"
    lines.append(f"  Demographics: age={age}, sex={sex}")
    nums = []
    for label, col in [('HR','heart_rate'),('PR','pr_interval'),('QRS','qrs_duration'),
                        ('QT','qt_interval'),('P_axis','p_axis'),('QRS_axis','qrs_axis'),('T_axis','t_axis')]:
        v = r.get(col, np.nan)
        if not pd.isna(v):
            unit = "°" if 'axis' in col else ("ms" if col != 'heart_rate' else "")
            nums.append(f"{label}={v:.0f}{unit}")
    if nums: lines.append(f"  Numerical: {', '.join(nums)}")
    rl = []
    for i in range(18):
        v = r.get(f'report_{i}', None)
        if pd.notna(v) and str(v).strip(): rl.append(f"    [{i+1:>2}] {str(v).strip()}")
    lines.append("  Report (raw, line-by-line):" if rl else "  Report: (empty)")
    lines.extend(rl)
    return "\n".join(lines)

SYSTEM_PROMPT = """You are an expert cardiac electrophysiologist analyzing 
SAE atom activations on 12-lead ECG data. Each atom is a learned feature 
in the foundation model's representation. We sample ECGs at three activation 
levels (HIGH, MEDIUM, ZERO) and ask you to identify the cardiac pattern the 
atom detects. You see demographics, numerical measurements, and the full 
machine-generated report line-by-line."""

def prompt_phase1(atom_id, high_ex, med_ex, zero_ex):
    lines = [f"ATOM {atom_id} activation analysis", "="*60, "",
             f"HIGH (n={len(high_ex)}):", "-"*60]
    for i,(idx,v) in enumerate(high_ex):
        lines += [f"\nECG_H{i+1}:", format_ecg_full(idx, "HIGH", v)]
    lines += [f"\nMEDIUM (n={len(med_ex)}):", "-"*60]
    for i,(idx,v) in enumerate(med_ex):
        lines += [f"\nECG_M{i+1}:", format_ecg_full(idx, "MEDIUM", v)]
    lines += [f"\nZERO (n={len(zero_ex)}):", "-"*60]
    for i,idx in enumerate(zero_ex):
        lines += [f"\nECG_Z{i+1}:", format_ecg_full(idx, "ZERO", 0.0)]
    lines.append("""

="*60
Find the cardiac pattern that distinguishes HIGH from ZERO. The pattern should:
  1. Be present in most HIGH (>=6/8) and largely absent in ZERO (<=2/8)
  2. Show graded prevalence in MEDIUM
  3. Be specific enough to test on held-out ECGs

Output STRICTLY JSON:
{
  "summary": "<one-sentence detector summary>",
  "description": "<2-4 sentences with specific features>",
  "key_evidence": ["<f1>","<f2>","<f3>"]
}""")
    return "\n".join(lines)

def prompt_phase2(description, ecgs_to_score):
    lines = ["You previously characterized an SAE atom with this description:","",
             f'"{description}"',"","="*60,
             f"Rate each of {len(ecgs_to_score)} held-out ECGs from 0 (no match) to 10 (perfect match) based on how well each matches the described pattern.",
             "="*60,""]
    for i, idx in enumerate(ecgs_to_score):
        lines += [f"\nECG_{i}:", format_ecg_full(idx)]
    lines.append(f"\nOutput STRICTLY a JSON array of {len(ecgs_to_score)} integers (0-10). No other text.")
    return "\n".join(lines)

def call_claude(prompt, max_tokens=2048, retries=3):
    for attempt in range(retries):
        try:
            r = client.messages.create(model=MODEL, max_tokens=max_tokens,
                                        system=SYSTEM_PROMPT,
                                        messages=[{"role":"user","content":prompt}])
            return r.content[0].text
        except Exception as e:
            err = str(e)
            if 'rate' in err.lower() or '429' in err:
                time.sleep(5 + 2**attempt)
            else:
                time.sleep(1 + attempt)
    return None

def parse_json_obj(text):
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m: return json.loads(m.group(0))
    except: pass
    return None

def parse_scores(text, n):
    try:
        m = re.search(r'\[[\d,\s]+\]', text)
        if m:
            arr = json.loads(m.group(0))
            if len(arr)==n: return arr
    except: pass
    return None

# ============================================================
# Process one atom (thread-safe)
# ============================================================
def process_atom(atom_id):
    st, en = acts.indptr[atom_id], acts.indptr[atom_id+1]
    if en - st < 100: return ('skip:too_few_fires', None)
    
    values = acts.data[st:en]; indices = acts.indices[st:en]
    train_mask = ~is_heldout[indices]
    train_idx = indices[train_mask]; train_vals = values[train_mask]
    n_train = len(train_idx)
    if n_train < 50: return ('skip:too_few_train', None)
    
    sorted_vals = np.sort(train_vals)
    q80 = sorted_vals[int(n_train*0.8)]; q50 = sorted_vals[int(n_train*0.5)]
    high_pool = train_idx[train_vals >= q80]; high_v = train_vals[train_vals >= q80]
    med_pool = train_idx[(train_vals >= q50) & (train_vals < q80)]
    med_v = train_vals[(train_vals >= q50) & (train_vals < q80)]
    all_train = np.where(~is_heldout)[0]
    zero_pool = np.setdiff1d(all_train, indices)
    if len(high_pool) < N_HIGH or len(med_pool) < N_MED or len(zero_pool) < N_ZERO:
        return ('skip:pool_too_small', None)
    
    rng = np.random.RandomState(atom_id)
    hi = rng.choice(len(high_pool), N_HIGH, replace=False)
    high_ex = [(int(high_pool[i]), float(high_v[i])) for i in hi]
    me = rng.choice(len(med_pool), N_MED, replace=False)
    med_ex = [(int(med_pool[i]), float(med_v[i])) for i in me]
    zero_ex = list(rng.choice(zero_pool, N_ZERO, replace=False).astype(int))
    
    p1 = prompt_phase1(atom_id, high_ex, med_ex, zero_ex)
    d_text = call_claude(p1, 1024)
    if d_text is None: return ('skip:phase1_failed', None)
    desc = parse_json_obj(d_text)
    if desc is None: return ('skip:phase1_unparseable', None)
    
    heldout_idx_a = indices[is_heldout[indices]]
    heldout_vals_a = values[is_heldout[indices]]
    if len(heldout_idx_a) < 6: return ('skip:heldout_too_few', None)
    
    sorted_ho = np.sort(heldout_vals_a)[::-1]
    ho_q70 = sorted_ho[min(int(len(heldout_idx_a)*0.3), len(sorted_ho)-1)]
    ho_high = heldout_idx_a[heldout_vals_a >= ho_q70]
    all_heldout = np.where(is_heldout)[0]
    ho_zero = np.setdiff1d(all_heldout, indices)
    if len(ho_high) < 6 or len(ho_zero) < 6: return ('skip:ho_too_small', None)
    
    rng2 = np.random.RandomState(atom_id + 999)
    n_h = min(N_HELDOUT_HIGH, len(ho_high))
    n_z = min(N_HELDOUT_ZERO, len(ho_zero))
    ho_high_pick = rng2.choice(ho_high, n_h, replace=False)
    ho_zero_pick = rng2.choice(ho_zero, n_z, replace=False)
    ho_idx = np.concatenate([ho_high_pick, ho_zero_pick])
    act_lookup = dict(zip(heldout_idx_a, heldout_vals_a))
    true_acts = np.array([act_lookup.get(i, 0.0) for i in ho_idx])
    
    p2 = prompt_phase2(desc.get('description',''), ho_idx.tolist())
    s_text = call_claude(p2, 512)
    if s_text is None: return ('skip:phase2_failed', None)
    scores = parse_scores(s_text, len(ho_idx))
    if scores is None: return ('skip:phase2_unparseable', None)
    
    if np.std(scores) < 1e-6 or np.std(true_acts) < 1e-6:
        r_val, p_val = np.nan, np.nan
    else:
        r_val, p_val = pearsonr(scores, true_acts)
    
    cat_row = tax[tax['atom_id']==atom_id]
    cat = cat_row[cat_col].iloc[0] if len(cat_row)>0 else 'Unknown'
    
    return ('ok', {
        'atom_id': atom_id, 'stage15_category': cat,
        'summary': desc.get('summary',''), 'description': desc.get('description',''),
        'key_evidence': '; '.join(desc.get('key_evidence',[])),
        'pearson_r': r_val, 'p_value': p_val,
        'n_heldout': len(ho_idx), 'n_train_fires': n_train,
    })

# ============================================================
# Parallel execution
# ============================================================
import threading
results_lock = threading.Lock()
done_count = [len(done_atoms)]
print(f"Running with {N_WORKERS} parallel workers...\n")
t_start = time.time()

def worker(atom_id):
    status, info = process_atom(atom_id)
    with results_lock:
        if status == 'ok':
            results.append(info)
            # Save after each completion
            pd.DataFrame(results).to_csv(results_path, index=False)
        done_count[0] += 1
        elapsed = time.time() - t_start
        cat = tax[tax['atom_id']==atom_id][cat_col].iloc[0] if (tax['atom_id']==atom_id).any() else '?'
        msg = f"r={info['pearson_r']:.3f}" if status == 'ok' else status
        print(f"[{done_count[0]}/{len(TARGET_ATOMS)}] atom {atom_id} ({cat[:12]}) {msg} | elapsed {elapsed:.0f}s", flush=True)
    return status

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    list(executor.map(worker, todo_atoms))

# ============================================================
# Final summary
# ============================================================
df = pd.DataFrame(results)
df.to_csv(results_path, index=False)

print(f"\n{'='*70}")
print(f"FINAL SUMMARY")
print(f"{'='*70}")
print(f"Total processed: {len(df)}/{len(TARGET_ATOMS)}")
print(f"Total time: {time.time()-t_start:.0f}s")

valid = df.dropna(subset=['pearson_r'])
if len(valid) > 0:
    print(f"\nOVERALL METRICS (n={len(valid)})")
    print(f"  Median r:  {valid['pearson_r'].median():+.3f}")
    print(f"  Mean r:    {valid['pearson_r'].mean():+.3f}")
    print(f"  Std:       {valid['pearson_r'].std():.3f}")
    print(f"  r > 0.5:   {(valid['pearson_r']>0.5).sum()} ({100*(valid['pearson_r']>0.5).mean():.0f}%)")
    print(f"  r > 0.3:   {(valid['pearson_r']>0.3).sum()} ({100*(valid['pearson_r']>0.3).mean():.0f}%)")
    print(f"  r < 0:     {(valid['pearson_r']<0).sum()} ({100*(valid['pearson_r']<0).mean():.0f}%)")
    
    print(f"\nBY STAGE 15 CATEGORY")
    for cat in ['Separable','Entangled-Related','Entangled-Mixed','Uninformative','Contributing']:
        sub = valid[valid['stage15_category'] == cat]
        if len(sub) > 0:
            print(f"  {cat:<22} n={len(sub):>3}  median={sub['pearson_r'].median():+.3f}  "
                  f"mean={sub['pearson_r'].mean():+.3f}  "
                  f">0.5: {(sub['pearson_r']>0.5).sum()}/{len(sub)}")

print(f"\nResults: {results_path}")
