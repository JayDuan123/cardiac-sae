"""
Stage 17: InterPLM-style Claude interpretation of Uninformative atoms.

For each frequently-firing atom that did NOT enrich for any concept
in our 40-concept inventory, we use Claude to:
  1. Generate a description from activation-stratified examples
  2. Validate by predicting activation on held-out ECGs (different subjects)
  3. Pearson r(predicted, true) quantifies description quality

This adapts the InterPLM protocol (Simon et al., arXiv:2412.12101)
to whole-ECG embeddings.
"""
import sys, os, json, time, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import pearsonr
import anthropic

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

# ============================================================
# Config
# ============================================================
SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
out_dir = sae_dir / "claude_interp"
out_dir.mkdir(parents=True, exist_ok=True)

N_TARGET_ATOMS = 50          # how many Uninformative atoms to interpret
N_EXAMPLES_PER_BIN = 8       # examples per activation bin (high/mid/zero)
N_HELDOUT_TEST = 20          # held-out ECGs for validation
HELDOUT_SUBJECT_FRAC = 0.2   # 20% subjects for held-out
MODEL = "claude-sonnet-4-5-20250929"
MAX_RETRIES = 3
SLEEP_BETWEEN = 1.0          # seconds between API calls

client = anthropic.Anthropic()

# ============================================================
# Load data
# ============================================================
print("=" * 60)
print("Stage 17: Claude interpretation of Uninformative atoms")
print("=" * 60)
print("\nLoading data ...")

acts = load_npz(sae_dir / "activations_all.npz").tocsc()
N, D = acts.shape
print(f"  activations: {acts.shape}")

tax = pd.read_csv(sae_dir / "taxonomy" / "atom_taxonomy.csv")
print(f"  taxonomy: {len(tax)} atoms")

# Meta + machine measurements
meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
meta = meta.reset_index().rename(columns={'index': 'record_idx'})
meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')

# Clinical (for subject_id)
CLINICAL_DIR = cfg.EMBEDDING_DIR.parent / "clinical"
clin = pd.read_csv(CLINICAL_DIR / "record_with_clinical.csv")
df_clin = clin.set_index('record_idx').reindex(range(N))

# Machine measurements
mm_num = ['study_id', 'rr_interval', 'p_onset', 'qrs_onset', 'qrs_end', 't_end',
          'p_axis', 'qrs_axis', 't_axis']
mm_rep = [f'report_{i}' for i in range(18)]
mm = pd.read_csv("/workspace/data/mimic-iv-ecg-aws/machine_measurements.csv",
                 usecols=mm_num + mm_rep, low_memory=False)

# Clean sentinel + derive features
for col in ['rr_interval', 'p_onset', 'qrs_onset', 'qrs_end', 't_end']:
    mm[col] = mm[col].where((mm[col] >= 0) & (mm[col] < 2000), np.nan)
for col in ['p_axis', 'qrs_axis', 't_axis']:
    mm[col] = mm[col].where((mm[col] >= -180) & (mm[col] <= 180), np.nan)
mm['heart_rate'] = 60000 / mm['rr_interval'].replace(0, np.nan)
mm['pr_interval'] = mm['qrs_onset'] - mm['p_onset']
mm['qrs_duration'] = mm['qrs_end'] - mm['qrs_onset']
mm['qt_interval'] = mm['t_end'] - mm['qrs_onset']

# Build per-record feature table
feat_cols_num = ['heart_rate', 'pr_interval', 'qrs_duration', 'qt_interval',
                 'p_axis', 'qrs_axis', 't_axis']
feat = meta[['record_idx', 'study_id']].merge(
    mm[['study_id'] + feat_cols_num + mm_rep], on='study_id', how='left'
).set_index('record_idx').reindex(range(N))

# Build report text per record
def join_reports(row):
    items = [str(s) for s in row.values if pd.notna(s) and str(s).strip()]
    return ' | '.join(items[:8])  # cap to keep prompt size sane
feat['report_text'] = feat[mm_rep].apply(join_reports, axis=1)

print(f"  feature table ready")

# ============================================================
# Select target atoms
# ============================================================
uninf = tax[tax['category'] == 'Uninformative'].copy()
uninf = uninf.sort_values('fire_pct', ascending=False).head(N_TARGET_ATOMS)
print(f"\nSelected {len(uninf)} Uninformative atoms (top by firing freq)")
print(f"  fire_pct range: {uninf['fire_pct'].min():.2f}% to {uninf['fire_pct'].max():.2f}%")

# ============================================================
# Helper: stratified sampling for one atom
# ============================================================
def sample_for_atom(atom_id, n_per_bin=N_EXAMPLES_PER_BIN, heldout_n=N_HELDOUT_TEST):
    """Returns: (train_examples [list of dicts], heldout_examples [list of dicts])"""
    st, en = acts.indptr[atom_id], acts.indptr[atom_id + 1]
    fired_idx = acts.indices[st:en]
    fired_vals = acts.data[st:en]
    if len(fired_idx) < 3 * n_per_bin:
        return None, None  # not enough activations

    # Subject split
    fired_subjects = df_clin.loc[fired_idx, 'subject_id'].values
    unique_subj = np.unique(fired_subjects[~pd.isna(fired_subjects)])
    rng = np.random.RandomState(atom_id)  # deterministic per atom
    rng.shuffle(unique_subj)
    n_ho = max(1, int(len(unique_subj) * HELDOUT_SUBJECT_FRAC))
    ho_subj = set(unique_subj[:n_ho].tolist())

    train_mask = ~pd.Series(fired_subjects).isin(ho_subj).values
    ho_mask = pd.Series(fired_subjects).isin(ho_subj).values

    # ---- TRAIN: stratified by activation strength ----
    train_idx = fired_idx[train_mask]
    train_vals = fired_vals[train_mask]
    if len(train_idx) < 2 * n_per_bin:
        return None, None
    # Quantile bins of activation strength
    q_high = np.quantile(train_vals, 0.80)
    q_mid_lo = np.quantile(train_vals, 0.20)
    q_mid_hi = np.quantile(train_vals, 0.50)
    high_idx = train_idx[train_vals >= q_high]
    mid_idx = train_idx[(train_vals >= q_mid_lo) & (train_vals <= q_mid_hi)]
    # Zero activation: sample from records where atom did NOT fire,
    # but from train subjects (use a random sample of records not in fired_idx)
    train_subj_records = [i for i, s in enumerate(df_clin['subject_id'].values)
                          if pd.notna(s) and s not in ho_subj]
    train_subj_records = np.array(train_subj_records)
    nonfired = np.setdiff1d(train_subj_records, fired_idx, assume_unique=False)
    zero_idx = rng.choice(nonfired, size=min(n_per_bin, len(nonfired)), replace=False)

    high_pick = rng.choice(high_idx, size=min(n_per_bin, len(high_idx)), replace=False)
    mid_pick = rng.choice(mid_idx, size=min(n_per_bin, len(mid_idx)), replace=False)

    def to_dict(idx, level, atom_id=atom_id):
        row = feat.loc[idx]
        rep = str(row.get('report_text', ''))[:300]
        return {
            'record_idx': int(idx),
            'activation_level': level,
            'activation_value': float(acts[idx, atom_id]) if level != 'zero' else 0.0,
            'heart_rate': None if pd.isna(row['heart_rate']) else float(row['heart_rate']),
            'pr_interval': None if pd.isna(row['pr_interval']) else float(row['pr_interval']),
            'qrs_duration': None if pd.isna(row['qrs_duration']) else float(row['qrs_duration']),
            'qt_interval': None if pd.isna(row['qt_interval']) else float(row['qt_interval']),
            'qrs_axis': None if pd.isna(row['qrs_axis']) else float(row['qrs_axis']),
            'report': rep,
        }

    train_ex = ([to_dict(i, 'HIGH') for i in high_pick] +
                [to_dict(i, 'MEDIUM') for i in mid_pick] +
                [to_dict(i, 'ZERO') for i in zero_idx])

    # ---- HELD-OUT: mix of fired (varying strength) + nonfired ----
    ho_fired = fired_idx[ho_mask]
    ho_fired_vals = fired_vals[ho_mask]
    if len(ho_fired) < heldout_n // 2:
        # take all available
        ho_fired_pick = ho_fired
    else:
        ho_fired_pick = rng.choice(ho_fired, size=heldout_n // 2, replace=False)

    ho_subj_records = [i for i, s in enumerate(df_clin['subject_id'].values)
                       if pd.notna(s) and s in ho_subj]
    ho_nonfired = np.setdiff1d(np.array(ho_subj_records), fired_idx, assume_unique=False)
    n_zero = heldout_n - len(ho_fired_pick)
    ho_zero_pick = rng.choice(ho_nonfired, size=min(n_zero, len(ho_nonfired)), replace=False)

    def to_dict_ho(idx, atom_id=atom_id):
        row = feat.loc[idx]
        rep = str(row.get('report_text', ''))[:300]
        true_act = float(acts[idx, atom_id]) if idx in fired_idx else 0.0
        return {
            'record_idx': int(idx),
            'true_activation': true_act,
            'heart_rate': None if pd.isna(row['heart_rate']) else float(row['heart_rate']),
            'pr_interval': None if pd.isna(row['pr_interval']) else float(row['pr_interval']),
            'qrs_duration': None if pd.isna(row['qrs_duration']) else float(row['qrs_duration']),
            'qt_interval': None if pd.isna(row['qt_interval']) else float(row['qt_interval']),
            'qrs_axis': None if pd.isna(row['qrs_axis']) else float(row['qrs_axis']),
            'report': rep,
        }

    ho_ex = ([to_dict_ho(i) for i in ho_fired_pick] +
             [to_dict_ho(i) for i in ho_zero_pick])

    return train_ex, ho_ex


# ============================================================
# Claude calls
# ============================================================
SYSTEM_DESC = """You are an expert cardiac electrophysiologist analyzing patterns in ECG data.
You will be shown a set of ECGs labeled with activation strength (HIGH/MEDIUM/ZERO) of a
particular signal pattern detector. Your task is to identify what ECG feature this detector
is capturing, by finding what distinguishes HIGH-activation ECGs from ZERO-activation ones.

Be specific and clinical. Avoid vague descriptions. Your description must be precise enough
that someone given a new ECG's report and measurements could predict whether the detector
would activate."""

def prompt_generate(train_examples):
    lines = []
    for ex in train_examples:
        nums = (f"HR={ex['heart_rate']:.0f}" if ex['heart_rate'] is not None else "HR=?") + \
               (f", PR={ex['pr_interval']:.0f}ms" if ex['pr_interval'] is not None else "") + \
               (f", QRS={ex['qrs_duration']:.0f}ms" if ex['qrs_duration'] is not None else "") + \
               (f", QT={ex['qt_interval']:.0f}ms" if ex['qt_interval'] is not None else "") + \
               (f", QRSaxis={ex['qrs_axis']:.0f}°" if ex['qrs_axis'] is not None else "")
        lines.append(f"[{ex['activation_level']:6s}] {nums} | {ex['report']}")
    examples_str = "\n".join(lines)
    return f"""I will show you {len(train_examples)} ECGs with their activation level for a specific atom detector.

ECG examples (format: [LEVEL] numerical features | report text):
{examples_str}

Identify what distinguishes HIGH-activation ECGs from ZERO-activation ones. Then output STRICTLY
in this JSON format (no other text):

{{
  "summary": "The atom activates on ...",
  "description": "<detailed clinical description, 2-4 sentences, including key numerical thresholds and report phrases that predict activation>",
  "key_evidence": ["<feature1>", "<feature2>", "<feature3>"]
}}"""

def prompt_predict(description, heldout_examples):
    lines = []
    for i, ex in enumerate(heldout_examples):
        nums = (f"HR={ex['heart_rate']:.0f}" if ex['heart_rate'] is not None else "HR=?") + \
               (f", PR={ex['pr_interval']:.0f}ms" if ex['pr_interval'] is not None else "") + \
               (f", QRS={ex['qrs_duration']:.0f}ms" if ex['qrs_duration'] is not None else "") + \
               (f", QT={ex['qt_interval']:.0f}ms" if ex['qt_interval'] is not None else "") + \
               (f", QRSaxis={ex['qrs_axis']:.0f}°" if ex['qrs_axis'] is not None else "")
        lines.append(f"ECG_{i}: {nums} | {ex['report']}")
    examples_str = "\n".join(lines)
    return f"""You previously generated this description for an ECG detector:

DESCRIPTION: {description}

Now, for each of these {len(heldout_examples)} ECGs, rate how strongly this detector would activate,
on a scale from 0 (definitely not) to 10 (strong match). Use the description as your rubric.

{examples_str}

Output STRICTLY a JSON array of {len(heldout_examples)} integers between 0 and 10, no other text:
[score_for_ECG_0, score_for_ECG_1, ..., score_for_ECG_{len(heldout_examples)-1}]"""

def call_claude(prompt, system=SYSTEM_DESC):
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=MODEL, max_tokens=1024, system=system,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text
        except Exception as e:
            print(f"    API error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
    return None

def parse_description(text):
    try:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception:
        pass
    return None

def parse_scores(text, n_expected):
    try:
        m = re.search(r'\[[\d,\s]+\]', text)
        if m:
            arr = json.loads(m.group(0))
            if len(arr) == n_expected:
                return arr
    except Exception:
        pass
    return None

# ============================================================
# Main loop
# ============================================================
results = []
t0 = time.time()
for ai, (_, row) in enumerate(uninf.iterrows()):
    atom_id = int(row['atom_id'])
    fire_pct = row['fire_pct']
    print(f"\n[{ai+1}/{len(uninf)}] atom {atom_id} (fire {fire_pct:.2f}%) "
          f"-- {time.time()-t0:.0f}s elapsed")

    train_ex, ho_ex = sample_for_atom(atom_id)
    if train_ex is None:
        print("  -- skip (not enough activations)")
        continue

    # Generate description
    print(f"  generating description from {len(train_ex)} train examples ...")
    desc_text = call_claude(prompt_generate(train_ex))
    if desc_text is None:
        print("  -- failed to get description"); continue
    desc = parse_description(desc_text)
    if desc is None:
        print(f"  -- failed to parse: {desc_text[:100]}"); continue
    print(f"  summary: {desc.get('summary', '')[:80]}")
    time.sleep(SLEEP_BETWEEN)

    # Predict on held-out
    print(f"  predicting on {len(ho_ex)} held-out ECGs ...")
    pred_text = call_claude(prompt_predict(desc.get('description', ''), ho_ex))
    if pred_text is None:
        print("  -- failed prediction"); continue
    scores = parse_scores(pred_text, len(ho_ex))
    if scores is None:
        print(f"  -- failed to parse scores: {pred_text[:100]}"); continue

    true_acts = [ex['true_activation'] for ex in ho_ex]
    if np.std(scores) < 1e-6 or np.std(true_acts) < 1e-6:
        r, p = np.nan, np.nan
    else:
        r, p = pearsonr(scores, true_acts)
    print(f"  Pearson r = {r:.3f}  (n_test = {len(ho_ex)})")

    results.append({
        'atom_id': atom_id,
        'fire_pct': fire_pct,
        'summary': desc.get('summary', ''),
        'description': desc.get('description', ''),
        'key_evidence': '; '.join(desc.get('key_evidence', [])),
        'pearson_r': r,
        'p_value': p,
        'n_heldout': len(ho_ex),
    })

    # Save incrementally
    pd.DataFrame(results).to_csv(out_dir / "atom_descriptions.csv", index=False)
    time.sleep(SLEEP_BETWEEN)

# ============================================================
# Summary
# ============================================================
df = pd.DataFrame(results)
df.to_csv(out_dir / "atom_descriptions.csv", index=False)

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
valid = df[df['pearson_r'].notna()]
print(f"  atoms attempted: {len(df)}")
print(f"  with valid r:    {len(valid)}")
if len(valid) > 0:
    print(f"  median r:        {valid['pearson_r'].median():.3f}")
    print(f"  mean r:          {valid['pearson_r'].mean():.3f}")
    print(f"  r > 0.5:         {(valid['pearson_r'] > 0.5).sum()} atoms")
    print(f"  r > 0.7:         {(valid['pearson_r'] > 0.7).sum()} atoms")

print(f"\nOutput: {out_dir}/atom_descriptions.csv")

# Top descriptions
print("\nTop 5 by Pearson r:")
for _, r in valid.nlargest(5, 'pearson_r').iterrows():
    print(f"  atom {int(r['atom_id'])}  r={r['pearson_r']:.2f}  {r['summary'][:70]}")
