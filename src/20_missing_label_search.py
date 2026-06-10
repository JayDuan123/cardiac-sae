"""
Stage 20: Systematic search for missing-annotation atoms (InterPLM Nudix equivalent).

Procedure:
  1. Filter Stage 17d atoms with r > 0.7
  2. For each, infer the "claimed concept" from Claude description
  3. Take top-50 activating ECGs
  4. Classify each into 4 categories based on:
     - Report text contains concept (regex match)
     - Patient ICD codes contain concept
  5. Identify atoms where Category C (report missing, ICD confirms) >= 3

This is the ECG analog of InterPLM's Nudix box discovery (Swiss-Prot
missed annotation confirmed by InterPro).

Caveat documented: ECG-level expression of chronic conditions is intermittent,
so 'missing report annotation' is more tentative than 'missing Swiss-Prot label'.
"""
import sys, re, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import fisher_exact

sys.path.insert(0, '/workspace/jay/stsae_project/Cardiac-Sensing-FM')
import config as cfg

SAE = cfg.SAE_DIR / "batchtopk_tiny_aws_k32_d1536"
out_dir = SAE / "missing_label_search"
out_dir.mkdir(parents=True, exist_ok=True)

TOP_K_ECG = 50         # search within top-50 activating ECGs per atom
MIN_R = 0.6            # only atoms with held-out r >= this
MIN_ANCHOR = 5         # need >=5 anchor cases (report+ICD agree)
MIN_MISSING = 3        # need >=3 missing-label cases

# ============================================================
# Concept catalog (claimed concept → ICD prefixes + report regex)
# ============================================================
CONCEPT_DEFS = {
    'AF': {
        'icd9_prefix': ['4273'],
        'icd10_prefix': ['I48'],
        'report_regex': re.compile(
            r'\b(atrial\s+fib(rillation)?|afib|a\.?\s*fib|atrial\s+flutter)\b',
            re.IGNORECASE),
        'desc_match': ['atrial fibrillation', 'afib', 'a-fib', 'atrial flutter'],
    },
    'RBBB': {
        'icd9_prefix': ['4261'],
        'icd10_prefix': ['I451'],
        'report_regex': re.compile(
            r'\b(right\s+bundle\s+branch|rbbb|r\.b\.b\.b)\b',
            re.IGNORECASE),
        'desc_match': ['right bundle branch', 'rbbb'],
    },
    'LBBB': {
        'icd9_prefix': ['4263'],
        'icd10_prefix': ['I447'],
        'report_regex': re.compile(
            r'\b(left\s+bundle\s+branch|lbbb|l\.b\.b\.b)\b',
            re.IGNORECASE),
        'desc_match': ['left bundle branch', 'lbbb'],
    },
    'MI': {
        'icd9_prefix': ['410','411','412','413','414'],
        'icd10_prefix': ['I20','I21','I22','I23','I24','I25'],
        'report_regex': re.compile(
            r'\b(infarct|infarction|myocardial|stemi|nstemi|q[\s-]*wave|t[\s-]*wave\s*inversion)\b',
            re.IGNORECASE),
        'desc_match': ['infarct', 'myocardial', 'mi', 'stemi'],
    },
    'HF': {
        'icd9_prefix': ['428'],
        'icd10_prefix': ['I50'],
        'report_regex': re.compile(
            r'\b(heart\s+failure|chf|congestive|pulmonary\s+edema)\b',
            re.IGNORECASE),
        'desc_match': ['heart failure', 'chf', 'congestive'],
    },
    'pacing': {
        'icd9_prefix': ['V450', 'V53'],
        'icd10_prefix': ['Z45', 'Z95'],
        'report_regex': re.compile(
            r'\b(pace[dr]?|pacing|pacemaker|ppm)\b', re.IGNORECASE),
        'desc_match': ['paced', 'pacing', 'pacemaker'],
    },
    'tachycardia': {
        'icd9_prefix': ['4270', '4271', '4272'],
        'icd10_prefix': ['I470', 'I471', 'I472', 'R000'],
        'report_regex': re.compile(
            r'\b(tachycardia|svt|vt|tachy)\b', re.IGNORECASE),
        'desc_match': ['tachycardia', 'svt', 'tachy'],
    },
    'bradycardia': {
        'icd9_prefix': ['4271', '4268'],
        'icd10_prefix': ['R001'],
        'report_regex': re.compile(
            r'\b(bradycardia|brady|slow\s+heart\s+rate)\b', re.IGNORECASE),
        'desc_match': ['bradycardia', 'brady'],
    },
    'LVH': {
        'icd9_prefix': ['4291', '4293'],
        'icd10_prefix': ['I515', 'I517'],
        'report_regex': re.compile(
            r'\b(ventricular\s+hypertrophy|lvh|hypertrophy)\b', re.IGNORECASE),
        'desc_match': ['ventricular hypertrophy', 'lvh', 'hypertrophy'],
    },
    'ischemia': {
        'icd9_prefix': ['414'],
        'icd10_prefix': ['I25'],
        'report_regex': re.compile(
            r'\b(ischemia|ischemi[ac]|st[\s-]*depression|st[\s-]*elevation|coronary)\b',
            re.IGNORECASE),
        'desc_match': ['ischemi', 'st-depression', 'st-elevation', 'coronary'],
    },
}

def infer_concept_from_description(desc, summary=''):
    """Map Claude description to one of the catalog concepts."""
    text = (str(desc) + ' ' + str(summary)).lower()
    matches = []
    for concept, info in CONCEPT_DEFS.items():
        for keyword in info['desc_match']:
            if keyword.lower() in text:
                matches.append(concept)
                break
    return matches

# ============================================================
# Load data
# ============================================================
print("Loading ...")
acts = load_npz(SAE / "activations_all.npz").tocsc()
N, D = acts.shape

clin = pd.read_csv(cfg.EMBEDDING_DIR.parent / "clinical" / "record_with_clinical.csv")
clin = clin.set_index('record_idx').reindex(range(N))

meta = pd.read_csv(cfg.EMBEDDING_DIR / f"csfm_{cfg.CSFM_VARIANT.lower()}_{cfg.RUN_TAG}_meta.csv")
meta = meta.reset_index().rename(columns={'index': 'record_idx'})
meta['study_id'] = meta['path'].str.extract(r'/s(\d+)/')[0].astype('Int64')

mm = pd.read_csv("/workspace/data/mimic-iv-ecg-aws/machine_measurements.csv",
                 usecols=['study_id'] + [f'report_{i}' for i in range(18)],
                 low_memory=False)
mm['study_id'] = mm['study_id'].astype('Int64')

# Combine all 18 report lines into one searchable string per ECG
mm['report_full'] = mm[[f'report_{i}' for i in range(18)]].apply(
    lambda r: ' | '.join(str(s) for s in r.values if pd.notna(s)),
    axis=1)

feat = meta[['record_idx','study_id']].merge(
    mm[['study_id','report_full']], on='study_id', how='left').set_index('record_idx').reindex(range(N))

# Stage 17d results
descs = pd.read_csv(SAE / "claude_interp_random200" / "atom_descriptions_random200.csv")
descs_high = descs[descs['pearson_r'] >= MIN_R].copy()
print(f"\nAtoms with r >= {MIN_R}: {len(descs_high)}")

# Infer concepts
descs_high['inferred_concepts'] = descs_high.apply(
    lambda r: infer_concept_from_description(r['description'], r['summary']),
    axis=1)
descs_high['n_concepts'] = descs_high['inferred_concepts'].str.len()
print(f"  with ≥1 catalog concept inferred: {(descs_high['n_concepts']>0).sum()}")
print(f"  Concept distribution:")
all_concepts = [c for cl in descs_high['inferred_concepts'] for c in cl]
from collections import Counter
for c, n in Counter(all_concepts).most_common():
    print(f"    {c}: {n}")

# ============================================================
# ICD lookup helper
# ============================================================
def has_icd_concept(rec_idx, concept):
    info = CONCEPT_DEFS[concept]
    s = clin.iloc[rec_idx]['icd_codes_str']
    v = clin.iloc[rec_idx]['icd_codes_v_str']
    if pd.isna(s): return False
    codes = str(s).split(',')
    vers = str(v).split(',') if pd.notna(v) else ['9']*len(codes)
    for c, ver in zip(codes, vers):
        c=c.strip(); ver=ver.strip()
        if ver=='9' and any(c.startswith(p) for p in info['icd9_prefix']):
            return True
        if ver=='10' and any(c.startswith(p) for p in info['icd10_prefix']):
            return True
    return False

def report_has_concept(rec_idx, concept):
    info = CONCEPT_DEFS[concept]
    rep = feat.iloc[rec_idx]['report_full']
    if pd.isna(rep) or not str(rep).strip(): return False
    return bool(info['report_regex'].search(str(rep)))

# ============================================================
# Per-atom analysis
# ============================================================
all_candidates = []
print(f"\n{'='*80}")
print(f"Scanning {len(descs_high)} atoms with r >= {MIN_R}")
print(f"{'='*80}\n")

for _, atom_row in descs_high.iterrows():
    atom_id = int(atom_row['atom_id'])
    inferred = atom_row['inferred_concepts']
    if len(inferred) == 0: continue
    
    # Get top-K activating ECGs
    st, en = acts.indptr[atom_id], acts.indptr[atom_id+1]
    if en - st < TOP_K_ECG: continue
    
    values = acts.data[st:en]
    indices = acts.indices[st:en]
    top_order = np.argsort(values)[::-1][:TOP_K_ECG]
    top_idx = indices[top_order]
    top_vals = values[top_order]
    
    # For each inferred concept, classify all top-K ECGs
    for concept in inferred:
        cats = {'A_both':[], 'B_report_only':[], 'C_icd_only':[], 'D_neither':[]}
        for ti, rec in zip(top_vals, top_idx):
            in_report = report_has_concept(rec, concept)
            in_icd = has_icd_concept(rec, concept)
            if in_report and in_icd: cats['A_both'].append((rec, ti))
            elif in_report and not in_icd: cats['B_report_only'].append((rec, ti))
            elif not in_report and in_icd: cats['C_icd_only'].append((rec, ti))
            else: cats['D_neither'].append((rec, ti))
        
        n_A = len(cats['A_both']); n_B = len(cats['B_report_only'])
        n_C = len(cats['C_icd_only']); n_D = len(cats['D_neither'])
        total_with_concept_in_report = n_A + n_B  # anchor evidence
        
        # Background rate of ICD concept in non-activating ECGs
        rng = np.random.RandomState(atom_id + hash(concept) % 1000)
        all_records = np.arange(N)
        bg_pool = np.setdiff1d(all_records, indices)
        bg_sample = rng.choice(bg_pool, 1000, replace=False)
        bg_has_icd = sum(has_icd_concept(r, concept) for r in bg_sample)
        bg_rate = bg_has_icd / 1000
        
        # Fisher exact: is C cases enriched vs background?
        # C cases: report missing but ICD has concept
        # vs Background: ICD has concept (without atom firing)
        # We need a stricter test:
        # Of top-K ECGs where report DOESN'T mention concept (C+D),
        # is ICD positive rate higher than in background?
        no_report = n_C + n_D
        try:
            odds, p_fisher = fisher_exact(
                [[n_C, no_report - n_C], [bg_has_icd, 1000 - bg_has_icd]],
                alternative='greater')
        except:
            odds, p_fisher = np.nan, 1.0
        
        # Anchor must be substantial
        anchor_rate = total_with_concept_in_report / TOP_K_ECG
        
        is_paper_grade = (
            anchor_rate >= 0.50 and
            n_C >= MIN_MISSING and
            p_fisher < 0.05
        )
        
        result = {
            'atom_id': atom_id,
            'claude_r': atom_row['pearson_r'],
            'stage15_category': atom_row['stage15_category'],
            'concept': concept,
            'top_K': TOP_K_ECG,
            'A_report_and_icd': n_A,
            'B_report_only': n_B,
            'C_icd_only_MISSING': n_C,
            'D_neither': n_D,
            'anchor_rate': anchor_rate,
            'bg_icd_rate': bg_rate,
            'C_fold_enrich': (n_C / no_report) / max(bg_rate, 1e-6) if no_report > 0 else np.nan,
            'fisher_p': p_fisher,
            'paper_grade': is_paper_grade,
            'summary': str(atom_row['summary'])[:100],
        }
        all_candidates.append(result)

cdf = pd.DataFrame(all_candidates)
cdf = cdf.sort_values(['paper_grade', 'C_icd_only_MISSING', 'C_fold_enrich'],
                       ascending=[False, False, False])
cdf.to_csv(out_dir / "missing_label_candidates.csv", index=False)

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*80}")
print(f"RESULTS")
print(f"{'='*80}\n")
print(f"Total atom-concept pairs scanned: {len(cdf)}")
print(f"Paper-grade missing label candidates: {cdf['paper_grade'].sum()}")
print(f"  (anchor_rate>=0.5, C>=3, Fisher p<0.05)\n")

if cdf['paper_grade'].sum() > 0:
    print(f"TOP PAPER-GRADE CANDIDATES:")
    print(f"{'='*80}")
    pg = cdf[cdf['paper_grade']].head(10)
    for _, r in pg.iterrows():
        print(f"\nAtom {int(r['atom_id'])} ({r['stage15_category']}, r={r['claude_r']:+.3f})")
        print(f"  Claimed concept: {r['concept']}")
        print(f"  Top-{r['top_K']} breakdown:")
        print(f"    (A) report ✓ ICD ✓     : {r['A_report_and_icd']:>2}  [anchor]")
        print(f"    (B) report ✓ ICD ✗     : {r['B_report_only']:>2}  [ECG only, no chronic dx]")
        print(f"    (C) report ✗ ICD ✓     : {r['C_icd_only_MISSING']:>2}  ★ MISSING LABEL")
        print(f"    (D) report ✗ ICD ✗     : {r['D_neither']:>2}  [other/uncertain]")
        print(f"  Anchor rate (A+B)/{r['top_K']} = {r['anchor_rate']:.0%}")
        print(f"  Missing ICD rate among C+D: {r['C_icd_only_MISSING']}/{r['C_icd_only_MISSING']+r['D_neither']} = {r['C_icd_only_MISSING']/(r['C_icd_only_MISSING']+r['D_neither']):.0%}")
        print(f"  Background ICD rate: {r['bg_icd_rate']:.0%}")
        print(f"  Fold enrichment: {r['C_fold_enrich']:.1f}x")
        print(f"  Fisher p = {r['fisher_p']:.4f}")
        print(f"  Claude summary: {r['summary'][:80]}")
else:
    print("No paper-grade candidates found. Showing closest ones (highest C):")
    print(cdf.head(10)[['atom_id', 'concept', 'A_report_and_icd', 'C_icd_only_MISSING',
                         'D_neither', 'anchor_rate', 'fisher_p', 'paper_grade']].to_string())

print(f"\nFull results: {out_dir}/missing_label_candidates.csv")
