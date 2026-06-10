"""
Stage 19c: Single-atom case study panels v2.

Improvements over 19b:
  1. Description shown in dedicated full-width panel (no truncation)
  2. Age plot REPLACED with QRS duration distribution (description-aligned)
  3. Added 3 ZERO-activation reference ECGs for contrast
  4. Metadata: explicit enriched concept names + AUROC values
  5. "Position in landscape" line in header
  
5 hero atoms covering paper Figure 5:
  Panel A: atom 1378  — high-r Separable (RBBB)
  Panel B: atom 785   — high-r Uninformative (LBBB) ★ HERO
  Panel C: atom 9     — Uninformative AF (missing annotation candidate)
  Panel D: atom 19    — Data corruption detector
  Panel E: atom 1251  — Negative-r failure case
"""
import sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.sparse import load_npz

sys.path.insert(0, '/workspace/jay/stsae_project/Cardiac-Sensing-FM')
import config as cfg

mpl.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'figure.dpi': 100,
    'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

SAE = cfg.SAE_DIR / "batchtopk_tiny_aws_k32_d1536"
out_dir = SAE / "single_atom_case_studies"
out_dir.mkdir(parents=True, exist_ok=True)

HERO_ATOMS = {
    1378: {'label': 'Panel A — Atom 1378: RBBB (Separable, high-r)',
           'short': 'rbbb_separable', 'color': '#2ca02c'},
    785:  {'label': 'Panel B — Atom 785: LBBB (Uninformative, high-r) ★ HERO',
           'short': 'lbbb_uninf_hero', 'color': '#d62728'},
    9:    {'label': 'Panel C — Atom 9: Atrial fibrillation (Uninformative) ★ Missing annotation',
           'short': 'af_uninf_missing', 'color': '#9467bd'},
    19:   {'label': 'Panel D — Atom 19: Data corruption detector',
           'short': 'data_corruption', 'color': '#7f7f7f'},
    1251: {'label': 'Panel E — Atom 1251: Negative-r failure case',
           'short': 'negative_failure', 'color': '#8B4513'},
}

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
                 usecols=['study_id', 'rr_interval', 'p_onset', 'qrs_onset',
                          'qrs_end', 't_end', 'p_axis', 'qrs_axis', 't_axis'] +
                         [f'report_{i}' for i in range(18)], low_memory=False)
mm['study_id'] = mm['study_id'].astype('Int64')

for c in ['rr_interval','p_onset','qrs_onset','qrs_end','t_end']:
    mm[c] = mm[c].where((mm[c]>=0)&(mm[c]<2000), np.nan)
for c in ['p_axis','qrs_axis','t_axis']:
    mm[c] = mm[c].where((mm[c]>=-180)&(mm[c]<=180), np.nan)

mm['heart_rate'] = 60000 / mm['rr_interval'].replace(0, np.nan)
mm['qrs_duration'] = mm['qrs_end'] - mm['qrs_onset']
mm['pr_interval'] = mm['qrs_onset'] - mm['p_onset']
mm['qt_interval'] = mm['t_end'] - mm['qrs_onset']

feat = meta[['record_idx','study_id']].merge(mm, on='study_id', how='left').set_index('record_idx').reindex(range(N))

descs = pd.read_csv(SAE / "claude_interp_random200" / "atom_descriptions_random200.csv")
descs = descs.set_index('atom_id')

enr = pd.read_csv(SAE / "taxonomy" / "enrichment_tests.csv")

# ICD lookup
def has_icd(s, v, p9, p10):
    if pd.isna(s): return False
    codes = str(s).split(','); vers = str(v).split(',') if pd.notna(v) else ['9']*len(codes)
    for c, ver in zip(codes, vers):
        c=c.strip(); ver=ver.strip()
        if ver=='9' and any(c.startswith(p) for p in p9): return True
        if ver=='10' and any(c.startswith(p) for p in p10): return True
    return False

icd_af_all = np.array([has_icd(s,v,['4273'],['I48'])
                        for s,v in zip(clin['icd_codes_str'], clin['icd_codes_v_str'])])
icd_mi_all = np.array([has_icd(s,v,['410','411','412','413','414'],
                                ['I20','I21','I22','I23','I24','I25'])
                       for s,v in zip(clin['icd_codes_str'], clin['icd_codes_v_str'])])
icd_hf_all = np.array([has_icd(s,v,['428'],['I50'])
                       for s,v in zip(clin['icd_codes_str'], clin['icd_codes_v_str'])])
mort_1y_all = ((clin['has_dod']==True) & (clin['days_to_death']<=365)).values

# Stage 17d landscape stats (for "position in landscape" annotation)
n_random200 = len(descs)
median_r = descs['pearson_r'].median()

# ============================================================
# Helpers
# ============================================================
def get_top_atoms_concepts(atom_id, top_n=3):
    """Get top enriched concepts for this atom with their AUROC."""
    atom_enr = enr[enr['atom_id'] == atom_id].sort_values('auroc', ascending=False)
    if len(atom_enr) == 0:
        return []
    top = atom_enr.head(top_n)
    return [(row['concept'], row['auroc'], row['enriched']) for _, row in top.iterrows()]


def format_ecg_box_text(rec_idx, atom_val, position='top'):
    """Format a single ECG into a compact text block."""
    r = feat.iloc[rec_idx]
    c = clin.iloc[rec_idx]
    
    age = f"{c['age_at_ecg']:.0f}" if pd.notna(c['age_at_ecg']) else "?"
    sex = str(c['gender'])[:1] if pd.notna(c['gender']) else "?"
    
    nums = []
    for lbl, col in [('HR','heart_rate'), ('QRS','qrs_duration'),
                      ('PR','pr_interval'), ('QT','qt_interval'),
                      ('QRSax','qrs_axis')]:
        v = r.get(col, np.nan)
        if not pd.isna(v):
            u = "°" if 'ax' in col else ("" if col=='heart_rate' else "ms")
            nums.append(f"{lbl}={v:.0f}{u}")
    
    # Report lines, full content
    report_lines = []
    for ri in range(18):
        v = r.get(f'report_{ri}', None)
        if pd.notna(v) and str(v).strip():
            line = str(v).strip()
            if len(line) > 42: line = line[:39] + "..."
            report_lines.append(f"[{ri+1}] {line}")
    
    # ICD flags
    flags = []
    if icd_af_all[rec_idx]: flags.append("AF")
    if icd_mi_all[rec_idx]: flags.append("MI")
    if icd_hf_all[rec_idx]: flags.append("HF")
    if mort_1y_all[rec_idx]: flags.append("☠1yr")
    icd_str = ", ".join(flags) if flags else "—"
    
    label = f"act={atom_val:.2f}" if position == 'top' else "act=0 (ZERO)"
    
    txt = (f"[{label}]   age={age}, sex={sex}\n"
           f"{', '.join(nums)}\n"
           f"ICD/outcome: {icd_str}\n"
           f"Report:\n" + "\n".join(report_lines[:7]))
    return txt


# ============================================================
# Build figure for one atom
# ============================================================
def build_atom_panel(atom_id, label_info):
    print(f"\n--- Atom {atom_id}: {label_info['label']} ---")
    
    # Get top-activating ECGs
    st, en = acts.indptr[atom_id], acts.indptr[atom_id+1]
    values = acts.data[st:en]; indices = acts.indices[st:en]
    if len(values) < 10:
        print(f"  too sparse, skip"); return
    
    top_order = np.argsort(values)[::-1][:8]
    top_idx = indices[top_order]; top_vals = values[top_order]
    
    # 3 ZERO activation ECGs (atom doesn't fire)
    all_records = np.arange(N)
    zero_pool = np.setdiff1d(all_records, indices)
    rng = np.random.RandomState(atom_id + 1000)
    zero_idx = rng.choice(zero_pool, 3, replace=False)
    
    # ----- Metadata -----
    desc_row = descs.loc[atom_id] if atom_id in descs.index else None
    fire_rate = 100 * len(indices) / N
    
    # Enriched concepts with AUROC
    top_concepts = get_top_atoms_concepts(atom_id, top_n=3)
    if top_concepts:
        enr_str_parts = []
        for c, auc, is_enr in top_concepts:
            tag = "★" if is_enr else ""
            c_short = c.replace('TXT:','').replace('NUM:','').replace('ICD:','')[:35]
            enr_str_parts.append(f"{c_short}{tag} (AUC {auc:.3f})")
        concepts_line = "; ".join(enr_str_parts)
    else:
        concepts_line = "no enriched concepts (Stage 15 silent)"
    
    if desc_row is not None:
        cat = desc_row['stage15_category']
        r_val = desc_row['pearson_r']
        summary = desc_row['summary']
        description = desc_row['description']
        p_val = desc_row['p_value']
        n_held = int(desc_row['n_heldout']) if not pd.isna(desc_row['n_heldout']) else 0
    else:
        cat='?'; r_val=np.nan; summary='?'; description='?'
        p_val=np.nan; n_held=0
    
    s15_atom = enr[enr['atom_id'] == atom_id]
    max_auc = s15_atom['auroc'].max() if len(s15_atom)>0 else np.nan
    n_enriched_concepts = s15_atom['enriched'].sum() if len(s15_atom)>0 else 0
    
    # Position in random-200 landscape
    if not pd.isna(r_val):
        r_rank = (descs['pearson_r'] > r_val).sum() + 1
        r_pctile = 100 * (descs['pearson_r'] < r_val).mean()
    else:
        r_rank = '?'; r_pctile = np.nan
    
    # ----- Build figure: 5-row layout -----
    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(5, 4, 
                          height_ratios=[0.55, 0.7, 1.25, 1.25, 1.1],
                          hspace=0.55, wspace=0.35)
    
    # ========== ROW 1: Metadata header ==========
    ax_h = fig.add_subplot(gs[0, :])
    ax_h.axis('off')
    
    header_lines = [
        f"ATOM {atom_id} — {label_info['label']}",
        "",
        f"Stage 15 category: {cat}   "
        f"Max AUROC (across all concepts): {max_auc:.3f}   "
        f"Enriched concepts (Stage 15): {n_enriched_concepts}   "
        f"Fire rate: {fire_rate:.2f}% ({len(indices):,} ECGs)",
        "",
        f"Top concepts:  {concepts_line}",
        "",
        f"Stage 17d held-out Pearson r = {r_val:+.3f}  (p = {p_val:.2e}, n held-out = {n_held})   "
        f"Rank: #{r_rank}/{n_random200} ({r_pctile:.0f}th pctile),  Overall median r = {median_r:+.3f}",
    ]
    ax_h.text(0.005, 0.98, "\n".join(header_lines),
              transform=ax_h.transAxes, fontsize=10.5,
              verticalalignment='top', family='monospace',
              bbox=dict(boxstyle='round,pad=0.5',
                        facecolor=label_info['color'], alpha=0.18,
                        edgecolor=label_info['color'], linewidth=2))
    
    # ========== ROW 2: Claude description (full, no truncation) ==========
    ax_d = fig.add_subplot(gs[1, :])
    ax_d.axis('off')
    
    desc_text = (
        f"Claude summary:  {summary}\n\n"
        f"Claude description (full):  {description}\n\n"
        f"Key evidence:  {desc_row.get('key_evidence','')}"
    )
    ax_d.text(0.005, 0.98, desc_text, transform=ax_d.transAxes,
              fontsize=9.5, verticalalignment='top',
              wrap=True, family='serif',
              bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='#FFFBEA', alpha=0.85,
                        edgecolor='#888', linewidth=1))
    
    # ========== ROW 3 & 4: 8 top-activating ECGs (4 + 4) ==========
    for i in range(4):
        ax = fig.add_subplot(gs[2, i])
        ax.axis('off')
        txt = format_ecg_box_text(top_idx[i], top_vals[i], position='top')
        ax.text(0.0, 0.98, f"Top-activating ECG #{i+1}\n{txt}",
                transform=ax.transAxes, fontsize=7.5,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#fff4f0', alpha=0.7,
                          edgecolor=label_info['color'], linewidth=1.2))
    
    for i in range(4):
        ax = fig.add_subplot(gs[3, i])
        ax.axis('off')
        if i < 3:
            # ECG #5-7 in top row
            txt = format_ecg_box_text(top_idx[i+4], top_vals[i+4], position='top')
            ax.text(0.0, 0.98, f"Top-activating ECG #{i+5}\n{txt}",
                    transform=ax.transAxes, fontsize=7.5,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#fff4f0', alpha=0.7,
                              edgecolor=label_info['color'], linewidth=1.2))
        else:
            # ECG #8 (last) — also a top-activating
            txt = format_ecg_box_text(top_idx[7], top_vals[7], position='top')
            ax.text(0.0, 0.98, f"Top-activating ECG #8\n{txt}",
                    transform=ax.transAxes, fontsize=7.5,
                    verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#fff4f0', alpha=0.7,
                              edgecolor=label_info['color'], linewidth=1.2))
    
    # ========== ROW 5: 4 quantitative panels ==========
    
    # --- Panel (1): Activation distribution ---
    ax1 = fig.add_subplot(gs[4, 0])
    ax1.hist(values, bins=40, color=label_info['color'], alpha=0.75,
             edgecolor='black', linewidth=0.5)
    ax1.axvline(top_vals[-1], color='red', linestyle='--', linewidth=1.5,
                label=f'top-8 floor = {top_vals[-1]:.2f}')
    ax1.set_xlabel('Atom activation value')
    ax1.set_ylabel('# firing ECGs')
    ax1.set_title(f'Activation distribution\n(n={len(values):,} firing)',
                  fontsize=9, fontweight='bold')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # --- Panel (2): QRS duration top-8 vs background ---
    ax2 = fig.add_subplot(gs[4, 1])
    top_qrs = feat.iloc[top_idx]['qrs_duration'].dropna()
    
    rng2 = np.random.RandomState(42)
    bg_size = 5000
    bg_idx_sample = rng2.choice(N, bg_size, replace=False)
    bg_qrs = feat.iloc[bg_idx_sample]['qrs_duration'].dropna()
    
    if len(top_qrs) > 0 and len(bg_qrs) > 0:
        bins = np.arange(60, 220, 5)
        ax2.hist(bg_qrs, bins=bins, color='gray', alpha=0.4,
                 density=True, label=f'Background (med {bg_qrs.median():.0f}ms)')
        ax2.hist(top_qrs, bins=bins, color=label_info['color'], alpha=0.75,
                 density=True, label=f'Top-8 (med {top_qrs.median():.0f}ms)')
        ax2.axvline(120, color='red', linestyle=':', alpha=0.7,
                    label='Wide QRS = 120ms')
    ax2.set_xlabel('QRS duration (ms)')
    ax2.set_ylabel('Density')
    ax2.set_title('QRS duration: top-8 vs background',
                  fontsize=9, fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    # --- Panel (3): Clinical outcomes ---
    ax3 = fig.add_subplot(gs[4, 2])
    n_top = len(top_idx)
    af_top = icd_af_all[top_idx].sum()
    mi_top = icd_mi_all[top_idx].sum()
    hf_top = icd_hf_all[top_idx].sum()
    mort_top = mort_1y_all[top_idx].sum()
    
    af_bg = icd_af_all[bg_idx_sample].mean() * 100
    mi_bg = icd_mi_all[bg_idx_sample].mean() * 100
    hf_bg = icd_hf_all[bg_idx_sample].mean() * 100
    mort_bg = mort_1y_all[bg_idx_sample].mean() * 100
    
    cats = ['ICD:AF', 'ICD:MI', 'ICD:HF', 'Mort 1yr']
    top_pcts = [af_top/n_top*100, mi_top/n_top*100, hf_top/n_top*100, mort_top/n_top*100]
    bg_pcts = [af_bg, mi_bg, hf_bg, mort_bg]
    
    x = np.arange(len(cats)); w = 0.38
    ax3.bar(x - w/2, top_pcts, w, color=label_info['color'],
            alpha=0.85, edgecolor='black', linewidth=0.5,
            label=f'Top-8')
    ax3.bar(x + w/2, bg_pcts, w, color='gray', alpha=0.6,
            edgecolor='black', linewidth=0.5,
            label='Background')
    ax3.set_xticks(x); ax3.set_xticklabels(cats, fontsize=8, rotation=15)
    ax3.set_ylabel('% positive')
    ax3.set_title('Clinical outcomes:\ntop-8 vs background',
                  fontsize=9, fontweight='bold')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis='y')
    
    for xi, (t, b) in enumerate(zip(top_pcts, bg_pcts)):
        ax3.text(xi - w/2, t+1, f'{t:.0f}', ha='center', fontsize=7)
        ax3.text(xi + w/2, b+1, f'{b:.0f}', ha='center', fontsize=7)
    
    # --- Panel (4): ZERO activation reference ECGs ---
    ax4 = fig.add_subplot(gs[4, 3])
    ax4.axis('off')
    
    zero_lines = ["ZERO-activation reference ECGs",
                  "(atom does NOT fire on these)\n"]
    for i, zi in enumerate(zero_idx):
        r = feat.iloc[zi]; c = clin.iloc[zi]
        age = f"{c['age_at_ecg']:.0f}" if pd.notna(c['age_at_ecg']) else "?"
        sex = str(c['gender'])[:1] if pd.notna(c['gender']) else "?"
        qrs = f"{r['qrs_duration']:.0f}" if not pd.isna(r['qrs_duration']) else "?"
        hr = f"{r['heart_rate']:.0f}" if not pd.isna(r['heart_rate']) else "?"
        # main diagnosis (first non-empty line)
        main_dx = "(no report)"
        for ri in range(18):
            v = r.get(f'report_{ri}', None)
            if pd.notna(v) and str(v).strip():
                main_dx = str(v).strip()[:40]
                break
        
        zero_lines.append(f"#{i+1}: age={age}, sex={sex}, HR={hr}, QRS={qrs}ms")
        zero_lines.append(f"    {main_dx}\n")
    
    ax4.text(0.0, 0.98, "\n".join(zero_lines),
             transform=ax4.transAxes, fontsize=8,
             verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor='#f0f0f0', alpha=0.85,
                       edgecolor='gray', linewidth=1))
    
    plt.suptitle(label_info['label'], fontsize=15, fontweight='bold', y=0.995)
    out_path = out_dir / f"atom_{atom_id}_{label_info['short']}.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  saved: {out_path.name}")


# ============================================================
# Run for all 5 hero atoms
# ============================================================
for atom_id, info in HERO_ATOMS.items():
    try:
        build_atom_panel(atom_id, info)
    except Exception as e:
        print(f"  ERROR atom {atom_id}: {e}")
        import traceback; traceback.print_exc()

print(f"\nDone. Output in: {out_dir}")
print(f"Files:")
for f in sorted(out_dir.glob('atom_*.png')):
    print(f"  {f.name}")
