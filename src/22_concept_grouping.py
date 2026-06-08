"""
Stage 22: Concept semantic grouping & refined taxonomy.

The Stage 15 atom-level taxonomy treats semantically redundant concepts
(e.g., NUM:tachycardia and TXT:Probable sinus tachycardia) as independent.
This inflates the Entangled fraction because monosemantic atoms enriched
for multiple variants of the same clinical phenomenon are mislabeled.

This stage:
  1. Defines clinical concept GROUPS (manually + by inspection of
     enrichment patterns)
  2. Reclassifies each atom by counting enriched GROUPS, not concepts
  3. Subdivides remaining Entangled into Redundant / Related / Mixed
"""
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)
import config as cfg

SAE_NAME = "batchtopk_tiny_aws_k32_d1536"
sae_dir = cfg.SAE_DIR / SAE_NAME
out_dir = sae_dir / "taxonomy_grouped"
out_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Manually define concept groups based on clinical knowledge
# Each group = one underlying clinical phenomenon
# ============================================================
CONCEPT_GROUPS = {
    'tachycardia': [
        'NUM:tachycardia',
        'TXT:Probable sinus tachycardia',
        'TXT:Probable supraventricular tachycardia',
        'TXT:Extreme tachycardia with wide complex',
    ],
    'bradycardia': [
        'NUM:bradycardia',
        'TXT:Marked sinus bradycardia',
    ],
    'right_axis': [
        'NUM:right_axis',
        'TXT:Severe right axis deviation',
    ],
    'left_axis': [
        'NUM:left_axis',
    ],
    'wide_qrs': [
        'NUM:wide_qrs',
    ],
    'atrial_flutter': [
        'TXT:Atrial flutter',
        'TXT:Atrial flutter with rapid ventricular re',
        'TXT:Atrial flutter with uncontrolled ventric',
        'TXT:Atrial flutter with 4:1 A-V block',
    ],
    'pacing': [
        'TXT:Ventricular pacing',
        'TXT:A-V sequential pacemaker',
        'TXT:Atrial pacing',
        'TXT:Atrial-sensed ventricular-paced rhythm',
    ],
    'junctional': [
        'TXT:Junctional rhythm',
        'TXT:Probable junctional rhythm',
        'TXT:Possible idioventricular rhythm with slo',
    ],
    'infarct_anterolateral': [
        'TXT:Anterolateral infarct - age undetermined',
        'TXT:Anterolateral ST-T changes may be due to',
        'TXT:Extensive infarct - age undetermined',
        'TXT:Possible extensive infarct - age undeter',
        'TXT:Lateral infarct - age undetermined',
    ],
    'st_changes': [
        'TXT:ST elev, probable normal early repol pat',
        'TXT:Lateral ST elevation, CONSIDER ACUTE INF',
        'TXT:Tall T waves - consider acute ischemia o',
    ],
    'bbb_block': [
        'TXT:RBBB and LAFB',
        'TXT:Incomplete RBBB',
    ],
    'pvc': [
        'TXT:Sinus rhythm with bigeminal PVCs',
    ],
    'rwave': [
        'TXT:Abnormal R-wave progression, early trans',
    ],
    'idioventricular': [
        'TXT:Accelerated idioventricular rhythm',
    ],
    'undetermined_rhythm': [
        'TXT:Undetermined rhythm',
    ],
    # ICD as own groups (patient-level, not redundant with TXT)
    'AF_disease': ['ICD:AF'],
    'HF_disease': ['ICD:HF'],
    'MI_disease': ['ICD:MI'],
    'DM_disease': ['ICD:DM'],
    'HTN_disease': ['ICD:HTN'],
}

# Build concept → group map
concept_to_group = {}
for group, concepts in CONCEPT_GROUPS.items():
    for c in concepts:
        concept_to_group[c] = group

print("=" * 70)
print("Stage 22: Concept grouping & refined taxonomy")
print("=" * 70)
print(f"\nConcept groups defined: {len(CONCEPT_GROUPS)}")
print(f"Concepts mapped: {len(concept_to_group)}")

# Find any unmapped concepts
enr = pd.read_csv(sae_dir / "taxonomy" / "enrichment_tests.csv")
all_concepts = enr['concept'].unique()
unmapped = [c for c in all_concepts if c not in concept_to_group]
if unmapped:
    print(f"\nWARNING: {len(unmapped)} concepts not in any group:")
    for c in unmapped[:20]:
        print(f"  · {c}")
    # Assign each unmapped concept to its own group
    for c in unmapped:
        concept_to_group[c] = c.replace(':', '_').replace(' ', '_')[:30]

# ============================================================
# Reclassify atoms by enriched GROUPS
# ============================================================
print("\nReclassifying atoms by group count ...")

# Old taxonomy
tax = pd.read_csv(sae_dir / "taxonomy" / "atom_taxonomy.csv")
D = len(tax)

# For each atom, count distinct enriched groups
enr_pass = enr[enr['enriched']].copy()
enr_pass['group'] = enr_pass['concept'].map(concept_to_group)

atom_groups = enr_pass.groupby('atom_id')['group'].apply(set).to_dict()
atom_concepts = enr_pass.groupby('atom_id')['concept'].apply(list).to_dict()

def reclassify(atom_id, old_cat):
    if old_cat == 'Dead':
        return 'Dead', 0, []
    groups = atom_groups.get(atom_id, set())
    concepts = atom_concepts.get(atom_id, [])
    n_groups = len(groups)
    n_concepts = len(concepts)

    if n_groups == 0:
        return 'Uninformative', 0, []
    if n_groups == 1:
        # All enriched concepts in one group → effectively monosemantic
        return 'Separable', 1, list(groups)

    # n_groups >= 2 → genuine multi-group enrichment
    # Sub-classify based on group composition
    return 'Entangled', n_groups, sorted(groups)

new_cats = []
group_lists = []
n_groups_list = []
for _, row in tax.iterrows():
    aid = row['atom_id']
    cat, ng, grps = reclassify(aid, row['category'])
    new_cats.append(cat)
    n_groups_list.append(ng)
    group_lists.append('; '.join(grps))

tax_new = tax.copy()
tax_new['category_grouped'] = new_cats
tax_new['n_enriched_groups'] = n_groups_list
tax_new['enriched_groups'] = group_lists

# ============================================================
# Sub-classify Entangled into Redundant/Related/Mixed
# (Optional but useful for paper)
# ============================================================
# Define "related" group pairs (clinically related but distinct concepts)
RELATED_PAIRS = {
    ('wide_qrs', 'bbb_block'),         # wide QRS often = BBB
    ('wide_qrs', 'right_axis'),        # less direct but related
    ('atrial_flutter', 'tachycardia'), # flutter is fast rhythm
    ('pacing', 'bbb_block'),           # paced beats look like BBB
    ('infarct_anterolateral', 'st_changes'),
    ('junctional', 'bradycardia'),     # junctional escape is slow
}
RELATED_PAIRS = {tuple(sorted(p)) for p in RELATED_PAIRS}

from itertools import combinations
def entangled_subtype(groups):
    if len(groups) < 2: return ''
    pairs = list(combinations(sorted(groups), 2))
    all_related = all(p in RELATED_PAIRS for p in pairs)
    if all_related:
        return 'Related'
    else:
        return 'Mixed'

subtypes = []
for cat, grps_str in zip(new_cats, group_lists):
    if cat == 'Entangled':
        grps = set(grps_str.split('; ')) if grps_str else set()
        subtypes.append(entangled_subtype(grps))
    else:
        subtypes.append('')
tax_new['entangled_subtype'] = subtypes

# ============================================================
# Compare before/after
# ============================================================
print("\nBEFORE grouping (Stage 15):")
print(tax['category'].value_counts().to_string())
print(f"  D={D}")

print("\nAFTER grouping (Stage 22):")
print(tax_new['category_grouped'].value_counts().to_string())

print("\nMovement:")
print(pd.crosstab(tax['category'], tax_new['category_grouped'],
                  margins=True).to_string())

# Entangled subtype breakdown
ent = tax_new[tax_new['category_grouped'] == 'Entangled']
print(f"\nEntangled subtypes (n={len(ent)}):")
print(ent['entangled_subtype'].value_counts().to_string())

# Save
tax_new.to_csv(out_dir / "atom_taxonomy_grouped.csv", index=False)

# ============================================================
# Final summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL TAXONOMY (concept-grouped + entangled subtyped):")
print("=" * 70)
cats = tax_new['category_grouped'].value_counts()
print(f"\n  Separable      : {cats.get('Separable', 0):>5}  ({100*cats.get('Separable',0)/D:.1f}%)")
ent_red = ((tax_new['category_grouped']=='Entangled') & 
           (tax_new['entangled_subtype']=='Related')).sum()
ent_mix = ((tax_new['category_grouped']=='Entangled') & 
           (tax_new['entangled_subtype']=='Mixed')).sum()
print(f"  Entangled-Related: {ent_red:>3}  ({100*ent_red/D:.1f}%)")
print(f"  Entangled-Mixed  : {ent_mix:>3}  ({100*ent_mix/D:.1f}%)")
print(f"  Uninformative  : {cats.get('Uninformative', 0):>5}  ({100*cats.get('Uninformative',0)/D:.1f}%)")
print(f"  Dead           : {cats.get('Dead', 0):>5}  ({100*cats.get('Dead',0)/D:.1f}%)")

print(f"\nOutput: {out_dir}/atom_taxonomy_grouped.csv")
