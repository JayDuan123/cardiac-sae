# Cardiac-SAE

Mechanistically interpretable analysis of cardiac foundation model embeddings via Sparse Autoencoders.

This repository contains code, results, and figures for applying Sparse Autoencoders (SAEs) to a frozen pre-trained ECG foundation model (CSFM-Tiny). We train a BatchTopK SAE on 800K MIMIC-IV ECG embeddings and characterize each of 1,536 dictionary atoms along three independent axes: subject-level ICD phenotypes, machine-extracted report text, and objective numerical ECG measurements.

## Key results

- Dictionary preserves task information: SAE atoms achieve PRS greater than 0.99 for atrial fibrillation, heart failure, and MI relative to the dense embedding baseline.
- Three-axis atom characterization: 75 percent of atoms align with specific report phrases (Stage 12); 29 percent have absolute Spearman r greater than 0.3 with at least one objective ECG measurement (Stage 13).
- Grid search benchmark: 20 SAE configurations (5 sparsity levels times 4 dictionary widths) evaluated across reconstruction, dictionary health, monosemanticity, and downstream task preservation.

## Repository structure

    cardiac-sae/
      src/         Pipeline scripts (stages 1-14 + grid search)
      results/     Quantitative outputs (CSV)
      figures/     Paper-ready figures (PNG)
      docs/        Methodology and reproduction guides
      config.py    Path configuration

## Pipeline

Stage 1: 01_test_pipeline.py - CSFM-Tiny smoke test
Stage 2: 02_extract_embedding.py - Extract 768-D embeddings for 800K ECGs
Stage 3: 03_train_sae.py - Train BatchTopK SAE (K=1536, k=32)
Stage 4: 04_extract_activations.py - Extract sparse activations
Stage 5: 05_analyze_dictionary.py - Dictionary health audit
Stage 6: 06_visualize_atoms.py - Per-atom top-activating ECG plots
Stage 7: 07_join_clinical.py - Join ECGs to ICD diagnoses
Stage 8: 08_explore_clinical.py - Phenotype distribution analysis
Stage 9: 09_probe_all.py - Sparse vs dense probing
Stage 10: 10_atom_phenotype_correlation.py - Atom-phenotype enrichment
Stage 11: 11_visualize_phenotype_atoms.py - Phenotype-atom visualizations
Stage 12: 12_atom_report_labeling.py - Text axis: report-based labels
Stage 13: 13_atom_feature_correlation.py - Numerical axis: Spearman r
Stage 14: 14_visualize_atom_features.py - 6 atom-feature figures
grid_search.py - 20 SAE configurations evaluation
grid_visualize.py - 7 grid search figures

## Setup

    git clone https://github.com/JayDuan123/cardiac-sae.git
    cd cardiac-sae
    conda create -n csfm python=3.11
    conda activate csfm
    pip install -r requirements.txt

## External dependencies

1. MIMIC-IV-ECG v1.0 - https://physionet.org/content/mimic-iv-ecg/ (PhysioNet credentialing required)
2. MIMIC-IV hosp module - https://physionet.org/content/mimiciv/ (for ICD codes)
3. CSFM-Tiny weights - https://github.com/guxiao0822/Cardiac-Sensing-FM

## Methodology

The SAE is trained ONLY on ECG embeddings. No text, no ICD codes enter the training loop. After training, each atom is characterized post-hoc along three orthogonal axes:

- Axis 1 (ICD): subject-level disease phenotypes from EHR
- Axis 2 (Report): ECG-level machine-extracted text phrases
- Axis 3 (Numerical): objective ECG measurements

This separation ensures atom emergence is genuinely unsupervised. See docs/METHODOLOGY.md.

## Related work

- CSFM: https://github.com/guxiao0822/Cardiac-Sensing-FM
- BatchTopK SAE: Bussmann et al., NeurIPS 2024 workshop
- SAEBench: Karvonen et al., ICML 2025
- EEG-SAE: Lehn-Schioler et al.

## License

MIT - see LICENSE file.
