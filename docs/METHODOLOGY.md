# Methodology

## Three-axis atom characterization

The SAE dictionary is trained purely on ECG embeddings. After training, each atom is characterized post-hoc along three independent axes.

### Axis 1: Subject-level ICD phenotypes
- Source: MIMIC-IV hosp/diagnoses_icd.csv
- Phenotypes: AF, HF, MI/IHD, DM, HTN
- Method: logistic probe (Stage 9) + per-atom lift enrichment (Stage 10)

### Axis 2: ECG-level machine reports
- Source: machine_measurements.csv report_0 to report_17
- Method (Stage 12): tokenize phrases, compute per-atom phrase lift in top-50 activating ECGs

### Axis 3: Numerical ECG measurements
- Source: machine_measurements.csv numerical columns
- 16 features (8 numerical: heart_rate, pr_interval, qrs_duration, qt_interval, qtc, p_axis, qrs_axis, t_axis; 8 binary: tachycardia, bradycardia, wide_qrs, long_qt, left_axis, right_axis, st_elevation, st_depression)
- Method (Stage 13): per-atom per-feature Spearman r

## SAE architecture
- Foundation: CSFM-Tiny ViT (frozen, 768-D)
- SAE: BatchTopK training, JumpReLU inference
- Default: K=1536 atoms, k=32 active per ECG, AUX-K=256

## Grid search
20 configs: 5 sparsity x 4 dictionary widths. Evaluation covers reconstruction, dictionary health, monosemanticity, downstream tasks, sparse probing.
