# Reproduction guide

## Prerequisites
1. PhysioNet credentialed access
2. CSFM-Tiny weights
3. GPU with at least 24 GB VRAM

## Setup
Edit config.py with your paths, then run scripts in src/ sequentially.

## Timing (A100)
- 02_extract_embedding: ~6h
- 03_train_sae: ~30m
- 04-06: ~15m
- 07-09: ~35m
- 10-14: ~15m
- grid_search: ~35m
