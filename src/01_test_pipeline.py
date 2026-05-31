"""
端到端验证:加载 CSFM、读一条 ECG、预处理、推理,看 embedding 维度对不对。
跑通这个再做批量抽取。
"""
import sys, os, glob
import numpy as np
import torch
import torch.nn as nn
import wfdb

# 把项目根加到 path
PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)

from network.model import CSFM_model
from utils import preprocess_ecg

import config as cfg

# === 1. 加载模型 ===
print("=" * 60)
print("Step 1: Load CSFM model")
print("=" * 60)

model = CSFM_model(cfg.CSFM_VARIANT)
ckpt = torch.load(cfg.WEIGHT_PATH, map_location='cpu')

# 按 README 的方式过滤 encoder 权重
state_dict = {
    k.replace('encoder.', ''): v
    for k, v in ckpt.items()
    if k.startswith('encoder.') and 'mlp_head' not in k
}

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print(f"Missing keys:    {len(missing)}")
print(f"Unexpected keys: {len(unexpected)}")
if missing[:3]:
    print(f"  missing example: {missing[:3]}")
if unexpected[:3]:
    print(f"  unexpected example: {unexpected[:3]}")

n_params = sum(p.numel() for p in model.parameters())
print(f"Total params: {n_params/1e6:.2f}M")

# 禁用 classification head
model.mlp_head = nn.Identity()
model = model.cuda().eval()

# === 2. 读一条 ECG ===
print("\n" + "=" * 60)
print("Step 2: Read one ECG record")
print("=" * 60)

import pandas as pd

df = pd.read_csv(cfg.DATA_ROOT / "record_list.csv")
df['path'] = df['path'].str.rstrip('/')
print(f"Total records in csv: {len(df)}")

test_path = str(cfg.DATA_ROOT / df.iloc[0]['path'])
print(f"Testing: {test_path}")

rec = wfdb.rdrecord(test_path)
print(f"  Channels: {rec.sig_name}")
print(f"  fs:       {rec.fs} Hz")
print(f"  Length:   {rec.sig_len} samples ({rec.sig_len/rec.fs:.1f}s)")
print(f"  Shape:    {rec.p_signal.shape}")

# NaN check
n_nan = np.isnan(rec.p_signal).sum()
print(f"  NaN:      {n_nan} ({100*n_nan/rec.p_signal.size:.2f}%)")

# === 3. 预处理 ===
print("\n" + "=" * 60)
print("Step 3: Preprocess")
print("=" * 60)

ecg_raw = rec.p_signal.T   # (12, 5000)
print(f"  Raw shape (channels, time): {ecg_raw.shape}")

# 处理 NaN(用 0 填,后面统一做 z-norm 不影响)
if n_nan > 0:
    ecg_raw = np.nan_to_num(ecg_raw, nan=0.0)

ecg_proc = preprocess_ecg(ecg_raw, fs=cfg.ECG_FS_RAW)
print(f"  Processed shape: {ecg_proc.shape}")  # 期望 (12, 2500)

# === 4. CSFM 前向 ===
print("\n" + "=" * 60)
print("Step 4: CSFM forward")
print("=" * 60)

signal = torch.tensor(ecg_proc, dtype=torch.float32).unsqueeze(0).cuda()  # (1, 12, 2500)
print(f"  Input tensor:    {signal.shape}")

if cfg.USE_ALL_12_LEADS:
    channels = np.arange(12)   # [0..11]
else:
    channels = np.array([cfg.LEAD_II_IDX])
    signal = signal[:, [cfg.LEAD_II_IDX], :]  # 只保留 II
print(f"  Channels:        {channels}")
print(f"  Final input:     {signal.shape}")

with torch.no_grad():
    features = model(signal, channels)

print(f"  Output (z_CLS):  {features.shape}")
print(f"  Expected:        (1, 768)")
print(f"  Mean / Std / Range: {features.mean():.4f} / {features.std():.4f} / "
      f"[{features.min():.4f}, {features.max():.4f}]")

# === 5. Sanity check ===
print("\n" + "=" * 60)
print("Step 5: Sanity check")
print("=" * 60)

assert features.shape == (1, cfg.CSFM_DIM), f"Wrong embedding dim! Got {features.shape}"
assert not torch.isnan(features).any(), "NaN in features!"
assert features.std() > 0, "Features are constant!"
print("All checks passed ✓")

print("\n" + "=" * 60)
print("Pipeline verified. Ready for batch extraction.")
print("=" * 60)