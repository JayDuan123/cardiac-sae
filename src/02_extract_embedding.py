"""
批量抽 80 万条 ECG → CSFM embedding,缓存到磁盘。

改动 v2:
- 加 RUN_TAG 区分不同运行
- 启动时做 N 一致性检查
- 优化 ETA 显示
"""
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import wfdb
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = "/workspace/jay/stsae_project/Cardiac-Sensing-FM"
sys.path.insert(0, PROJECT_ROOT)

from network.model import CSFM_model
from utils import preprocess_ecg
import config as cfg


# ============ Dataset ============
class MIMICECGDataset(Dataset):
    def __init__(self, record_paths, data_root, fs_raw=500):
        self.record_paths = record_paths
        self.data_root = str(data_root)
        self.fs_raw = fs_raw

    def __len__(self):
        return len(self.record_paths)

    def __getitem__(self, idx):
        path = self.record_paths[idx]
        try:
            full_path = os.path.join(self.data_root, path)
            rec = wfdb.rdrecord(full_path)
            ecg = rec.p_signal.T  # (12, 5000)

            if np.isnan(ecg).any():
                ecg = np.nan_to_num(ecg, nan=0.0)
            if ecg.shape[0] != 12:
                raise ValueError(f"Expected 12 leads, got {ecg.shape[0]}")

            ecg_proc = preprocess_ecg(ecg, fs=self.fs_raw)
            return torch.tensor(ecg_proc, dtype=torch.float32), idx, True
        except Exception as e:
            return torch.zeros(12, cfg.ECG_LEN_TARGET), idx, False


def collate_fn(batch):
    signals = torch.stack([b[0] for b in batch])
    indices = [b[1] for b in batch]
    oks = [b[2] for b in batch]
    return signals, indices, oks


def main():
    variant = cfg.CSFM_VARIANT.lower()
    tag = cfg.RUN_TAG

    EMB_FILE  = cfg.EMBEDDING_DIR / f"csfm_{variant}_{tag}_embeddings.npy"
    META_FILE = cfg.EMBEDDING_DIR / f"csfm_{variant}_{tag}_meta.csv"
    DONE_FILE = cfg.EMBEDDING_DIR / f"csfm_{variant}_{tag}_done_idx.npy"
    FAIL_LOG  = cfg.LOG_DIR       / f"csfm_{variant}_{tag}_failures.log"

    print(f"=== Run tag: {tag} ===")
    print(f"EMB_FILE:  {EMB_FILE}")
    print(f"META_FILE: {META_FILE}")
    print(f"DONE_FILE: {DONE_FILE}")

    # === 加载 record_list ===
    print("\nLoading record_list.csv ...")
    df = pd.read_csv(cfg.DATA_ROOT / "record_list.csv")
    df['path'] = df['path'].str.rstrip('/')
    df = df.reset_index(drop=True)
    N = len(df)
    print(f"Total records: {N}")

    # === sanity check:已有缓存的 N 是否一致 ===
    if EMB_FILE.exists():
        size_bytes = EMB_FILE.stat().st_size
        expected_bytes = N * cfg.CSFM_DIM * 2  # float16
        if size_bytes != expected_bytes:
            print(f"\n[ERROR] Existing {EMB_FILE.name} size mismatch:")
            print(f"  on disk:  {size_bytes / 1e9:.3f} GB ({size_bytes // (cfg.CSFM_DIM*2)} records)")
            print(f"  expected: {expected_bytes / 1e9:.3f} GB ({N} records)")
            print(f"  This usually means previous run used a different N (e.g. dry-run).")
            print(f"  Solutions:")
            print(f"    1. Change RUN_TAG in config.py to a new value")
            print(f"    2. Or delete: rm {EMB_FILE} {META_FILE} {DONE_FILE}")
            sys.exit(1)
        if DONE_FILE.exists():
            done_mask = np.load(DONE_FILE)
            if done_mask.shape[0] != N:
                print(f"\n[ERROR] DONE_FILE shape {done_mask.shape} != N={N}")
                sys.exit(1)
        print(f"\nResuming from existing cache.")

    # === 准备 memmap 输出文件 ===
    if not EMB_FILE.exists():
        print(f"\nCreating embedding memmap: {EMB_FILE}")
        print(f"  Allocated size: {N * cfg.CSFM_DIM * 2 / 1e9:.2f} GB float16")
        emb_mmap = np.memmap(EMB_FILE, dtype=np.float16, mode='w+',
                             shape=(N, cfg.CSFM_DIM))
        emb_mmap[:] = 0
        emb_mmap.flush()
        del emb_mmap
        df[['path']].to_csv(META_FILE, index=False)

    emb_mmap = np.memmap(EMB_FILE, dtype=np.float16, mode='r+',
                         shape=(N, cfg.CSFM_DIM))

    # === 断点续算 ===
    if DONE_FILE.exists():
        done_mask = np.load(DONE_FILE)
        n_done = done_mask.sum()
        print(f"Resuming: {n_done}/{N} already done ({100*n_done/N:.2f}%)")
    else:
        done_mask = np.zeros(N, dtype=bool)
        n_done = 0

    todo_idx = np.where(~done_mask)[0]
    if len(todo_idx) == 0:
        print("All records done. Exiting.")
        return
    print(f"To process: {len(todo_idx)} records")

    # === 加载模型 ===
    print(f"\nLoading CSFM-{cfg.CSFM_VARIANT} ...")
    model = CSFM_model(cfg.CSFM_VARIANT)
    ckpt = torch.load(cfg.WEIGHT_PATH, map_location='cpu')
    state_dict = {k.replace('encoder.', ''): v for k, v in ckpt.items()
                  if k.startswith('encoder.') and 'mlp_head' not in k}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    model.mlp_head = nn.Identity()
    model = model.cuda().eval()
    torch.backends.cudnn.benchmark = True

    # === DataLoader ===
    todo_paths = df.iloc[todo_idx]['path'].tolist()
    dataset = MIMICECGDataset(todo_paths, cfg.DATA_ROOT, fs_raw=cfg.ECG_FS_RAW)
    loader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=True if cfg.NUM_WORKERS > 0 else False,
        prefetch_factor=4 if cfg.NUM_WORKERS > 0 else None,
    )

    channels = np.arange(12) if cfg.USE_ALL_12_LEADS else np.array([cfg.LEAD_II_IDX])

    # === 推理 ===
    fail_log = open(FAIL_LOG, 'a')
    n_failed = 0
    n_processed_session = 0
    flush_every = 200

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float16):
        pbar = tqdm(loader, total=len(loader), desc=f"CSFM-{cfg.CSFM_VARIANT}-{tag}")
        for batch_i, (signals, todo_local_idx, oks) in enumerate(pbar):
            signals = signals.cuda(non_blocking=True)
            emb = model(signals, channels)  # (B, 768)
            emb_np = emb.detach().float().cpu().numpy().astype(np.float16)

            for j, (local_i, ok) in enumerate(zip(todo_local_idx, oks)):
                global_i = todo_idx[local_i]
                if ok:
                    emb_mmap[global_i] = emb_np[j]
                    done_mask[global_i] = True
                else:
                    n_failed += 1
                    fail_log.write(f"{df.iloc[global_i]['path']}\n")
                    fail_log.flush()
                n_processed_session += 1

            if (batch_i + 1) % flush_every == 0:
                emb_mmap.flush()
                np.save(DONE_FILE, done_mask)

            elapsed = time.time() - t0
            speed = n_processed_session / elapsed if elapsed > 0 else 0
            remaining = len(todo_idx) - n_processed_session
            eta_min = remaining / speed / 60 if speed > 0 else 0
            pbar.set_postfix({
                "fail": n_failed,
                "rec/s": f"{speed:.1f}",
                "eta_h": f"{eta_min/60:.1f}",
            })

    # 收尾
    emb_mmap.flush()
    np.save(DONE_FILE, done_mask)
    fail_log.close()

    print(f"\n{'='*60}")
    print(f"Done. Total: {N}, Success: {done_mask.sum()}, Failed: {n_failed}")
    print(f"Embeddings: {EMB_FILE}")
    print(f"  shape: ({N}, {cfg.CSFM_DIM}), float16")
    print(f"  size:  {EMB_FILE.stat().st_size / 1e9:.2f} GB")
    print(f"Meta:       {META_FILE}")
    print(f"Done mask:  {DONE_FILE}")
    if n_failed > 0:
        print(f"Failures:   {FAIL_LOG}")


if __name__ == "__main__":
    main()