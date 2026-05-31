"""公共配置"""
from pathlib import Path

PROJECT_ROOT = Path("/workspace/jay/stsae_project/Cardiac-Sensing-FM")
DATA_ROOT    = Path("/workspace/data/mimic-iv-ecg-aws")
WEIGHT_PATH  = PROJECT_ROOT / "CSFM_tiny.pth"

OUTPUT_ROOT = PROJECT_ROOT / "stsae" / "outputs"
EMBEDDING_DIR = OUTPUT_ROOT / "embeddings"
SAE_DIR       = OUTPUT_ROOT / "sae_checkpoints"
LOG_DIR       = OUTPUT_ROOT / "logs"
for d in [OUTPUT_ROOT, EMBEDDING_DIR, SAE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CSFM_VARIANT = "Tiny"
CSFM_DIM = 768

ECG_FS_RAW = 500
ECG_FS_TARGET = 250
ECG_LEN_TARGET = 2500
N_LEADS = 12
USE_ALL_12_LEADS = True
LEAD_II_IDX = 1

BATCH_SIZE = 256
NUM_WORKERS = 12

RUN_TAG = "aws"
