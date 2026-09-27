from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_CSV_GZ = ROOT_DIR / "data" / "raw" / "criteo-research-uplift-v2.1.csv.gz"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FULL_PARQUET = PROCESSED_DIR / "full"
SAMPLE_PARQUET = PROCESSED_DIR / "sample_10pct"

FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
LABEL_COLS = ["visit", "conversion"]
LEAK_COL = "exposure"  # post-treatment — never use as a feature

RANDOM_SEED = 42
SAMPLE_FRACTION = 0.10

# ── Stage 4 — Split & Scaling ─────────────────────────────────────────────────
SPLITS_DIR = PROCESSED_DIR / "splits"          # written by feature_engineering.py
SCALER_PATH = ROOT_DIR / "models" / "feature_scaler.joblib"  # fit on train only
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15
