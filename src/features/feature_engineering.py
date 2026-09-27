"""
Stage 4 — Feature Engineering & Split.

Does exactly two things, both load-bearing for every model in Stage 5:
  1. Stratified 70/15/15 split on (treatment, conversion) jointly — NOT just
     treatment, and NOT visit — so the rare conversion outcome is represented
     proportionally in every split, not just the denser visit signal.
  2. StandardScaler fit on TRAIN ONLY, then applied to val/test. Fitting on
     the full dataset before splitting would leak test-set statistics into
     training — small in effect here (anonymized dense features) but a real
     methodology error to avoid on principle.

Test set is written once and should not be touched again until Stage 6 (final
evaluation) — treat it as read-only from this point forward.

Run:
    python -m src.features.feature_engineering --which sample
    python -m src.features.feature_engineering --which full
"""
import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.config import (
    FEATURE_COLS,
    FULL_PARQUET,
    LABEL_COLS,
    RANDOM_SEED,
    SAMPLE_PARQUET,
    SCALER_PATH,
    SPLITS_DIR,
    TEST_FRAC,
    TRAIN_FRAC,
    TREATMENT_COL,
    VAL_FRAC,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9, "Fractions must sum to 1.0"


def load_df(which: str) -> pd.DataFrame:
    path = FULL_PARQUET if which == "full" else SAMPLE_PARQUET
    log.info("Reading parquet from: %s", path)
    df = pd.read_parquet(path)
    # treatment arrives as `category` dtype (Parquet partition column) — cast to int
    df[TREATMENT_COL] = df[TREATMENT_COL].astype(int)
    for c in LABEL_COLS:
        df[c] = df[c].astype(int)
    log.info("Loaded shape: %s   dtypes: %s", df.shape, df.dtypes.to_dict())
    return df


def stratified_three_way_split(df: pd.DataFrame, seed: int):
    """
    70/15/15 split stratified on (treatment, conversion) jointly.

    Stratifying on conversion — not visit — preserves the rarer outcome
    (~0.3%) proportionally across splits. Visit (~4.7%) is dense enough
    that a split balanced on conversion will also balance visit implicitly.
    """
    # Joint stratification key: "0_0", "0_1", "1_0", "1_1"
    strat_key = df[TREATMENT_COL].astype(str) + "_" + df["conversion"].astype(str)

    train_df, temp_df, strat_train, strat_temp = train_test_split(
        df,
        strat_key,
        train_size=TRAIN_FRAC,
        stratify=strat_key,
        random_state=seed,
    )

    # Split remaining (val + test) proportionally
    rel_val = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=rel_val,
        stratify=strat_temp,
        random_state=seed,
    )
    return train_df, val_df, test_df


def log_split_summary(name: str, split_df: pd.DataFrame) -> dict:
    n = len(split_df)
    t_ratio = split_df[TREATMENT_COL].mean()
    visit_rate = split_df["visit"].mean()
    conv_rate = split_df["conversion"].mean()
    conv_count = int(split_df["conversion"].sum())
    log.info(
        "%-6s  n=%9s  treat_ratio=%.4f  visit_rate=%.5f  conv_rate=%.5f  conv_count=%d",
        name, f"{n:,}", t_ratio, visit_rate, conv_rate, conv_count,
    )
    return {
        "n": n,
        "treat_ratio": round(t_ratio, 6),
        "visit_rate": round(visit_rate, 6),
        "conv_rate": round(conv_rate, 6),
        "conv_count": conv_count,
    }


def run(which: str):
    df = load_df(which)
    log.info("Total rows: %s", f"{len(df):,}")

    # ── 1. Stratified split ───────────────────────────────────────────────────
    train_df, val_df, test_df = stratified_three_way_split(df, RANDOM_SEED)

    log.info("=" * 70)
    log.info("Split summary — verify treat_ratio & conv_rate are consistent:")
    summaries = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        summaries[name] = log_split_summary(name, split_df)
    log.info("=" * 70)

    # Warn if conversion counts are dangerously low (Qini/AUUC unstable <50)
    for name, s in summaries.items():
        if s["conv_count"] < 50:
            log.warning(
                "⚠️  %s conv_count=%d < 50 — Qini/AUUC confidence intervals will be "
                "unstable. Consider revisiting split ratios.", name, s["conv_count"]
            )

    # ── 2. Fit scaler on TRAIN ONLY ───────────────────────────────────────────
    log.info("Fitting StandardScaler on train set only...")
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_df[FEATURE_COLS].values)
    val_X   = scaler.transform(val_df[FEATURE_COLS].values)
    test_X  = scaler.transform(test_df[FEATURE_COLS].values)
    log.info(
        "Scaler fitted: mean[:3]=%s  std[:3]=%s",
        scaler.mean_[:3].round(4), scaler.scale_[:3].round(4),
    )

    # ── 3. Save splits as compressed .npz ────────────────────────────────────
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)

    for name, split_df, X in [
        ("train", train_df, train_X),
        ("val",   val_df,   val_X),
        ("test",  test_df,  test_X),
    ]:
        out_path = SPLITS_DIR / f"{which}_{name}.npz"
        np.savez_compressed(
            out_path,
            X=X.astype(np.float32),
            treatment=split_df[TREATMENT_COL].values.astype(np.int8),
            visit=split_df["visit"].values.astype(np.int8),
            conversion=split_df["conversion"].values.astype(np.int8),
        )
        log.info("Wrote %s  (%s rows)", out_path.name, f"{len(split_df):,}")

    # ── 4. Save scaler ────────────────────────────────────────────────────────
    joblib.dump(scaler, SCALER_PATH)
    log.info(
        "Scaler saved → %s  (fit on train only — reuse for any new data, never refit)",
        SCALER_PATH,
    )

    log.info("Stage 4 complete for '%s'.", which)
    log.info("Test set is now READ-ONLY until Stage 6 final evaluation.")
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Feature Engineering & Split")
    parser.add_argument(
        "--which",
        choices=["full", "sample"],
        required=True,
        help="Which dataset to process: 'sample' (10%%, dev) or 'full' (14M, final)",
    )
    args = parser.parse_args()
    run(args.which)
