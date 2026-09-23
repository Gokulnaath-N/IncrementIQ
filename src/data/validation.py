"""
Stage 1 — Validation.

Checks the ingested Parquet against the documented dataset stats:
  rows ~13,979,592 | treatment ratio ~0.85 | visit rate ~0.047 | conversion rate ~0.0029

Writes reports/exports/ingestion_validation.json. Run this on FULL_PARQUET after
ingestion, and again on SAMPLE_PARQUET to sanity-check the sample is representative.

Run:
    python -m src.data.validation --which full
    python -m src.data.validation --which sample
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.config import (
    FEATURE_COLS,
    FULL_PARQUET,
    LABEL_COLS,
    SAMPLE_PARQUET,
    TREATMENT_COL,
)
from src.data.ingestion import get_spark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

EXPECTED = {
    "treatment_ratio": 0.846,
    "visit_rate": 0.04699,
    "conversion_rate": 0.00292,
}
TOLERANCE = 0.02  # absolute tolerance on ratios


def run(which: str):
    path = FULL_PARQUET if which == "full" else SAMPLE_PARQUET
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run ingestion first.")

    spark = get_spark("criteo-validation")
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(str(path))
    n = df.count()
    log.info("%s: %s rows", which, f"{n:,}")

    report = {"dataset": which, "row_count": n, "checks": {}}

    expected_cols = set(FEATURE_COLS + [TREATMENT_COL] + LABEL_COLS)
    actual_cols = set(df.columns)
    report["checks"]["schema_matches"] = expected_cols == actual_cols
    report["checks"]["missing_columns"] = list(expected_cols - actual_cols)
    report["checks"]["unexpected_columns"] = list(actual_cols - expected_cols)

    null_counts = df.select(
        [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    ).collect()[0].asDict()
    report["checks"]["null_counts"] = null_counts
    report["checks"]["any_nulls"] = any(v > 0 for v in null_counts.values())

    agg = df.agg(
        F.mean(TREATMENT_COL).alias("treatment_ratio"),
        F.mean("visit").alias("visit_rate"),
        F.mean("conversion").alias("conversion_rate"),
    ).collect()[0].asDict()

    for key, expected_val in EXPECTED.items():
        actual_val = agg[key]
        within_tol = abs(actual_val - expected_val) <= TOLERANCE
        report["checks"][key] = {
            "expected": expected_val,
            "actual": round(actual_val, 5),
            "within_tolerance": within_tol,
        }
        if not within_tol:
            log.warning("%s out of tolerance: expected %.5f, got %.5f", key, expected_val, actual_val)

    feature_ranges = df.select(
        [F.min(c).alias(f"{c}_min") for c in FEATURE_COLS]
        + [F.max(c).alias(f"{c}_max") for c in FEATURE_COLS]
    ).collect()[0].asDict()
    report["checks"]["feature_ranges"] = feature_ranges

    out_path = Path("reports/exports")
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / f"ingestion_validation_{which}.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info("Validation report written to %s", out_file)

    all_passed = (
        report["checks"]["schema_matches"]
        and not report["checks"]["any_nulls"]
        and all(report["checks"][k]["within_tolerance"] for k in EXPECTED)
    )
    log.info("ALL CHECKS PASSED: %s", all_passed)

    spark.stop()
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    args = parser.parse_args()
    passed = run(args.which)
    sys.exit(0 if passed else 1)
