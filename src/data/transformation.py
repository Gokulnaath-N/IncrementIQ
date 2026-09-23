"""
Stage 2 — Covariate Balance (SMD).

Computes Standardized Mean Difference for every feature f0-f11 between the
treatment and control groups. SMD < 0.1 is the conventional "balanced"
threshold — this table is the evidence that randomization actually worked.

Everything here is a Spark groupBy/agg (returns 2 rows x 24 numbers). The
full 14M-row dataframe is NEVER collected to the driver.

Run:
    python -m src.data.transformation --which full
    python -m src.data.transformation --which sample
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path

from pyspark.sql import functions as F

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.config import FEATURE_COLS, FULL_PARQUET, SAMPLE_PARQUET, TREATMENT_COL
from src.data.ingestion import get_spark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SMD_BALANCED_THRESHOLD = 0.1


def compute_smd_table(df) -> list[dict]:
    """
    Returns one dict per feature: {feature, mean_treat, mean_ctrl, var_treat,
    var_ctrl, smd, balanced}. Uses a single groupBy pass (not one query per
    feature) so this is O(1) Spark jobs regardless of feature count.
    """
    agg_exprs = []
    for c in FEATURE_COLS:
        agg_exprs.append(F.mean(c).alias(f"{c}__mean"))
        agg_exprs.append(F.variance(c).alias(f"{c}__var"))

    # Small result: 2 rows (treatment=0, treatment=1) x 24 columns.
    grouped = df.groupBy(TREATMENT_COL).agg(*agg_exprs).collect()

    stats_by_group = {row[TREATMENT_COL]: row.asDict() for row in grouped}
    treat_stats = stats_by_group[1]
    ctrl_stats = stats_by_group[0]

    rows = []
    for c in FEATURE_COLS:
        mean_t = treat_stats[f"{c}__mean"]
        mean_c = ctrl_stats[f"{c}__mean"]
        var_t = treat_stats[f"{c}__var"]
        var_c = ctrl_stats[f"{c}__var"]

        pooled_sd = math.sqrt((var_t + var_c) / 2)
        smd = (mean_t - mean_c) / pooled_sd if pooled_sd > 0 else 0.0

        rows.append(
            {
                "feature": c,
                "mean_treatment": round(mean_t, 6),
                "mean_control": round(mean_c, 6),
                "var_treatment": round(var_t, 6),
                "var_control": round(var_c, 6),
                "smd": round(smd, 6),
                "balanced": abs(smd) < SMD_BALANCED_THRESHOLD,
            }
        )
    return rows


def run(which: str):
    path = FULL_PARQUET if which == "full" else SAMPLE_PARQUET
    spark = get_spark("criteo-transformation")
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(str(path))
    log.info("Computing SMD table on %s dataset", which)

    smd_rows = compute_smd_table(df)

    n_unbalanced = sum(1 for r in smd_rows if not r["balanced"])
    log.info("%d/%d features unbalanced (|SMD| >= %.1f)", n_unbalanced, len(smd_rows), SMD_BALANCED_THRESHOLD)
    for r in smd_rows:
        flag = "" if r["balanced"] else "  <-- FLAGGED"
        log.info("  %-6s smd=%+.4f%s", r["feature"], r["smd"], flag)

    out_dir = Path("reports/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"balance_table_{which}.json"
    with open(out_file, "w") as f:
        json.dump(
            {"dataset": which, "smd_threshold": SMD_BALANCED_THRESHOLD, "features": smd_rows},
            f, indent=2,
        )
    log.info("Balance table written to %s", out_file)

    spark.stop()
    return smd_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    args = parser.parse_args()
    run(args.which)
