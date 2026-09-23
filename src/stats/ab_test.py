"""
Stage 2 — Power Analysis & Classical A/B Test (SRS Phase 2).

Runs on BOTH outcomes: visit (dense, ~4.7%) and conversion (very sparse,
~0.29%). Conversion CIs will be wide — that's a real, reportable finding
about this dataset, not a bug.

Only aggregated counts (4 numbers per outcome: n_treat, n_ctrl, x_treat,
x_ctrl) ever leave Spark. All statistics run in statsmodels/scipy on the
driver from those counts.

Run:
    python -m src.stats.ab_test --which full
    python -m src.stats.ab_test --which sample
"""
import argparse
import json
import logging
import math
import sys
from pathlib import Path

from pyspark.sql import functions as F
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.config import FULL_PARQUET, SAMPLE_PARQUET, TREATMENT_COL
from src.data.ingestion import get_spark

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ALPHA = 0.05
POWER = 0.80
Z_CRIT_95 = 1.959963985


def get_group_counts(df, outcome_col: str) -> dict:
    """Single Spark aggregation -> 4 numbers. Safe to call per outcome."""
    agg = df.groupBy(TREATMENT_COL).agg(
        F.count("*").alias("n"),
        F.sum(outcome_col).alias("x"),
    ).collect()
    by_group = {row[TREATMENT_COL]: row.asDict() for row in agg}
    return {
        "n_treatment": by_group[1]["n"],
        "x_treatment": int(by_group[1]["x"]),
        "n_control": by_group[0]["n"],
        "x_control": int(by_group[0]["x"]),
    }


def two_proportion_ztest(counts: dict) -> dict:
    n1, x1 = counts["n_treatment"], counts["x_treatment"]
    n0, x0 = counts["n_control"], counts["x_control"]

    p1, p0 = x1 / n1, x0 / n0
    abs_lift = p1 - p0
    rel_lift = abs_lift / p0 if p0 > 0 else float("nan")

    # statsmodels: order = [treatment, control]
    z_stat, p_value = proportions_ztest([x1, x0], [n1, n0])

    # Non-pooled SE for the CI on the difference (standard for reporting a lift interval)
    se_diff = math.sqrt((p1 * (1 - p1)) / n1 + (p0 * (1 - p0)) / n0)
    ci_low = abs_lift - Z_CRIT_95 * se_diff
    ci_high = abs_lift + Z_CRIT_95 * se_diff

    return {
        "p_treatment": round(p1, 6),
        "p_control": round(p0, 6),
        "absolute_lift": round(abs_lift, 6),
        "relative_lift_pct": round(rel_lift * 100, 3) if not math.isnan(rel_lift) else None,
        "z_statistic": round(float(z_stat), 4),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < ALPHA),
        "ci_95_absolute_lift": [round(ci_low, 6), round(ci_high, 6)],
    }


def retrospective_mde(counts: dict) -> dict:
    """
    Given the ACTUAL sample sizes, what's the smallest effect this experiment
    could reliably detect at alpha=0.05, power=0.8? This is retrospective —
    framed correctly, not as a pre-registration power calc.
    """
    n1, n0 = counts["n_treatment"], counts["n_control"]
    p0 = counts["x_control"] / n0

    analysis = NormalIndPower()
    ratio = n1 / n0  # treatment:control allocation ratio (~0.85/0.15 here)

    effect_size_h = analysis.solve_power(
        effect_size=None, nobs1=n0, alpha=ALPHA, power=POWER, ratio=ratio
    )

    # Invert Cohen's h = 2*(asin(sqrt(p1)) - asin(sqrt(p0))) to get p1, holding p0 fixed.
    phi0 = math.asin(math.sqrt(p0))
    phi1 = phi0 + effect_size_h / 2
    p1_mde = math.sin(phi1) ** 2
    mde_absolute = p1_mde - p0

    return {
        "baseline_control_rate": round(p0, 6),
        "cohens_h_detectable": round(float(effect_size_h), 6),
        "mde_absolute": round(mde_absolute, 6),
        "mde_relative_pct": round((mde_absolute / p0) * 100, 3) if p0 > 0 else None,
        "alpha": ALPHA,
        "power": POWER,
        "note": "Retrospective MDE given actual observed sample sizes, not a pre-registration calc.",
    }


def run(which: str):
    path = FULL_PARQUET if which == "full" else SAMPLE_PARQUET
    spark = get_spark("criteo-ab-test")
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(str(path))

    results = {"dataset": which, "outcomes": {}}
    for outcome in ["visit", "conversion"]:
        log.info("Aggregating counts for outcome=%s", outcome)
        counts = get_group_counts(df, outcome)
        log.info("  counts: %s", counts)

        ztest_result = two_proportion_ztest(counts)
        mde_result = retrospective_mde(counts)

        log.info(
            "  %s: p_treat=%.5f p_ctrl=%.5f lift=%+.5f p_value=%.2e sig=%s",
            outcome, ztest_result["p_treatment"], ztest_result["p_control"],
            ztest_result["absolute_lift"], ztest_result["p_value"],
            ztest_result["significant_at_0.05"],
        )

        results["outcomes"][outcome] = {
            "raw_counts": counts,
            "ztest": ztest_result,
            "retrospective_power": mde_result,
        }

    out_dir = Path("reports/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"ab_test_results_{which}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info("A/B test results written to %s", out_file)

    spark.stop()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    args = parser.parse_args()
    run(args.which)
