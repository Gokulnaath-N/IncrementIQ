"""
Finishes the A/B analysis from the raw counts produced by the SageMaker
Processing job (sagemaker_pyspark_job.py -> ab_test_counts.json). This is
the SAME math as src/stats/ab_test.py's two_proportion_ztest/retrospective_mde,
just decoupled from Spark so it can run anywhere with just statsmodels.

Expected input JSON shape:
{
  "visit":      {"n_treatment": int, "x_treatment": int, "n_control": int, "x_control": int},
  "conversion": {"n_treatment": int, "x_treatment": int, "n_control": int, "x_control": int}
}

Run:
    python -m src.stats.ab_test_from_counts --counts-file path/to/ab_test_counts.json
"""
import argparse
import json
import logging
import math
from pathlib import Path

from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

ALPHA, POWER, Z_CRIT_95 = 0.05, 0.80, 1.959963985


def two_proportion_ztest(counts: dict) -> dict:
    n1, x1 = counts["n_treatment"], counts["x_treatment"]
    n0, x0 = counts["n_control"], counts["x_control"]
    p1, p0 = x1 / n1, x0 / n0
    abs_lift = p1 - p0
    z_stat, p_value = proportions_ztest([x1, x0], [n1, n0])
    se_diff = math.sqrt((p1 * (1 - p1)) / n1 + (p0 * (1 - p0)) / n0)
    return {
        "p_treatment": round(p1, 6),
        "p_control": round(p0, 6),
        "absolute_lift": round(abs_lift, 6),
        "relative_lift_pct": round((abs_lift / p0) * 100, 3) if p0 > 0 else None,
        "z_statistic": round(float(z_stat), 4),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < ALPHA),
        "ci_95_absolute_lift": [
            round(abs_lift - Z_CRIT_95 * se_diff, 6),
            round(abs_lift + Z_CRIT_95 * se_diff, 6),
        ],
    }


def retrospective_mde(counts: dict) -> dict:
    n1, n0 = counts["n_treatment"], counts["n_control"]
    p0 = counts["x_control"] / n0
    effect_size_h = NormalIndPower().solve_power(
        effect_size=None, nobs1=n0, alpha=ALPHA, power=POWER, ratio=n1 / n0
    )
    phi0 = math.asin(math.sqrt(p0))
    p1_mde = math.sin(phi0 + effect_size_h / 2) ** 2
    mde_absolute = p1_mde - p0
    return {
        "baseline_control_rate": round(p0, 6),
        "mde_absolute": round(mde_absolute, 6),
        "mde_relative_pct": round((mde_absolute / p0) * 100, 3) if p0 > 0 else None,
        "alpha": ALPHA,
        "power": POWER,
    }


def run(counts_file: str):
    with open(counts_file) as f:
        data = json.load(f)

    # Support either direct dictionary or nested {"counts": {...}}
    all_counts = data.get("counts", data)

    results = {"source": counts_file, "outcomes": {}}
    for outcome, counts in all_counts.items():
        if outcome not in ["visit", "conversion"]:
            continue
        ztest_result = two_proportion_ztest(counts)
        mde_result = retrospective_mde(counts)
        results["outcomes"][outcome] = {
            "raw_counts": counts,
            "ztest": ztest_result,
            "retrospective_power": mde_result,
        }
        log.info(
            "%s: lift=%+.5f p_value=%.3e sig=%s mde_rel=%.2f%%",
            outcome,
            ztest_result["absolute_lift"],
            ztest_result["p_value"],
            ztest_result["significant_at_0.05"],
            mde_result["mde_relative_pct"],
        )

    out_dir = Path("reports/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ab_test_results_from_sagemaker.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Written to %s", out_file)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts-file", required=True)
    args = parser.parse_args()
    run(args.counts_file)
