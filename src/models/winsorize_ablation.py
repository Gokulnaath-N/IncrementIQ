"""
Winsorization ablation (SRS O5 sensitivity analysis). Clips a model's tau to
the [1st, 99th] percentile range -- capping the extreme outlier predictions
identified in the decile-10 diagnostic -- and recomputes Qini/AUUC/decile
table via the SAME shared evaluator used for the main leaderboard, so the
comparison is apples-to-apples.

This does NOT retrain the model. It only asks: "if we simply refuse to trust
the model's most extreme tail predictions when RANKING users, does the
decile table become monotonic, and does Qini improve or get worse?"

Run:
    python -m src.models.winsorize_ablation --model t_learner --which full \
        --tau-path models/t_learner_full_tau_test.npy
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run(model_name: str, which: str, tau_path: str, lower_pct: float = 1.0, upper_pct: float = 99.0):
    from src.config.config import SPLITS_DIR
    from src.models.evaluate_uplift import (
        qini_coefficient, auuc_score, uplift_at_k, decile_table,
        evaluate_random_baseline_sanity, bootstrap_ci,
    )

    test_data = np.load(SPLITS_DIR / f"{which}_test.npz")
    treatment, outcome = test_data["treatment"], test_data["conversion"]
    tau_raw = np.load(tau_path)

    lo_val, hi_val = np.percentile(tau_raw, [lower_pct, upper_pct])
    tau_clipped = np.clip(tau_raw, lo_val, hi_val)
    n_clipped_low = int((tau_raw < lo_val).sum())
    n_clipped_high = int((tau_raw > hi_val).sum())

    log.info("Clipping tau to [%.6f, %.6f] (%.0fth/%.0fth pct)", lo_val, hi_val, lower_pct, upper_pct)
    log.info("Rows clipped: %d below, %d above (%.3f%% of %d total)",
              n_clipped_low, n_clipped_high, 100 * (n_clipped_low + n_clipped_high) / len(tau_raw), len(tau_raw))

    results = {}
    for label, tau in [("raw", tau_raw), ("winsorized", tau_clipped)]:
        qini = qini_coefficient(tau, treatment, outcome)
        auuc = auuc_score(tau, treatment, outcome)
        up10 = uplift_at_k(tau, treatment, outcome, 0.10)
        deciles = decile_table(tau, treatment, outcome)
        results[label] = {"qini": round(qini, 6), "auuc": round(auuc, 6), "uplift_at_10pct": round(up10, 6), "decile_table": deciles}
        log.info("[%s] Qini=%.6f  AUUC=%.6f  Uplift@10=%.6f", label, qini, auuc, up10)

    log.info("=" * 70)
    log.info("DECILE 9 vs 10 COMPARISON (raw vs winsorized)")
    for d_idx in [8, 9]:  # 0-indexed: deciles 9 and 10
        raw_d = results["raw"]["decile_table"][d_idx]
        win_d = results["winsorized"]["decile_table"][d_idx]
        log.info("Decile %d | raw: mean_tau=%+.6f obs_lift=%+.6f | winsorized: mean_tau=%+.6f obs_lift=%+.6f",
                  d_idx + 1, raw_d["mean_predicted_tau"], raw_d["observed_lift"] or 0,
                  win_d["mean_predicted_tau"], win_d["observed_lift"] or 0)

    qini_delta = results["winsorized"]["qini"] - results["raw"]["qini"]
    log.info("=" * 70)
    log.info("Qini delta (winsorized - raw): %+.6f", qini_delta)
    log.info("Verdict: %s", "winsorizing HELPED" if qini_delta > 0 else "winsorizing HURT or no change")

    out_dir = Path("reports/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"winsorize_ablation_{model_name}_{which}.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": model_name, "dataset": which,
            "clip_bounds": [round(float(lo_val), 6), round(float(hi_val), 6)],
            "n_clipped": {"low": n_clipped_low, "high": n_clipped_high},
            "results": results,
            "qini_delta_winsorized_minus_raw": round(qini_delta, 6),
        }, f, indent=2)
    log.info("Written to %s", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    parser.add_argument("--tau-path", required=True)
    parser.add_argument("--lower-pct", type=float, default=1.0)
    parser.add_argument("--upper-pct", type=float, default=99.0)
    args = parser.parse_args()
    run(args.model, args.which, args.tau_path, args.lower_pct, args.upper_pct)
