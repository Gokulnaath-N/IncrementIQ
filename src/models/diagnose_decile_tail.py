"""
Diagnoses the decile-10 non-monotonicity flagged in the Stage 5 leaderboard.
Two questions:
  1. Is the extreme decile-10 mean tau driven by a few outlier predictions
     (mean >> median in magnitude), or is it a genuinely broad shift?
  2. Is decile-10's reversed observed lift even distinguishable from zero/
     noise, given how few conversions land in any single decile (~200k rows
     at a 0.29% base rate is only ~600 conversions total, spread across both
     arms of one decile)?

Run:
    python -m src.models.diagnose_decile_tail --which full --tau-path models/causal_forest_full_500k_tau_test.npy --model causal_forest
"""
import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run(model_name: str, which: str, tau_path: str, target_deciles=(1, 9, 10)):
    from src.config.config import SPLITS_DIR

    test_data = np.load(SPLITS_DIR / f"{which}_test.npz")
    treatment, outcome = test_data["treatment"], test_data["conversion"]
    tau = np.load(tau_path)

    n = len(tau)
    order = np.argsort(-tau)
    tau_o, t_o, y_o = tau[order], treatment[order], outcome[order]
    edges = np.linspace(0, n, 11, dtype=int)

    rng = np.random.RandomState(42)

    for d in target_deciles:
        lo, hi = edges[d - 1], edges[d]
        tau_d, t_d, y_d = tau_o[lo:hi], t_o[lo:hi], y_o[lo:hi]

        pct = np.percentile(tau_d, [0, 5, 25, 50, 75, 95, 100])
        n_t, n_c = int(t_d.sum()), int((1 - t_d).sum())
        conv_t, conv_c = int(y_d[t_d == 1].sum()), int(y_d[t_d == 0].sum())

        log.info("=" * 70)
        log.info("DECILE %d  (n=%d, n_treat=%d, n_ctrl=%d)", d, len(tau_d), n_t, n_c)
        log.info("tau percentiles [min, p5, p25, median, p75, p95, max]:")
        log.info("  %s", [round(float(v), 6) for v in pct])
        log.info("mean tau = %.6f  |  median tau = %.6f  <- large gap means a few outliers drive the mean",
                  float(tau_d.mean()), float(np.median(tau_d)))
        log.info("conversions: treat=%d (rate=%.5f)  ctrl=%d (rate=%.5f)",
                  conv_t, conv_t / n_t if n_t else float("nan"),
                  conv_c, conv_c / n_c if n_c else float("nan"))

        if n_t > 0 and n_c > 0:
            observed_lift = y_d[t_d == 1].mean() - y_d[t_d == 0].mean()
            boot = []
            for _ in range(500):
                idx = rng.randint(0, len(tau_d), len(tau_d))
                td_b, yd_b = t_d[idx], y_d[idx]
                if td_b.sum() > 0 and (1 - td_b).sum() > 0:
                    boot.append(yd_b[td_b == 1].mean() - yd_b[td_b == 0].mean())
            lo_ci, hi_ci = np.percentile(boot, [2.5, 97.5])
            log.info("observed_lift = %+.6f  95%% bootstrap CI = [%+.6f, %+.6f]  contains_zero=%s",
                      observed_lift, lo_ci, hi_ci, lo_ci <= 0 <= hi_ci)

    log.info("=" * 70)
    log.info("READ: if decile-10's CI is wide and contains zero (or overlaps decile-9's CI),")
    log.info("the 'reversal' is noise from too few conversions per decile -- not a real effect.")
    log.info("If mean tau >> median tau in magnitude, a handful of outlier predictions are")
    log.info("driving the whole decile's summary stat -- worth reporting as an estimator")
    log.info("instability limitation, and considering winsorizing tau in the tails.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    parser.add_argument("--tau-path", required=True)
    args = parser.parse_args()
    run(args.model, args.which, args.tau_path)
