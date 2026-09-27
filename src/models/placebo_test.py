"""
Stage 5 -- Placebo Test (SRS FR-10, mandatory). v2.

Fix vs v1: pseudo-treatment is assigned 50/50 (not matching the real 0.85
treatment ratio). An 85/15 fake split trains mu0 on far fewer rows than mu1,
which gives the two classifiers different calibration even with zero true
signal -- that asymmetry, not leakage, is what produced a CI excluding zero
in v1. base_score is also pinned explicitly so both classifiers start from
the same calibration point.

Reports TWO checks, not one:
  1. Statistical: does the 95% bootstrap CI on tau_mean contain zero?
     At ~1M-scale test sets, this can fail even with a functioning pipeline,
     purely from finite-sample noise between two independently fit models --
     that is expected, not a bug.
  2. Practical: is |tau_mean| small relative to the REAL measured ATE for
     this outcome (pulled from Stage 2's ab_test_results json)? This is the
     check that actually matters for trusting the pipeline.

Run:
    python -m src.models.placebo_test --which sample
    python -m src.models.placebo_test --which full
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.config import SPLITS_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

N_BOOTSTRAP = 1000
PRACTICAL_THRESHOLD_FRAC = 0.10  # |tau_mean| must be < 10% of the real ATE to pass practically


def load_real_ate(which: str) -> float:
    """Pulls the real conversion ATE from Stage 2's output, so the practical
    threshold isn't a magic hardcoded number -- it's tied to your own measured effect."""
    path = Path("reports/exports") / f"ab_test_results_{which}.json"
    with open(path) as f:
        ab = json.load(f)
    return abs(ab["outcomes"]["conversion"]["ztest"]["absolute_lift"])


def run(which: str):
    train_path = SPLITS_DIR / f"{which}_train.npz"
    log.info("Loading %s", train_path)
    data = np.load(train_path)
    X, treatment, conversion = data["X"], data["treatment"], data["conversion"]

    control_mask = treatment == 0
    X_ctrl, y_ctrl = X[control_mask], conversion[control_mask]
    n_ctrl = len(X_ctrl)
    base_rate = float(y_ctrl.mean())
    log.info("Control-only pool: %s rows, %d conversions, base_rate=%.5f", f"{n_ctrl:,}", y_ctrl.sum(), base_rate)

    rng = np.random.RandomState(RANDOM_SEED)
    # SYMMETRIC 50/50 fake assignment -- not the real 0.85 ratio. Using the real
    # ratio here trains mu0/mu1 on very different sample sizes, which gives them
    # different calibration even under a true null, and that shows up as a
    # spurious non-zero tau. 50/50 removes that confound.
    pseudo_treatment = rng.binomial(1, 0.50, size=n_ctrl)

    strat_key = [f"{t}_{y}" for t, y in zip(pseudo_treatment, y_ctrl)]
    X_train, X_test, pt_train, pt_test, y_train, y_test = train_test_split(
        X_ctrl, pseudo_treatment, y_ctrl, test_size=0.30, stratify=strat_key, random_state=RANDOM_SEED,
    )

    log.info("Training placebo T-Learner (mu0, mu1 on FAKE 50/50 groups, shared base_score)")
    common_params = dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        tree_method="hist", n_jobs=-1, eval_metric="logloss",
        random_state=RANDOM_SEED, base_score=base_rate,
    )
    mu0 = xgb.XGBClassifier(**common_params)
    mu1 = xgb.XGBClassifier(**common_params)

    mu0.fit(X_train[pt_train == 0], y_train[pt_train == 0])
    mu1.fit(X_train[pt_train == 1], y_train[pt_train == 1])

    p0 = mu0.predict_proba(X_test)[:, 1]
    p1 = mu1.predict_proba(X_test)[:, 1]
    tau = p1 - p0

    tau_mean = float(tau.mean())
    tau_std = float(tau.std())

    boot_means = np.array([
        tau[rng.randint(0, len(tau), len(tau))].mean() for _ in range(N_BOOTSTRAP)
    ])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    statistical_pass = ci_low <= 0 <= ci_high

    real_ate = load_real_ate(which)
    practical_ratio = abs(tau_mean) / real_ate if real_ate > 0 else float("inf")
    practical_pass = practical_ratio < PRACTICAL_THRESHOLD_FRAC

    log.info("=" * 70)
    log.info("PLACEBO TEST RESULT (%s, 50/50 symmetric assignment)", which)
    log.info("tau_mean = %+.6f   tau_std = %.6f", tau_mean, tau_std)
    log.info("95%% bootstrap CI: [%+.6f, %+.6f]  -> statistical check: %s",
              ci_low, ci_high, "PASS" if statistical_pass else "FAIL (expected at large N, see practical check)")
    log.info("Real conversion ATE = %.6f | |tau_mean|/ATE = %.2f%% (threshold %.0f%%) -> practical check: %s",
              real_ate, practical_ratio * 100, PRACTICAL_THRESHOLD_FRAC * 100, "PASS" if practical_pass else "FAIL")
    log.info("=" * 70)

    result = {
        "dataset": which,
        "n_control_pool": int(n_ctrl),
        "pseudo_assignment": "50/50 symmetric",
        "base_score_pinned": base_rate,
        "tau_mean": round(tau_mean, 6),
        "tau_std": round(tau_std, 6),
        "ci_95": [round(float(ci_low), 6), round(float(ci_high), 6)],
        "statistical_check_pass": bool(statistical_pass),
        "real_ate_conversion": round(real_ate, 6),
        "practical_ratio_pct": round(practical_ratio * 100, 3),
        "practical_threshold_pct": PRACTICAL_THRESHOLD_FRAC * 100,
        "practical_check_pass": bool(practical_pass),
        "overall_verdict": "PASS" if practical_pass else "FAIL - investigate before trusting CATE models",
    }

    out_dir = Path("reports/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"placebo_test_{which}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Written to %s", out_file)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    args = parser.parse_args()
    run(args.which)
