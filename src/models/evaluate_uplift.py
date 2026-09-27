"""
Shared Stage 5/6 uplift evaluator (SRS FR-11, FR-12). Every model -- T-Learner,
X-Learner, CausalForest, TARNet -- calls THIS to get its metrics, so the
leaderboard compares numbers computed one single way.

BUGFIX vs the per-model qini_score() in t_learner.py: the random baseline for
this rate-based uplift curve (mean_treated - mean_control within each top-k
prefix) is a FLAT line at the overall ATE, not a triangle rising to it. Under
a genuinely uninformative tau, this curve sits flat at the ATE for every k --
it does not start at 0. The triangle-baseline formula belongs to the
different, cumulative-SUM Qini curve definition and does not apply here.
Using the triangle baseline gives a useless model a spuriously positive Qini
of ~0.5*ATE. This is proven empirically by evaluate_random_baseline() below,
which shuffles the model's own tau (destroying its ranking signal) and checks
that the corrected Qini collapses to ~0.

Run:
    python -m src.models.evaluate_uplift --model x_learner --which full \
        --tau-path models/x_learner_full_tau_test.npy
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def uplift_curve(tau: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> np.ndarray:
    n = len(tau)
    order = np.argsort(-tau)
    t_o, y_o = treatment[order], outcome[order]

    cum_t = np.cumsum(t_o)
    cum_c = np.cumsum(1 - t_o)

    with np.errstate(divide="ignore", invalid="ignore"):
        curve = np.where(
            (cum_t > 0) & (cum_c > 0),
            np.cumsum(y_o * t_o) / cum_t - np.cumsum(y_o * (1 - t_o)) / cum_c,
            0.0,
        )
    return curve


def auuc_score(tau, treatment, outcome) -> float:
    n = len(tau)
    curve = uplift_curve(tau, treatment, outcome)
    return float(np.trapezoid(curve, dx=1.0 / n))


def qini_coefficient(tau, treatment, outcome) -> float:
    """CORRECTED: baseline is the flat-line ATE integral, computed with the
    SAME trapz call/dx as the model curve so there's no rounding mismatch
    between the two integrals being subtracted."""
    n = len(tau)
    model_auc = auuc_score(tau, treatment, outcome)
    ate = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    random_auc = float(np.trapezoid(np.full(n, ate), dx=1.0 / n))
    return model_auc - random_auc


def uplift_at_k(tau, treatment, outcome, k: float = 0.10) -> float:
    n = len(tau)
    cutoff = int(n * k)
    idx = np.argsort(-tau)[:cutoff]
    t_k, y_k = treatment[idx], outcome[idx]
    if t_k.sum() == 0 or (1 - t_k).sum() == 0:
        return float("nan")
    return float(y_k[t_k == 1].mean() - y_k[t_k == 0].mean())


def bootstrap_ci(tau, treatment, outcome, metric_fn, n_boot: int = 200, seed: int = 42):
    rng = np.random.RandomState(seed)
    n = len(tau)
    boot_vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        boot_vals[i] = metric_fn(tau[idx], treatment[idx], outcome[idx])
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return float(lo), float(hi)


def decile_table(tau, treatment, outcome) -> list[dict]:
    """FR-12: non-cumulative per-decile observed lift, ranked by predicted tau
    descending. Should show roughly monotonic decrease top -> bottom decile."""
    n = len(tau)
    order = np.argsort(-tau)
    t_o, y_o, tau_o = treatment[order], outcome[order], tau[order]

    rows = []
    edges = np.linspace(0, n, 11, dtype=int)
    for d in range(10):
        lo, hi = edges[d], edges[d + 1]
        t_d, y_d = t_o[lo:hi], y_o[lo:hi]
        n_t, n_c = t_d.sum(), (1 - t_d).sum()
        observed_lift = (
            y_d[t_d == 1].mean() - y_d[t_d == 0].mean() if n_t > 0 and n_c > 0 else float("nan")
        )
        rows.append({
            "decile": d + 1,
            "n": int(hi - lo),
            "mean_predicted_tau": round(float(tau_o[lo:hi].mean()), 6),
            "observed_lift": round(float(observed_lift), 6) if not np.isnan(observed_lift) else None,
        })
    return rows


def evaluate_random_baseline_sanity(tau, treatment, outcome, seed: int = 42) -> dict:
    """Shuffle tau to destroy any real ranking signal, then check the
    CORRECTED qini collapses to ~0. This is the empirical proof the baseline
    fix is right, not just an argument on paper."""
    rng = np.random.RandomState(seed)
    shuffled_tau = tau.copy()
    rng.shuffle(shuffled_tau)
    qini_shuffled = qini_coefficient(shuffled_tau, treatment, outcome)
    ate = float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())
    return {
        "shuffled_tau_qini": round(qini_shuffled, 6),
        "ate_for_reference": round(ate, 6),
        "sanity_pass": abs(qini_shuffled) < 0.1 * abs(ate) if ate != 0 else abs(qini_shuffled) < 1e-4,
    }


def evaluate(model_name: str, which: str, tau_path: str, n_boot: int = 200):
    from src.config.config import SPLITS_DIR

    test_data = np.load(SPLITS_DIR / f"{which}_test.npz")
    treatment, outcome = test_data["treatment"], test_data["conversion"]
    tau = np.load(tau_path)

    assert len(tau) == len(treatment), f"Row mismatch: tau={len(tau)} vs test={len(treatment)}"

    log.info("Evaluating %s on %s (N=%s)", model_name, which, f"{len(tau):,}")

    qini = qini_coefficient(tau, treatment, outcome)
    auuc = auuc_score(tau, treatment, outcome)
    up10 = uplift_at_k(tau, treatment, outcome, 0.10)
    up20 = uplift_at_k(tau, treatment, outcome, 0.20)

    log.info("Running bootstrap CIs (n_boot=%d) -- this is the slow part on full data", n_boot)
    qini_ci = bootstrap_ci(tau, treatment, outcome, qini_coefficient, n_boot=n_boot)
    auuc_ci = bootstrap_ci(tau, treatment, outcome, auuc_score, n_boot=n_boot)

    deciles = decile_table(tau, treatment, outcome)
    sanity = evaluate_random_baseline_sanity(tau, treatment, outcome)

    log.info("Qini=%.6f (CI %s)  AUUC=%.6f (CI %s)  Uplift@10=%.6f  Uplift@20=%.6f",
              qini, qini_ci, auuc, auuc_ci, up10, up20)
    log.info("Sanity check (shuffled tau): qini=%.6f  pass=%s", sanity["shuffled_tau_qini"], sanity["sanity_pass"])

    result = {
        "model": model_name, "dataset": which, "n_test": len(tau),
        "qini": round(qini, 6), "qini_95ci": [round(qini_ci[0], 6), round(qini_ci[1], 6)],
        "auuc": round(auuc, 6), "auuc_95ci": [round(auuc_ci[0], 6), round(auuc_ci[1], 6)],
        "uplift_at_10pct": round(up10, 6), "uplift_at_20pct": round(up20, 6),
        "tau_mean": round(float(tau.mean()), 6), "tau_std": round(float(tau.std()), 6),
        "decile_table": deciles,
        "random_baseline_sanity_check": sanity,
    }

    out_dir = Path("reports/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"uplift_metrics_{model_name}_{which}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Written to %s", out_file)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="name for output filename, e.g. t_learner")
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    parser.add_argument("--tau-path", required=True, help="path to the model's tau_test.npy")
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()
    evaluate(args.model, args.which, args.tau_path, args.n_boot)
