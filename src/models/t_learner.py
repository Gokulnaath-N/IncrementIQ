"""
Stage 5a — T-Learner Uplift Model (XGBoost).

The T-Learner is the cheapest, most interpretable uplift baseline:
  - Train mu0(x) on control rows  (treatment == 0)
  - Train mu1(x) on treatment rows (treatment == 1)
  - CATE: tau(x) = mu1(x) - mu0(x)

Works on the full 14M dataset. Runs in ~5-10 min on CPU.

Run:
    python -m src.models.t_learner --which sample   # dev (fast, ~30s)
    python -m src.models.t_learner --which full     # final (~10 min)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.config import RANDOM_SEED, SCALER_PATH, SPLITS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── XGBoost hyperparameters ───────────────────────────────────────────────────
# Conservative defaults: tree_method='hist' is fast on large datasets.
# n_estimators=300 with early stopping is standard for tabular uplift.
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=50,   # conservative — avoids overfitting on imbalanced outcome
    scale_pos_weight=1,    # we handle class imbalance via eval_metric, not upweighting
    tree_method="hist",    # fast histogram method — handles 10M+ rows fine
    eval_metric="logloss",
    early_stopping_rounds=20,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=0,
)


def load_split(which: str, name: str) -> dict:
    path = SPLITS_DIR / f"{which}_{name}.npz"
    log.info("Loading %s ...", path.name)
    data = np.load(path)
    return {k: data[k] for k in data.files}


def train_t_learner(train: dict, val: dict) -> tuple:
    """Train mu0 on control, mu1 on treatment. Return (mu0, mu1)."""
    ctrl_mask  = train["treatment"] == 0
    treat_mask = train["treatment"] == 1

    X_ctrl,  y_ctrl  = train["X"][ctrl_mask],  train["conversion"][ctrl_mask]
    X_treat, y_treat = train["X"][treat_mask], train["conversion"][treat_mask]

    # Validation sets for early stopping
    val_ctrl_mask  = val["treatment"] == 0
    val_treat_mask = val["treatment"] == 1
    Xv_ctrl,  yv_ctrl  = val["X"][val_ctrl_mask],  val["conversion"][val_ctrl_mask]
    Xv_treat, yv_treat = val["X"][val_treat_mask], val["conversion"][val_treat_mask]

    log.info(
        "mu0 training set: %s rows  (%d conversions, %.4f%% rate)",
        f"{len(X_ctrl):,}", y_ctrl.sum(), 100 * y_ctrl.mean(),
    )
    log.info(
        "mu1 training set: %s rows  (%d conversions, %.4f%% rate)",
        f"{len(X_treat):,}", y_treat.sum(), 100 * y_treat.mean(),
    )

    log.info("Fitting mu0 (control model)...")
    t0 = time.time()
    mu0 = XGBClassifier(**XGB_PARAMS)
    mu0.fit(
        X_ctrl, y_ctrl,
        eval_set=[(Xv_ctrl, yv_ctrl)],
        verbose=False,
    )
    log.info("mu0 fitted in %.1fs  (best iteration: %d)", time.time() - t0, mu0.best_iteration)

    log.info("Fitting mu1 (treatment model)...")
    t0 = time.time()
    mu1 = XGBClassifier(**XGB_PARAMS)
    mu1.fit(
        X_treat, y_treat,
        eval_set=[(Xv_treat, yv_treat)],
        verbose=False,
    )
    log.info("mu1 fitted in %.1fs  (best iteration: %d)", time.time() - t0, mu1.best_iteration)

    return mu0, mu1


def predict_cate(mu0, mu1, X: np.ndarray) -> np.ndarray:
    """tau(x) = P(Y=1|T=1,X=x) - P(Y=1|T=0,X=x)"""
    p0 = mu0.predict_proba(X)[:, 1]
    p1 = mu1.predict_proba(X)[:, 1]
    return p1 - p0


def qini_score(tau: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
    """
    Qini coefficient — area between the uplift curve and random baseline,
    normalized by the area of the perfect model.

    Higher is better. 0 = random. 1 = perfect (theoretical upper bound).
    """
    n = len(tau)
    order = np.argsort(-tau)   # descending CATE — target top persuadables first
    treatment_o = treatment[order]
    outcome_o   = outcome[order]

    # Incremental uplift at each rank k
    cumulative_treat = np.cumsum(treatment_o)
    cumulative_ctrl  = np.cumsum(1 - treatment_o)

    # Avoid division by zero in early ranks where one arm may be empty
    with np.errstate(divide="ignore", invalid="ignore"):
        uplift_curve = np.where(
            (cumulative_treat > 0) & (cumulative_ctrl > 0),
            np.cumsum(outcome_o * treatment_o) / cumulative_treat
            - np.cumsum(outcome_o * (1 - treatment_o)) / cumulative_ctrl,
            0.0,
        )

    qini = np.trapezoid(uplift_curve, dx=1.0 / n)

    # Random baseline: flat line at overall ATE
    n_t = treatment.sum()
    n_c = (1 - treatment).sum()
    ate = (outcome[treatment == 1].mean() - outcome[treatment == 0].mean())
    random_qini = ate * 0.5   # triangle under random line

    return float(qini - random_qini)


def auuc_score(tau: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
    """
    Area Under the Uplift Curve (AUUC).
    Measures the raw area under the uplift-at-rank-k curve (unnormalized).
    """
    n = len(tau)
    order = np.argsort(-tau)
    treatment_o = treatment[order]
    outcome_o   = outcome[order]

    cumulative_treat = np.cumsum(treatment_o)
    cumulative_ctrl  = np.cumsum(1 - treatment_o)

    with np.errstate(divide="ignore", invalid="ignore"):
        uplift_curve = np.where(
            (cumulative_treat > 0) & (cumulative_ctrl > 0),
            np.cumsum(outcome_o * treatment_o) / cumulative_treat
            - np.cumsum(outcome_o * (1 - treatment_o)) / cumulative_ctrl,
            0.0,
        )

    return float(np.trapezoid(uplift_curve, dx=1.0 / n))


def uplift_at_k(tau: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, k: float = 0.20) -> float:
    """
    Average CATE for the top-k% of users ranked by predicted tau.
    Business metric: 'if we target the top 20%, what's the expected lift?'
    """
    n = len(tau)
    cutoff = int(n * k)
    top_k_idx = np.argsort(-tau)[:cutoff]
    T_k = treatment[top_k_idx]
    Y_k = outcome[top_k_idx]

    n_t = T_k.sum()
    n_c = (1 - T_k).sum()
    if n_t == 0 or n_c == 0:
        return float("nan")

    return float(Y_k[T_k == 1].mean() - Y_k[T_k == 0].mean())


def run(which: str):
    log.info("=== Stage 5a: T-Learner (XGBoost) — %s ===", which)

    train = load_split(which, "train")
    val   = load_split(which, "val")
    test  = load_split(which, "test")

    log.info(
        "Data loaded | train=%s  val=%s  test=%s",
        f"{len(train['X']):,}", f"{len(val['X']):,}", f"{len(test['X']):,}",
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    mu0, mu1 = train_t_learner(train, val)

    # ── Predict CATE on val & test ────────────────────────────────────────────
    log.info("Predicting CATE on val set...")
    tau_val  = predict_cate(mu0, mu1, val["X"])

    log.info("Predicting CATE on test set...")
    tau_test = predict_cate(mu0, mu1, test["X"])

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("=== Evaluation Results (T-Learner / XGBoost) ===")

    for split_name, tau, data in [("val", tau_val, val), ("test", tau_test, test)]:
        T, Y = data["treatment"].astype(int), data["conversion"].astype(int)
        q  = qini_score(tau, T, Y)
        au = auuc_score(tau, T, Y)
        u20 = uplift_at_k(tau, T, Y, k=0.20)
        u10 = uplift_at_k(tau, T, Y, k=0.10)
        log.info(
            "%-5s  Qini=%.6f  AUUC=%.6f  Uplift@10%%=%.6f  Uplift@20%%=%.6f"
            "  tau_mean=%.6f  tau_std=%.6f",
            split_name, q, au, u10, u20, tau.mean(), tau.std(),
        )

    log.info("=" * 70)

    # ── Save models & predictions ─────────────────────────────────────────────
    models_dir = SCALER_PATH.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    mu0_path = models_dir / f"t_learner_{which}_mu0.joblib"
    mu1_path = models_dir / f"t_learner_{which}_mu1.joblib"
    tau_path = models_dir / f"t_learner_{which}_tau_test.npy"

    joblib.dump(mu0, mu0_path)
    joblib.dump(mu1, mu1_path)
    np.save(tau_path, tau_test)

    log.info("Saved mu0 → %s", mu0_path.name)
    log.info("Saved mu1 → %s", mu1_path.name)
    log.info("Saved tau_test → %s", tau_path.name)
    log.info("Stage 5a complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5a: T-Learner (XGBoost)")
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    args = parser.parse_args()
    run(args.which)
