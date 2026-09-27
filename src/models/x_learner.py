"""
Stage 5b — X-Learner Uplift Model (XGBoost).

The X-Learner refines the T-Learner by leveraging imputed counterfactual
outcomes — especially valuable when treatment arms are imbalanced (85/15 here).

Algorithm (Kunzel et al., 2019):
  Stage 1:  Train mu0(x) and mu1(x)  [reused from T-Learner — loaded from disk]
  Stage 2:  Compute pseudo-outcomes:
              D1(x) = Y1 - mu0(x)    for treated units  (actual outcome minus CF control pred)
              D0(x) = mu1(x) - Y0    for control units  (CF treatment pred minus actual outcome)
  Stage 3:  Fit tau1 on (X_treat, D1)  — regressor, not classifier
            Fit tau0 on (X_ctrl,  D0)  — regressor, not classifier
  Stage 4:  tau(x) = g(x)*tau0(x) + (1-g(x))*tau1(x)
              where g(x) = P(T=1|X) = propensity score

Propensity choice:
  In a randomized experiment, g(x) = treatment_rate (constant) is unbiased
  and avoids propensity model variance. We use g = 0.85 (empirical rate).
  The weighting down-weights tau1 (large-arm estimate) and up-weights tau0
  (small-arm estimate), correcting for the 85/15 imbalance.

Run:
    python -m src.models.x_learner --which sample   # dev (~30s)
    python -m src.models.x_learner --which full     # final (~5 min)
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.config import RANDOM_SEED, SCALER_PATH, SPLITS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = SCALER_PATH.parent

# XGBoost regressor params for pseudo-outcome fitting
# Using max_depth=4 (shallower than T-Learner) — pseudo-outcomes are noisier,
# deeper trees overfit. min_child_weight=20 is looser since targets are continuous.
XGB_REG_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    tree_method="hist",
    eval_metric="rmse",
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


def load_t_learner_models(which: str):
    mu0_path = MODELS_DIR / f"t_learner_{which}_mu0.joblib"
    mu1_path = MODELS_DIR / f"t_learner_{which}_mu1.joblib"
    if not mu0_path.exists() or not mu1_path.exists():
        raise FileNotFoundError(
            f"T-Learner models not found for '{which}'. "
            f"Run: python -m src.models.t_learner --which {which}"
        )
    log.info("Loading T-Learner mu0 from %s ...", mu0_path.name)
    mu0 = joblib.load(mu0_path)
    log.info("Loading T-Learner mu1 from %s ...", mu1_path.name)
    mu1 = joblib.load(mu1_path)
    return mu0, mu1


def compute_pseudo_outcomes(mu0, mu1, train: dict) -> tuple:
    """
    Stage 2 of X-Learner: compute imputed treatment effects per arm.

    D1(x) = Y1 - mu0(x)  for treated   — 'what extra did treatment add beyond control prediction?'
    D0(x) = mu1(x) - Y0  for control   — 'what would treatment have added over their actual outcome?'
    """
    ctrl_mask  = train["treatment"] == 0
    treat_mask = train["treatment"] == 1

    X_ctrl,  Y_ctrl  = train["X"][ctrl_mask],  train["conversion"][ctrl_mask].astype(float)
    X_treat, Y_treat = train["X"][treat_mask], train["conversion"][treat_mask].astype(float)

    log.info("Computing pseudo-outcomes for %s treated, %s control...",
             f"{treat_mask.sum():,}", f"{ctrl_mask.sum():,}")

    # mu0 predicts P(Y=1|T=0,X) — use prob of positive class
    log.info("Predicting mu0(X_treat) ...")
    mu0_on_treat = mu0.predict_proba(X_treat)[:, 1]
    D1 = Y_treat - mu0_on_treat   # pseudo-outcome for treated arm

    log.info("Predicting mu1(X_ctrl) ...")
    mu1_on_ctrl  = mu1.predict_proba(X_ctrl)[:, 1]
    D0 = mu1_on_ctrl - Y_ctrl     # pseudo-outcome for control arm

    log.info(
        "D1 (treated pseudo-outcome): mean=%.6f  std=%.6f", D1.mean(), D1.std()
    )
    log.info(
        "D0 (control pseudo-outcome): mean=%.6f  std=%.6f", D0.mean(), D0.std()
    )

    return X_treat, D1, X_ctrl, D0


def train_tau_models(X_treat, D1, X_ctrl, D0, val: dict, mu0, mu1) -> tuple:
    """Stage 3: fit tau1 on (X_treat, D1) and tau0 on (X_ctrl, D0)."""
    val_treat_mask = val["treatment"] == 1
    val_ctrl_mask  = val["treatment"] == 0

    Xv_treat = val["X"][val_treat_mask]
    Yv_treat = val["conversion"][val_treat_mask].astype(float)
    Xv_ctrl  = val["X"][val_ctrl_mask]
    Yv_ctrl  = val["conversion"][val_ctrl_mask].astype(float)

    # Val pseudo-outcomes for early stopping
    mu0_on_val_treat = mu0.predict_proba(Xv_treat)[:, 1]
    Dv1 = Yv_treat - mu0_on_val_treat

    mu1_on_val_ctrl = mu1.predict_proba(Xv_ctrl)[:, 1]
    Dv0 = mu1_on_val_ctrl - Yv_ctrl

    log.info("Fitting tau1 (treatment-side CATE) on %s rows ...", f"{len(X_treat):,}")
    t0 = time.time()
    tau1 = XGBRegressor(**XGB_REG_PARAMS)
    tau1.fit(X_treat, D1, eval_set=[(Xv_treat, Dv1)], verbose=False)
    log.info("tau1 fitted in %.1fs  (best iter: %d)", time.time() - t0, tau1.best_iteration)

    log.info("Fitting tau0 (control-side CATE) on %s rows ...", f"{len(X_ctrl):,}")
    t0 = time.time()
    tau0 = XGBRegressor(**XGB_REG_PARAMS)
    tau0.fit(X_ctrl, D0, eval_set=[(Xv_ctrl, Dv0)], verbose=False)
    log.info("tau0 fitted in %.1fs  (best iter: %d)", time.time() - t0, tau0.best_iteration)

    return tau0, tau1


def predict_cate(tau0, tau1, X: np.ndarray, propensity: float) -> np.ndarray:
    """
    Stage 4: tau(x) = g(x)*tau0(x) + (1-g(x))*tau1(x)
    g = P(T=1|X) = propensity score (constant = treatment_rate for RCT).
    """
    t0_pred = tau0.predict(X)
    t1_pred = tau1.predict(X)
    return propensity * t0_pred + (1.0 - propensity) * t1_pred


def qini_score(tau: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
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
    qini = np.trapezoid(uplift_curve, dx=1.0 / n)
    ate  = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    return float(qini - ate * 0.5)


def auuc_score(tau: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
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


def uplift_at_k(tau: np.ndarray, treatment: np.ndarray, outcome: np.ndarray, k: float) -> float:
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
    log.info("=== Stage 5b: X-Learner (XGBoost) — %s ===", which)

    # ── 1. Load splits ─────────────────────────────────────────────────────────
    train = load_split(which, "train")
    val   = load_split(which, "val")
    test  = load_split(which, "test")
    log.info(
        "Data loaded | train=%s  val=%s  test=%s",
        f"{len(train['X']):,}", f"{len(val['X']):,}", f"{len(test['X']):,}",
    )

    # ── 2. Load mu0, mu1 from T-Learner (Stage 1 of X-Learner) ────────────────
    mu0, mu1 = load_t_learner_models(which)

    # ── 3. Compute pseudo-outcomes (Stage 2) ───────────────────────────────────
    X_treat, D1, X_ctrl, D0 = compute_pseudo_outcomes(mu0, mu1, train)

    # ── 4. Train tau0, tau1 regressors (Stage 3) ───────────────────────────────
    tau0, tau1 = train_tau_models(X_treat, D1, X_ctrl, D0, val, mu0, mu1)

    # ── 5. Propensity: constant for RCT (empirical treatment rate) ─────────────
    propensity = float(train["treatment"].mean())
    log.info("Using constant propensity g = %.4f (RCT empirical rate)", propensity)

    # ── 6. Predict CATE (Stage 4) ──────────────────────────────────────────────
    log.info("Predicting CATE on val set...")
    tau_val  = predict_cate(tau0, tau1, val["X"],  propensity)

    log.info("Predicting CATE on test set...")
    tau_test = predict_cate(tau0, tau1, test["X"], propensity)

    # ── 7. Evaluate ────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("=== Evaluation Results (X-Learner / XGBoost) ===")
    for split_name, tau, data in [("val", tau_val, val), ("test", tau_test, test)]:
        T = data["treatment"].astype(int)
        Y = data["conversion"].astype(int)
        q   = qini_score(tau, T, Y)
        au  = auuc_score(tau, T, Y)
        u10 = uplift_at_k(tau, T, Y, k=0.10)
        u20 = uplift_at_k(tau, T, Y, k=0.20)
        log.info(
            "%-5s  Qini=%.6f  AUUC=%.6f  Uplift@10%%=%.6f  Uplift@20%%=%.6f"
            "  tau_mean=%.6f  tau_std=%.6f",
            split_name, q, au, u10, u20, tau.mean(), tau.std(),
        )
    log.info("=" * 70)

    # ── 8. Save ────────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tau0_path = MODELS_DIR / f"x_learner_{which}_tau0.joblib"
    tau1_path = MODELS_DIR / f"x_learner_{which}_tau1.joblib"
    tau_path  = MODELS_DIR / f"x_learner_{which}_tau_test.npy"

    joblib.dump(tau0, tau0_path)
    joblib.dump(tau1, tau1_path)
    np.save(tau_path, tau_test)

    log.info("Saved tau0 → %s", tau0_path.name)
    log.info("Saved tau1 → %s", tau1_path.name)
    log.info("Saved tau_test → %s", tau_path.name)
    log.info("Stage 5b complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5b: X-Learner (XGBoost)")
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    args = parser.parse_args()
    run(args.which)
