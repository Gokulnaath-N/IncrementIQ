"""
Stage 5c — Causal Forest (EconML CausalForestDML).

CausalForestDML uses Double Machine Learning (DML) to orthogonalise the CATE
estimate, removing nuisance variation in both Y and T before fitting the forest:

  Step 1: Fit E[Y|X] with a first-stage ML model (XGBoost)
  Step 2: Fit E[T|X] with a propensity model (XGBoost)
  Step 3: Regress residuals:
            (Y - Ŷ) ~ tau(X) * (T - T̂)
          using a causal forest (gradient-boosted regression trees)

Key advantage over T/X-Learner: cross-fitting (k-fold) avoids in-sample bias
in the nuisance models — no val/test leakage, no early-stopping tuning.

Scalability: CausalForestDML does not scale to 14M rows (O(n log n) forest
memory). We subsample SUBSAMPLE_N rows stratified on (treatment, conversion)
from full_train. CATE is predicted on the full test set (2.1M rows).

Run:
    python -m src.models.causal_forest --which sample          # dev (~2 min)
    python -m src.models.causal_forest --which full            # ~5-8 min on 500k
    python -m src.models.causal_forest --which full --n 200000 # faster, less accurate
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBRegressor

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.config import RANDOM_SEED, SCALER_PATH, SPLITS_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = SCALER_PATH.parent

# Default subsample size for full dataset
# 500k rows balances accuracy vs runtime (~5-8 min on CPU)
DEFAULT_SUBSAMPLE = {
    "sample": None,   # use all of sample_train (~980k — still fast for forest)
    "full": 500_000,
}


def load_split(which: str, name: str) -> dict:
    path = SPLITS_DIR / f"{which}_{name}.npz"
    log.info("Loading %s ...", path.name)
    data = np.load(path)
    return {k: data[k] for k in data.files}


def stratified_subsample(data: dict, n: int, seed: int) -> dict:
    """Subsample n rows stratified on (treatment, conversion) to preserve rates."""
    total = len(data["X"])
    if n >= total:
        log.info("Subsample n=%s >= total %s — using full split.", f"{n:,}", f"{total:,}")
        return data

    strat_key = data["treatment"].astype(str) + "_" + data["conversion"].astype(str)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(sss.split(data["X"], strat_key))

    log.info(
        "Subsampled %s → %s rows  (treat_ratio=%.4f  conv_rate=%.5f)",
        f"{total:,}", f"{n:,}",
        data["treatment"][idx].mean(), data["conversion"][idx].mean(),
    )
    return {k: data[k][idx] for k in data}


def qini_score(tau, treatment, outcome) -> float:
    n = len(tau)
    order = np.argsort(-tau)
    t_o, y_o = treatment[order], outcome[order]
    ct = np.cumsum(t_o)
    cc = np.cumsum(1 - t_o)
    with np.errstate(divide="ignore", invalid="ignore"):
        curve = np.where((ct > 0) & (cc > 0),
                         np.cumsum(y_o * t_o) / ct - np.cumsum(y_o * (1 - t_o)) / cc, 0.0)
    qini = np.trapezoid(curve, dx=1.0 / n)
    ate = outcome[treatment == 1].mean() - outcome[treatment == 0].mean()
    return float(qini - ate * 0.5)


def auuc_score(tau, treatment, outcome) -> float:
    n = len(tau)
    order = np.argsort(-tau)
    t_o, y_o = treatment[order], outcome[order]
    ct = np.cumsum(t_o)
    cc = np.cumsum(1 - t_o)
    with np.errstate(divide="ignore", invalid="ignore"):
        curve = np.where((ct > 0) & (cc > 0),
                         np.cumsum(y_o * t_o) / ct - np.cumsum(y_o * (1 - t_o)) / cc, 0.0)
    return float(np.trapezoid(curve, dx=1.0 / n))


def uplift_at_k(tau, treatment, outcome, k: float) -> float:
    n = len(tau)
    idx = np.argsort(-tau)[:int(n * k)]
    T_k, Y_k = treatment[idx], outcome[idx]
    nt, nc = T_k.sum(), (1 - T_k).sum()
    if nt == 0 or nc == 0:
        return float("nan")
    return float(Y_k[T_k == 1].mean() - Y_k[T_k == 0].mean())


def run(which: str, subsample_n: int | None = None):
    log.info("=== Stage 5c: CausalForestDML (EconML) — %s ===", which)

    try:
        from econml.dml import CausalForestDML
    except ImportError:
        log.error("econml not installed. Run: pip install econml")
        sys.exit(1)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    train = load_split(which, "train")
    test  = load_split(which, "test")

    # ── 2. Subsample train if needed ───────────────────────────────────────────
    n_sub = subsample_n if subsample_n is not None else DEFAULT_SUBSAMPLE[which]
    if n_sub is not None:
        train_sub = stratified_subsample(train, n_sub, RANDOM_SEED)
    else:
        train_sub = train
        log.info("Using full train split: %s rows", f"{len(train['X']):,}")

    X_tr = train_sub["X"].astype(np.float64)  # EconML needs float64
    T_tr = train_sub["treatment"].astype(np.float64)
    Y_tr = train_sub["conversion"].astype(np.float64)

    log.info(
        "Training CausalForestDML on %s rows | treat_ratio=%.4f  conv_rate=%.5f",
        f"{len(X_tr):,}", T_tr.mean(), Y_tr.mean(),
    )

    # ── 3. Build CausalForestDML ───────────────────────────────────────────────
    # EconML 0.17 CausalForestDML calls .predict() on BOTH nuisance models,
    # expecting continuous probability outputs from both — never class labels.
    #   model_y → E[Y|X] = P(Y=1|X)   XGBRegressor (MSE → conditional mean)
    #   model_t → E[T|X] = P(T=1|X)   XGBRegressor (MSE → propensity score)
    # Residuals T - T̂ and Y - Ŷ are then used for the causal forest fit.
    model_y = XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", eval_metric="rmse",
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
    )
    model_t = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, tree_method="hist", eval_metric="rmse",
        random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
    )

    cf = CausalForestDML(
        model_y=model_y,
        model_t=model_t,
        n_estimators=200,
        min_samples_leaf=10,
        max_depth=None,
        max_features="auto",
        honest=True,           # honest splitting: prevents overfitting of CATE
        cv=3,                  # 3-fold cross-fitting for nuisance models
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )

    # ── 4. Fit ─────────────────────────────────────────────────────────────────
    log.info("Fitting CausalForestDML (this includes 3-fold nuisance cross-fitting)...")
    t0 = time.time()
    cf.fit(Y_tr, T_tr, X=X_tr)
    elapsed = time.time() - t0
    log.info("CausalForestDML fitted in %.1fs", elapsed)

    # ── 5. Predict CATE on test ────────────────────────────────────────────────
    X_test = test["X"].astype(np.float64)
    T_test = test["treatment"].astype(int)
    Y_test = test["conversion"].astype(int)

    log.info("Predicting CATE on test set (%s rows)...", f"{len(X_test):,}")
    t0 = time.time()
    tau_test = cf.effect(X_test).flatten()
    log.info("Prediction done in %.1fs", time.time() - t0)

    # ── 6. Evaluate ────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("=== Evaluation Results (CausalForestDML) ===")
    q   = qini_score(tau_test, T_test, Y_test)
    au  = auuc_score(tau_test, T_test, Y_test)
    u10 = uplift_at_k(tau_test, T_test, Y_test, k=0.10)
    u20 = uplift_at_k(tau_test, T_test, Y_test, k=0.20)
    log.info(
        "test   Qini=%.6f  AUUC=%.6f  Uplift@10%%=%.6f  Uplift@20%%=%.6f"
        "  tau_mean=%.6f  tau_std=%.6f",
        q, au, u10, u20, tau_test.mean(), tau_test.std(),
    )
    log.info("=" * 70)

    # ── 7. Save ────────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    n_tag = f"{len(X_tr) // 1000}k"
    cf_path  = MODELS_DIR / f"causal_forest_{which}_{n_tag}.joblib"
    tau_path = MODELS_DIR / f"causal_forest_{which}_{n_tag}_tau_test.npy"

    joblib.dump(cf, cf_path)
    np.save(tau_path, tau_test)
    log.info("Saved model → %s", cf_path.name)
    log.info("Saved tau_test → %s", tau_path.name)
    log.info("Stage 5c complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5c: CausalForestDML (EconML)")
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    parser.add_argument(
        "--n", type=int, default=None,
        help="Override subsample size (default: 500k for full, all for sample)",
    )
    args = parser.parse_args()
    run(args.which, subsample_n=args.n)
