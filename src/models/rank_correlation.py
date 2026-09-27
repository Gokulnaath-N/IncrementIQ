"""
Stage 5 -- Model Agreement / Spearman Rank Correlation (SRS FR-9).

Loads per-row test-set tau predictions from T-Learner, X-Learner, and
CausalForestDML -- all confirmed positionally aligned to the same
{which}_test.npz row order -- and computes pairwise Spearman correlation.

High agreement (conventionally >0.5-0.6) is evidence the models are
converging on the same real uplift signal rather than each fitting noise
independently. Low agreement doesn't necessarily mean one is wrong, but it
means the "who to target" business recommendation would change materially
depending which model you picked -- worth flagging explicitly either way.

NOTE on CausalForest: it was trained on a 500k subsample while T/X-Learner
trained on the full set, but all three are SCORED on the same full test.npz,
so this comparison is valid despite the different training sizes.

Run:
    python -m src.models.rank_correlation --which full \
        --causal-forest-tag causal_forest_full_500k_tau_test.npy
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def run(which: str, causal_forest_filename: str):
    paths = {
        "t_learner": MODELS_DIR / f"t_learner_{which}_tau_test.npy",
        "x_learner": MODELS_DIR / f"x_learner_{which}_tau_test.npy",
        "causal_forest": MODELS_DIR / causal_forest_filename,
        "tarnet": MODELS_DIR / f"tarnet_{which}_tau_test.npy",
    }

    tau = {}
    for name, path in paths.items():
        if not path.exists():
            log.warning("Missing %s -- skipping %s from correlation matrix", path, name)
            continue
        arr = np.load(path)
        tau[name] = arr
        log.info("Loaded %s: n=%s", name, f"{len(arr):,}")

    lengths = {name: len(arr) for name, arr in tau.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"Row count mismatch across models -- these are NOT the same test set: {lengths}. "
            "Do not proceed with correlation until this is resolved."
        )

    names = list(tau.keys())
    results = {"dataset": which, "n_test": lengths[names[0]], "pairs": {}}

    log.info("=" * 60)
    log.info("SPEARMAN RANK CORRELATION -- pairwise model agreement")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            corr, p_value = spearmanr(tau[a], tau[b])
            key = f"{a}_vs_{b}"
            results["pairs"][key] = {"spearman_corr": round(float(corr), 4), "p_value": float(p_value)}
            agreement = "strong" if corr > 0.6 else "moderate" if corr > 0.3 else "weak"
            log.info("%-30s rho=%.4f  p=%.2e  (%s agreement)", key, corr, p_value, agreement)
    log.info("=" * 60)

    out_dir = Path("reports/exports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"rank_correlation_{which}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Written to %s", out_file)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    parser.add_argument(
        "--causal-forest-tag", required=True,
        help="Exact filename in models/, e.g. causal_forest_full_500k_tau_test.npy",
    )
    args = parser.parse_args()
    run(args.which, args.causal_forest_tag)
