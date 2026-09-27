"""
Stage 5d — TARNet Uplift Model (PyTorch).

Treatment-Agnostic Representation Network (Shalit et al., 2017).
A deep learning architecture designed for counterfactual estimation:
  - Shared representation Phi(X): maps features into a common latent space,
    learning representations from ALL units (treatment and control).
  - Branching heads:
      mu0_head(Phi(X)): predicts control outcome Y0
      mu1_head(Phi(X)): predicts treatment outcome Y1
  - CATE prediction: tau(x) = mu1(Phi(x)) - mu0(Phi(x))

Key Advantages:
  1. Solves the sample-imbalance problem: the shared trunk trains on all 14M rows,
     so the control arm benefits from feature representations learned from treatment.
  2. Balanced arm loss: gradients for mu0 and mu1 are normalized per arm,
     preventing the 85% treatment arm from dominating control head updates.
  3. Inductive bias: shared representation prevents heads from learning wildly
     divergent latent feature maps.

Run:
    python -m src.models.tarnet --which sample   # dev (~1-2 min)
    python -m src.models.tarnet --which full     # final full dataset (~8-12 min)
"""
import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config.config import RANDOM_SEED, SCALER_PATH, SPLITS_DIR
from src.models.evaluate_uplift import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODELS_DIR = SCALER_PATH.parent


def seed_everything(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TARNet(nn.Module):
    """
    Treatment-Agnostic Representation Network.
    Shared representation layers -> separate potential outcome heads.
    """
    def __init__(self, in_features: int = 12, shared_dim: int = 64, head_dim: int = 32,
                 dropout: float = 0.10, base_rate: float = 0.0029):
        super().__init__()

        # Shared representation trunk Phi(X)
        self.shared = nn.Sequential(
            nn.Linear(in_features, shared_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, shared_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        # Control potential outcome head mu0
        self.head0 = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ELU(),
            nn.Linear(head_dim, 1),
        )

        # Treatment potential outcome head mu1
        self.head1 = nn.Sequential(
            nn.Linear(shared_dim, head_dim),
            nn.ELU(),
            nn.Linear(head_dim, 1),
        )

        # Initialize output biases to empirical log-odds of base rate for rapid calibration
        init_bias = math.log(base_rate / (1.0 - base_rate))
        nn.init.constant_(self.head0[-1].bias, init_bias)
        nn.init.constant_(self.head1[-1].bias, init_bias)

    def forward(self, x: torch.Tensor):
        phi = self.shared(x)
        logit0 = self.head0(phi).squeeze(-1)
        logit1 = self.head1(phi).squeeze(-1)
        return logit0, logit1

    @torch.no_grad()
    def predict_cate(self, x: torch.Tensor) -> np.ndarray:
        self.eval()
        logit0, logit1 = self.forward(x)
        p0 = torch.sigmoid(logit0).cpu().numpy()
        p1 = torch.sigmoid(logit1).cpu().numpy()
        return p1 - p0


def compute_tarnet_loss(logit0: torch.Tensor, logit1: torch.Tensor,
                       treatment: torch.Tensor, outcome: torch.Tensor,
                       bce: nn.BCEWithLogitsLoss) -> torch.Tensor:
    """
    Balanced loss across arms: averages loss separately per arm so the 85% treatment
    group doesn't overwhelm the 15% control group.
    """
    ctrl_mask = (treatment == 0)
    treat_mask = (treatment == 1)

    loss_ctrl = bce(logit0[ctrl_mask], outcome[ctrl_mask]) if ctrl_mask.any() else torch.tensor(0.0, device=logit0.device)
    loss_treat = bce(logit1[treat_mask], outcome[treat_mask]) if treat_mask.any() else torch.tensor(0.0, device=logit1.device)

    return 0.5 * loss_ctrl + 0.5 * loss_treat


def load_data_split(which: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = SPLITS_DIR / f"{which}_{split}.npz"
    log.info("Loading %s ...", path.name)
    data = np.load(path)
    return data["X"], data["treatment"].astype(np.float32), data["conversion"].astype(np.float32)


def train_tarnet(model: TARNet, train_loader: DataLoader, val_loader: DataLoader,
                 epochs: int = 8, lr: float = 1e-3, device: str = "cpu") -> TARNet:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    bce = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience = 3
    patience_counter = 0

    log.info("Starting TARNet training (%d epochs, device=%s) ...", epochs, device)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x_b, t_b, y_b in train_loader:
            x_b, t_b, y_b = x_b.to(device), t_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logit0, logit1 = model(x_b)
            loss = compute_tarnet_loss(logit0, logit1, t_b, y_b, bce)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(1, n_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for xv_b, tv_b, yv_b in val_loader:
                xv_b, tv_b, yv_b = xv_b.to(device), tv_b.to(device), yv_b.to(device)
                logit0_v, logit1_v = model(xv_b)
                loss_v = compute_tarnet_loss(logit0_v, logit1_v, tv_b, yv_b, bce)
                val_loss += loss_v.item()
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        elapsed = time.time() - t0
        scheduler.step(avg_val_loss)

        log.info(
            "Epoch %d/%d (%.1fs) | Train Loss: %.6f | Val Loss: %.6f | LR: %.2e",
            epoch, epochs, elapsed, avg_train_loss, avg_val_loss, optimizer.param_groups[0]["lr"]
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info("Early stopping triggered at epoch %d.", epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def run(which: str, epochs: int = 8, batch_size: int = 4096, lr: float = 1e-3, n_boot: int = 200):
    seed_everything(RANDOM_SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("=== Stage 5d: TARNet (PyTorch) — %s (Device: %s) ===", which, device)

    # 1. Load splits
    X_train, t_train, y_train = load_data_split(which, "train")
    X_val, t_val, y_val = load_data_split(which, "val")
    X_test, t_test, y_test = load_data_split(which, "test")

    log.info("Data loaded: train=%s, val=%s, test=%s",
             f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}")

    base_rate = float(y_train.mean())
    log.info("Empirical base conversion rate: %.5f", base_rate)

    # 2. DataLoaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(t_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(t_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Model
    model = TARNet(in_features=X_train.shape[1], shared_dim=64, head_dim=32, dropout=0.10, base_rate=base_rate)
    trained_model = train_tarnet(model, train_loader, val_loader, epochs=epochs, lr=lr, device=device)

    # 4. Predict CATE on Test Set
    log.info("Predicting CATE on test set (%s rows) ...", f"{len(X_test):,}")
    t0 = time.time()
    test_tensor = torch.from_numpy(X_test).to(device)
    
    # Process in chunks if needed to avoid CPU memory spikes
    chunk_size = 131072
    tau_chunks = []
    for i in range(0, len(test_tensor), chunk_size):
        chunk = test_tensor[i:i + chunk_size]
        tau_chunks.append(trained_model.predict_cate(chunk))
    tau_test = np.concatenate(tau_chunks)
    log.info("Prediction completed in %.1fs", time.time() - t0)

    # 5. Save model and tau_test
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"tarnet_{which}.pt"
    tau_path = MODELS_DIR / f"tarnet_{which}_tau_test.npy"

    torch.save(trained_model.state_dict(), model_path)
    np.save(tau_path, tau_test)
    log.info("Saved model weights → %s", model_path.name)
    log.info("Saved tau_test → %s", tau_path.name)

    # 6. Evaluate via shared evaluate_uplift (FR-11, FR-12)
    log.info("Evaluating TARNet via shared evaluate_uplift...")
    metrics = evaluate(model_name="tarnet", which=which, tau_path=str(tau_path), n_boot=n_boot)

    log.info("=== Stage 5d Complete for '%s' ===", which)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5d: TARNet Uplift Model")
    parser.add_argument("--which", choices=["full", "sample"], required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()
    run(args.which, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, n_boot=args.n_boot)
