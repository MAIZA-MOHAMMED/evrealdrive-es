"""
Evaluation metrics for the SoC LSTM.

Reports both:
    - Per-segment errors (percentage points, pp) — used in Figure 3.
    - Per-trip cumulative errors (Wh) — used in Tables 1 and 4.

Reference:
    Cherif et al. (2026), Sections 3.3 and 5.2.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.lstm_model import SoCLSTM


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
@torch.no_grad()
def predict(model: SoCLSTM, X: np.ndarray, batch_size: int = 256,
            device: str = "cpu") -> np.ndarray:
    """Run batched inference and return predictions as numpy array."""
    model.eval().to(device)
    preds = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32).to(device)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)


# ----------------------------------------------------------------------------
# Full evaluation
# ----------------------------------------------------------------------------
def evaluate_all(model: SoCLSTM, X: np.ndarray, y: np.ndarray,
                 split_name: str = "test") -> dict:
    """Compute all metrics for a single split."""
    y_pred = predict(model, X)
    y_true = y.reshape(-1, 1) if y.ndim == 1 else y

    metrics = {
        "split": split_name,
        "n_samples": int(len(y_true)),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
    return metrics


def print_metrics(m: dict, unit: str = "") -> None:
    """Pretty-print a metrics dict."""
    u = f" {unit}" if unit else ""
    print(f"--- {m['split']} (n={m['n_samples']}) ---")
    print(f"  RMSE : {m['rmse']:.4f}{u}")
    print(f"  MAE  : {m['mae']:.4f}{u}")
    print(f"  R^2  : {m['r2']:.4f}")
    print(f"  MAPE : {m['mape']:.2f} %")


def load_model(ckpt_path: str, device: str = "cpu") -> SoCLSTM:
    """Load a trained SoCLSTM from a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = SoCLSTM(
        input_dim=cfg["input_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate the SoC LSTM.")
    p.add_argument("--ckpt", type=str, default="checkpoints/best_lstm.pt")
    p.add_argument("--data-dir", type=str, default="data/")
    p.add_argument("--out-csv", type=str, default="results/evaluation.csv")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device=device)

    data_dir = Path(args.data_dir)
    results = []
    for split in ["train", "val", "test"]:
        X = np.load(data_dir / f"X_{split}.npy")
        y = np.load(data_dir / f"y_{split}.npy")
        m = evaluate_all(model, X, y, split_name=split)
        print_metrics(m)
        results.append(m)

    # Save to CSV
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[evaluate] results written to {out}")
