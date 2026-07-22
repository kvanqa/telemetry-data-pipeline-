"""
threshold_scan.py

Sweeps threshold percentiles to find the precision/recall/F1 trade-off curve,
instead of guessing one threshold at a time. Also saves two plots useful for
a README results section:

  - threshold_sweep.png: precision/recall/F1 vs. threshold percentile
  - score_distribution.png: histogram of reconstruction error, normal vs.
    anomalous windows — the clearest single plot for showing the model
    actually separates the two classes

Usage (run from the project root):
    python src/threshold_scan.py --channel A-1
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

from model import LSTMAutoencoder, reconstruction_error
from data_loader import load_channel, make_windows, get_anomaly_windows
from evaluate import build_ground_truth, smooth_scores

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def scan(channel_id: str, window_size: int = 100, feature_indices: list = None,
          smoothing_window: int = 15, percentiles: np.ndarray = None) -> None:
    if percentiles is None:
        percentiles = np.arange(80, 99.9, 1.0)

    test_arr = load_channel(channel_id, split="test")
    windows = make_windows(test_arr, window_size=window_size)
    x = torch.tensor(windows, dtype=torch.float32)

    model = LSTMAutoencoder(n_features=x.shape[-1])
    model.load_state_dict(torch.load(MODEL_DIR / f"{channel_id}_autoencoder.pt"))
    model.eval()

    scores = reconstruction_error(model, x, feature_indices=feature_indices).numpy()
    if smoothing_window > 1:
        scores = smooth_scores(scores, smoothing_window)

    anomaly_ranges = get_anomaly_windows(channel_id)
    y_true = build_ground_truth(len(windows), window_size, anomaly_ranges)
    anomaly_rate = y_true.mean() * 100
    print(f"True anomaly rate: {anomaly_rate:.2f}% of windows\n")

    results = []
    for pct in percentiles:
        threshold = np.percentile(scores, pct)
        predictions = (scores > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, predictions, average="binary", zero_division=0
        )
        results.append((pct, threshold, precision, recall, f1, predictions.sum()))

    print(f"{'pct':>6} {'threshold':>10} {'precision':>10} {'recall':>8} {'f1':>8} {'flagged':>8}")
    for pct, threshold, precision, recall, f1, flagged in results:
        print(f"{pct:6.1f} {threshold:10.4f} {precision:10.3f} {recall:8.3f} {f1:8.3f} {flagged:8d}")

    best = max(results, key=lambda r: r[4])
    print(f"\nBest F1: pct={best[0]:.1f} precision={best[2]:.3f} recall={best[3]:.3f} f1={best[4]:.3f}")

    # --- Plot 1: threshold sweep ---
    pcts = [r[0] for r in results]
    precisions = [r[2] for r in results]
    recalls = [r[3] for r in results]
    f1s = [r[4] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(pcts, precisions, marker="o", label="precision")
    plt.plot(pcts, recalls, marker="o", label="recall")
    plt.plot(pcts, f1s, marker="o", label="F1")
    plt.axvline(best[0], color="gray", linestyle="--", alpha=0.6, label=f"best F1 @ {best[0]:.1f}th pct")
    plt.xlabel("threshold percentile")
    plt.ylabel("score")
    plt.title(f"Threshold sweep — channel {channel_id}")
    plt.legend()
    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    sweep_path = OUTPUT_DIR / f"{channel_id}_threshold_sweep.png"
    plt.savefig(sweep_path, dpi=120)
    plt.close()
    print(f"\nSaved {sweep_path}")

    # --- Plot 2: score distribution, normal vs anomalous ---
    plt.figure(figsize=(10, 6))
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    plt.hist(normal_scores, bins=60, alpha=0.6, label=f"normal windows (n={len(normal_scores)})", color="steelblue")
    plt.hist(anomaly_scores, bins=60, alpha=0.6, label=f"anomalous windows (n={len(anomaly_scores)})", color="orange")
    plt.axvline(best[1], color="red", linestyle="--", label=f"best threshold ({best[1]:.4f})")
    plt.xlabel("reconstruction error (smoothed)" if smoothing_window > 1 else "reconstruction error")
    plt.ylabel("count")
    plt.title(f"Score distribution — channel {channel_id}")
    plt.legend()
    plt.tight_layout()
    dist_path = OUTPUT_DIR / f"{channel_id}_score_distribution.png"
    plt.savefig(dist_path, dpi=120)
    plt.close()
    print(f"Saved {dist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="A-1")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--feature-indices", type=int, nargs="*", default=[0])
    parser.add_argument("--smoothing-window", type=int, default=15)
    args = parser.parse_args()
    feature_indices = args.feature_indices if args.feature_indices else None
    scan(args.channel, window_size=args.window_size, feature_indices=feature_indices,
         smoothing_window=args.smoothing_window)