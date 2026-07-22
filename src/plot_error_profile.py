"""
plot_error_profile.py

Plots reconstruction error across all test windows for a channel, with the
labeled anomaly region shaded. This is the key diagnostic for a
precision=0.000/recall=0.000 result: it shows whether the model's error
actually rises near the real anomaly (just under an over-strict threshold)
or is flat/uninformative there (a genuinely undertrained or misaligned model).

Usage (run from the project root):
    python src/plot_error_profile.py --channel A-1
"""

import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from model import LSTMAutoencoder, reconstruction_error
from data_loader import load_channel, make_windows, get_anomaly_windows

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def plot_error_profile(channel_id: str, window_size: int = 100, threshold_pct: float = 99.0,
                        feature_indices: list = None) -> None:
    test_arr = load_channel(channel_id, split="test")
    windows = make_windows(test_arr, window_size=window_size)
    x = torch.tensor(windows, dtype=torch.float32)

    model = LSTMAutoencoder(n_features=x.shape[-1])
    model.load_state_dict(torch.load(MODEL_DIR / f"{channel_id}_autoencoder.pt"))
    model.eval()

    scores = reconstruction_error(model, x, feature_indices=feature_indices).numpy()
    threshold = np.percentile(scores, threshold_pct)

    try:
        anomaly_ranges = get_anomaly_windows(channel_id)
    except (FileNotFoundError, ValueError):
        anomaly_ranges = []

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Full profile
    ax = axes[0]
    ax.plot(scores, linewidth=0.7, color="steelblue", label="reconstruction error")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, label=f"{threshold_pct}th pct threshold")
    for start, end in anomaly_ranges:
        ax.axvspan(start, end, color="orange", alpha=0.3, label="labeled anomaly")
    ax.set_title(f"Reconstruction error — channel {channel_id} (full test set)")
    ax.set_xlabel("window index")
    ax.set_ylabel("MSE reconstruction error")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Zoomed view around the first anomaly, if any
    ax2 = axes[1]
    if anomaly_ranges:
        start, end = anomaly_ranges[0]
        pad = (end - start) * 3
        lo, hi = max(0, start - pad), min(len(scores), end + pad)
        ax2.plot(range(lo, hi), scores[lo:hi], linewidth=1.2, color="steelblue")
        ax2.axhline(threshold, color="red", linestyle="--", linewidth=1)
        ax2.axvspan(start, end, color="orange", alpha=0.3)
        ax2.set_title(f"Zoomed view around anomaly [{start}, {end})")
    else:
        ax2.text(0.5, 0.5, "No labeled anomaly for this channel", ha="center", va="center")
    ax2.set_xlabel("window index")
    ax2.set_ylabel("MSE reconstruction error")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{channel_id}_error_profile.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved plot to {out_path}")

    if anomaly_ranges:
        start, end = anomaly_ranges[0]
        window_start = max(0, start - window_size + 1)
        window_end = min(len(scores), end)
        near_anomaly_scores = scores[window_start:window_end]
        if len(near_anomaly_scores):
            print(f"\nScore stats in the anomaly region [{window_start}, {window_end}):")
            print(f"  min={near_anomaly_scores.min():.4f} max={near_anomaly_scores.max():.4f} "
                  f"mean={near_anomaly_scores.mean():.4f}")
            print(f"Overall score stats: min={scores.min():.4f} max={scores.max():.4f} "
                  f"mean={scores.mean():.4f} threshold={threshold:.4f}")
            print(f"Percentile rank of max anomaly-region score: "
                  f"{(scores < near_anomaly_scores.max()).mean() * 100:.1f}th percentile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="A-1")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--threshold-pct", type=float, default=99.0)
    parser.add_argument("--feature-indices", type=int, nargs="*", default=[0],
                         help="Feature column(s) to score on. Default: [0]. Pass with no "
                              "values to score across all features instead.")
    args = parser.parse_args()
    feature_indices = args.feature_indices if args.feature_indices else None
    plot_error_profile(args.channel, window_size=args.window_size, threshold_pct=args.threshold_pct,
                        feature_indices=feature_indices)