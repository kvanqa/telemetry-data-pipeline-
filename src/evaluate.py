"""
evaluate.py

Loads a trained autoencoder, scores test windows, and compares reconstruction
error against the labeled anomaly windows to compute precision/recall/F1.

Ground truth: each window covers test-array timesteps [i, i + window_size).
A window is labeled anomalous if it overlaps any labeled [start, end] range
from labeled_anomalies.csv for that channel.

Scoring: by default this scores using ONLY feature 0 (--feature-indices 0),
matching the original SMAP/MSL paper's approach. For these channels, feature
0 is the actual telemetry value; the remaining columns are one-hot encoded
command context. Averaging reconstruction error across all features dilutes
a real, localized telemetry deviation with noise from unpredictable command
timing elsewhere in the window — pass --feature-indices with no values (or
edit the default) to score across all features instead, for comparison.

Smoothing: raw reconstruction error can still be noisy even after focusing
on the right feature — a rolling-mean pass before thresholding suppresses
brief spikes while preserving sustained elevations. Both un-smoothed and
smoothed results are printed so you can compare.
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from model import LSTMAutoencoder, reconstruction_error
from data_loader import load_channel, make_windows, get_anomaly_windows

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def build_ground_truth(n_windows: int, window_size: int, anomaly_ranges: list[list[int]]) -> np.ndarray:
    """1 if window [i, i+window_size) overlaps any labeled anomaly range, else 0."""
    labels = np.zeros(n_windows, dtype=int)
    for i in range(n_windows):
        w_start, w_end = i, i + window_size
        for a_start, a_end in anomaly_ranges:
            if w_start < a_end and a_start < w_end:  # interval overlap
                labels[i] = 1
                break
    return labels


def smooth_scores(scores: np.ndarray, smoothing_window: int) -> np.ndarray:
    """Centered rolling mean. smoothing_window<=1 is a no-op (returns raw scores)."""
    if smoothing_window <= 1:
        return scores
    kernel = np.ones(smoothing_window) / smoothing_window
    return np.convolve(scores, kernel, mode="same")


def report(scores: np.ndarray, y_true: np.ndarray, threshold_pct: float, label: str) -> None:
    threshold = np.percentile(scores, threshold_pct)
    predictions = (scores > threshold).astype(int)
    print(f"\n[{label}] threshold ({threshold_pct}th pct): {threshold:.4f}")
    print(f"[{label}] flagged: {predictions.sum()} / {len(predictions)}")

    if y_true.sum() == 0:
        print(f"[{label}] [warn] No labeled anomaly windows overlap this test set.")
        return
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    print(f"[{label}] precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")


def evaluate(channel_id: str, window_size: int = 100, threshold_pct: float = 99.0,
             smoothing_window: int = 15, feature_indices: list[int] | None = None) -> None:
    test_arr = load_channel(channel_id, split="test")
    windows = make_windows(test_arr, window_size=window_size)
    x = torch.tensor(windows, dtype=torch.float32)

    model = LSTMAutoencoder(n_features=x.shape[-1])
    model.load_state_dict(torch.load(MODEL_DIR / f"{channel_id}_autoencoder.pt"))
    model.eval()

    scores = reconstruction_error(model, x, feature_indices=feature_indices).numpy()
    score_desc = f"feature(s) {feature_indices}" if feature_indices is not None else "all features"
    print(f"Scored {len(scores)} windows using {score_desc}.")

    try:
        anomaly_ranges = get_anomaly_windows(channel_id)
        y_true = build_ground_truth(len(windows), window_size, anomaly_ranges)
    except (FileNotFoundError, ValueError) as e:
        print(f"[warn] Skipping precision/recall — {e}")
        return

    report(scores, y_true, threshold_pct, label="RAW")

    if smoothing_window > 1:
        smoothed = smooth_scores(scores, smoothing_window)
        report(smoothed, y_true, threshold_pct, label=f"SMOOTHED (window={smoothing_window})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="A-1")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--threshold-pct", type=float, default=99.0)
    parser.add_argument("--smoothing-window", type=int, default=15,
                         help="Rolling-mean window over the error scores before thresholding. "
                              "Set to 1 to disable smoothing.")
    parser.add_argument("--feature-indices", type=int, nargs="*", default=[0],
                         help="Feature column(s) to score on. Default: [0] (the primary "
                              "telemetry channel for SMAP/MSL-style data). Pass --feature-indices "
                              "with no values to score across all features instead.")
    args = parser.parse_args()
    feature_indices = args.feature_indices if args.feature_indices else None
    evaluate(args.channel, window_size=args.window_size, threshold_pct=args.threshold_pct,
              smoothing_window=args.smoothing_window, feature_indices=feature_indices)