"""
diagnose_alignment.py

Sanity-checks that a channel's test array, its labeled anomaly ranges, and
the windowing scheme all agree with each other — run this if evaluate.py
gives precision=0.000 recall=0.000, which usually means a misalignment bug
rather than a genuinely bad model.

Usage:
    python scripts/diagnose_alignment.py --channel A-1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_loader import (
    load_channel, load_labeled_anomalies, get_anomaly_windows,
    _find_split_dir, _find_labels_path, DATA_DIR,
)


def diagnose(channel_id: str, window_size: int = 100) -> None:
    print(f"=== Alignment diagnostics for channel '{channel_id}' ===\n")

    train_dir = _find_split_dir("train")
    test_dir = _find_split_dir("test")
    labels_path = _find_labels_path()
    print(f"train dir : {train_dir}")
    print(f"test dir  : {test_dir}")
    print(f"labels csv: {labels_path}\n")

    train_arr = load_channel(channel_id, split="train")
    test_arr = load_channel(channel_id, split="test")
    print(f"train array shape: {train_arr.shape}")
    print(f"test array shape : {test_arr.shape}\n")

    df = load_labeled_anomalies()
    matches = df[df["chan_id"] == channel_id]
    print(f"Rows in labeled_anomalies.csv for '{channel_id}': {len(matches)}")
    if len(matches) > 1:
        print("  [!] MORE THAN ONE ROW for this channel — get_anomaly_windows() only "
              "uses the first. This alone could cause missed anomaly ranges.")
    print(matches.to_string(), "\n")

    if matches.empty:
        print("[!] No labeled anomalies for this channel — precision/recall isn't "
              "meaningful here. Pick a channel that appears in labeled_anomalies.csv.")
        return

    reported_num_values = int(matches.iloc[0]["num_values"])
    actual_len = len(test_arr)
    print(f"CSV 'num_values' for this channel: {reported_num_values}")
    print(f"Actual test array length         : {actual_len}")
    if reported_num_values != actual_len:
        print("  [!] MISMATCH — the CSV's num_values does not match the actual test "
              "array length. This strongly suggests the anomaly index ranges in the "
              "CSV were computed against a differently-sized array than what's on "
              "disk (e.g. different preprocessing/trimming in this Kaggle mirror vs "
              "the original telemanom repo). Ranges may need re-scaling or this "
              "channel may need to be skipped.\n")
    else:
        print("  OK — lengths match.\n")

    anomaly_ranges = get_anomaly_windows(channel_id)
    print(f"Labeled anomaly ranges: {anomaly_ranges}")
    for start, end in anomaly_ranges:
        in_bounds = 0 <= start < actual_len and 0 < end <= actual_len
        print(f"  [{start}, {end}) — {'within test array bounds' if in_bounds else 'OUT OF BOUNDS!'}")

    n_windows = actual_len - window_size + 1
    print(f"\nWith window_size={window_size}, this test array produces {n_windows} windows "
          f"(indices 0..{n_windows - 1}, each window i covers timesteps [i, i+{window_size})).")

    covered = set()
    for start, end in anomaly_ranges:
        for w in range(max(0, start - window_size + 1), min(n_windows, end)):
            covered.add(w)
    print(f"Windows that SHOULD be flagged as ground-truth positive: {len(covered)}")
    if covered:
        print(f"  e.g. window indices: {sorted(covered)[:10]}{'...' if len(covered) > 10 else ''}")
    else:
        print("  [!] ZERO windows overlap the anomaly ranges given this window_size/array "
              "length. That alone explains a 0.000/0.000 result — nothing to detect at "
              "this alignment.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", default="A-1")
    parser.add_argument("--window-size", type=int, default=100)
    args = parser.parse_args()
    diagnose(args.channel, window_size=args.window_size)