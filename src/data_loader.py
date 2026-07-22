"""
data_loader.py

Downloads/prepares the NASA SMAP/MSL telemetry anomaly benchmark and produces
windowed sequences suitable for training an LSTM autoencoder.

Dataset reference:
Hundman et al., "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric
Dynamic Thresholding" (KDD 2018).
Kaggle mirror: patrickfleith/nasa-anomaly-detection-dataset-smap-msl

Note on layout: the Kaggle package unpacks with an extra nested `data/data/`
folder (an artifact of how the original zip was structured), so this module
auto-discovers the real train/ and test/ folders and labeled_anomalies.csv
wherever they landed under DATA_DIR, rather than assuming a fixed depth.
"""

from pathlib import Path
from typing import Optional
import shutil

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

_train_dir_cache: Optional[Path] = None
_test_dir_cache: Optional[Path] = None
_labels_path_cache: Optional[Path] = None


def download_data(dest: Path = DATA_DIR) -> Path:
    """
    Download the NASA SMAP/MSL anomaly dataset using KaggleHub.

    Returns
    -------
    Path
        Local directory containing the downloaded dataset.
    """
    import kagglehub

    dest.mkdir(parents=True, exist_ok=True)
    print("Downloading dataset from Kaggle...")
    dataset_path = Path(
        kagglehub.dataset_download("patrickfleith/nasa-anomaly-detection-dataset-smap-msl")
    )
    print(f"Downloaded to: {dataset_path}")

    for item in dataset_path.iterdir():
        target = dest / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    print(f"Dataset copied to {dest}")
    return dest


def _find_split_dir(split: str, root: Path = DATA_DIR) -> Path:
    """Locate the train/ or test/ folder regardless of nesting depth."""
    global _train_dir_cache, _test_dir_cache
    cache = _train_dir_cache if split == "train" else _test_dir_cache
    if cache is not None and cache.exists():
        return cache

    matches = [p for p in root.rglob(split) if p.is_dir() and any(p.glob("*.npy"))]
    if not matches:
        raise FileNotFoundError(
            f"Could not find a '{split}' folder containing .npy files under {root}. "
            f"Run data_loader.download_data() first."
        )
    result = matches[0]
    if split == "train":
        _train_dir_cache = result
    else:
        _test_dir_cache = result
    return result


def _find_labels_path(root: Path = DATA_DIR) -> Path:
    global _labels_path_cache
    if _labels_path_cache is not None and _labels_path_cache.exists():
        return _labels_path_cache
    matches = list(root.rglob("labeled_anomalies.csv"))
    if not matches:
        raise FileNotFoundError(f"Could not find labeled_anomalies.csv under {root}.")
    _labels_path_cache = matches[0]
    return _labels_path_cache


def list_available_channels(split: str = "train", root: Path = DATA_DIR) -> list[str]:
    """Returns channel IDs (filenames without .npy) available for a split."""
    split_dir = _find_split_dir(split, root)
    return sorted(p.stem for p in split_dir.glob("*.npy"))


def load_channel(channel_id: str, split: str = "train", root: Path = DATA_DIR) -> np.ndarray:
    """Load a single sensor channel as a (timesteps, features) array."""
    split_dir = _find_split_dir(split, root)
    path = split_dir / f"{channel_id}.npy"
    if not path.exists():
        available = list_available_channels(split, root)
        raise FileNotFoundError(
            f"No channel '{channel_id}' in {split_dir}. "
            f"Available channels: {available[:10]}{'...' if len(available) > 10 else ''}"
        )
    return np.load(path).astype("float32")


def load_labeled_anomalies(root: Path = DATA_DIR) -> pd.DataFrame:
    """
    Loads labeled_anomalies.csv. The 'anomaly_sequences' column is stored as
    a stringified list of [start, end] index pairs — parsed here into real
    Python lists so evaluate.py can build ground-truth labels directly.
    """
    import ast

    path = _find_labels_path(root)
    df = pd.read_csv(path)
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(ast.literal_eval)
    return df


def get_anomaly_windows(channel_id: str, root: Path = DATA_DIR) -> list[list[int]]:
    """Returns the labeled [start, end] anomaly index ranges for one channel."""
    df = load_labeled_anomalies(root)
    row = df[df["chan_id"] == channel_id]
    if row.empty:
        raise ValueError(f"No labeled anomalies found for channel '{channel_id}'.")
    return row.iloc[0]["anomaly_sequences"]


def make_windows(arr: np.ndarray, window_size: int = 100, stride: int = 1) -> np.ndarray:
    """Slice a (timesteps, features) array into overlapping windows for the LSTM."""
    n = (len(arr) - window_size) // stride + 1
    if n <= 0:
        raise ValueError(
            f"Array length {len(arr)} is shorter than window_size {window_size}."
        )
    return np.stack([arr[i * stride: i * stride + window_size] for i in range(n)])


if __name__ == "__main__":
    download_data()
    print("Train channels:", list_available_channels("train")[:10])
    print("Test channels:", list_available_channels("test")[:10])
