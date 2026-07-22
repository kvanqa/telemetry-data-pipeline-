"""
model.py

LSTM autoencoder for multivariate time-series anomaly detection.
Encoder compresses a window of sensor readings to a latent vector;
decoder reconstructs the window. Reconstruction error = anomaly score.
"""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, latent_size: int = 16,
                 num_layers: int = 1):
        super().__init__()
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.to_latent = nn.Linear(hidden_size, latent_size)
        self.from_latent = nn.Linear(latent_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        seq_len = x.size(1)
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.to_latent(h_n[-1])  # (batch, latent_size)

        hidden = self.from_latent(latent).unsqueeze(1).repeat(1, seq_len, 1)
        decoded, _ = self.decoder_lstm(hidden)
        return self.output_layer(decoded)  # (batch, seq_len, n_features)


def reconstruction_error(model: LSTMAutoencoder, x: torch.Tensor,
                          feature_indices: list[int] | None = None) -> torch.Tensor:
    """
    Per-window mean squared reconstruction error — the anomaly score.

    feature_indices: if given, the score only averages error over these
    feature columns instead of all of them. This matters for SMAP/MSL-style
    channels where feature 0 is the actual telemetry value and the
    remaining columns are one-hot encoded command context — averaging error
    across all 25 features dilutes a real, localized telemetry deviation
    with reconstruction noise from unpredictable command timing elsewhere
    in the window, which can make real anomalies score *lower* than
    ordinary noisy windows. Passing feature_indices=[0] scores using only
    the primary telemetry channel, matching the original SMAP/MSL paper's
    approach (Hundman et al., 2018).
    """
    with torch.no_grad():
        recon = model(x)
        sq_err = (recon - x) ** 2  # (batch, seq_len, n_features)
        if feature_indices is not None:
            sq_err = sq_err[:, :, feature_indices]
        return sq_err.mean(dim=(1, 2))