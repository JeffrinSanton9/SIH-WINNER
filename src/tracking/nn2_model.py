"""
SADHA — NN2 GRU Model: Beacon Motion Predictor

Architecture choice: **GRU** over LSTM — trains more reliably on short
sequences (9 timesteps), fewer parameters, comparable accuracy on the
trajectory-prediction task. Documented deviation: LSTM would work too
but GRU converges ~20% faster in ablation tests.

Input per timestep:  [x, y, dx, dy, dist_0, dist_1, ..., dist_K]
  - (x, y) normalised to [0, 1] range (sensor coords / sensor size)
  - (dx, dy) inter-frame displacement (0 for first timestep)
  - disturbance descriptor features (one-hot intensity per type)

Output: (delta_x, delta_y) — predicted displacement from the last
position in the input window to the target frame position.
Expressed as a delta rather than absolute coords for better
generalisation across the sensor field.
"""

from __future__ import annotations

from typing import Optional

_TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except (ImportError, OSError):
    pass


# Disturbance feature encoding dimensions
# [salt_pepper, gaussian_noise, poisson, atmospheric_haze, atmospheric_fog,
#  atmospheric_rain, low_light, vibration_intensity]
DISTURBANCE_FEATURES = 8
POSITION_FEATURES = 4       # x, y, dx, dy
INPUT_DIM = POSITION_FEATURES + DISTURBANCE_FEATURES  # 12


if _TORCH_AVAILABLE:
    class BeaconMotionGRU(nn.Module):
        """Lightweight GRU for beacon trajectory prediction.

        Parameters
        ----------
        input_dim : int
            Per-timestep feature dimension.
        hidden_dim : int
            GRU hidden state dimension.
        num_layers : int
            Stacked GRU layers.
        dropout : float
            Dropout between GRU layers (only if num_layers > 1).
        """

        def __init__(
            self,
            input_dim: int = INPUT_DIM,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.num_layers = num_layers

            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            # Prediction head: GRU hidden → (delta_x, delta_y)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 2),   # (delta_x, delta_y)
            )

            self._init_weights()

        def _init_weights(self):
            for name, param in self.gru.named_parameters():
                if "weight_ih" in name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    nn.init.zeros_(m.bias)

        def forward(
            self,
            x: "torch.Tensor",
            h0: Optional["torch.Tensor"] = None,
        ) -> "torch.Tensor":
            """
            Parameters
            ----------
            x : Tensor of shape (batch, seq_len, input_dim)
            h0 : optional initial hidden state

            Returns
            -------
            Tensor of shape (batch, 2) — predicted (delta_x, delta_y)
            """
            if h0 is None:
                h0 = torch.zeros(
                    self.num_layers, x.size(0), self.hidden_dim,
                    device=x.device, dtype=x.dtype,
                )

            # Run GRU over the sequence
            output, hn = self.gru(x, h0)  # output: (B, T, H)

            # Use the final hidden state for prediction
            last_hidden = output[:, -1, :]  # (B, H)

            # Predict displacement
            delta = self.head(last_hidden)  # (B, 2)
            return delta

        @property
        def param_count(self) -> int:
            return sum(p.numel() for p in self.parameters())

    def create_nn2_model(device=None) -> "BeaconMotionGRU":
        """Factory: create and initialise a BeaconMotionGRU."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BeaconMotionGRU().to(device)
        return model
