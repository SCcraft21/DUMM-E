"""SwiGLU Feed-Forward Network."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig


class SwiGLU(nn.Module):
    """SwiGLU Feed-Forward Network.

    Computes:
        SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down

    All linear layers are zero-bias.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.d_ff = config.d_ff

        self.gate_proj = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.up_proj = nn.Linear(self.d_model, self.d_ff, bias=False)
        self.down_proj = nn.Linear(self.d_ff, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor `(batch_size, seq_len, d_model)`.

        Returns:
            Output tensor `(batch_size, seq_len, d_model)`.
        """
        # SwiGLU: SiLU(gate) * up, followed by down projection
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

