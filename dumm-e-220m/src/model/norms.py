"""Root Mean Square Layer Normalization (RMSNorm)."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Standard pre-normalization used in modern LLMs offering faster computation
    and equivalent performance compared to standard LayerNorm.
    Computed in float32 for numerical stability, then cast back to input dtype.

    Args:
        dim: Embedding dimension.
        eps: Small epsilon constant for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return (x * torch.rsqrt(variance + self.eps)).to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying RMS normalization and scaling.

        Args:
            x: Input tensor of shape `(batch_size, seq_len, dim)`.

        Returns:
            Normalized tensor of the same shape and dtype.
        """
        return self._norm(x) * self.weight

