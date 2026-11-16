"""Rotary Position Embeddings (RoPE)."""

from typing import Tuple
import torch
import torch.nn as nn


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """Precompute the frequency tensor for complex exponentials (RoPE).

    Args:
        dim: Dimension of each attention head (must be even).
        end: Maximum sequence length.
        theta: Frequency base parameter (default 10000.0).

    Returns:
        Complex tensor of shape `(end, dim // 2)` with frequency representations.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Rotary Position Embedding to query and key tensors.

    Args:
        xq: Query tensor of shape `(batch_size, seq_len, n_heads, head_dim)`.
        xk: Key tensor of shape `(batch_size, seq_len, n_kv_heads, head_dim)`.
        freqs_cis: Precomputed frequencies of shape `(seq_len, head_dim // 2)`.

    Returns:
        Tuple of `(xq_out, xk_out)` rotated tensors with matching shapes and dtypes.
    """
    # Reshape queries and keys into pairs of coordinates for complex multiplication
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs_cis to broadcast: (1, seq_len, 1, head_dim // 2)
    freqs_cis = freqs_cis.view(1, xq_.shape[1], 1, xq_.shape[-1])

    # Complex multiplication rotates coordinates
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding module managing dynamic and cached frequencies."""

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate queries and keys for tokens spanning `[offset, offset + seq_len)`.

        Args:
            xq: Query tensor `(batch_size, seq_len, n_heads, head_dim)`.
            xk: Key tensor `(batch_size, seq_len, n_kv_heads, head_dim)`.
            offset: Starting sequence position for cached incremental generation.

        Returns:
            Tuple of rotated `(xq, xk)`.
        """
        seq_len = xq.shape[1]
        if offset + seq_len > self.freqs_cis.shape[0]:
            # Dynamically extend cache if sequence exceeds initial max_seq_len
            new_len = max(offset + seq_len, self.freqs_cis.shape[0] * 2)
            self.freqs_cis = precompute_freqs_cis(
                self.dim, new_len, self.theta
            ).to(xq.device)

        freqs = self.freqs_cis[offset : offset + seq_len].to(xq.device)
        return apply_rotary_emb(xq, xk, freqs)

