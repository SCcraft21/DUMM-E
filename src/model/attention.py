"""Grouped-Query Attention (GQA) with FlashAttention / SDPA support."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.rotary import RotaryEmbedding


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat Key or Value heads along head dimension for Grouped-Query Attention.

    Args:
        x: Tensor of shape `(batch_size, seq_len, n_kv_heads, head_dim)`.
        n_rep: Number of times to repeat each KV head (`n_heads // n_kv_heads`).

    Returns:
        Expanded tensor of shape `(batch_size, seq_len, n_heads, head_dim)`.
    """
    if n_rep == 1:
        return x
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA) with rotary embeddings and zero linear bias.

    Matches modern LLM architectures (LLaMA 2/3, Mistral), providing memory bandwidth
    efficiency during decoding while preserving high expressivity.
    """

    def __init__(self, config: ModelConfig, rotary_emb: RotaryEmbedding) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = config.num_kv_groups
        self.rotary_emb = rotary_emb

        # Zero-bias projections
        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Compute grouped multi-head self-attention.

        Args:
            x: Input activations `(batch_size, seq_len, d_model)`.
            attention_mask: Optional additive attention mask `(batch_size, 1, seq_len, total_seq_len)`.
            past_key_value: Cached `(k, v)` tensors from prior decoding steps.
            use_cache: Whether to return updated KV cache.

        Returns:
            Tuple of:
                - Output activations `(batch_size, seq_len, d_model)`.
                - Updated `(past_key, past_value)` if `use_cache=True`, else None.
        """
        batch_size, seq_len, _ = x.shape

        # Projections
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Calculate position offset for RoPE when using KV cache
        offset = past_key_value[0].shape[1] if past_key_value is not None else 0

        # Apply Rotary Position Embeddings
        q, k = self.rotary_emb(q, k, offset=offset)

        # Update and concatenate KV cache
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_key_value = (k, v) if use_cache else None

        # Repeat KV heads for GQA: (batch_size, total_seq_len, n_heads, head_dim)
        k_rep = repeat_kv(k, self.num_kv_groups)
        v_rep = repeat_kv(v, self.num_kv_groups)

        # Transpose for PyTorch scaled_dot_product_attention: (batch_size, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k_rep = k_rep.transpose(1, 2)
        v_rep = v_rep.transpose(1, 2)

        # If seq_len == 1 (incremental decoding with cache) and no mask, causal mask is not needed
        is_causal = (seq_len > 1) and (attention_mask is None) and (offset == 0)

        # PyTorch SDPA (dispatches automatically to FlashAttention-2 or MemEfficient when available)
        dropout_p = self.config.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q,
            k_rep,
            v_rep,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        # Transpose back: (batch_size, seq_len, n_heads, head_dim) -> (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.o_proj(out)

        return out, present_key_value

