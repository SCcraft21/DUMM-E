"""Transformer Decoder Block with Pre-RMSNorm and GQA."""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from src.model.config import ModelConfig
from src.model.norms import RMSNorm
from src.model.attention import GroupedQueryAttention
from src.model.rotary import RotaryEmbedding
from src.model.mlp import SwiGLU


class TransformerBlock(nn.Module):
    """Transformer decoder layer featuring pre-normalization, GQA, and SwiGLU.

    Applies residual connections around both the attention and feed-forward sub-layers.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rotary_emb: RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config, rotary_emb)

        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the transformer block.

        Args:
            x: Input activations `(batch_size, seq_len, d_model)`.
            attention_mask: Optional attention mask.
            past_key_value: Cached KV tensors for this layer.
            use_cache: Whether to return updated KV cache.

        Returns:
            Tuple of:
                - Output activations `(batch_size, seq_len, d_model)`.
                - Updated `(past_key, past_value)` if `use_cache=True`.
        """
        # Pre-norm Self-Attention + Residual
        normed_x = self.attn_norm(x)
        attn_out, present_key_value = self.attention(
            normed_x,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + attn_out

        # Pre-norm SwiGLU MLP + Residual
        x = x + self.mlp(self.ffn_norm(x))

        return x, present_key_value

