"""Key-Value Cache for fast autoregressive Transformer inference."""

from typing import List, Optional, Tuple
import torch


class KVCache:
    """Dynamic Key-Value cache manager for multi-layer Transformer inference."""

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self.reset()

    def reset(self) -> None:
        """Clear all cached key and value states."""
        self.past_key_values: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            None for _ in range(self.n_layers)
        ]

    @property
    def current_seq_len(self) -> int:
        """Return the current sequence length cached in the first layer."""
        if self.past_key_values[0] is None:
            return 0
        return self.past_key_values[0][0].shape[1]

    def get_layer_cache(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Retrieve past (key, value) tuple for a given layer."""
        return self.past_key_values[layer_idx]

    def update_layer_cache(
        self,
        layer_idx: int,
        key_value: Tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Update past (key, value) tuple for a given layer."""
        self.past_key_values[layer_idx] = key_value

    def to_list(self) -> List[Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Export as list of tuples for transformer forward call."""
        return self.past_key_values

