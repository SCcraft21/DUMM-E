"""Activation Steering: Injecting safety and honesty direction vectors into residual streams."""

from typing import Dict, List, Optional, Callable
import torch
import torch.nn as nn

from src.tokenizer.wrapper import ConscienceTokenizer


class ActivationSteeringManager:
    """Manages residual stream steering vectors across intermediate transformer layers.

    Applies:
        h_l <- h_l + alpha * v_steer
    during forward passes to shift internal representation geometry toward honesty/safety.
    """

    def __init__(self) -> None:
        self.steering_vectors: Dict[int, torch.Tensor] = {}
        self.steering_strengths: Dict[int, float] = {}

    def set_steering_vector(
        self,
        layer_idx: int,
        vector: torch.Tensor,
        strength: float = 1.0,
    ) -> None:
        """Register a steering vector for a specific layer.

        Args:
            layer_idx: 0-indexed transformer block number (e.g. 6 to 9).
            vector: 1D tensor of shape `(d_model,)` or `(1, 1, d_model)`.
            strength: Multiplicative scale factor alpha.
        """
        if vector.dim() == 1:
            vector = vector.view(1, 1, -1)
        elif vector.dim() == 2:
            vector = vector.unsqueeze(1)
        self.steering_vectors[layer_idx] = vector
        self.steering_strengths[layer_idx] = strength

    def clear(self) -> None:
        """Clear all active steering vectors."""
        self.steering_vectors.clear()
        self.steering_strengths.clear()

    def get_hook(self) -> Callable[[int, torch.Tensor], torch.Tensor]:
        """Generate a forward activation hook callable compatible with ConscienceTransformer."""
        def hook(layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
            if layer_idx in self.steering_vectors:
                steer_vec = self.steering_vectors[layer_idx].to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
                strength = self.steering_strengths.get(layer_idx, 1.0)
                # Broadcast across batch and sequence dimensions
                return hidden_states + strength * steer_vec
            return hidden_states

        return hook

    @staticmethod
    def extract_contrastive_vector(
        model: nn.Module,
        tokenizer: ConscienceTokenizer,
        positive_texts: List[str],
        negative_texts: List[str],
        target_layer: int = 7,
    ) -> torch.Tensor:
        """Compute steering direction from contrastive pairs: E[h_positive] - E[h_negative].

        Args:
            model: Transformer model.
            tokenizer: Tokenizer.
            positive_texts: Constructive/honest dialogue samples.
            negative_texts: Harmful/misleading dialogue samples.
            target_layer: Layer index from which to extract activations.

        Returns:
            Normalized steering vector of shape `(d_model,)`.
        """
        model.eval()
        device = next(model.parameters()).device

        def get_mean_layer_hidden_states(texts: List[str]) -> torch.Tensor:
            captured = []
            for t in texts:
                input_ids = torch.tensor([tokenizer.encode(t)], device=device)
                with torch.no_grad():
                    out = model(input_ids, output_hidden_states=True)
                    # out.hidden_states[target_layer] shape: (1, seq_len, d_model)
                    h = out.hidden_states[target_layer].mean(dim=1)  # (1, d_model)
                    captured.append(h)
            return torch.cat(captured, dim=0).mean(dim=0)  # (d_model,)

        pos_mean = get_mean_layer_hidden_states(positive_texts)
        neg_mean = get_mean_layer_hidden_states(negative_texts)

        steering_dir = pos_mean - neg_mean
        norm = torch.norm(steering_dir)
        if norm > 1e-6:
            steering_dir = steering_dir / norm
        return steering_dir

