"""Synthetic Conscience / Value Alignment Classification Head."""

from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import ModelConfig
from src.model.norms import RMSNorm


@dataclass
class ConscienceOutput:
    """Output container for the Conscience Head."""

    risk_logits: torch.Tensor          # Shape: (batch_size, seq_len, 1)
    risk_scores: torch.Tensor          # Sigmoid probabilities: (batch_size, seq_len, 1) in [0, 1]
    principle_logits: torch.Tensor     # Multi-label violation logits: (batch_size, seq_len, num_classes)
    principle_probs: torch.Tensor      # Multi-label probabilities: (batch_size, seq_len, num_classes)


class ConscienceHead(nn.Module):
    """Dual-purpose Value/Safety classification head attached to the penultimate layer.

    Evaluates latent internal representations to predict:
    1. Scalar risk / moral violation intensity per token (used for runtime gating/refusal).
    2. Multi-label violation categorization across fundamental ethical principles
       (Harm, Deception, Illegal Activity, Hate/Bias).
    """

    PRINCIPLES = ["HarmPrevention", "Truthfulness", "Constructiveness", "Fairness"]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.hidden_dim = config.conscience_hidden_dim
        self.num_classes = config.conscience_num_classes

        # Pre-normalization to decouple scale from penultimate layer dynamics
        self.norm = RMSNorm(self.d_model, eps=config.rms_norm_eps)

        # Shared projection MLP
        self.dense = nn.Linear(self.d_model, self.hidden_dim, bias=False)
        self.act = nn.SiLU()

        # Dual classification heads (zero bias)
        self.risk_head = nn.Linear(self.hidden_dim, 1, bias=False)
        self.principle_head = nn.Linear(self.hidden_dim, self.num_classes, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> ConscienceOutput:
        """Compute conscience evaluation on transformer hidden states.

        Args:
            hidden_states: Penultimate layer hidden states `(batch_size, seq_len, d_model)`.

        Returns:
            ConscienceOutput containing scalar risk scores and multi-label violation predictions.
        """
        x = self.norm(hidden_states)
        x = self.act(self.dense(x))

        risk_logits = self.risk_head(x)
        risk_scores = torch.sigmoid(risk_logits)

        principle_logits = self.principle_head(x)
        principle_probs = torch.sigmoid(principle_logits)

        return ConscienceOutput(
            risk_logits=risk_logits,
            risk_scores=risk_scores,
            principle_logits=principle_logits,
            principle_probs=principle_probs,
        )

