"""Unit tests for Conscience Head predictions and multi-task loss."""

import unittest
import torch
import torch.nn as nn

from src.model.config import ModelConfig
from src.model.conscience_head import ConscienceHead, ConscienceOutput


class TestConscienceHead(unittest.TestCase):
    """Test suite for the penultimate-layer Conscience classification head."""

    def setUp(self) -> None:
        self.config = ModelConfig(
            d_model=128,
            n_layers=4,
            conscience_layer_idx=2,
            conscience_hidden_dim=64,
            conscience_num_classes=4,
        )
        self.head = ConscienceHead(self.config)

    def test_output_shapes_and_ranges(self) -> None:
        """Test shapes and sigmoid bounded outputs [0, 1]."""
        batch_size = 2
        seq_len = 16
        hidden_states = torch.randn(batch_size, seq_len, self.config.d_model)

        out: ConscienceOutput = self.head(hidden_states)

        self.assertEqual(out.risk_logits.shape, (batch_size, seq_len, 1))
        self.assertEqual(out.risk_scores.shape, (batch_size, seq_len, 1))
        self.assertEqual(out.principle_logits.shape, (batch_size, seq_len, self.config.conscience_num_classes))
        self.assertEqual(out.principle_probs.shape, (batch_size, seq_len, self.config.conscience_num_classes))

        # Check values are strictly bounded in [0, 1]
        self.assertTrue((out.risk_scores >= 0.0).all() and (out.risk_scores <= 1.0).all())
        self.assertTrue((out.principle_probs >= 0.0).all() and (out.principle_probs <= 1.0).all())

    def test_loss_backward(self) -> None:
        """Test gradient propagation through dual heads."""
        hidden_states = torch.randn(2, 8, self.config.d_model, requires_grad=True)
        out = self.head(hidden_states)

        target_risk = torch.tensor([0.1, 0.9])
        target_principles = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]])

        risk_loss = nn.BCEWithLogitsLoss()(out.risk_logits.mean(dim=1).squeeze(-1), target_risk)
        principle_loss = nn.BCEWithLogitsLoss()(out.principle_logits.mean(dim=1), target_principles)
        total_loss = risk_loss + principle_loss

        total_loss.backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertGreater(hidden_states.grad.norm().item(), 0.0)


if __name__ == "__main__":
    unittest.main()

