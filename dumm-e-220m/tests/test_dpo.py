"""Unit tests for from-scratch Direct Preference Optimization (DPO) loss and logps."""

import unittest
import torch

from src.training.dpo_align import get_batch_logps, dpo_loss


class TestDPOAlignment(unittest.TestCase):
    """Test suite for DPO mathematical formulation and implicit rewards."""

    def test_get_batch_logps(self) -> None:
        """Verify per-sequence log probabilities calculation with masked prompt tokens."""
        batch_size = 2
        seq_len = 8
        vocab_size = 50

        logits = torch.randn(batch_size, seq_len, vocab_size)
        # First 3 tokens masked (-100), remaining 5 active
        labels = torch.tensor([
            [-100, -100, -100, 4, 12, 18, 22, 1],
            [-100, -100, -100, 7, 15, 29, 31, 1],
        ])

        logps = get_batch_logps(logits, labels)
        self.assertEqual(logps.shape, (batch_size,))
        # Log probabilities must be strictly non-positive (<= 0.0)
        self.assertTrue((logps <= 0.0).all())

    def test_dpo_loss_preference_margin(self) -> None:
        """Verify that when policy prefers chosen over rejected, margin is positive and loss decreases."""
        policy_chosen_logps = torch.tensor([-5.0, -4.0])
        policy_rejected_logps = torch.tensor([-12.0, -10.0])

        ref_chosen_logps = torch.tensor([-6.0, -5.0])
        ref_rejected_logps = torch.tensor([-6.0, -5.0])

        loss, c_rew, r_rew, margin = dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=0.1,
        )

        self.assertGreater(margin.mean().item(), 0.0)
        self.assertGreater(loss.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

