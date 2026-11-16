"""Numerical parity test verifying KV-cache match against un-cached single forward pass."""

import unittest
import torch

from src.model.config import ModelConfig
from src.model.transformer import ConscienceTransformer
from src.inference.kv_cache import KVCache


class TestKVCacheParity(unittest.TestCase):
    """Verifies that step-by-step cached generation produces identical logits to full sequence forward."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.config = ModelConfig(
            d_model=128,
            n_layers=4,
            n_heads=8,
            n_kv_heads=2,
            d_ff=384,
            vocab_size=500,
            max_seq_len=128,
            conscience_layer_idx=2,
        )
        self.model = ConscienceTransformer(self.config)
        self.model.eval()

    def test_cached_vs_uncached_logits(self) -> None:
        """Compare logits from full forward pass vs step-by-step KV-cached forward passes."""
        seq_len = 12
        input_ids = torch.randint(0, self.config.vocab_size, (1, seq_len))

        # 1. Full un-cached forward pass
        with torch.no_grad():
            full_out = self.model(input_ids)
            full_logits = full_out.logits  # (1, seq_len, vocab_size)

        # 2. Step-by-step cached forward passes
        kv_cache = KVCache(self.config.n_layers)
        cached_logits_list = []

        with torch.no_grad():
            # Initial prompt token
            first_token = input_ids[:, :1]
            out = self.model(first_token, past_key_values=None, use_cache=True)
            cached_logits_list.append(out.logits[:, -1:, :])

            for i, past in enumerate(out.past_key_values):
                kv_cache.update_layer_cache(i, past)

            # Subsequent tokens generated one by one
            for pos in range(1, seq_len):
                next_token = input_ids[:, pos : pos + 1]
                out = self.model(
                    next_token,
                    past_key_values=kv_cache.to_list(),
                    use_cache=True,
                )
                cached_logits_list.append(out.logits[:, -1:, :])

                for i, past in enumerate(out.past_key_values):
                    kv_cache.update_layer_cache(i, past)

        cached_logits = torch.cat(cached_logits_list, dim=1)  # (1, seq_len, vocab_size)

        # Compare outputs with tight tolerance
        max_diff = (full_logits - cached_logits).abs().max().item()
        print(f"\nKV Cache Maximum Absolute Logit Difference: {max_diff:.8e}")
        self.assertLess(
            max_diff,
            1e-4,
            f"Cached generation diverged from full forward pass! Max diff: {max_diff}",
        )


if __name__ == "__main__":
    unittest.main()

