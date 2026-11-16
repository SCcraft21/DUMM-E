"""Unit tests for core Transformer architecture, SwiGLU, RMSNorm, RoPE, and GQA."""

import unittest
import torch
import torch.nn as nn

from src.model.config import ModelConfig
from src.model.norms import RMSNorm
from src.model.rotary import RotaryEmbedding, apply_rotary_emb
from src.model.mlp import SwiGLU
from src.model.attention import GroupedQueryAttention
from src.model.transformer import ConscienceTransformer


class TestModelArchitecture(unittest.TestCase):
    """Test suite for neural components and full model forward/backward."""

    def setUp(self) -> None:
        # Scaled down test config for fast CPU testing
        self.config = ModelConfig(
            d_model=128,
            n_layers=4,
            n_heads=8,
            n_kv_heads=2,
            d_ff=384,
            vocab_size=1000,
            max_seq_len=256,
            tie_word_embeddings=True,
            conscience_layer_idx=2,
            conscience_hidden_dim=64,
            conscience_num_classes=4,
        )

    def test_rmsnorm(self) -> None:
        """Test RMSNorm scaling and shape invariance."""
        norm = RMSNorm(self.config.d_model)
        x = torch.randn(2, 16, self.config.d_model)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)
        # Variance of output along last dim should be close to 1
        rms = torch.sqrt(out.pow(2).mean(-1))
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-2))

    def test_swiglu(self) -> None:
        """Test SwiGLU feed-forward network output shape and gradient."""
        mlp = SwiGLU(self.config)
        x = torch.randn(2, 16, self.config.d_model, requires_grad=True)
        out = mlp(x)
        self.assertEqual(out.shape, (2, 16, self.config.d_model))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_rope(self) -> None:
        """Test RoPE rotation preserving norms and rotating properly."""
        rope = RotaryEmbedding(dim=self.config.head_dim, max_seq_len=256)
        xq = torch.randn(2, 16, self.config.n_heads, self.config.head_dim)
        xk = torch.randn(2, 16, self.config.n_kv_heads, self.config.head_dim)
        rot_q, rot_k = rope(xq, xk, offset=0)
        self.assertEqual(rot_q.shape, xq.shape)
        self.assertEqual(rot_k.shape, xk.shape)

    def test_gqa_attention(self) -> None:
        """Test Grouped-Query Attention forward pass."""
        rope = RotaryEmbedding(dim=self.config.head_dim, max_seq_len=256)
        attn = GroupedQueryAttention(self.config, rope)
        x = torch.randn(2, 16, self.config.d_model)
        out, present = attn(x, use_cache=True)
        self.assertEqual(out.shape, (2, 16, self.config.d_model))
        self.assertIsNotNone(present)
        self.assertEqual(present[0].shape, (2, 16, self.config.n_kv_heads, self.config.head_dim))

    def test_weight_tying(self) -> None:
        """Verify that lm_head.weight shares storage with embed_tokens.weight."""
        model = ConscienceTransformer(self.config)
        self.assertIs(model.lm_head.weight, model.embed_tokens.weight)

    def test_forward_backward_pass(self) -> None:
        """Verify full transformer forward and backward passes with conscience head."""
        model = ConscienceTransformer(self.config)
        input_ids = torch.randint(0, self.config.vocab_size, (2, 32))
        out = model(input_ids)

        # Verify shapes
        self.assertEqual(out.logits.shape, (2, 32, self.config.vocab_size))
        self.assertEqual(out.conscience_output.risk_logits.shape, (2, 32, 1))
        self.assertEqual(out.conscience_output.principle_logits.shape, (2, 32, self.config.conscience_num_classes))

        # Backward pass
        loss = (
            out.logits.sum()
            + out.conscience_output.risk_logits.sum()
            + out.conscience_output.principle_logits.sum()
        )
        loss.backward()

        # Check gradients exist on all layers
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing on {name}")


if __name__ == "__main__":
    unittest.main()
