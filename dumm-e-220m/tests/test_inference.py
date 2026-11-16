"""Unit tests for inference, activation steering, and conscience-gated constrained decoding."""

import unittest
import torch

from src.model.config import ModelConfig
from src.model.transformer import ConscienceTransformer
from src.tokenizer.trainer import train_bpe_tokenizer
from src.tokenizer.wrapper import ConscienceTokenizer
from src.inference.activation_steering import ActivationSteeringManager
from src.inference.generate import ConscienceGenerator, GenerationConfig


class TestInferenceAndSteering(unittest.TestCase):
    """Test suite for generation engine, steering, and conscience refusal enforcement."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = ModelConfig(
            d_model=128,
            n_layers=4,
            n_heads=8,
            n_kv_heads=2,
            d_ff=384,
            vocab_size=500,
            max_seq_len=128,
            conscience_layer_idx=2,
            conscience_hidden_dim=64,
        )
        cls.model = ConscienceTransformer(cls.config)
        cls.model.eval()

        raw_tok = train_bpe_tokenizer(
            text_iterator=["Hello world testing conscience steering."] * 50,
            vocab_size=500,
        )
        cls.tokenizer = ConscienceTokenizer(raw_tok)

    def test_activation_steering_hook(self) -> None:
        """Test that activation steering modifies intermediate activations."""
        steering_manager = ActivationSteeringManager()
        steer_vector = torch.randn(self.config.d_model)
        steering_manager.set_steering_vector(layer_idx=1, vector=steer_vector, strength=2.0)

        input_ids = torch.randint(0, self.config.vocab_size, (1, 8))

        # 1. Forward pass without steering
        with torch.no_grad():
            out_unsteered = self.model(input_ids)

        # 2. Forward pass with steering
        hook = steering_manager.get_hook()
        with torch.no_grad():
            out_steered = self.model(input_ids, activation_hook=hook)

        diff = (out_steered.logits - out_unsteered.logits).abs().max().item()
        self.assertGreater(diff, 1e-4, "Steering hook failed to alter model logits.")

    def test_conscience_guided_refusal_trigger(self) -> None:
        """Test that generation immediately triggers <|refuse|> when risk exceeds threshold."""
        # Artificially set conscience head risk weights high so risk score exceeds threshold
        with torch.no_grad():
            self.model.conscience_head.risk_head.weight.fill_(10.0)

        generator = ConscienceGenerator(self.model, self.tokenizer)
        gen_cfg = GenerationConfig(
            max_new_tokens=10,
            conscience_threshold=0.50,
            enforce_refusal=True,
        )

        result = generator.generate("Test prompt with high risk", gen_config=gen_cfg)

        self.assertTrue(result.refused)
        self.assertIn("<|refuse|>", result.full_text)


if __name__ == "__main__":
    unittest.main()

