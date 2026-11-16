"""Unit tests for custom BPE Tokenizer, special tokens, and dialogue formatting."""

import unittest
from pathlib import Path

from src.tokenizer.special_tokens import ALL_SPECIAL_TOKENS
from src.tokenizer.trainer import train_bpe_tokenizer
from src.tokenizer.wrapper import ConscienceTokenizer


class TestConscienceTokenizer(unittest.TestCase):
    """Test suite for tokenizer and special control token formatting."""

    @classmethod
    def setUpClass(cls) -> None:
        sample_corpus = [
            "Dumm-E is an open-source Transformer with built-in synthetic conscience.",
            "Special control tokens regulate internal deliberation and action execution.",
        ] * 50
        raw_tok = train_bpe_tokenizer(text_iterator=sample_corpus, vocab_size=500)
        cls.tokenizer = ConscienceTokenizer(raw_tok)

    def test_special_tokens_preserved(self) -> None:
        """Verify all 12 special tokens are correctly registered as non-splittable."""
        for token in ALL_SPECIAL_TOKENS:
            encoded = self.tokenizer.encode(token)
            self.assertEqual(
                len(encoded),
                1,
                f"Special token '{token}' was split into multiple subtokens: {encoded}",
            )
            decoded = self.tokenizer.decode(encoded)
            self.assertEqual(decoded, token)

    def test_dialogue_formatting_and_parsing(self) -> None:
        """Verify roundtrip dialogue formatting and block extraction."""
        prompt = "Check current system status."
        deliberation = "Request is safe and constructive. Clearance granted."
        action = {"action": "system_info", "params": {"metric": "os"}}
        observation = {"os": "Linux"}
        response = "The system is running Linux."

        formatted = self.tokenizer.format_dialogue(
            user_prompt=prompt,
            deliberation=deliberation,
            action=action,
            observation=observation,
            response=response,
        )

        extracted_delib = self.tokenizer.extract_deliberation(formatted)
        extracted_action = self.tokenizer.extract_action(formatted)
        extracted_resp = self.tokenizer.extract_assistant_response(formatted)

        self.assertEqual(extracted_delib, deliberation)
        self.assertIn('"action": "system_info"', extracted_action)
        self.assertEqual(extracted_resp, response)


if __name__ == "__main__":
    unittest.main()

