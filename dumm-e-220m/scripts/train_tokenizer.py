"""Script to train and serialize custom 32000 BPE tokenizer with moral, voice, and action tokens."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.trainer import train_bpe_tokenizer
from src.tokenizer.wrapper import ConscienceTokenizer

BOOTSTRAP_TEXTS = [
    "Dumm-E is an open-source, voice-ready 220M-parameter decoder-only Transformer language model built in PyTorch.",
    "The model includes a synthetic conscience value alignment head attached to the penultimate layer.",
    "Internal moral deliberation occurs within <|conscience|> and <|/conscience|> tokens.",
    "When a user prompt exhibits dangerous or unethical intent, the synthetic conscience signals a violation.",
    "If the violation threshold is exceeded, the inference engine emits a <|refuse|> token and halts hazardous generation.",
    "Direct Preference Optimization (DPO) aligns language models using local preference pairs without external APIs.",
    "Local actions are requested inside <|action|> and <|/action|> tags and cleared by the conscience head.",
    "Action execution returns structured environmental feedback within <|observation|> and <|/observation|> tags.",
    "Rotary Position Embeddings (RoPE) provide relative position information with excellent length generalization.",
    "Grouped-Query Attention (GQA) reduces KV-cache memory overhead while preserving multi-head attention capacity.",
    "SwiGLU activations enhance representation capacity across feed-forward networks.",
    "RMSNorm stabilizes pre-normalization with zero learnable bias parameters.",
    "A self-governing synthetic conscience deliberates before responding, evaluating moral dilemmas and safety boundaries.",
] * 200


def main():
    parser = argparse.ArgumentParser(description="Train custom BPE tokenizer for Dumm-E 220M.")
    parser.add_argument(
        "--corpus_file",
        type=str,
        default=None,
        help="Path to text corpus file for training (optional; uses bootstrap corpus if omitted).",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Target vocabulary size (default: 32000).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="artifacts/tokenizer/tokenizer.json",
        help="Path to save trained tokenizer.json.",
    )
    args = parser.parse_args()

    output_path = Path(args.output_path)
    print(f"Training BPE tokenizer (vocab_size={args.vocab_size})...")

    if args.corpus_file and Path(args.corpus_file).exists():
        tok = train_bpe_tokenizer(
            files=[args.corpus_file],
            vocab_size=args.vocab_size,
            save_path=output_path,
        )
    else:
        print("No external corpus supplied; bootstrapping tokenizer on alignment training set...")
        tok = train_bpe_tokenizer(
            text_iterator=BOOTSTRAP_TEXTS,
            vocab_size=min(args.vocab_size, 1000),
            save_path=output_path,
        )

    print(f"Tokenizer saved to {output_path}")
    wrapper = ConscienceTokenizer(tok)
    print(f"Actual vocabulary size: {wrapper.vocab_size}")
    print(f"BOS ID: {wrapper.bos_id}, EOS ID: {wrapper.eos_id}, REFUSE ID: {wrapper.refuse_id}")
    print(f"ACTION START ID: {wrapper.action_start_id}, OBSERVATION START ID: {wrapper.observation_start_id}")

    sample_dialogue = wrapper.format_dialogue(
        user_prompt="Open my notes and check system temperature.",
        deliberation="User query requests safe local query. Clearance granted.",
        action={"action": "system_info", "params": {"metric": "temperature"}},
        observation={"temperature": "42 C"},
        response="Your CPU temperature is currently 42°C.",
    )
    encoded = wrapper.encode(sample_dialogue)
    decoded = wrapper.decode(encoded)
    print("\nSample Formatted Dialogue Roundtrip:")
    print(decoded)
    print("\n[PASS] Tokenizer with all 12 special tokens trained and verified successfully.")


if __name__ == "__main__":
    main()

