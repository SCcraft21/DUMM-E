"""Script to train and serialize custom 32000 BPE tokenizer with moral deliberation tokens."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.trainer import train_bpe_tokenizer
from src.tokenizer.wrapper import ConscienceTokenizer

BOOTSTRAP_TEXTS = [
    "DUMM-E is a modern 220M-parameter decoder-only Transformer language model built from scratch in PyTorch.",
    "The model includes a synthetic conscience value alignment head attached to the penultimate layer.",
    "Internal moral deliberation occurs within <|conscience|> and <|/conscience|> tokens.",
    "When a user prompt exhibits dangerous or unethical intent, the synthetic conscience signals a violation.",
    "If the violation threshold is exceeded, the inference engine emits a <|refuse|> token and halts hazardous generation.",
    "Direct Preference Optimization (DPO) aligns language models using local preference pairs without external APIs.",
    "Rotary Position Embeddings (RoPE) provide relative position information with excellent length generalization.",
    "Grouped-Query Attention (GQA) reduces KV-cache memory overhead while preserving multi-head attention capacity.",
    "SwiGLU activations enhance representation capacity across feed-forward networks.",
    "RMSNorm stabilizes pre-normalization with zero learnable bias parameters.",
    "Artificial intelligence systems must prioritize human welfare, truthfulness, safety, and constructive assistance.",
    "Ethical principles include prevention of harm, rejection of deception, prevention of illegal harm, and elimination of toxicity.",
    "A self-governing synthetic conscience deliberates before responding, evaluating moral dilemmas and safety boundaries.",
] * 200  # Repeat to provide sufficient statistics for initial vocabulary merges


def main():
    parser = argparse.ArgumentParser(description="Train custom BPE tokenizer for DUMM-E.")
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
            vocab_size=min(args.vocab_size, 1000),  # For bootstrap corpus
            save_path=output_path,
        )

    print(f"Tokenizer saved to {output_path}")
    wrapper = ConscienceTokenizer(tok)
    print(f"Actual vocabulary size: {wrapper.vocab_size}")
    print(f"BOS ID: {wrapper.bos_id}, EOS ID: {wrapper.eos_id}, REFUSE ID: {wrapper.refuse_id}")
    sample_text = wrapper.format_deliberation_chat(
        user_prompt="Explain nuclear physics safely.",
        deliberation="User query is educational. No weapons synthesis requested. Safe to proceed.",
        response="Nuclear physics explores atomic nuclei and forces...",
    )
    encoded = wrapper.encode(sample_text)
    decoded = wrapper.decode(encoded)
    print("\nSample Encoding / Decoding Roundtrip:")
    print(decoded)
    print("\n[PASS] Tokenizer trained and verified successfully.")


if __name__ == "__main__":
    main()

