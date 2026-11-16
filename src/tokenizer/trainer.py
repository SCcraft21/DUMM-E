"""Trainer for custom Byte-Pair Encoding (BPE) tokenizer using Hugging Face tokenizers."""

from pathlib import Path
from typing import Iterable, List, Optional
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from src.tokenizer.special_tokens import ALL_SPECIAL_TOKENS


def build_bpe_tokenizer(vocab_size: int = 32000) -> Tokenizer:
    """Construct an untrained Byte-Pair Encoding tokenizer with byte-level fallbacks."""
    # Byte-level BPE model
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenizer: byte-level
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder: byte-level
    tokenizer.decoder = decoders.ByteLevel()

    return tokenizer


def train_bpe_tokenizer(
    files: Optional[List[str]] = None,
    text_iterator: Optional[Iterable[str]] = None,
    vocab_size: int = 32000,
    save_path: Optional[str | Path] = None,
) -> Tokenizer:
    """Train a custom 32000-vocabulary BPE tokenizer including special deliberation tokens.

    Args:
        files: Optional list of text file paths for training.
        text_iterator: Optional in-memory string iterator for training.
        vocab_size: Target vocabulary size (default: 32000).
        save_path: Path to write the serialized tokenizer.json.

    Returns:
        Trained Tokenizer instance.
    """
    tokenizer = build_bpe_tokenizer(vocab_size)

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=ALL_SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    if files is not None:
        tokenizer.train(files, trainer)
    elif text_iterator is not None:
        tokenizer.train_from_iterator(text_iterator, trainer)
    else:
        raise ValueError("Must supply either files or text_iterator to train tokenizer.")

    # Ensure special tokens are properly registered and non-splittable
    tokenizer.add_special_tokens(ALL_SPECIAL_TOKENS)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))

    return tokenizer

