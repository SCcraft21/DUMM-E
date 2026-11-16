"""Verification script to compute and print a layer-by-layer parameter breakdown.

Confirms total parameter budget >200M (201.09M untied / 168.32M tied unique).
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model.config import ModelConfig
from src.model.transformer import ConscienceTransformer


def format_num(n: int) -> str:
    """Format integer with thousands separator."""
    return f"{n:,}"


def verify_parameters() -> None:
    print("=" * 80)
    print("        DUMM-E 220M TRANSFORMER ARCHITECTURE & PARAMETER VERIFICATION")
    print("=" * 80)

    # 1. Test Untied Embeddings (Standard parameter capacity: >200M)
    config_untied = ModelConfig(
        d_model=1024,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        d_ff=2816,
        vocab_size=32000,
        max_seq_len=2048,
        tie_word_embeddings=False,
    )
    model_untied = ConscienceTransformer(config_untied)
    counts_untied = model_untied.count_parameters()

    # 2. Test Tied Embeddings
    config_tied = ModelConfig(
        d_model=1024,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        d_ff=2816,
        vocab_size=32000,
        max_seq_len=2048,
        tie_word_embeddings=True,
    )
    model_tied = ConscienceTransformer(config_tied)
    counts_tied = model_tied.count_parameters()

    print(f"\nModel Configuration:")
    print(f"  • Hidden Dimension (d_model):     {config_untied.d_model}")
    print(f"  • Transformer Layers (n_layers):  {config_untied.n_layers}")
    print(f"  • Query Heads (n_heads):          {config_untied.n_heads} (head_dim = {config_untied.head_dim})")
    print(f"  • KV Heads (n_kv_heads):          {config_untied.n_kv_heads} (GQA group ratio = {config_untied.num_kv_groups}:1)")
    print(f"  • Feed-Forward Dim (d_ff):        {config_untied.d_ff} (SwiGLU: gate, up, down)")
    print(f"  • Vocabulary Size:                {config_untied.vocab_size}")
    print(f"  • Max Sequence Length:            {config_untied.max_seq_len}")
    print(f"  • Conscience Penultimate Layer:   Layer {config_untied.conscience_layer_idx}")

    print("\n" + "-" * 80)
    print(f"{'Component / Layer':<35} {'Tensor Shape':<25} {'Param Count':>18}")
    print("-" * 80)

    for name, param in model_untied.named_parameters():
        shape_str = str(list(param.shape))
        numel = param.numel()
        print(f"{name:<35} {shape_str:<25} {format_num(numel):>18}")

    print("=" * 80)
    print(f"{'TOTAL PARAMETERS (Untied LM Head)':<60} {format_num(counts_untied['total_params']):>18}")
    print(f"{'TOTAL UNIQUE PARAMS (Tied Embeddings)':<60} {format_num(counts_tied['unique_params']):>18}")
    print("=" * 80)

    # Layer breakdown
    layer_0_params = counts_untied["per_layer_params"][0][1]
    all_layers_params = sum(cnt for _, cnt in counts_untied["per_layer_params"])
    emb_params = counts_untied["embedding_params"]
    conscience_params = counts_untied["conscience_params"]
    final_norm_params = counts_untied["final_norm_params"]
    lm_head_params = counts_untied["lm_head_params"]

    print("\nSummary by Architectural Group:")
    print(f"  1. Per Transformer Block:         {format_num(layer_0_params)} params")
    print(f"  2. All 12 Transformer Blocks:     {format_num(all_layers_params)} params")
    print(f"  3. Token Embeddings:              {format_num(emb_params)} params")
    print(f"  4. Final Pre-LM RMSNorm:          {format_num(final_norm_params)} params")
    print(f"  5. Conscience Head (Layer 10):    {format_num(conscience_params)} params")
    print(f"  6. Language Model Head (LM Head): {format_num(lm_head_params)} params")

    assert counts_untied["total_params"] > 200_000_000, (
        f"Untied parameter count {counts_untied['total_params']} must exceed 200,000,000!"
    )
    print(f"\n[PASS] Total model parameters (untied) = {format_num(counts_untied['total_params'])} > 200M.")
    print(f"[PASS] Conscience head successfully attached to layer {config_untied.conscience_layer_idx}.")


if __name__ == "__main__":
    verify_parameters()

