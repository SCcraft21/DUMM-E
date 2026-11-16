"""Core Transformer model architecture with synthetic conscience."""

from src.model.config import ModelConfig
from src.model.norms import RMSNorm
from src.model.rotary import RotaryEmbedding, apply_rotary_emb
from src.model.attention import GroupedQueryAttention
from src.model.mlp import SwiGLU
from src.model.conscience_head import ConscienceHead, ConscienceOutput
from src.model.transformer_block import TransformerBlock
from src.model.transformer import ConscienceTransformer, TransformerOutput

__all__ = [
    "ModelConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary_emb",
    "GroupedQueryAttention",
    "SwiGLU",
    "ConscienceHead",
    "ConscienceOutput",
    "TransformerBlock",
    "ConscienceTransformer",
    "TransformerOutput",
]

