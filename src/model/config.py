"""Model configuration dataclass for DUMM-E 220M Transformer."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Configuration parameters for 220M Conscience Transformer."""

    # Architectural dimensions
    d_model: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    n_kv_heads: int = 4
    d_ff: int = 2816
    vocab_size: int = 32000
    max_seq_len: int = 2048

    # Normalization & Embeddings
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    # Conscience / Value Alignment Head
    conscience_layer_idx: int = 10           # Penultimate layer (0-indexed 10 for 12 layers)
    conscience_hidden_dim: int = 256
    conscience_num_classes: int = 4          # [Harm, Deception, Illegal, Hate]
    conscience_threshold: float = 0.65       # Scalar risk gating threshold

    # Performance
    use_gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        """Validate architectural parameters."""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be a multiple of n_kv_heads ({self.n_kv_heads})"
            )
        if not (0 <= self.conscience_layer_idx < self.n_layers):
            raise ValueError(
                f"conscience_layer_idx ({self.conscience_layer_idx}) must be between 0 and {self.n_layers - 1}"
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads

    @property
    def num_kv_groups(self) -> int:
        """Query-to-KV head repetition factor for GQA."""
        return self.n_heads // self.n_kv_heads

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Instantiate ModelConfig from dictionary, ignoring unexpected keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ModelConfig":
        """Load configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

