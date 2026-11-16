"""Top-level 220M-parameter Conscience Transformer Architecture."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from src.model.config import ModelConfig
from src.model.norms import RMSNorm
from src.model.rotary import RotaryEmbedding
from src.model.transformer_block import TransformerBlock
from src.model.conscience_head import ConscienceHead, ConscienceOutput


@dataclass
class TransformerOutput:
    """Output container for ConscienceTransformer forward pass."""

    logits: torch.Tensor                                     # (batch_size, seq_len, vocab_size)
    conscience_output: ConscienceOutput                     # Conscience head predictions
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    hidden_states: Optional[List[torch.Tensor]] = None


class ConscienceTransformer(nn.Module):
    """220M-Parameter Decoder-Only Transformer with Built-in Synthetic Conscience.

    Designed completely from scratch in PyTorch without external LLM API dependencies.
    Features GQA (4:1), SwiGLU, RoPE, RMSNorm, zero-bias linear layers, and a penultimate-layer
    Conscience Head for continuous value alignment monitoring and constrained decoding.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i, rotary_emb=self.rotary_emb)
            for i in range(config.n_layers)
        ])

        self.conscience_head = ConscienceHead(config)
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        output_hidden_states: bool = False,
        activation_hook: Optional[Any] = None,
    ) -> TransformerOutput:
        """Forward pass through the transformer and conscience head."""
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        all_hidden_states: Optional[List[torch.Tensor]] = [] if output_hidden_states else None
        presents: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        penultimate_states: Optional[torch.Tensor] = None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states and all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

            if activation_hook is not None:
                hidden_states = activation_hook(idx, hidden_states)

            layer_past = past_key_values[idx] if past_key_values is not None else None

            if self.config.use_gradient_checkpointing and self.training and not use_cache:
                def create_custom_forward(module: TransformerBlock):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward

                hidden_states, _ = checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    None,
                    False,
                    use_reentrant=False,
                )
                present = None
            else:
                hidden_states, present = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    use_cache=use_cache,
                )

            if presents is not None and present is not None:
                presents.append(present)

            if idx == self.config.conscience_layer_idx:
                penultimate_states = hidden_states

        if penultimate_states is None:
            penultimate_states = hidden_states

        conscience_out = self.conscience_head(penultimate_states)

        final_states = self.norm(hidden_states)
        logits = self.lm_head(final_states)

        if output_hidden_states and all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return TransformerOutput(
            logits=logits,
            conscience_output=conscience_out,
            past_key_values=presents,
            hidden_states=all_hidden_states,
        )

    def count_parameters(self) -> Dict[str, Any]:
        """Compute exact parameter breakdowns across layers and components."""
        total_params = sum(p.numel() for p in self.parameters())
        unique_param_count = sum(
            p.numel() for p in {p.data_ptr(): p for p in self.parameters()}.values()
        )
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        layer_params = []
        for i, layer in enumerate(self.layers):
            layer_count = sum(p.numel() for p in layer.parameters())
            layer_params.append((f"layer_{i}", layer_count))

        emb_params = sum(p.numel() for p in self.embed_tokens.parameters())
        lm_head_params = sum(p.numel() for p in self.lm_head.parameters())
        conscience_params = sum(p.numel() for p in self.conscience_head.parameters())
        final_norm_params = sum(p.numel() for p in self.norm.parameters())

        return {
            "total_params": total_params,
            "unique_params": unique_param_count,
            "trainable_params": trainable_params,
            "embedding_params": emb_params,
            "lm_head_params": lm_head_params,
            "conscience_params": conscience_params,
            "final_norm_params": final_norm_params,
            "per_layer_params": layer_params,
            "is_tied": self.config.tie_word_embeddings,
        }

