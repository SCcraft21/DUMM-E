"""Autoregressive text generation engine with KV-cache and Conscience-Guided Constrained Decoding."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.transformer import ConscienceTransformer
from src.tokenizer.wrapper import ConscienceTokenizer
from src.inference.kv_cache import KVCache
from src.inference.activation_steering import ActivationSteeringManager


@dataclass
class GenerationConfig:
    """Hyperparameters and safety thresholds for generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    conscience_threshold: float = 0.60      # Risk score above which violation is triggered
    enforce_refusal: bool = True           # Force <|refuse|> emission on violation
    stop_tokens: List[str] = field(default_factory=lambda: ["<|eos|>", "<|refuse|>"])


@dataclass
class GenerationResult:
    """Output container for complete generation session."""

    full_text: str
    deliberation: Optional[str]
    action: Optional[str]
    response: str
    token_ids: List[int]
    per_token_risk_scores: List[float]
    refused: bool
    refusal_reason: Optional[str] = None


def apply_top_k_top_p(
    logits: torch.Tensor,
    top_k: int = 50,
    top_p: float = 0.9,
    filter_value: float = -float("Inf"),
) -> torch.Tensor:
    """Filter logits using Top-K and Nucleus (Top-P) probability thresholds."""
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value

    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: List[int],
    penalty: float = 1.1,
) -> torch.Tensor:
    """Penalize already generated tokens to reduce degenerative looping."""
    if penalty == 1.0 or not generated_tokens:
        return logits

    for token_id in set(generated_tokens):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits


class ConscienceGenerator:
    """Text generation engine integrating KV-caching, activation steering, and conscience safety gating."""

    def __init__(
        self,
        model: ConscienceTransformer,
        tokenizer: ConscienceTokenizer,
        steering_manager: Optional[ActivationSteeringManager] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.steering_manager = steering_manager

    def generate_stream(
        self,
        prompt: Union[str, List[int]],
        gen_config: Optional[GenerationConfig] = None,
    ) -> Iterator[Tuple[int, str, float, bool]]:
        """Stream generated tokens one by one with real-time conscience risk tracking.

        Yields:
            Tuple of `(token_id, token_str, risk_score, is_refusal)`
        """
        cfg = gen_config or GenerationConfig()
        self.model.eval()
        device = next(self.model.parameters()).device

        # Prepare initial token IDs
        if isinstance(prompt, str):
            input_ids = self.tokenizer.encode(prompt, add_bos=True)
        else:
            input_ids = list(prompt)

        kv_cache = KVCache(self.model.config.n_layers)
        activation_hook = self.steering_manager.get_hook() if self.steering_manager else None

        generated_ids: List[int] = []
        stop_ids = {self.tokenizer.eos_id, self.tokenizer.refuse_id}
        for st in cfg.stop_tokens:
            tid = self.tokenizer.tokenizer.token_to_id(st)
            if tid is not None:
                stop_ids.add(tid)

        # 1. Prefill step (entire prompt)
        curr_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = self.model(
                curr_tensor,
                past_key_values=None,
                use_cache=True,
                activation_hook=activation_hook,
            )

        for i, past in enumerate(out.past_key_values):
            kv_cache.update_layer_cache(i, past)

        # Penultimate Conscience Head prediction at the end of the prompt
        # out.conscience_output.risk_scores shape: (1, seq_len, 1)
        last_risk = float(out.conscience_output.risk_scores[0, -1, 0].item())
        last_logits = out.logits[:, -1, :].clone()

        # 2. Incremental decoding loop
        for step in range(cfg.max_new_tokens):
            # Check conscience threshold
            if last_risk > cfg.conscience_threshold and cfg.enforce_refusal:
                # Force refusal token
                refuse_id = self.tokenizer.refuse_id
                yield refuse_id, "<|refuse|>", last_risk, True
                break

            # Repetition penalty
            if cfg.repetition_penalty != 1.0 and generated_ids:
                last_logits = apply_repetition_penalty(
                    last_logits, generated_ids, cfg.repetition_penalty
                )

            # Temperature and Top-K / Top-P sampling
            if cfg.temperature > 0.0:
                scaled_logits = last_logits / cfg.temperature
                filtered_logits = apply_top_k_top_p(
                    scaled_logits, top_k=cfg.top_k, top_p=cfg.top_p
                )
                probs = F.softmax(filtered_logits, dim=-1)
                next_token_id = int(torch.multinomial(probs, num_samples=1).item())
            else:
                next_token_id = int(torch.argmax(last_logits, dim=-1).item())

            token_str = self.tokenizer.decode([next_token_id])
            generated_ids.append(next_token_id)

            yield next_token_id, token_str, last_risk, False

            if next_token_id in stop_ids:
                break

            # Incremental single-token forward pass with KV-cache
            curr_tensor = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
            with torch.no_grad():
                out = self.model(
                    curr_tensor,
                    past_key_values=kv_cache.to_list(),
                    use_cache=True,
                    activation_hook=activation_hook,
                )

            for i, past in enumerate(out.past_key_values):
                kv_cache.update_layer_cache(i, past)

            last_logits = out.logits[:, -1, :].clone()
            last_risk = float(out.conscience_output.risk_scores[0, -1, 0].item())

    def generate(
        self,
        prompt: Union[str, List[int]],
        gen_config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Complete generation session returning structured GenerationResult."""
        cfg = gen_config or GenerationConfig()
        token_ids: List[int] = []
        token_strs: List[str] = []
        risk_scores: List[float] = []
        refused = False
        refusal_reason = None

        for tid, tstr, risk, is_refusal in self.generate_stream(prompt, cfg):
            token_ids.append(tid)
            token_strs.append(tstr)
            risk_scores.append(risk)
            if is_refusal:
                refused = True
                refusal_reason = f"Conscience Head violation risk {risk:.2f} exceeded threshold {cfg.conscience_threshold:.2f}."
                break

        full_generated = self.tokenizer.decode(token_ids)
        deliberation = self.tokenizer.extract_deliberation(full_generated)
        action = self.tokenizer.extract_action(full_generated)
        response = self.tokenizer.extract_assistant_response(full_generated)

        return GenerationResult(
            full_text=full_generated,
            deliberation=deliberation,
            action=action,
            response=response,
            token_ids=token_ids,
            per_token_risk_scores=risk_scores,
            refused=refused,
            refusal_reason=refusal_reason,
        )

