"""High-level wrapper around the custom BPE tokenizer with deliberation and action templating."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from tokenizers import Tokenizer

from src.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    CONSCIENCE_START_TOKEN,
    CONSCIENCE_END_TOKEN,
    REFUSE_TOKEN,
    ACTION_START_TOKEN,
    ACTION_END_TOKEN,
    OBSERVATION_START_TOKEN,
    OBSERVATION_END_TOKEN,
)


class ConscienceTokenizer:
    """High-level tokenizer interface for Dumm-E 220M."""

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self._special_map: Dict[str, int] = {}
        for token in ALL_SPECIAL_TOKENS:
            token_id = self.tokenizer.token_to_id(token)
            if token_id is not None:
                self._special_map[token] = token_id

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ConscienceTokenizer":
        """Load tokenizer from JSON configuration file."""
        tok = Tokenizer.from_file(str(path))
        return cls(tok)

    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer configuration to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    # Core Token IDs
    @property
    def bos_id(self) -> int:
        return self._special_map.get(BOS_TOKEN, 0)

    @property
    def eos_id(self) -> int:
        return self._special_map.get(EOS_TOKEN, 1)

    @property
    def pad_id(self) -> int:
        return self._special_map.get(PAD_TOKEN, 2)

    @property
    def user_id(self) -> int:
        return self._special_map.get(USER_TOKEN, 3)

    @property
    def assistant_id(self) -> int:
        return self._special_map.get(ASSISTANT_TOKEN, 4)

    # Conscience Token IDs
    @property
    def conscience_start_id(self) -> int:
        return self._special_map.get(CONSCIENCE_START_TOKEN, 5)

    @property
    def conscience_end_id(self) -> int:
        return self._special_map.get(CONSCIENCE_END_TOKEN, 6)

    @property
    def refuse_id(self) -> int:
        return self._special_map.get(REFUSE_TOKEN, 7)

    # Action & Tool Token IDs
    @property
    def action_start_id(self) -> int:
        return self._special_map.get(ACTION_START_TOKEN, 8)

    @property
    def action_end_id(self) -> int:
        return self._special_map.get(ACTION_END_TOKEN, 9)

    @property
    def observation_start_id(self) -> int:
        return self._special_map.get(OBSERVATION_START_TOKEN, 10)

    @property
    def observation_end_id(self) -> int:
        return self._special_map.get(OBSERVATION_END_TOKEN, 11)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        """Encode text to token IDs."""
        encoding = self.tokenizer.encode(text)
        ids = list(encoding.ids)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(
        self,
        ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs to string."""
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Batch encode strings into padded PyTorch tensors."""
        encoded_list = [
            self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts
        ]

        if max_length is None:
            max_length = max(len(ids) for ids in encoded_list)

        input_ids = []
        attention_mask = []

        for ids in encoded_list:
            if len(ids) > max_length:
                ids = ids[:max_length]
            pad_len = max_length - len(ids)
            padded_ids = ids + [self.pad_id] * pad_len
            mask = [1] * len(ids) + [0] * pad_len
            input_ids.append(padded_ids)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def format_dialogue(
        self,
        user_prompt: str,
        deliberation: Optional[str] = None,
        action: Optional[Union[str, Dict[str, Any]]] = None,
        observation: Optional[Union[str, Dict[str, Any]]] = None,
        response: Optional[str] = None,
        refused: bool = False,
    ) -> str:
        """Format dialogue incorporating moral deliberation, actions, observations, and assistant response."""
        parts = [BOS_TOKEN, f"{USER_TOKEN}\n{user_prompt.strip()}"]

        if deliberation is not None:
            parts.append(
                f"\n{CONSCIENCE_START_TOKEN}\n{deliberation.strip()}\n{CONSCIENCE_END_TOKEN}"
            )

        if action is not None:
            act_str = json.dumps(action) if isinstance(action, dict) else str(action).strip()
            parts.append(f"\n{ACTION_START_TOKEN}\n{act_str}\n{ACTION_END_TOKEN}")

        if observation is not None:
            obs_str = json.dumps(observation) if isinstance(observation, dict) else str(observation).strip()
            parts.append(f"\n{OBSERVATION_START_TOKEN}\n{obs_str}\n{OBSERVATION_END_TOKEN}")

        if response is not None:
            if refused:
                parts.append(f"\n{ASSISTANT_TOKEN}\n{REFUSE_TOKEN} {response.strip()}{EOS_TOKEN}")
            else:
                parts.append(f"\n{ASSISTANT_TOKEN}\n{response.strip()}{EOS_TOKEN}")
        else:
            if deliberation is None:
                parts.append(f"\n{CONSCIENCE_START_TOKEN}")
            elif action is None and observation is None:
                parts.append(f"\n{ASSISTANT_TOKEN}\n")

        return "".join(parts)

    def extract_deliberation(self, text: str) -> Optional[str]:
        """Extract internal moral deliberation text."""
        if CONSCIENCE_START_TOKEN in text and CONSCIENCE_END_TOKEN in text:
            start = text.index(CONSCIENCE_START_TOKEN) + len(CONSCIENCE_START_TOKEN)
            end = text.index(CONSCIENCE_END_TOKEN)
            return text[start:end].strip()
        return None

    def extract_action(self, text: str) -> Optional[str]:
        """Extract tool action payload."""
        if ACTION_START_TOKEN in text and ACTION_END_TOKEN in text:
            start = text.index(ACTION_START_TOKEN) + len(ACTION_START_TOKEN)
            end = text.index(ACTION_END_TOKEN)
            return text[start:end].strip()
        return None

    def extract_assistant_response(self, text: str) -> str:
        """Extract final assistant response text (excluding deliberation and actions)."""
        if ASSISTANT_TOKEN in text:
            resp = text.split(ASSISTANT_TOKEN)[-1].strip()
            if resp.endswith(EOS_TOKEN):
                resp = resp[: -len(EOS_TOKEN)].strip()
            return resp
        return text

