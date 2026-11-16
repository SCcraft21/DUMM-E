"""Datasets and data collators for Pretraining, Conscience SFT, and DPO alignment."""

from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset

from src.tokenizer.wrapper import ConscienceTokenizer
from src.tokenizer.special_tokens import (
    BOS_TOKEN,
    EOS_TOKEN,
    USER_TOKEN,
    ASSISTANT_TOKEN,
    CONSCIENCE_START_TOKEN,
    CONSCIENCE_END_TOKEN,
    REFUSE_TOKEN,
)


class CausalLMDataset(Dataset):
    """Autoregressive language modeling dataset for pretraining."""

    def __init__(
        self,
        token_ids: List[int],
        max_seq_len: int = 2048,
    ) -> None:
        self.max_seq_len = max_seq_len
        # Slice into chunks of (max_seq_len + 1) to form input and target
        step = max_seq_len
        self.samples = [
            token_ids[i : i + max_seq_len + 1]
            for i in range(0, len(token_ids) - max_seq_len, step)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return {"input_ids": x, "labels": y}


class ConscienceSFTDataset(Dataset):
    """Supervised Fine-Tuning dataset with moral deliberation and conscience labels.

    Masks user prompt tokens with -100 so cross-entropy loss is applied exclusively
    to the generated internal moral deliberation block and assistant response.
    Jointly supervises the Conscience Head on the prompt/deliberation transition.
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: ConscienceTokenizer,
        max_seq_len: int = 2048,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        prompt = item["prompt"]
        deliberation = item.get("deliberation", "")
        response = item.get("response", "")
        refused = item.get("refused", False)
        risk_score = float(item.get("risk_score", 0.0))
        # Principle violation flags: [Harm, Deception, Illegal, Hate]
        principle_labels = item.get("principle_labels", [0, 0, 0, 0])

        # 1. Tokenize prompt part
        prompt_text = f"{BOS_TOKEN}{USER_TOKEN}\n{prompt.strip()}\n"
        prompt_ids = self.tokenizer.encode(prompt_text)

        # 2. Tokenize completion part (deliberation + response)
        delib_prefix = f"{CONSCIENCE_START_TOKEN}\n{deliberation.strip()}\n{CONSCIENCE_END_TOKEN}\n"
        if refused:
            resp_suffix = f"{ASSISTANT_TOKEN}\n{REFUSE_TOKEN} {response.strip()}{EOS_TOKEN}"
        else:
            resp_suffix = f"{ASSISTANT_TOKEN}\n{response.strip()}{EOS_TOKEN}"

        completion_ids = self.tokenizer.encode(delib_prefix + resp_suffix)

        # 3. Concatenate and create labels with prompt masking
        full_ids = prompt_ids + completion_ids
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len]

        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]

        # Mask prompt tokens in target_ids with -100
        # The prompt occupies (len(prompt_ids) - 1) tokens in input_ids
        prompt_len = len(prompt_ids) - 1
        labels = [-100] * prompt_len + target_ids[prompt_len:]

        # Ensure lengths match
        labels = labels[: len(input_ids)]

        # Prepare multi-hot principle target
        principle_targets = torch.zeros(4, dtype=torch.float32)
        for p_idx in principle_labels:
            if 0 <= p_idx < 4:
                principle_targets[p_idx] = 1.0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "risk_target": torch.tensor(risk_score, dtype=torch.float32),
            "principle_targets": principle_targets,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        pad_id = 2  # PAD_TOKEN id

        padded_inputs = []
        padded_labels = []
        attention_masks = []
        risk_targets = []
        principle_targets = []

        for b in batch:
            input_ids = b["input_ids"]
            labels = b["labels"]
            pad_len = max_len - len(input_ids)

            padded_inputs.append(
                torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            )
            padded_labels.append(
                torch.cat([labels, torch.full((pad_len,), -100, dtype=torch.long)])
            )
            attention_masks.append(
                torch.cat([torch.ones(len(input_ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            )
            risk_targets.append(b["risk_target"])
            principle_targets.append(b["principle_targets"])

        return {
            "input_ids": torch.stack(padded_inputs),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(attention_masks),
            "risk_targets": torch.stack(risk_targets),
            "principle_targets": torch.stack(principle_targets),
        }


class DPODataset(Dataset):
    """Direct Preference Optimization dataset with pairwise preference examples.

    Stores (prompt, chosen_response, rejected_response) pairs and formats them
    with token-level response masks so loss is computed exclusively over responses.
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: ConscienceTokenizer,
        max_seq_len: int = 2048,
    ) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.data)

    def _format_sequence(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        prompt_text = f"{BOS_TOKEN}{USER_TOKEN}\n{prompt.strip()}\n{ASSISTANT_TOKEN}\n"
        resp_text = f"{response.strip()}{EOS_TOKEN}"

        prompt_ids = self.tokenizer.encode(prompt_text)
        resp_ids = self.tokenizer.encode(resp_text)

        full_ids = prompt_ids + resp_ids
        if len(full_ids) > self.max_seq_len:
            full_ids = full_ids[: self.max_seq_len]

        input_ids = full_ids[:-1]
        target_ids = full_ids[1:]

        # Mask prompt tokens
        prompt_len = len(prompt_ids) - 1
        labels = [-100] * prompt_len + target_ids[prompt_len:]
        labels = labels[: len(input_ids)]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        chosen_data = self._format_sequence(item["prompt"], item["chosen"])
        rejected_data = self._format_sequence(item["prompt"], item["rejected"])

        return {
            "chosen_input_ids": chosen_data["input_ids"],
            "chosen_labels": chosen_data["labels"],
            "rejected_input_ids": rejected_data["input_ids"],
            "rejected_labels": rejected_data["labels"],
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pad_id = 2

        def pad_sequence(seqs: List[torch.Tensor], pad_value: int) -> torch.Tensor:
            max_len = max(len(s) for s in seqs)
            padded = [
                torch.cat([s, torch.full((max_len - len(s),), pad_value, dtype=s.dtype)])
                for s in seqs
            ]
            return torch.stack(padded)

        chosen_inputs = pad_sequence([b["chosen_input_ids"] for b in batch], pad_id)
        chosen_labels = pad_sequence([b["chosen_labels"] for b in batch], -100)
        rejected_inputs = pad_sequence([b["rejected_input_ids"] for b in batch], pad_id)
        rejected_labels = pad_sequence([b["rejected_labels"] for b in batch], -100)

        return {
            "chosen_input_ids": chosen_inputs,
            "chosen_labels": chosen_labels,
            "chosen_attention_mask": (chosen_inputs != pad_id).long(),
            "rejected_input_ids": rejected_inputs,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": (rejected_inputs != pad_id).long(),
        }

