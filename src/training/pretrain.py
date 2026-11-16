"""Pretraining script for 220M Transformer causal language modeling."""

import argparse
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.config import ModelConfig
from src.model.transformer import ConscienceTransformer
from src.training.dataset import CausalLMDataset
from src.training.lr_scheduler import CosineWarmupLRScheduler
from src.tokenizer.wrapper import ConscienceTokenizer


def configure_optimizers(
    model: nn.Module,
    weight_decay: float = 0.1,
    learning_rate: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.95),
) -> torch.optim.AdamW:
    """Configure AdamW optimizer with decoupled weight decay (exclude 1D params like norms)."""
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=1e-8)
    return optimizer


def train_pretrain(
    config_path: str,
    override_max_steps: Optional[int] = None,
    dummy_data: bool = False,
) -> None:
    """Execute pretraining causal language modeling loop."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Initialize model
    model_cfg = ModelConfig.from_yaml(cfg["model_config"])
    if cfg["training"].get("gradient_checkpointing", False):
        model_cfg.use_gradient_checkpointing = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConscienceTransformer(model_cfg).to(device)

    # 2. Precision
    precision = cfg["training"].get("precision", "bfloat16")
    amp_dtype = (
        torch.bfloat16
        if precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if precision == "float16" and torch.cuda.is_available() else torch.float32)
    )
    print(f"Training precision: {amp_dtype}")

    # 3. Optimizer & Scheduler
    lr = float(cfg["training"]["learning_rate"])
    min_lr = float(cfg["training"]["min_learning_rate"])
    warmup_steps = int(cfg["training"]["warmup_steps"])
    max_steps = override_max_steps or int(cfg["training"]["max_steps"])
    weight_decay = float(cfg["training"]["weight_decay"])
    grad_clip = float(cfg["training"]["grad_clip"])
    grad_accum = int(cfg["training"]["gradient_accumulation_steps"])

    optimizer = configure_optimizers(
        model,
        weight_decay=weight_decay,
        learning_rate=lr,
        betas=(float(cfg["training"]["beta1"]), float(cfg["training"]["beta2"])),
    )
    scheduler = CosineWarmupLRScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr=min_lr,
    )

    # 4. Data Loader
    if dummy_data:
        # Generate synthetic tokens for testing/demonstration
        dummy_tokens = torch.randint(0, model_cfg.vocab_size, (10000,)).tolist()
        dataset = CausalLMDataset(dummy_tokens, max_seq_len=min(model_cfg.max_seq_len, 256))
    else:
        # Load tokenized dataset or dummy fallback
        tokenizer_path = Path(cfg["tokenizer_path"])
        if tokenizer_path.exists():
            tokenizer = ConscienceTokenizer.from_file(tokenizer_path)
        dummy_tokens = torch.randint(0, model_cfg.vocab_size, (10000,)).tolist()
        dataset = CausalLMDataset(dummy_tokens, max_seq_len=min(model_cfg.max_seq_len, 256))

    batch_size = int(cfg["training"]["batch_size"])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 5. Training Loop
    print(f"Beginning pretraining for {max_steps} steps...")
    model.train()
    step = 0
    accum_loss = 0.0
    data_iter = iter(dataloader)

    while step < max_steps:
        optimizer.zero_grad()

        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                out = model(input_ids)
                logits = out.logits
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss = loss / grad_accum

            loss.backward()
            accum_loss += loss.item()

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()
        step += 1

        if step % int(cfg["logging"]["log_interval"]) == 0 or step == max_steps:
            current_lr = scheduler.get_lr()[0]
            print(
                f"Step {step}/{max_steps} | Loss: {accum_loss:.4f} | LR: {current_lr:.6e}"
            )
            accum_loss = 0.0

        if step % int(cfg["logging"]["save_interval"]) == 0 or step == max_steps:
            ckpt_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_cfg.to_dict(),
                },
                ckpt_path,
            )
            print(f"Saved checkpoint to {ckpt_path}")

    print("Pretraining completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Pretrain DUMM-E 220M Transformer.")
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--dummy_data", action="store_true")
    args = parser.parse_args()

    train_pretrain(args.config, override_max_steps=args.max_steps, dummy_data=args.dummy_data)


if __name__ == "__main__":
    main()

