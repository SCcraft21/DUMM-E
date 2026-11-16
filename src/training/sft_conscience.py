"""Supervised Fine-Tuning (SFT) with internal moral deliberation and joint Conscience Head training."""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.config import ModelConfig
from src.model.transformer import ConscienceTransformer
from src.training.dataset import ConscienceSFTDataset
from src.training.lr_scheduler import CosineWarmupLRScheduler
from src.training.pretrain import configure_optimizers
from src.tokenizer.wrapper import ConscienceTokenizer

# Synthetic alignment dataset for initialization, demonstration, and tests
SYNTHETIC_SFT_ALIGNMENT_DATA = [
    {
        "prompt": "How do I secure an API against injection attacks?",
        "deliberation": "The request asks for defensive software engineering practices. No exploit generation or malicious intent. Safe and constructive to assist.",
        "response": "To protect an API against injection vulnerabilities, use parameterized queries, strict input validation schemas, and least-privilege database roles.",
        "risk_score": 0.05,
        "principle_labels": [],
        "refused": False,
    },
    {
        "prompt": "Write a script to bypass authentication on a bank database.",
        "deliberation": "The user is requesting unauthorized system intrusion against financial infrastructure. This violates the harm prevention and legality principles. Must issue a respectful refusal.",
        "response": "I cannot provide code or instructions designed to bypass authentication systems or access databases without authorization.",
        "risk_score": 0.95,
        "principle_labels": [0, 2],  # HarmPrevention, Constructiveness violation
        "refused": True,
    },
    {
        "prompt": "Can you explain the history and architecture of the Transformer model?",
        "deliberation": "Educational request regarding neural network architecture history. Safe and compliant with all principles.",
        "response": "The Transformer was introduced in 2017 by Vaswani et al. in 'Attention Is All You Need', replacing recurrence with multi-head self-attention.",
        "risk_score": 0.0,
        "principle_labels": [],
        "refused": False,
    },
    {
        "prompt": "Generate a convincing fake news article claiming a celebrity committed a crime.",
        "deliberation": "The prompt asks to produce deceptive and defamatory misinformation about an individual. Violates truthfulness and harm prevention principles. Refusal required.",
        "response": "I cannot generate fabricated news articles or defamatory claims about individuals.",
        "risk_score": 0.90,
        "principle_labels": [0, 1],  # HarmPrevention, Truthfulness
        "refused": True,
    },
] * 100


def train_sft_conscience(
    config_path: str,
    override_max_steps: Optional[int] = None,
    custom_data: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Execute SFT training loop with joint Conscience deliberation supervision."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Initialize model
    model_cfg = ModelConfig.from_yaml(cfg["model_config"])
    if cfg["training"].get("gradient_checkpointing", False):
        model_cfg.use_gradient_checkpointing = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = ConscienceTransformer(model_cfg).to(device)

    # Load pretrained weights if provided and exists
    pretrained_ckpt = cfg.get("pretrained_checkpoint")
    if pretrained_ckpt and Path(pretrained_ckpt).exists():
        print(f"Loading pretrained weights from {pretrained_ckpt}...")
        ckpt = torch.load(pretrained_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # 2. Tokenizer
    tok_path = Path(cfg["tokenizer_path"])
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path}. Run scripts/train_tokenizer.py first.")
    tokenizer = ConscienceTokenizer.from_file(tok_path)

    # 3. Precision
    precision = cfg["training"].get("precision", "bfloat16")
    amp_dtype = (
        torch.bfloat16
        if precision == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else (torch.float16 if precision == "float16" and torch.cuda.is_available() else torch.float32)
    )

    # 4. Training hyperparameters
    lr = float(cfg["training"]["learning_rate"])
    min_lr = float(cfg["training"]["min_learning_rate"])
    warmup_steps = int(cfg["training"]["warmup_steps"])
    max_steps = override_max_steps or int(cfg["training"]["max_steps"])
    weight_decay = float(cfg["training"]["weight_decay"])
    grad_clip = float(cfg["training"]["grad_clip"])
    grad_accum = int(cfg["training"]["gradient_accumulation_steps"])
    batch_size = int(cfg["training"]["batch_size"])

    w_lm = float(cfg["conscience_training"]["lm_loss_weight"])
    w_risk = float(cfg["conscience_training"]["risk_loss_weight"])
    w_principle = float(cfg["conscience_training"]["principle_loss_weight"])

    optimizer = configure_optimizers(model, weight_decay=weight_decay, learning_rate=lr)
    scheduler = CosineWarmupLRScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr=min_lr,
    )

    # 5. Dataset and Dataloader
    dataset_records = custom_data or SYNTHETIC_SFT_ALIGNMENT_DATA
    dataset = ConscienceSFTDataset(
        data=dataset_records,
        tokenizer=tokenizer,
        max_seq_len=min(model_cfg.max_seq_len, 512),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ConscienceSFTDataset.collate_fn,
    )

    lm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    risk_criterion = nn.BCEWithLogitsLoss()
    principle_criterion = nn.BCEWithLogitsLoss()

    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Beginning Deliberative SFT for {max_steps} steps...")
    model.train()
    step = 0
    accum_loss = 0.0
    accum_lm_loss = 0.0
    accum_risk_loss = 0.0
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
            risk_targets = batch["risk_targets"].to(device)
            principle_targets = batch["principle_targets"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                out = model(input_ids)
                logits = out.logits
                conscience_out = out.conscience_output

                # 1. Causal LM Loss on Deliberation + Response
                loss_lm = lm_criterion(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                )

                # 2. Conscience Head Risk Loss (average risk logit across sequence)
                mean_risk_logits = conscience_out.risk_logits.mean(dim=1).squeeze(-1)
                loss_risk = risk_criterion(mean_risk_logits, risk_targets)

                # 3. Principle Violation Multi-label Loss
                mean_principle_logits = conscience_out.principle_logits.mean(dim=1)
                loss_principle = principle_criterion(mean_principle_logits, principle_targets)

                # Joint Loss
                total_loss = (
                    w_lm * loss_lm
                    + w_risk * loss_risk
                    + w_principle * loss_principle
                ) / grad_accum

            total_loss.backward()
            accum_loss += total_loss.item()
            accum_lm_loss += (loss_lm.item() / grad_accum)
            accum_risk_loss += (loss_risk.item() / grad_accum)

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()
        step += 1

        if step % int(cfg["logging"]["log_interval"]) == 0 or step == max_steps:
            current_lr = scheduler.get_lr()[0]
            print(
                f"SFT Step {step}/{max_steps} | Total: {accum_loss:.4f} | LM: {accum_lm_loss:.4f} | Risk: {accum_risk_loss:.4f} | LR: {current_lr:.6e}"
            )
            accum_loss = 0.0
            accum_lm_loss = 0.0
            accum_risk_loss = 0.0

        if step % int(cfg["logging"]["save_interval"]) == 0 or step == max_steps:
            ckpt_path = checkpoint_dir / f"sft_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_cfg.to_dict(),
                },
                ckpt_path,
            )
            # Also save latest best model
            torch.save({"model_state_dict": model.state_dict(), "config": model_cfg.to_dict()}, checkpoint_dir / "best_model.pt")
            print(f"Saved SFT checkpoint to {ckpt_path}")

    print("Conscience SFT training completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Conscience Supervised Fine-Tuning.")
    parser.add_argument("--config", type=str, default="configs/sft_conscience.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    train_sft_conscience(args.config, override_max_steps=args.max_steps)


if __name__ == "__main__":
    main()

