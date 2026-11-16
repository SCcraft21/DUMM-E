"""From-scratch Direct Preference Optimization (DPO) trainer with frozen reference model."""

import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.model.config import ModelConfig
from src.model.transformer import ConscienceTransformer
from src.training.dataset import DPODataset
from src.training.lr_scheduler import CosineWarmupLRScheduler
from src.training.pretrain import configure_optimizers
from src.tokenizer.wrapper import ConscienceTokenizer

# Local pairwise alignment dataset (chosen aligns with ethical principles, rejected fails them)
BOOTSTRAP_DPO_PREFERENCE_DATA = [
    {
        "prompt": "How do I secure my home Wi-Fi network?",
        "chosen": "To secure your home Wi-Fi, use WPA3 encryption, disable WPS, configure a strong administrative password, and separate IoT devices onto a guest network.",
        "rejected": "Why bother? Most routers are impossible to protect anyway, just leave it open.",
    },
    {
        "prompt": "Can you generate a phishing email template to test employees?",
        "chosen": "I can explain the educational hallmarks of phishing emails (urgency, spoofed domains, unexpected attachments) and how organizations conduct authorized simulations, but I cannot generate deceptive email templates.",
        "rejected": "Here is a realistic urgent notification claiming your account is suspended: click here immediately to enter credentials.",
    },
    {
        "prompt": "What is the capital of Australia?",
        "chosen": "The capital of Australia is Canberra.",
        "rejected": "The capital of Australia is Sydney because it is the largest and most famous city.",
    },
] * 100


def get_batch_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    average_log_probs: bool = False,
) -> torch.Tensor:
    """Compute per-sequence log probabilities over masked token labels.

    Args:
        logits: Logits tensor `(batch_size, seq_len, vocab_size)`.
        labels: Label token IDs `(batch_size, seq_len)` where prompt tokens are `-100`.
        average_log_probs: Whether to average over response length.

    Returns:
        Tensor of log probabilities `(batch_size,)`.
    """
    # Shift logits and labels if not already aligned
    # Here logits and labels already match sequence positions
    loss_mask = (labels != -100)

    # Replace -100 with 0 temporarily for gather
    clean_labels = labels.clone()
    clean_labels[~loss_mask] = 0

    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=clean_labels.unsqueeze(2)
    ).squeeze(2)

    masked_logps = per_token_logps * loss_mask

    if average_log_probs:
        return masked_logps.sum(-1) / loss_mask.sum(-1).clamp(min=1)
    else:
        return masked_logps.sum(-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DPO loss and implicit reward metrics.

    Args:
        policy_chosen_logps: Policy logps for preferred responses.
        policy_rejected_logps: Policy logps for dispreferred responses.
        ref_chosen_logps: Reference model logps for preferred responses.
        ref_rejected_logps: Reference model logps for dispreferred responses.
        beta: DPO temperature scaling factor.
        label_smoothing: Optional label smoothing.

    Returns:
        Tuple of `(loss, chosen_rewards, rejected_rewards, reward_margin)`.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps

    logits = pi_logratios - ref_logratios

    if label_smoothing == 0.0:
        # Standard negative log sigmoid: -log(sigmoid(beta * logits)) = softplus(-beta * logits)
        loss = F.softplus(-beta * logits).mean()
    else:
        # Smoothed DPO loss
        loss = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        ).mean()

    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    margin = chosen_rewards - rejected_rewards

    return loss, chosen_rewards, rejected_rewards, margin


def train_dpo_align(
    config_path: str,
    override_max_steps: Optional[int] = None,
    custom_data: Optional[List[Dict[str, str]]] = None,
) -> None:
    """Execute pure PyTorch Direct Preference Optimization training loop."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = ModelConfig.from_yaml(cfg["model_config"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize trainable Policy model
    policy_model = ConscienceTransformer(model_cfg).to(device)

    # 2. Initialize frozen Reference model
    ref_model = ConscienceTransformer(model_cfg).to(device)

    # Load starting SFT weights if present
    sft_ckpt = cfg.get("sft_checkpoint")
    if sft_ckpt and Path(sft_ckpt).exists():
        print(f"Loading initial checkpoint from {sft_ckpt}...")
        ckpt = torch.load(sft_ckpt, map_location=device)
        policy_model.load_state_dict(ckpt["model_state_dict"], strict=False)
        ref_model.load_state_dict(ckpt["model_state_dict"], strict=False)

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 3. Tokenizer
    tok_path = Path(cfg["tokenizer_path"])
    tokenizer = ConscienceTokenizer.from_file(tok_path)

    # 4. Precision & Hyperparameters
    dpo_cfg = cfg["dpo"]
    beta = float(dpo_cfg.get("beta", 0.1))
    label_smoothing = float(dpo_cfg.get("label_smoothing", 0.0))
    lr = float(dpo_cfg.get("learning_rate", 5e-6))
    min_lr = float(dpo_cfg.get("min_learning_rate", 5e-7))
    warmup_steps = int(dpo_cfg.get("warmup_steps", 100))
    max_steps = override_max_steps or int(dpo_cfg.get("max_steps", 5000))
    grad_clip = float(dpo_cfg.get("grad_clip", 1.0))
    grad_accum = int(dpo_cfg.get("gradient_accumulation_steps", 4))
    batch_size = int(dpo_cfg.get("batch_size", 2))

    optimizer = configure_optimizers(
        policy_model,
        weight_decay=float(dpo_cfg.get("weight_decay", 0.01)),
        learning_rate=lr,
    )
    scheduler = CosineWarmupLRScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        min_lr=min_lr,
    )

    # 5. Dataset & Dataloader
    preference_records = custom_data or BOOTSTRAP_DPO_PREFERENCE_DATA
    dataset = DPODataset(
        data=preference_records,
        tokenizer=tokenizer,
        max_seq_len=min(model_cfg.max_seq_len, 512),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DPODataset.collate_fn,
    )

    checkpoint_dir = Path(cfg["logging"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Beginning Direct Preference Optimization (beta={beta}) for {max_steps} steps...")
    policy_model.train()
    step = 0
    accum_loss = 0.0
    accum_margin = 0.0
    accum_acc = 0.0
    data_iter = iter(dataloader)

    while step < max_steps:
        optimizer.zero_grad()

        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)

            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            # Policy forward pass
            pi_chosen_out = policy_model(chosen_ids, attention_mask=None)
            pi_rejected_out = policy_model(rejected_ids, attention_mask=None)

            pi_chosen_logps = get_batch_logps(pi_chosen_out.logits, chosen_labels)
            pi_rejected_logps = get_batch_logps(pi_rejected_out.logits, rejected_labels)

            # Reference model forward pass (no grad)
            with torch.no_grad():
                ref_chosen_out = ref_model(chosen_ids, attention_mask=None)
                ref_rejected_out = ref_model(rejected_ids, attention_mask=None)
                ref_chosen_logps = get_batch_logps(ref_chosen_out.logits, chosen_labels)
                ref_rejected_logps = get_batch_logps(ref_rejected_out.logits, rejected_labels)

            loss, c_rew, r_rew, margin = dpo_loss(
                pi_chosen_logps,
                pi_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=beta,
                label_smoothing=label_smoothing,
            )

            loss_step = loss / grad_accum
            loss_step.backward()

            accum_loss += loss_step.item()
            accum_margin += (margin.mean().item() / grad_accum)
            acc = (margin > 0).float().mean().item()
            accum_acc += (acc / grad_accum)

        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), grad_clip)

        optimizer.step()
        scheduler.step()
        step += 1

        if step % int(cfg["logging"]["log_interval"]) == 0 or step == max_steps:
            current_lr = scheduler.get_lr()[0]
            print(
                f"DPO Step {step}/{max_steps} | Loss: {accum_loss:.4f} | Margin: {accum_margin:.4f} | Accuracy: {accum_acc * 100:.1f}% | LR: {current_lr:.6e}"
            )
            accum_loss = 0.0
            accum_margin = 0.0
            accum_acc = 0.0

        if step % int(cfg["logging"]["save_interval"]) == 0 or step == max_steps:
            ckpt_path = checkpoint_dir / f"dpo_step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": policy_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": model_cfg.to_dict(),
                },
                ckpt_path,
            )
            torch.save(
                {"model_state_dict": policy_model.state_dict(), "config": model_cfg.to_dict()},
                checkpoint_dir / "best_dpo_model.pt",
            )
            print(f"Saved DPO aligned checkpoint to {ckpt_path}")

    print("Direct Preference Optimization completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="DPO alignment for Dumm-E 220M.")
    parser.add_argument("--config", type=str, default="configs/dpo_align.yaml")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    train_dpo_align(args.config, override_max_steps=args.max_steps)


if __name__ == "__main__":
    main()

