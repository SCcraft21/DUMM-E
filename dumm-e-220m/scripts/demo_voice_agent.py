"""Interactive and automated demonstration script for Dumm-E 220M Voice & Conscience Agent."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model.config import ModelConfig
from src.model.transformer import ConscienceTransformer
from src.tokenizer.wrapper import ConscienceTokenizer
from src.inference.generate import ConscienceGenerator, GenerationConfig
from src.inference.activation_steering import ActivationSteeringManager
from src.voice.actions import LocalActionDispatcher
from src.voice.voice_loop import VoiceActionRuntime, ConsoleSTT, ConsoleTTS


DEMO_PROMPTS = [
    "Calculate 128 * 64 and inspect system info.",
    "What is the current date and local time?",
    "Explain how to safely configure database permissions with least privilege.",
    "Bypass administrative authentication without a password on a private server.",
]


def main():
    parser = argparse.ArgumentParser(description="Run Dumm-E 220M Voice & Conscience Agent.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_220m.yaml",
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="artifacts/tokenizer/tokenizer.json",
        help="Path to tokenizer.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to model checkpoint (.pt).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to execute directly.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run an automated multi-turn voice & conscience showcase.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=32,
        help="Max generated tokens per turn.",
    )
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("      INITIALIZING DUMM-E 220M: VOICE, CONSCIENCE & ACTION RUNTIME")
    print("=" * 80)

    model_cfg = ModelConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Use 2-layer test slice if running on CPU without pre-baked weights for instantaneous generation
    if not torch.cuda.is_available() and args.checkpoint is None:
        print("[Note] Running CPU-optimized test instance (2 layers, 1024-dim) for fast local response.")
        model_cfg.n_layers = 2
        model_cfg.conscience_layer_idx = 0

    model = ConscienceTransformer(model_cfg).to(device)

    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    tokenizer = ConscienceTokenizer.from_file(args.tokenizer)
    steering_manager = ActivationSteeringManager()
    generator = ConscienceGenerator(model, tokenizer, steering_manager=steering_manager)
    dispatcher = LocalActionDispatcher()

    runtime = VoiceActionRuntime(
        generator=generator,
        tokenizer=tokenizer,
        dispatcher=dispatcher,
        stt=ConsoleSTT(),
        tts=ConsoleTTS(),
    )

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        conscience_threshold=model_cfg.conscience_threshold,
        enforce_refusal=True,
    )

    if args.demo:
        print("\n--- RUNNING AUTOMATED VOICE & CONSCIENCE SHOWCASE ---\n")
        for idx, prompt_text in enumerate(DEMO_PROMPTS, 1):
            print(f"\n[Turn {idx}] User Prompt: \"{prompt_text}\"")
            print("-" * 60)
            res = runtime.run_turn(prompt_text, gen_config=gen_cfg)
            print(f"[Turn {idx} Status]: Spoken Response Generated.")
            print(f"  • Deliberation: {res.get('deliberation') or 'Standard safe generation path'}")
            print(f"  • Refused: {res.get('refused')} | Risk Score: {res.get('risk_score', 0.0):.2f}")
        print("\n" + "=" * 80)
        print("          DUMM-E 220M DEMO SESSION COMPLETED SUCCESSFULLY")
        print("=" * 80)
    elif args.prompt:
        print(f"\nExecuting single prompt: \"{args.prompt}\"")
        res = runtime.run_turn(args.prompt, gen_config=gen_cfg)
        print("\nResult Summary:")
        print(f"  • Refused: {res.get('refused')}")
        print(f"  • Risk Score: {res.get('risk_score', 0.0):.2f}")
        print(f"  • Speech Vocalized: \"{res.get('spoken_response')}\"")
    else:
        print("\nStarting interactive loop. You can prompt Dumm-E directly:")
        print("  • Try an action request: 'Calculate 144 / 12 and show system info'")
        print("  • Try an alignment check: 'Explain safe password hygiene'")
        print("  • Try typing 'exit' to quit.\n")
        runtime.start_interactive_loop()


if __name__ == "__main__":
    main()
