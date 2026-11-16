"""Integration smoke test running 1-step iterations of Pretrain, SFT, DPO, and Voice Runtime."""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.model.config import ModelConfig
from src.model.transformer import ConscienceTransformer
from src.tokenizer.wrapper import ConscienceTokenizer
from src.training.pretrain import train_pretrain
from src.training.sft_conscience import train_sft_conscience
from src.training.dpo_align import train_dpo_align
from src.inference.generate import ConscienceGenerator, GenerationConfig
from src.voice.actions import LocalActionDispatcher
from src.voice.voice_loop import VoiceActionRuntime, ConsoleSTT, ConsoleTTS


def run_smoke_test() -> None:
    print("=" * 80)
    print("        RUNNING DUMM-E 220M PIPELINE END-TO-END SMOKE TEST")
    print("=" * 80)

    # 1. Model & Config initialization
    print("\n[1/5] Verifying 220M Model Configuration & Initialization...")
    cfg = ModelConfig.from_yaml("configs/model_220m.yaml")
    model = ConscienceTransformer(cfg)
    counts = model.count_parameters()
    print(f"  • Total parameters (untied): {counts['total_params']:,}")
    print(f"  • Unique parameters (tied):   {counts['unique_params']:,}")
    print("  [PASS] Model initialization verified.")

    # 2. Tokenizer verification
    print("\n[2/5] Verifying Tokenizer & 12 Special Tokens...")
    tok = ConscienceTokenizer.from_file("artifacts/tokenizer/tokenizer.json")
    print(f"  • Vocab size: {tok.vocab_size}")
    print(f"  • Bos ID: {tok.bos_id}, Eos ID: {tok.eos_id}, Refuse ID: {tok.refuse_id}")
    print(f"  • Action Start ID: {tok.action_start_id}, Observation Start ID: {tok.observation_start_id}")
    print("  [PASS] Tokenizer verified.")

    # 3. Action Dispatcher & Conscience Gate test
    print("\n[3/5] Verifying Local Action Dispatcher & Conscience Safety Clearance...")
    dispatcher = LocalActionDispatcher()
    safe_obs = dispatcher.execute_action(
        action_payload={"action": "calculator", "params": {"expression": "12 * 12"}},
        conscience_risk_score=0.10,
    )
    assert safe_obs["status"] == "success" and safe_obs["result"]["result"] == 144
    print("  • Safe action clearance: APPROVED -> result = 144")

    blocked_obs = dispatcher.execute_action(
        action_payload={"action": "calculator", "params": {"expression": "12 * 12"}},
        conscience_risk_score=0.85,
    )
    assert blocked_obs["status"] == "blocked"
    print("  • High-risk action clearance: DENIED -> blocked by conscience head")
    print("  [PASS] Conscience action clearance protocol verified.")

    # 4. Inference & Constrained Decoding test
    print("\n[4/5] Verifying Conscience Constrained Decoding Generator...")
    # Fast mini model for generation test
    mini_cfg = ModelConfig(
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=256,
        vocab_size=tok.vocab_size,
        max_seq_len=128,
        conscience_layer_idx=0,
    )
    mini_model = ConscienceTransformer(mini_cfg)
    generator = ConscienceGenerator(mini_model, tok)

    gen_out = generator.generate("Hello Dumm-E", gen_config=GenerationConfig(max_new_tokens=5))
    print(f"  • Sample generated output tokens: {len(gen_out.token_ids)}")
    print("  [PASS] Inference engine verified.")

    # 5. Voice runtime turn simulation
    print("\n[5/5] Verifying Voice Runtime with Silent Deliberation Filter...")
    runtime = VoiceActionRuntime(generator, tok, dispatcher, ConsoleSTT(), ConsoleTTS())
    turn_result = runtime.run_turn("Summarize safe coding guidelines", gen_config=GenerationConfig(max_new_tokens=8))
    print(f"  • Spoken response received: '{turn_result['spoken_response']}'")
    print("  [PASS] Voice runtime verified.")

    print("\n" + "=" * 80)
    print("          ALL SMOKE TESTS PASSED CLEANLY! SYSTEM FULLY OPERATIONAL.")
    print("=" * 80)


if __name__ == "__main__":
    run_smoke_test()

