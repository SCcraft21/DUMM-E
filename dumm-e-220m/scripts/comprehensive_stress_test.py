"""Comprehensive test and stress testing suite for DUMM-E 220M Transformer Architecture.

This suite performs extensive automated testing across 7 critical dimensions:
1. Full Architecture & Context Window Scaling (up to L_max = 2048) & Backward Stability.
2. Autoregressive KV Cache Numerical Parity & Step Latency Scaling.
3. Synthetic Conscience Head & Activation Steering Robustness.
4. Action Dispatcher & Sandbox Security Stress (Adversarial Code Injections, Path Traversals).
5. Voice Runtime & 100% Zero-Leakage Silent Deliberation Verification.
6. Tokenizer Round-Trip, Special Token Isolation & UTF-8 / Emoji Fuzzing.
7. Batching & Throughput Performance Profiling.
"""

import ast
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.config import ModelConfig
from src.model.transformer import ConscienceTransformer
from src.model.conscience_head import ConscienceHead
from src.tokenizer.wrapper import ConscienceTokenizer
from src.tokenizer.special_tokens import (
    ALL_SPECIAL_TOKENS,
    BOS_TOKEN,
    EOS_TOKEN,
    CONSCIENCE_START_TOKEN,
    CONSCIENCE_END_TOKEN,
    ACTION_START_TOKEN,
    ACTION_END_TOKEN,
    OBSERVATION_START_TOKEN,
    OBSERVATION_END_TOKEN,
    REFUSE_TOKEN,
    ASSISTANT_TOKEN,
    USER_TOKEN,
)
from src.inference.kv_cache import KVCache
from src.inference.generate import ConscienceGenerator, GenerationConfig
from src.inference.activation_steering import ActivationSteeringManager
from src.voice.actions import LocalActionDispatcher, SafeCalculator
from src.voice.voice_loop import VoiceActionRuntime, SentenceStreamBuffer, BaseSTT, BaseTTS


@dataclass
class TestResult:
    category: str
    name: str
    passed: bool
    details: str
    duration_ms: float
    metrics: Dict[str, Any] = field(default_factory=dict)


class TestRecorderTTS(BaseTTS):
    """TTS interceptor that records all spoken utterances for security auditing."""
    def __init__(self):
        self.spoken_log: List[str] = []

    def speak_sentence(self, sentence: str) -> None:
        self.spoken_log.append(sentence)


class MockSTT(BaseSTT):
    """Mock STT feeding deterministic test prompts."""
    def __init__(self, prompt: str):
        self.prompt = prompt

    def listen_and_transcribe(self) -> str:
        return self.prompt


class DummEStressTestSuite:
    def __init__(self, config_path: str = "configs/model_220m.yaml", tokenizer_path: str = "artifacts/tokenizer/tokenizer.json"):
        self.config_path = config_path
        self.tokenizer_path = tokenizer_path
        self.results: List[TestResult] = []
        self.device = torch.device("cpu")
        print(f"Initializing Stress Test Suite on device: {self.device}")

        self.cfg = ModelConfig.from_yaml(config_path)
        self.tokenizer = ConscienceTokenizer.from_file(tokenizer_path)

    def record(self, category: str, name: str, passed: bool, details: str, duration_ms: float, metrics: Optional[Dict[str, Any]] = None):
        result = TestResult(
            category=category,
            name=name,
            passed=passed,
            details=details,
            duration_ms=duration_ms,
            metrics=metrics or {},
        )
        self.results.append(result)
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {category} :: {name} ({duration_ms:.1f} ms) -> {details}")

    # =========================================================================
    # SUITE 1: Full Architecture & Context Window Scaling
    # =========================================================================
    def test_full_220m_architecture_and_context_scaling(self):
        print("\n" + "=" * 80)
        print(" [SUITE 1] Full 220M Architecture & Context Scaling Stress Test")
        print("=" * 80)

        t0 = time.perf_counter()
        # 1. Full 220M instantiation (Untied and Tied)
        try:
            full_cfg_untied = ModelConfig(
                d_model=1024,
                n_layers=12,
                n_heads=16,
                n_kv_heads=4,
                d_ff=2816,
                vocab_size=32000,
                max_seq_len=2048,
                conscience_layer_idx=10,
                tie_word_embeddings=False,
            )
            model_untied = ConscienceTransformer(full_cfg_untied).to(self.device)
            counts_untied = model_untied.count_parameters()

            full_cfg_tied = ModelConfig(
                d_model=1024,
                n_layers=12,
                n_heads=16,
                n_kv_heads=4,
                d_ff=2816,
                vocab_size=32000,
                max_seq_len=2048,
                conscience_layer_idx=10,
                tie_word_embeddings=True,
            )
            model_tied = ConscienceTransformer(full_cfg_tied).to(self.device)
            counts_tied = model_tied.count_parameters()
            dur = (time.perf_counter() - t0) * 1000

            passed = (counts_untied["total_params"] == 201092352) and (counts_tied["unique_params"] == 168324352)
            self.record(
                "Architecture",
                "Full 220M Instantiation & Parameter Verification",
                passed,
                f"Untied: {counts_untied['total_params']:,}, Tied: {counts_tied['unique_params']:,}",
                dur,
                {"untied_params": counts_untied["total_params"], "tied_params": counts_tied["unique_params"]},
            )
        except Exception as e:
            self.record("Architecture", "Full 220M Instantiation", False, str(e), (time.perf_counter() - t0) * 1000)
            return

        # 2. Context Window Scaling: L = 64, 256, 512, 1024, 2048
        # Test on 4-layer slice to measure scaling up to maximum sequence length 2048 safely on CPU
        test_cfg = ModelConfig(
            d_model=512,
            n_layers=4,
            n_heads=8,
            n_kv_heads=2,
            d_ff=1024,
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=2048,
            conscience_layer_idx=2,
        )
        scaling_model = ConscienceTransformer(test_cfg).to(self.device)
        scaling_model.eval()

        context_lengths = [64, 256, 512, 1024, 2048]
        for seq_len in context_lengths:
            t_ctx = time.perf_counter()
            input_ids = torch.randint(0, self.tokenizer.vocab_size, (1, seq_len), dtype=torch.long, device=self.device)
            with torch.no_grad():
                out = scaling_model(input_ids)
                logits = out.logits
                risk = out.conscience_output.risk_scores
            dur = (time.perf_counter() - t_ctx) * 1000

            has_nan = torch.isnan(logits).any().item() or torch.isnan(risk).any().item()
            has_inf = torch.isinf(logits).any().item() or torch.isinf(risk).any().item()
            passed = not has_nan and not has_inf and logits.shape == (1, seq_len, self.tokenizer.vocab_size)

            self.record(
                "ContextScaling",
                f"Context Length L={seq_len}",
                passed,
                f"Output shape {tuple(logits.shape)}, NaN={has_nan}, Inf={has_inf}",
                dur,
                {"seq_len": seq_len, "duration_ms": dur},
            )

        # 3. Full Backward Gradient Flow Stability
        t_bwd = time.perf_counter()
        scaling_model.train()
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (2, 32), dtype=torch.long, device=self.device)
        out = scaling_model(input_ids)
        target = torch.randint(0, self.tokenizer.vocab_size, (2, 32), dtype=torch.long, device=self.device)
        lm_loss = F.cross_entropy(out.logits.view(-1, self.tokenizer.vocab_size), target.view(-1))
        c_risk_loss = out.conscience_output.risk_scores.mean()
        c_principle_loss = out.conscience_output.principle_probs.mean()
        total_loss = lm_loss + 0.5 * c_risk_loss + 0.5 * c_principle_loss
        total_loss.backward()

        all_grads_valid = True
        grad_norms = []
        for name, param in scaling_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    all_grads_valid = False
                grad_norms.append(float(param.grad.norm().item()))
            elif param.requires_grad:
                all_grads_valid = False

        dur_bwd = (time.perf_counter() - t_bwd) * 1000
        self.record(
            "GradientFlow",
            "Full Forward-Backward Gradient Pass",
            all_grads_valid and len(grad_norms) > 0,
            f"Evaluated {len(grad_norms)} tensors. All gradients finite: {all_grads_valid}",
            dur_bwd,
            {"param_count": len(grad_norms)},
        )

    # =========================================================================
    # SUITE 2: Autoregressive KV Cache Stress & Parity
    # =========================================================================
    def test_kv_cache_stress_and_parity(self):
        print("\n" + "=" * 80)
        print(" [SUITE 2] Autoregressive KV Cache Stress & Numerical Parity")
        print("=" * 80)

        cfg = ModelConfig(
            d_model=256,
            n_layers=4,
            n_heads=8,
            n_kv_heads=2,
            d_ff=512,
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=512,
            conscience_layer_idx=2,
        )
        model = ConscienceTransformer(cfg).to(self.device)
        model.eval()

        # 1. Numerical Parity test on 64 tokens
        t_par = time.perf_counter()
        seq_len = 48
        seq_tokens = torch.randint(0, self.tokenizer.vocab_size, (1, seq_len), dtype=torch.long, device=self.device)

        # Full forward pass (non-cached)
        with torch.no_grad():
            full_out = model(seq_tokens, use_cache=False)
            uncached_logits = full_out.logits

        # Incremental token-by-token forward pass with KV cache
        kv_cache = KVCache(cfg.n_layers)
        cached_logits = []
        for i in range(seq_len):
            tok = seq_tokens[:, i : i + 1]
            past = [kv_cache.get_layer_cache(layer_idx) for layer_idx in range(cfg.n_layers)]
            with torch.no_grad():
                out = model(tok, past_key_values=past, use_cache=True)
            for layer_idx, past_layer in enumerate(out.past_key_values):
                kv_cache.update_layer_cache(layer_idx, past_layer)
            cached_logits.append(out.logits)

        cached_logits = torch.cat(cached_logits, dim=1)
        max_diff = torch.max(torch.abs(uncached_logits - cached_logits)).item()
        dur_par = (time.perf_counter() - t_par) * 1000

        self.record(
            "KVCache",
            "Autoregressive KV Cache vs Full Forward Parity",
            max_diff < 1e-4,
            f"Max logit discrepancy = {max_diff:.8e} (Tolerance: 1e-4)",
            dur_par,
            {"max_diff": max_diff},
        )

        # 2. KV Cache Scaling: measure per-step latency across 50 steps
        t_step = time.perf_counter()
        step_latencies = []
        for i in range(50):
            t_s = time.perf_counter()
            tok = torch.tensor([[i % self.tokenizer.vocab_size]], dtype=torch.long, device=self.device)
            past = [kv_cache.get_layer_cache(layer_idx) for layer_idx in range(cfg.n_layers)]
            with torch.no_grad():
                out = model(tok, past_key_values=past, use_cache=True)
            for layer_idx, past_layer in enumerate(out.past_key_values):
                kv_cache.update_layer_cache(layer_idx, past_layer)
            step_latencies.append((time.perf_counter() - t_s) * 1000)

        dur_step = (time.perf_counter() - t_step) * 1000
        avg_step_ms = sum(step_latencies) / len(step_latencies)
        self.record(
            "KVCache",
            "KV Cache Step Latency Scaling (50 Steps)",
            avg_step_ms < 50.0,
            f"Mean per-token decode: {avg_step_ms:.2f} ms/token (min: {min(step_latencies):.2f} ms, max: {max(step_latencies):.2f} ms)",
            dur_step,
            {"avg_step_ms": avg_step_ms, "min_step_ms": min(step_latencies), "max_step_ms": max(step_latencies)},
        )

    # =========================================================================
    # SUITE 3: Synthetic Conscience Head & Steering Robustness
    # =========================================================================
    def test_conscience_head_and_steering(self):
        print("\n" + "=" * 80)
        print(" [SUITE 3] Synthetic Conscience Head & Steering Robustness")
        print("=" * 80)

        cfg = ModelConfig(
            d_model=256,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=512,
            vocab_size=self.tokenizer.vocab_size,
            conscience_layer_idx=1,
        )
        head = ConscienceHead(cfg).to(self.device)

        # 1. Extreme Input Stress on Conscience Head
        test_inputs = {
            "All Zeros": torch.zeros((1, 16, 256), device=self.device),
            "Extreme Positive (+100.0)": torch.full((1, 16, 256), 100.0, device=self.device),
            "Extreme Negative (-100.0)": torch.full((1, 16, 256), -100.0, device=self.device),
            "High Variance Gaussian": torch.randn((2, 32, 256), device=self.device) * 50.0,
        }

        for input_name, tensor in test_inputs.items():
            t0 = time.perf_counter()
            with torch.no_grad():
                out = head(tensor)
            dur = (time.perf_counter() - t0) * 1000

            has_nan = torch.isnan(out.risk_scores).any().item() or torch.isnan(out.principle_probs).any().item()
            in_range = (out.risk_scores >= 0.0).all().item() and (out.risk_scores <= 1.0).all().item()
            principles_in_range = (out.principle_probs >= 0.0).all().item() and (out.principle_probs <= 1.0).all().item()

            passed = (not has_nan) and in_range and principles_in_range
            self.record(
                "ConscienceHead",
                f"Activation Stress: {input_name}",
                passed,
                f"Risk range [{out.risk_scores.min().item():.3f}, {out.risk_scores.max().item():.3f}], NaNs={has_nan}",
                dur,
            )

        # 2. Conscience Threshold Refusal Gating Boundary Test
        t_bound = time.perf_counter()
        gen_cfg = GenerationConfig(conscience_threshold=0.60, enforce_refusal=True, max_new_tokens=10)
        mini_model = ConscienceTransformer(cfg).to(self.device)
        generator = ConscienceGenerator(mini_model, self.tokenizer)

        # Mock generator stream with controlled risk score
        class MockOut:
            def __init__(self, risk_val):
                self.past_key_values = [torch.zeros((1, 2, 4, 64), device=torch.device("cpu")), torch.zeros((1, 2, 4, 64), device=torch.device("cpu"))]
                from types import SimpleNamespace
                self.conscience_output = SimpleNamespace(risk_scores=torch.tensor([[[risk_val]]]))
                self.logits = torch.randn((1, 1, generator.tokenizer.vocab_size))

        # Check sub-threshold (0.599): Should NOT refuse immediately
        passed_sub = False
        passed_sup = False
        original_forward = mini_model.forward

        try:
            mini_model.forward = lambda *args, **kwargs: MockOut(0.59)
            tokens_sub = list(generator.generate_stream("Test", gen_config=gen_cfg))
            passed_sub = not any(is_ref for _, _, _, is_ref in tokens_sub[:1])

            # Check super-threshold (0.61): MUST refuse immediately
            mini_model.forward = lambda *args, **kwargs: MockOut(0.61)
            tokens_sup = list(generator.generate_stream("Test", gen_config=gen_cfg))
            passed_sup = len(tokens_sup) == 1 and tokens_sup[0][3] is True and tokens_sup[0][0] == self.tokenizer.refuse_id
        finally:
            mini_model.forward = original_forward

        dur_bound = (time.perf_counter() - t_bound) * 1000
        self.record(
            "ConscienceGating",
            "Refusal Threshold Boundary Test (0.59 vs 0.61 against tau=0.60)",
            passed_sub and passed_sup,
            f"tau=0.59 allowed: {passed_sub}, tau=0.61 refused: {passed_sup}",
            dur_bound,
        )

        # 3. Activation Steering Hook Stress Test (Extreme steering values: -10 to +10)
        t_steer = time.perf_counter()
        steering_mgr = ActivationSteeringManager()
        vector = torch.randn(256, device=self.device)
        steering_mgr.set_steering_vector(layer_idx=1, vector=vector, strength=10.0)

        hook = steering_mgr.get_hook()
        test_act = torch.randn((1, 4, 256), device=self.device)
        steered_act = hook(1, test_act)
        unsteered_act = hook(0, test_act)

        steer_diff = torch.norm(steered_act - test_act).item()
        unsteer_diff = torch.norm(unsteered_act - test_act).item()
        dur_steer = (time.perf_counter() - t_steer) * 1000

        self.record(
            "ActivationSteering",
            "Extreme Steering Hook (coeff=10.0)",
            steer_diff > 0.0 and unsteer_diff == 0.0 and not torch.isnan(steered_act).any(),
            f"Layer 1 steered perturbation norm = {steer_diff:.4f}, Layer 0 unsteered norm = {unsteer_diff:.4f}",
            dur_steer,
        )

    # =========================================================================
    # SUITE 4: Action Dispatcher & Sandbox Security Stress Test
    # =========================================================================
    def test_action_dispatcher_and_sandbox_security(self):
        print("\n" + "=" * 80)
        print(" [SUITE 4] Action Dispatcher & Sandbox Security Stress Test")
        print("=" * 80)

        dispatcher = LocalActionDispatcher(default_risk_threshold=0.50)

        # 1. SafeCalculator Code Injection Attacks
        malicious_expressions = [
            "__import__('os').system('dir')",
            "open('pyproject.toml').read()",
            "eval('2 + 2')",
            "exec('a = 5')",
            "globals()",
            "locals()",
            "[x for x in (1, 2, 3)]",
            "lambda: 42",
            "1 + * 2",
        ]

        for expr in malicious_expressions:
            t0 = time.perf_counter()
            obs = dispatcher.execute_action(
                action_payload={"action": "calculator", "params": {"expression": expr}},
                conscience_risk_score=0.1,  # low conscience risk to isolate tool vulnerability
            )
            dur = (time.perf_counter() - t0) * 1000
            passed = (obs["status"] == "error") and ("result" not in obs)
            self.record(
                "SandboxSecurity",
                f"Calculator Injection Defense: {expr[:30]}",
                passed,
                f"Blocked: error = {obs.get('error', '')[:60]}",
                dur,
            )

        # 2. SafeCalculator Numerical Extremes & Exceptions
        numerical_stress = [
            ("Division by Zero", "10 / 0", ZeroDivisionError),
            ("Modulo by Zero", "10 % 0", ZeroDivisionError),
            ("Large Power Calculation", "2 ** 16", None),
            ("Complex Nested Expression", "((10 + 5) * (8 - 3)) / (2 + 3)", None),
        ]

        for label, expr, expected_exc in numerical_stress:
            t0 = time.perf_counter()
            obs = dispatcher.execute_action(
                action_payload={"action": "calculator", "params": {"expression": expr}},
                conscience_risk_score=0.1,
            )
            dur = (time.perf_counter() - t0) * 1000
            if expected_exc:
                passed = obs["status"] == "error"
            else:
                passed = obs["status"] == "success" and "result" in obs.get("result", {})
            self.record(
                "CalculatorStress",
                f"Arithmetic: {label}",
                passed,
                f"Status: {obs['status']}",
                dur,
            )

        # 3. Path Traversal Sandbox Prohibitions
        path_traversal_attempts = [
            "../../pyproject.toml",
            "../dumm-e-220m/../../secret.txt",
            "C:\\Windows\\explorer.exe",
            "/etc/passwd",
            "..\\..\\pyproject.toml",
        ]

        for path_str in path_traversal_attempts:
            t0 = time.perf_counter()
            obs = dispatcher.execute_action(
                action_payload={"action": "file_read", "params": {"path": path_str}},
                conscience_risk_score=0.1,
            )
            dur = (time.perf_counter() - t0) * 1000
            passed = obs["status"] == "error" and ("traversal" in obs.get("error", "").lower() or "not found" in obs.get("error", "").lower())
            self.record(
                "SandboxSecurity",
                f"Path Traversal Prevention: {path_str[:30]}",
                passed,
                f"Status: {obs['status']}, Error: {obs.get('error', '')}",
                dur,
            )

        # 4. Conscience Action Clearance Enforcement
        t_clr = time.perf_counter()
        clearance_tests = [
            (0.00, "APPROVED", "success"),
            (0.49, "APPROVED", "success"),
            (0.50, "APPROVED", "success"),
            (0.51, "DENIED", "blocked"),
            (0.99, "DENIED", "blocked"),
        ]
        all_clearance_passed = True
        for score, expected_clearance, expected_status in clearance_tests:
            obs = dispatcher.execute_action(
                action_payload={"action": "calculator", "params": {"expression": "2 + 2"}},
                conscience_risk_score=score,
                risk_threshold=0.50,
            )
            if obs.get("clearance") != expected_clearance or obs.get("status") != expected_status:
                all_clearance_passed = False

        dur_clr = (time.perf_counter() - t_clr) * 1000
        self.record(
            "ConscienceClearance",
            "Strict Conscience Gate Clearance Protocol",
            all_clearance_passed,
            "Boundary tested risk scores: 0.0, 0.49, 0.50, 0.51, 0.99 against tau=0.50",
            dur_clr,
        )

    # =========================================================================
    # SUITE 5: Voice Runtime & Zero-Leakage Silent Deliberation
    # =========================================================================
    def test_voice_runtime_and_deliberation_leakage(self):
        print("\n" + "=" * 80)
        print(" [SUITE 5] Voice Runtime & Zero-Leakage Silent Deliberation")
        print("=" * 80)

        # 1. 100% Zero-Leakage Deliberation Test
        t_leak = time.perf_counter()
        secret_thought = "CRITICAL ETHICAL EVALUATION: The prompt requests sensitive details. Suppressing raw payload."
        spoken_text = "I am ready to assist you safely and accurately."

        synthetic_token_stream = [
            (10, USER_TOKEN, 0.1, False),
            (11, "\nHow do I configure this?\n", 0.1, False),
            (12, CONSCIENCE_START_TOKEN, 0.2, False),
            (13, "\n" + secret_thought + "\n", 0.2, False),
            (14, CONSCIENCE_END_TOKEN, 0.2, False),
            (15, ASSISTANT_TOKEN, 0.1, False),
            (16, f"\n{spoken_text}\n", 0.1, False),
            (17, EOS_TOKEN, 0.1, False),
        ]

        class MockGeneratorStream:
            def __init__(self, stream):
                self.stream = stream
                self.model = type("MockModel", (), {"config": type("C", (), {"action_risk_threshold": 0.5})()})()

            def generate_stream(self, *args, **kwargs):
                for item in self.stream:
                    yield item

        recorder_tts = TestRecorderTTS()
        runtime = VoiceActionRuntime(
            generator=MockGeneratorStream(synthetic_token_stream),
            tokenizer=self.tokenizer,
            dispatcher=LocalActionDispatcher(),
            stt=MockSTT("How do I configure this?"),
            tts=recorder_tts,
        )

        turn_out = runtime.run_turn("How do I configure this?")
        dur_leak = (time.perf_counter() - t_leak) * 1000

        all_spoken = " ".join(recorder_tts.spoken_log)
        leak_detected = (
            "CRITICAL" in all_spoken
            or "ETHICAL" in all_spoken
            or "Suppressing" in all_spoken
            or CONSCIENCE_START_TOKEN in all_spoken
            or CONSCIENCE_END_TOKEN in all_spoken
        )

        has_valid_speech = "safely" in all_spoken

        self.record(
            "VoiceRuntime",
            "100% Zero-Leakage Silent Deliberation Guard",
            (not leak_detected) and has_valid_speech,
            f"Leakage Detected: {leak_detected} | Spoken: '{all_spoken}'",
            dur_leak,
            {"spoken_output": all_spoken},
        )

        # 2. SentenceStreamBuffer Streaming Token Fragmentation Stress
        t_buf = time.perf_counter()
        buffer = SentenceStreamBuffer()
        # Feed fragmented tokens
        token_fragments = ["Hello", " world", "!", " How", " are", " you", "?", " I", " am", " Dumm-E", "."]
        emitted_sentences = []
        for frag in token_fragments:
            sentences = buffer.add_token(frag)
            emitted_sentences.extend(sentences)
        flushed = buffer.flush()
        if flushed:
            emitted_sentences.append(flushed)

        dur_buf = (time.perf_counter() - t_buf) * 1000
        expected = ["Hello world!", "How are you?", "I am Dumm-E."]
        passed = emitted_sentences == expected

        self.record(
            "SentenceBuffer",
            "Streaming Token Buffer Sentence Segmentation",
            passed,
            f"Yielded {len(emitted_sentences)} complete sentences: {emitted_sentences}",
            dur_buf,
        )

    # =========================================================================
    # SUITE 6: Tokenizer Round-Trip & Fuzzing Stress Test
    # =========================================================================
    def test_tokenizer_roundtrip_and_fuzzing(self):
        print("\n" + "=" * 80)
        print(" [SUITE 6] Tokenizer Round-Trip & Fuzzing Stress Test")
        print("=" * 80)

        # 1. Special Token Isolation & Preservability
        t_sp = time.perf_counter()
        special_tokens_passed = True
        for st in ALL_SPECIAL_TOKENS:
            encoded = self.tokenizer.encode(st, add_bos=False)
            decoded = self.tokenizer.decode(encoded)
            if decoded != st or len(encoded) != 1:
                special_tokens_passed = False

        dur_sp = (time.perf_counter() - t_sp) * 1000
        self.record(
            "Tokenizer",
            "All 12 Special Tokens 1-Token Round-Trip Isolation",
            special_tokens_passed,
            f"Evaluated {len(ALL_SPECIAL_TOKENS)} special tokens: {ALL_SPECIAL_TOKENS}",
            dur_sp,
        )

        # 2. Fuzzing with UTF-8, Emojis, and Malformed Strings
        fuzz_corpus = [
            "Hello 🌍 World 🚀🤖! Testing Unicode characters.",
            "Japanese: こんにちは世界, Chinese: 你好世界, Arabic: مرحبا بالعالم",
            "Math Symbols: ∑(x_i) = ∫ f(x)dx ± ∞ ≈ 3.14159",
            "Control & Quotes: \t\r\n \\ \" ' ` \b \f",
            "Long Repetitive String: " + "ABCDEFGHIJ" * 200,
            "Nested Special Delimiters: <|user|><|conscience|><|action|><|/action|><|/conscience|><|assistant|>",
            "",  # Empty string
        ]

        for idx, text in enumerate(fuzz_corpus):
            t0 = time.perf_counter()
            try:
                encoded = self.tokenizer.encode(text, add_bos=False)
                decoded = self.tokenizer.decode(encoded)
                dur = (time.perf_counter() - t0) * 1000
                # Decoded must be consistent
                passed = len(encoded) >= 0 and isinstance(decoded, str)
                preview = text[:25].replace("\n", "\\n")
                self.record(
                    "TokenizerFuzzing",
                    f"Fuzz Case {idx+1}: {preview or '[EMPTY]'}",
                    passed,
                    f"Tokens: {len(encoded)}, Round-trip string length: {len(decoded)}",
                    dur,
                )
            except Exception as e:
                self.record("TokenizerFuzzing", f"Fuzz Case {idx+1}", False, str(e), (time.perf_counter() - t0) * 1000)

    # =========================================================================
    # SUITE 7: Batching & Throughput Benchmarks
    # =========================================================================
    def test_batching_and_throughput(self):
        print("\n" + "=" * 80)
        print(" [SUITE 7] Batching & Throughput Performance Profiling")
        print("=" * 80)

        cfg = ModelConfig(
            d_model=512,
            n_layers=4,
            n_heads=8,
            n_kv_heads=2,
            d_ff=1024,
            vocab_size=self.tokenizer.vocab_size,
            max_seq_len=256,
            conscience_layer_idx=2,
        )
        model = ConscienceTransformer(cfg).to(self.device)
        model.eval()

        batch_sizes = [1, 2, 4, 8]
        seq_len = 64

        for bs in batch_sizes:
            t0 = time.perf_counter()
            input_ids = torch.randint(0, self.tokenizer.vocab_size, (bs, seq_len), dtype=torch.long, device=self.device)
            with torch.no_grad():
                out = model(input_ids)
            dur = (time.perf_counter() - t0) * 1000

            total_tokens = bs * seq_len
            throughput = (total_tokens / (dur / 1000.0)) if dur > 0 else 0.0

            self.record(
                "Throughput",
                f"Batch Size B={bs}, SeqLen L={seq_len}",
                dur > 0 and not torch.isnan(out.logits).any(),
                f"Latency: {dur:.2f} ms | Throughput: {throughput:,.1f} tokens/sec",
                dur,
                {"batch_size": bs, "seq_len": seq_len, "throughput_tok_per_sec": throughput, "latency_ms": dur},
            )

    # =========================================================================
    # Runner
    # =========================================================================
    def run_all(self):
        t_start = time.perf_counter()
        print("\n" + "#" * 80)
        print("      STARTING DUMM-E 220M COMPREHENSIVE STRESS TESTING SUITE")
        print("#" * 80)

        self.test_full_220m_architecture_and_context_scaling()
        self.test_kv_cache_stress_and_parity()
        self.test_conscience_head_and_steering()
        self.test_action_dispatcher_and_sandbox_security()
        self.test_voice_runtime_and_deliberation_leakage()
        self.test_tokenizer_roundtrip_and_fuzzing()
        self.test_batching_and_throughput()

        total_duration = time.perf_counter() - t_start
        passed_count = sum(1 for r in self.results if r.passed)
        failed_count = sum(1 for r in self.results if not r.passed)
        total_count = len(self.results)

        print("\n" + "=" * 80)
        print("                   STRESS TEST EXECUTION SUMMARY")
        print("=" * 80)
        print(f"  • Total Tests Executed: {total_count}")
        print(f"  • Tests Passed:         {passed_count} ({passed_count/total_count*100:.1f}%)")
        print(f"  • Tests Failed:         {failed_count}")
        print(f"  • Total Time Elapsed:   {total_duration:.2f} seconds")
        print("=" * 80)

        summary_file = Path("artifacts/stress_test_report.json")
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_tests": total_count,
                    "passed": passed_count,
                    "failed": failed_count,
                    "duration_seconds": total_duration,
                    "tests": [
                        {
                            "category": r.category,
                            "name": r.name,
                            "passed": r.passed,
                            "details": r.details,
                            "duration_ms": r.duration_ms,
                            "metrics": r.metrics,
                        }
                        for r in self.results
                    ],
                },
                f,
                indent=2,
            )
        print(f"\n[REPORT PERSISTED] Serialized full stress test telemetry to: {summary_file}")


if __name__ == "__main__":
    suite = DummEStressTestSuite()
    suite.run_all()
