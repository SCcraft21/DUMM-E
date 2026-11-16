# Dumm-E (220M): Voice-Ready Decoder-Only Transformer with Built-In Synthetic Conscience

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://python.org)

**Dumm-E** (**D**iscernment-**U**nified **M**oral & **M**ultimodal **E**ngine) is an open-source, 220M-parameter voice-ready decoder-only Transformer language model built **completely from scratch in pure PyTorch** with zero external LLM API dependencies.

Dumm-E combines high-performance dense autoregressive architecture with a built-in synthetic **Conscience Value Head** tapped directly into penultimate internal representations, local tool execution with moral safety clearance, sentence-level streaming voice I/O, and silent ethical deliberation.

---

## 🏛️ Architectural Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **$d_{\text{model}}$** | `1024` | Hidden representation dimension |
| **$n_{\text{layers}}$** | `12` | Transformer decoder blocks |
| **$n_{\text{heads}}$** | `16` | Query attention heads ($d_k = 64$) |
| **$n_{\text{kv\_heads}}$** | `4` | Grouped-Query Attention (GQA 4:1 query-to-KV ratio) |
| **$d_{\text{ff}}$** | `2816` | SwiGLU feed-forward intermediate dimension |
| **$\text{vocab\_size}$** | `32,000` | Byte-Pair Encoding (BPE) with byte fallback |
| **$L_{\max}$** | `2048` | Maximum context length |
| **Positional Encoding** | Rotary | RoPE ($\theta = 10000.0$) |
| **Normalization** | Pre-RMSNorm | Zero-bias linear projections |
| **Total Parameters (Untied)** | **201,092,352** | Full unshared capacity (>200M) |
| **Total Parameters (Tied)** | **168,324,352** | Unique parameters sharing embedding and LM head |
| **Conscience Tap** | Layer 10 | Penultimate layer evaluation |

---

## 🧠 Synthetic Conscience Architecture

Unlike post-hoc prompt engineering or external guardrails, Dumm-E integrates an architectural **Conscience Value Head** directly onto the hidden activations of layer 10 (the penultimate transformer layer):

```
Penultimate Hidden State (h_10 ∈ R^1024)
       │
       ▼
   RMSNorm(1024)
       │
       ▼
  Linear(1024 -> 256) + SiLU
       │
       ├───► Scalar Risk Head (256 -> 1) ────► Sigmoid ──► Risk Intensity s ∈ [0, 1]
       │
       └───► Principle Head  (256 -> 4) ────► Sigmoid ──► [Harm, Truth, Constructive, Fair]
```

### Conscience-Guided Constrained Decoding & Refusal
1. **Real-Time Token Monitoring**: At every generation step, the Conscience Head outputs a scalar violation risk $s_t$.
2. **Refusal Gating**: If $s_t > \tau_{\text{violation}}$ (e.g. 0.60), the decoder suppresses candidate logits, halts risky completion, and immediately emits `<|refuse|>`.
3. **Action Clearance Gate**: Before executing any local tool or system command, the action payload must receive moral clearance ($s < \tau_{\text{action}}$).

---

## 🎙️ Voice Runtime & Silent Deliberation

Dumm-E is voice-ready for local deployment using open-source audio engines (abstracting `faster-whisper` for STT and `Piper` / `Kokoro` for TTS):

```
User Voice Input
       │ (faster-whisper STT)
       ▼
<|bos|><|user|>\nPrompt\n
       │
       ▼
<|conscience|>\nInternal Moral Deliberation\n<|/conscience|>
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[INTERNAL LOGS ONLY]               <|action|>\nLocal Tool Payload\n<|/action|>
(Silent Thinking: Not Spoken)            │
                                         ▼ (Conscience Clearance Protocol)
                                   Local Tool Dispatcher
                                         │
                                         ▼
                                   <|observation|>\nTool Result\n<|/observation|>
                                         │
                                         ▼
                                   <|assistant|>\nSpoken Response\n<|eos|>
                                         │ (Piper / Kokoro TTS)
                                         ▼
                                   Real-Time Streaming Voice Output 🔊
```

- **Silent Deliberation Filter**: Tokens between `<|conscience|>` and `<|/conscience|>` are strictly withheld from audio synthesis. Dumm-E reasons ethically in silence before speaking.
- **Sentence-Level Streaming Buffer**: Tokens emitted inside `<|assistant|>` blocks are buffered into complete sentence boundaries (`.`, `!`, `?`) and dispatched to TTS with minimal voice latency.

---

## 🏷️ Control Tokens (12 Special Tokens)

| Token | Purpose |
| :--- | :--- |
| `<|bos|>`, `<|eos|>`, `<|pad|>` | Sequence delimiters and padding |
| `<|user|>`, `<|assistant|>` | Conversational turn-taking |
| `<|conscience|>`, `<|/conscience|>` | Deliberative reasoning boundary |
| `<|refuse|>` | Principled refusal trigger |
| `<|action|>`, `<|/action|>` | Local tool invocation delimiters |
| `<|observation|>`, `<|/observation|>` | Environment and tool observation delimiters |

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/dumm-e/dumm-e-220m.git
cd dumm-e-220m
pip install -r requirements.txt
```

### 2. Verify Model Architecture & Exact Parameter Count
```bash
python scripts/verify_params.py
```

### 3. Bootstrap BPE Tokenizer
```bash
python scripts/train_tokenizer.py --vocab_size 32000
```

### 4. Run the Full Test Suite
```bash
python -m unittest discover tests
# Or with pytest:
pytest tests/
```

### 5. Launch Interactive Voice / Action Agent
```bash
python scripts/demo_voice_agent.py
```

---

## 🔄 Multi-Stage Alignment Pipeline

Dumm-E includes three end-to-end, API-free training stages:

1. **Pretraining (`src/training/pretrain.py`)**:
   - Causal language modeling with AdamW decoupled weight decay, cosine annealing with warmup, gradient clipping ($1.0$), and `bfloat16` mixed precision.
   ```bash
   python -m src.training.pretrain --config configs/pretrain.yaml
   ```

2. **Conscience Deliberative SFT (`src/training/sft_conscience.py`)**:
   - Supervised fine-tuning formatting prompts with `<|conscience|>` reasoning blocks and `<|action|>` calls. Prompt tokens are masked with `-100`. Jointly optimizes next-token cross-entropy and Conscience Head BCE risk loss.
   ```bash
   python -m src.training.sft_conscience --config configs/sft_conscience.yaml
   ```

3. **Pure-PyTorch Direct Preference Optimization (`src/training/dpo_align.py`)**:
   - API-free preference alignment with a frozen reference model $\pi_{\text{ref}}$:
     $$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$
   ```bash
   python -m src.training.dpo_align --config configs/dpo_align.yaml
   ```

---

## 📁 Repository Structure

```
dumm-e-220m/
├── configs/
│   ├── model_220m.yaml          # 220M Model & Conscience parameters
│   ├── pretrain.yaml            # Causal LM pretraining config
│   ├── sft_conscience.yaml      # SFT Deliberation & Action config
│   └── dpo_align.yaml           # Pure-PyTorch DPO configuration
├── src/
│   ├── model/                   # Pure PyTorch 220M Transformer
│   │   ├── config.py            # ModelConfig dataclass
│   │   ├── norms.py             # Pre-RMSNorm
│   │   ├── rotary.py            # RoPE embeddings
│   │   ├── attention.py         # Grouped-Query Attention (GQA, SDPA)
│   │   ├── mlp.py               # SwiGLU activation
│   │   ├── conscience_head.py   # Penultimate layer Conscience Head
│   │   ├── transformer_block.py # Transformer decoder block
│   │   └── transformer.py       # Full ConscienceTransformer
│   ├── tokenizer/               # Custom BPE Tokenizer
│   │   ├── special_tokens.py    # 12 special tokens
│   │   ├── trainer.py           # HF Tokenizers BPE trainer
│   │   └── wrapper.py           # ConscienceTokenizer with chat templating
│   ├── training/                # Multi-stage training pipeline
│   │   ├── dataset.py           # CausalLM, ConscienceSFT, DPODataset
│   │   ├── lr_scheduler.py      # Cosine with warmup
│   │   ├── pretrain.py          # Pretraining loop
│   │   ├── sft_conscience.py    # Deliberative SFT loop
│   │   └── dpo_align.py         # Pure-PyTorch DPO trainer
│   ├── inference/               # Inference & constrained decoding
│   │   ├── kv_cache.py          # Dynamic KV cache
│   │   ├── activation_steering.py # Residual stream steering vector manager
│   │   └── generate.py          # Conscience-guided constrained decoding
│   └── voice/                   # Voice & local action runtime
│       ├── actions.py           # Safe local action dispatcher with moral clearance
│       └── voice_loop.py        # STT, silent deliberation filter, TTS streaming
├── scripts/
│   ├── verify_params.py         # Layer-by-layer parameter counter (>200M check)
│   ├── train_tokenizer.py       # Tokenizer training script
│   └── demo_voice_agent.py      # Interactive voice demo
├── tests/                       # Complete pytest/unittest test suite
│   ├── test_model.py            # Model forward/backward, RoPE, RMSNorm, GQA
│   ├── test_kv_cache.py         # Exact numerical parity test
│   ├── test_conscience_head.py  # Conscience risk scoring & loss
│   ├── test_tokenizer.py        # Special token preservation
│   ├── test_inference.py        # Activation steering & refusal gating
│   ├── test_actions.py          # Action parsing & conscience clearance
│   └── test_dpo.py              # DPO loss and preference ranking
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 📜 License
MIT License. Free for open research, educational, and commercial development. Zero proprietary LLM API dependencies.

