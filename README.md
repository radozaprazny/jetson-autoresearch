# autoresearch — Jetson Orin Nano Super (ARM64)

![teaser](progress.png)

Experiment branch on top of [radozaprazny/autoresearch](https://github.com/radozaprazny/autoresearch) (RTX 4070 Laptop GPU run), which is itself a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). Full credit to [@karpathy](https://github.com/karpathy) for the core idea: give an AI agent a small but real LLM training setup and let it experiment autonomously. It modifies `train.py`, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. The metric is **val_bpb** (validation bits per byte) — lower is better.

This branch documents a run on a **NVIDIA Jetson Orin Nano Super Developer Kit** — an edge AI platform with ARM64 CPU and unified GPU/CPU memory.

## Hardware

NVIDIA Jetson Orin Nano Super Developer Kit — 8 GB unified memory (CPU + GPU shared), 1024 CUDA cores,
JetPack 6.1 / CUDA 12.6, ARM64 (aarch64).
SDPA (no Flash Attention 3). `torch.compile` unavailable — Triton is not supported on ARM64.
~1.4 GB model footprint, ~1350 steps per 5-min experiment.

## Results: 1.625 → 1.406 (102 experiments, ~13.5% improvement)

## Final best config (exp98, commit `66aaf22`, val_bpb = 1.406354)

```python
DEPTH = 3
ASPECT_RATIO = 96         # model_dim=384, 3 Q heads, 1 KV head (GQA)
HEAD_DIM = 128
N_KV_HEAD = 1             # GQA: 3 Q heads sharing 1 KV head
TOTAL_BATCH_SIZE = 2**13  # 8K tokens
DEVICE_BATCH_SIZE = 16
WARMUP_RATIO = 0.02
WARMDOWN_RATIO = 0.4
FINAL_LR_FRAC = 0.1
WEIGHT_DECAY = 0.03       # linear decay to 0 during warmdown
EMBEDDING_LR = 0.3
UNEMBEDDING_LR = 0.0015
MATRIX_LR = 0.03
SCALAR_LR = 1.0
ADAM_BETAS = (0.8, 0.98)
# MUON:
momentum = 0.95, beta2 = 0.98, ns_steps = 7
# Architecture:
SiLU MLP (3x hidden), no logit softcap
VE gate channels = 16, init scale 0.68x
K+V token shift, 0.75 channels
RoPE base = 10000
x0_lambdas init = 0.1, resid_lambdas = 1.0
# Separate AdamW WD param groups (from #43):
embed_wd = 0.001, value_embeds_wd = 0.003, lm_head_wd = 0.01
```

## Progress curve (all keeps)

| commit  | val_bpb  | description |
|---------|----------|-------------|
| a43a3e7 | 1.624905 | baseline |
| e3aab3d | 1.568024 | DEPTH=6 |
| d1a7ead | 1.543826 | DEPTH=4 |
| 3225f9f | 1.534246 | TOTAL_BATCH_SIZE=2^13 |
| 42521b2 | 1.522878 | WARMDOWN_RATIO=0.7 |
| 6245c75 | 1.521740 | MATRIX_LR=0.03 |
| fa2c24f | 1.519965 | MATRIX_LR=0.04 |
| 5c08d8c | 1.519722 | EMBEDDING_LR=2.0 |
| 9589d31 | 1.514597 | EMBEDDING_LR=1.5 |
| e891044 | 1.506383 | EMBEDDING_LR=1.0 |
| 252a28f | 1.503142 | WEIGHT_DECAY=0.1 |
| d4b8aec | 1.495860 | ASPECT_RATIO=64 (model_dim=256) |
| 24fdce7 | 1.495167 | ASPECT_RATIO=96 (model_dim=384) |
| 52d5216 | 1.486708 | GeLU activation |
| fd34a9e | 1.486253 | SiLU activation |
| 8a1a3d9 | 1.481041 | remove logit softcap |
| 78d7730 | 1.461496 | MATRIX_LR=0.05 re-tune |
| 097ce8c | 1.457142 | GQA n_kv_head=1 |
| 3764fcb | 1.448032 | DEPTH=3 |
| 101d57e | 1.447731 | EMBEDDING_LR=1.5 re-tune |
| 36609c2 | 1.446879 | WARMDOWN_RATIO=0.6 |
| 73f11b0 | 1.446775 | WARMDOWN_RATIO=0.5 |
| b5dfc81 | 1.442340 | WEIGHT_DECAY=0.05 |
| 776ef92 | 1.441970 | FINAL_LR_FRAC=0.1 |
| d245b43 | 1.441009 | token shift K-only 0.5 ch |
| 395cadd | 1.437406 | token shift V added (K+V 0.5/0.5) |
| 1d53865 | 1.433786 | K+V token shift 0.75 ch |
| 053354c | 1.432096 | MATRIX_LR=0.04 re-tune |
| 6189d5c | 1.430267 | MATRIX_LR=0.03 re-tune |
| eb4761d | 1.430150 | VE gate channels 32→16 |
| ff42d66 | 1.429577 | init scale 0.68x |
| a17f15c | 1.429235 | WD linear decay in warmdown |
| e273443 | 1.426910 | EMBEDDING_LR=1.0 re-tune |
| f404733 | 1.425457 | EMBEDDING_LR=0.7 |
| 7bc43c4 | 1.425324 | EMBEDDING_LR=0.5 |
| 1035202 | 1.423790 | EMBEDDING_LR=0.3 |
| c1c1f0e | 1.422538 | WARMDOWN_RATIO=0.4 |
| eaf74d2 | 1.421779 | Muon ns_steps=7 |
| 26d0830 | 1.416517 | WARMUP_RATIO=0.02 (big win) |
| 525e7fe | 1.416003 | WEIGHT_DECAY=0.03 |
| 66aaf22 | 1.406354 | UNEMBEDDING_LR=0.0015 **FINAL BEST** |

## Key findings vs H100 upstream (#43)

| Hyperparameter | H100 optimal | Jetson optimal |
|---|---|---|
| Context length | 2048 | 512 (memory constraint) |
| Depth | 9+ | 3 (step-count limited: ~1350 steps) |
| MLP size | 4× | 3× |
| Logit softcap | beneficial | harmful (remove is a win) |
| GQA KV heads | MHA | n_kv_head=1 |
| Token shift | — | K+V at 0.75 channels |
| Warmdown ratio | 0.75 | 0.4 |
| Embedding LR | ~0.9 | 0.3 |
| Weight decay | 0.2 | 0.03 |
| torch.compile | works | unavailable (no Triton on ARM64) |

## Confirmed universal (cross-platform)

- SiLU ≥ ReLU²
- VE is load-bearing (~+0.05 bpb penalty when removed)
- ADAM_BETAS (0.8, 0.98) — consistent across all platforms
- Smaller batch → more steps → better val_bpb (within memory budget)
- Warmdown tuning is the biggest LR schedule lever

## Community findings tested on this hardware

**From Karpathy's H100 session ([#43](https://github.com/karpathy/autoresearch/discussions/43)):**

| exp | val_bpb | result | note |
|-----|---------|--------|------|
| ff42d66: init scale 0.68x | 1.429577 | ✅ keep | win here; loss on RTX 4070 — platform-specific |
| separate AdamW WD groups (embed/VE/lm_head) | — | ✅ included in base setup | |

**Token shift concept (from cerebustech-dev [#108](https://github.com/karpathy/autoresearch/discussions/108)):**

| exp | val_bpb | result | note |
|-----|---------|--------|------|
| d245b43: K-only shift 0.5 ch | 1.441009 | ✅ keep | starting point |
| 395cadd: V shift added (K+V 0.5) | 1.437406 | ✅ **−0.004** | extending to V is a Jetson win |
| 1d53865: K+V shift 0.75 ch | 1.433786 | ✅ **−0.004** | 3/4 channels optimal (RTX: K-only 1/4) |

## Notable failures (specific to this hardware)

- DEPTH=4+ (fewer steps kills it — Jetson is step-count constrained, not capacity constrained)
- TOTAL_BATCH_SIZE=2^14 (only 719 steps; too few)
- TOTAL_BATCH_SIZE=2^15 (noisy gradients, worse despite more data)
- SwiGLU (too many params, fewer steps per 5 min)
- Logit softcap (beneficial on H100 and RTX, harmful here)
- GQA MHA mode / n_kv_head=3 (more params → fewer steps → worse)
- RoPE base sweep (neither 500 nor 5000 helped, 10000 is optimal)
- Muon ns_steps>7 (slower per-step, fewer effective steps in 5 min)

## Platform note: Jetson Orin Nano Super (ARM64)

Two code changes from upstream were required:

- **`train.py`**: removed Flash Attention 3 kernel import (`from kernels import get_kernel`), replaced with `F.scaled_dot_product_attention`. Disabled `torch.compile` (Triton not supported on ARM64). The `step > 10` guard is still needed to exclude startup overhead from the time budget.
- **`prepare.py`**: `MAX_SEQ_LEN=512` (vs upstream 2048) and reduced eval tokens to fit Jetson VRAM.

## Quick start

**Requirements:** Python 3.10+, [uv](https://docs.astral.sh/uv/), NVIDIA Jetson with JetPack 6.1+.

```bash
# 1. Install uv (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (JetPack PyTorch wheel uses a non-standard filename)
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
UV_SKIP_WHEEL_FILENAME_CHECK=1 uv run train.py
```

Then point Claude Code or another coding agent at `program.md` and let it run the loop.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — [karpathy/autoresearch](https://github.com/karpathy/autoresearch) and the original idea
- [radozaprazny](https://github.com/radozaprazny) — [RTX 4070 Laptop GPU run](https://github.com/radozaprazny/autoresearch) (base branch for this Jetson run)
- [cerebustech-dev](https://github.com/cerebustech-dev) — GB10 Blackwell run, token shift K-only finding ([#108](https://github.com/karpathy/autoresearch/discussions/108))
- [heidiEC](https://github.com/heidiEC) — cross-platform graph prior ([#195](https://github.com/karpathy/autoresearch/discussions/195))

## License

MIT
