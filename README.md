# Relaxed Recursive Transformers (JAX/Flax)

A JAX/Flax implementation of [**"Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA"**](https://arxiv.org/abs/2410.20672) (Bae et al., ICLR 2025, Google DeepMind).

Companion project to [**Mixture-of-Recursions (PyTorch)**](https://github.com/ahmedtaha100/MoR), the NeurIPS 2025 follow-up paper.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONVERSION PIPELINE                          │
│                                                                 │
│  Vanilla Transformer          Recursive Transformer             │
│  ┌──────────────────┐         ┌──────────────────┐              │
│  │ Layer 0          │         │                  │              │
│  │ Layer 1          │  ──►    │  Shared Block    │ ×num_loops   │
│  │ Layer 2          │  avg    │  (block_size     │              │
│  │ Layer 3          │  init   │   unique layers) │              │
│  │  ...             │         │                  │              │
│  │ Layer N-1        │         └──────────────────┘              │
│  └──────────────────┘                                           │
│         │                                                       │
│         │  SVD of residuals                                     │
│         ▼                                                       │
│  Relaxed Recursive Transformer                                  │
│  ┌──────────────────────────────────────┐                       │
│  │  Shared Block + LoRA[loop_0]         │  Loop 0               │
│  │  Shared Block + LoRA[loop_1]         │  Loop 1               │
│  │  Shared Block + LoRA[loop_2]         │  Loop 2               │
│  │   ...                                │                       │
│  └──────────────────────────────────────┘                       │
│  Each loop uses the SAME shared weights + UNIQUE LoRA adapters  │
│  LoRA initialized via truncated SVD: residual ≈ B^T × A        │
└─────────────────────────────────────────────────────────────────┘
```

## Key Results

| Model | Params | LoRA Overhead | Perplexity (WikiText-2) |
|-------|--------|---------------|------------------------|
| Gemma 2B (Vanilla) | 2.51B | — | XX.X |
| Recursive Gemma (6×3) | 0.84B | 0% | XX.X |
| Relaxed Recursive Gemma (6×3, r=64) | 0.87B | ~3.5% | XX.X |

> *Results pending Gemma evaluation — small-scale validation passes all tests.*

## Quick Start

### Install

```bash
git clone https://github.com/ahmedtaha100/Relaxed-Recursive-Transformers-JAX.git
cd Relaxed-Recursive-Transformers-JAX
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run Demo (no downloads needed)

```bash
PYTHONPATH=. python scripts/demo.py
```

This creates a small random model, converts it to relaxed recursive form, runs inference, and prints parameter comparisons — all on CPU in seconds.

### Run Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

### Profile Models

```bash
PYTHONPATH=. python scripts/profile.py --config configs/test_small.yaml
```

## Gemma 2B Conversion

To convert the pretrained Gemma 2B model:

```bash
# 1. Authenticate with HuggingFace (Gemma is a gated model)
huggingface-cli login
# Accept the license at https://huggingface.co/google/gemma-2b

# 2. Convert to relaxed recursive form
PYTHONPATH=. python scripts/convert.py \
    --source google/gemma-2b \
    --method relaxed \
    --lora_rank 64 \
    --num_loops 3 \
    --output checkpoints/relaxed_gemma

# 3. Evaluate
PYTHONPATH=. python scripts/evaluate.py \
    --checkpoint checkpoints/relaxed_gemma \
    --eval efficiency,perplexity
```

For testing without Gemma access:

```bash
PYTHONPATH=. python scripts/convert.py \
    --source random \
    --config configs/test_small.yaml \
    --output checkpoints/test
```

## Project Structure

```
├── configs/                    # YAML configuration files
│   ├── test_small.yaml         # Small config for CPU testing
│   ├── recursive_gemma_2b.yaml # Gemma 2B recursive (no LoRA)
│   └── relaxed_recursive_gemma_2b.yaml  # Gemma 2B with LoRA
├── src/
│   ├── model/                  # Core model components
│   │   ├── config.py           # Configuration factories
│   │   ├── layers.py           # RMSNorm, RoPE, Attention, FFN, TransformerBlock
│   │   ├── lora.py             # LoRA layers and SVD initialization
│   │   ├── recursive_block.py  # Shared block with loop application
│   │   └── relaxed_recursive_transformer.py  # Full model
│   ├── conversion/             # Weight conversion pipeline
│   │   ├── weight_init.py      # Average and select-k initialization
│   │   ├── svd_init.py         # SVD-based LoRA initialization
│   │   └── convert_gemma.py    # End-to-end conversion
│   ├── inference/              # Generation utilities
│   │   ├── kv_cache.py         # KV cache with recursive depth indexing
│   │   ├── early_exit.py       # Early exit based on prediction convergence
│   │   └── depth_wise_batching.py  # Throughput simulation
│   ├── evaluation/             # Metrics
│   │   ├── perplexity.py       # Sliding-window perplexity
│   │   └── efficiency.py       # Parameter counting and FLOPs estimation
│   └── utils/                  # Shared utilities
│       ├── config_utils.py     # Dataclass configs and YAML loading
│       └── checkpoint.py       # Save/load checkpoints
├── scripts/                    # CLI entry points
│   ├── demo.py                 # Self-contained demo (no downloads)
│   ├── convert.py              # Model conversion
│   ├── evaluate.py             # Evaluation
│   └── profile.py              # Profiling and comparison
└── tests/                      # Test suite (56 tests)
```

## How It Works

**Layer Tying.** A standard transformer with *N* layers is converted into a *recursive* transformer by grouping layers into a shared block of *K* unique layers, applied *L* times (*K × L = N*). The shared block weights are initialized by averaging the original layers that map to each position — for example, in a 18-layer model with *K=6, L=3*, the shared layer at position 0 is the average of original layers 0, 6, and 12. This reduces parameters by a factor of *L* but hurts quality due to the loss of depth-wise specialization (Section 3.1, Equation 2).

**SVD-Initialized LoRA.** To recover the lost specialization, each recursive loop gets its own set of Low-Rank Adaptation (LoRA) modules on all linear projections. Rather than random initialization, the LoRA matrices are initialized via truncated SVD of the *residual* between the original layer weights and the shared weights: *W_original - W_shared ≈ B^T A* (Section 3.2, Equation 5). At full rank, this exactly recovers the original model; at reduced rank, it captures the most important directions of layer-specific variation.

**Early Exit and Depth-wise Batching.** Because the recursive structure means each loop produces a valid hidden state, the model can exit early if predictions converge before the final loop — e.g., if the argmax token after loop *l* matches loop *l-1*. This enables *depth-wise batching* (Section 4.2), where tokens at different recursion depths are batched together, improving serving throughput proportional to the early exit rate.

## Citation

```bibtex
@inproceedings{bae2025relaxed,
    title={Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA},
    author={Bae, Sangmin and Kudugunta, Sneha and Agarwal, Ankush and Mu, Jongwuk and Cao, Jilin and Firat, Orhan and Gupta, Manas},
    booktitle={International Conference on Learning Representations (ICLR)},
    year={2025}
}
```

## Related Work

- **[Mixture-of-Recursions (MoR)](https://github.com/ahmedtaha100/MoR)** — PyTorch implementation of the NeurIPS 2025 follow-up paper
- **[Original Paper](https://arxiv.org/abs/2410.20672)** — Relaxed Recursive Transformers (ICLR 2025)
- **[Gemma](https://ai.google.dev/gemma)** — Google's open-source language model family

## Author

**Ahmed Taha** | [ahmedtaha.io](https://ahmedtaha.io) | MS Computer Science (ML), Columbia University
