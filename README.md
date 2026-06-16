# Hybrid LLM-Guided Reinforcement Learning for MiniGrid

> Reproduced and extended from Park et al. (2025) — *Optimizing Agent Behavior in the MiniGrid Environment Using Reinforcement Learning Based on Large Language Models*, Applied Sciences.

**Authors:** Hamza Sabih · Muhammad Ahmad · Muhammad Zaid · Maaz Ali  
**Institution:** FAST-NUCES, Department of Artificial Intelligence & Data Science, Islamabad

---

## Overview

This project proposes a hybrid architecture that repositions a large language model (LLM) from a **direct action selector** (as in the base paper) to an **auxiliary reward shaper** for a PPO agent. A DistilGPT-2 model (~82M parameters) provides a step-level semantic bonus signal, while a custom CNN-based PPO policy handles all trainable decision-making. Progressive curriculum learning across three MiniGrid grid sizes (5×5 → 8×8 → 16×16) with policy weight transfer between stages demonstrates scalability well beyond the original benchmark.

---

## Key Results

| Method | Success % | Avg Reward | Std |
|---|---|---|---|
| PPO — Park et al. | 46.1% | 0.44 | 0.48 |
| DQN — Park et al. | 49.4% | 0.47 | 0.48 |
| LLM-only — Park et al. | 95.3% | 0.66 | 0.24 |
| Baseline PPO (ours) | 98.7% | 0.92 | — |
| **Hybrid PPO+LLM (ours, 5×5)** | **98.8%** | **0.92** | **0.14** |
| Hybrid PPO+LLM (8×8) | 100% | 0.96 | 0.01 |
| Hybrid PPO+LLM (16×16) | 100% | 0.98 | 0.01 |

- ~50× faster LLM inference: DistilGPT-2 at ~3ms vs LLaMA-3B at ~150ms per step
- Lower reward variance (std 0.14) vs LLM-only baseline (std 0.24)
- Perfect 100% success on 8×8 and 16×16 — results the base paper never reported

---

## Architecture

The system has three components working together:

```
MiniGrid Environment (image obs)
        │
        ▼
MinigridFeaturesExtractor (3-layer CNN → 128-dim vector)
        │
        ▼
PPO Policy Network ──── action ──→ Environment Step
        ▲                                │
        │                          r_env (sparse)
        │                                │
        └──── r_total = r_env + 0.1 × r_LLM
                                         ▲
                                  DistilGPT-2
                              (yes/no logit scoring)
```

**Composite reward:**

```
r_total = r_env + λ · r_LLM    where λ = 0.1
```

At each step, the agent's position, goal direction, and step count are converted to a natural language prompt. DistilGPT-2's final token logits for yes-type vs no-type tokens are compared via sigmoid to produce `r_LLM ∈ (0, 1)`.

---

## Curriculum Learning

Training progresses across three stages with policy weights transferred between each — no cold restart:

| Stage | Environment | Timesteps |
|---|---|---|
| 1 | MiniGrid-Empty-5×5-v0 | 60,000 |
| 2 | MiniGrid-Empty-8×8-v0 | 100,000 |
| 3 | MiniGrid-Empty-16×16-v0 | 200,000 |

---

## Project Structure

```
├── i221948_i221929_i221937_i221934.ipynb   # Main notebook (full pipeline)
├── hybrid_ppo_llm_minigrid.zip             # Saved PPO model weights (after training)
├── results.png                             # Generated plots
└── README.md
```

**Notebook cells at a glance:**

| Cell | Purpose |
|---|---|
| Cell 1 | Install dependencies |
| Cell 2 | Imports and device setup |
| Cell 3 | `MinigridFeaturesExtractor` — custom 3-layer CNN |
| Cell 4 | `LLMRewardShaper` — DistilGPT-2 bonus computation |
| Cell 5 | `MetricsCallback` — per-episode logging |
| Cell 6 | Environment factory + curriculum config |
| Cell 7 | Main training loop (curriculum + LLM shaping) |
| Cell 8 | Baseline PPO ablation (no LLM, same hyperparameters) |
| Cell 9 | Plotting — 4-panel results figure |
| Cell 10 | Save model + print paper-ready results table |

---

## Setup and Usage

### Requirements

```
Python 3.8+
CUDA GPU recommended (tested on NVIDIA T4 via Google Colab)
```

### Install dependencies

```bash
pip install minigrid stable-baselines3[extra] gymnasium matplotlib torch transformers
```

### Run on Google Colab (recommended)

1. Upload the notebook to [colab.research.google.com](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells in order — full training takes approximately 20–30 minutes

### Run locally

```bash
jupyter notebook i221948_i221929_i221937_i221934.ipynb
```

Run cells sequentially. Training time depends on hardware — a GPU is strongly recommended.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Algorithm | PPO (CnnPolicy) |
| Feature extractor | MinigridFeaturesExtractor |
| Feature dimension | 128 |
| Learning rate | 3 × 10⁻⁴ |
| n_steps | 256 |
| Batch size | 64 |
| n_epochs | 4 |
| Discount γ | 0.99 |
| GAE λ | 0.95 |
| Clip range ε | 0.2 |
| Entropy coefficient | 0.01 |
| LLM model | DistilGPT-2 (~82M params) |
| Shaping coefficient λ_LLM | 0.1 |
| Random seed | 42 |

---

## How the LLM Reward Shaper Works

```python
# Natural language prompt constructed at each step
"Agent at (2, 3), facing right. Goal is right by 2, down by 1. Step 14. Should agent move forward?"

# DistilGPT-2 final token logits extracted
yes_tokens = [" yes", " Yes", " forward"]
no_tokens  = [" no",  " No",  " backward", " stop"]

# Sigmoid of mean logit difference → bonus in (0, 1)
r_LLM = sigmoid(mean(yes_logits) - mean(no_logits))
bonus  = 0.1 * r_LLM
```

No text generation is required — only a single forward pass through the model, keeping inference at ~3ms per step.

---

## Differences from Base Paper (Park et al. 2025)

| Aspect | Park et al. | This Project |
|---|---|---|
| LLM role | Full action selector | Reward shaper only |
| Policy learning | None (static oracle) | PPO — trainable |
| LLM size | LLaMA-3B (3B params) | DistilGPT-2 (82M) |
| Inference speed | ~150ms per step | ~3ms per step |
| Training stability | Stochastic LLM output | PPO gradient clipping |
| Environments | 5×5 only | 5×5, 8×8, 16×16 |
| Transfer learning | None | Weight transfer between stages |
| Reproducibility | Limited | Full open-source notebook |

---

## Limitations

- MiniGrid-Empty environments contain no obstacles, keys, or doors — the hybrid design's advantage is most visible in harder sparse-reward settings
- The LLM bonus uses a simple yes/no logit contrast; more expressive sub-goal reasoning could improve sample efficiency further
- On the 5×5 task, PPO alone reaches 98.7% — close enough that isolating the LLM contribution requires looking at variance rather than success rate

---

## Future Work

- Evaluate on procedurally generated layouts: FourRooms, MultiRoom, BabyAI
- More expressive LLM scoring with multi-step sub-goal decomposition
- Probability calibration techniques for the LLM bonus signal across distribution shifts

---

## References

Park, B.-J., Yong, S.-J., Hwang, H.-S., & Moon, I.-Y. (2025). Optimizing Agent Behavior in the MiniGrid Environment Using Reinforcement Learning Based on Large Language Models. *Applied Sciences*, 15(4), 1860. https://doi.org/10.3390/app15041860

Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347

Chevalier-Boisvert, M., et al. (2023). MiniGrid & MiniWorld. arXiv:2306.13831

Raffin, A., et al. (2021). Stable-Baselines3. *JMLR*, 22(268), 1–8.
