# Agentic-JEPA

A proof-of-concept self-supervised world model for text-based agent planning, based on the Joint Embedding Predictive Architecture (JEPA).

The agent learns to predict the latent consequences of actions by observing state transitions, then plans at inference time by scoring candidate actions against a goal embedding. No reinforcement learning, no reward signal.

**Paper**: *Agentic-JEPA: A Self-Supervised World Model for Planning in Text-Based Agent Environments* ([arXiv](https://arxiv.org/abs/XXXX.XXXXX))

## Architecture

```
State s_t  ──→ [State Encoder φ_θ]  ──→ h_s_t ──┐
                                                  ├──→ [Predictor P_ω] ──→ ĥ_s_{t+1}
Action a_t ──→ [Action Encoder ψ_θ] ──→ h_a_t ──┘

Next state s_{t+1} ──→ [Target Encoder φ_θ̄ (EMA)] ──→ h_s_{t+1}  (training target)

Planning: score(a) = cos_sim(ĥ_s_{t+1}, h_goal)
```

- **State encoder**: GPT-2 backbone, last-token pooling
- **Action encoder**: Independent GPT-2 backbone, last-token pooling
- **Predictor**: Lightweight transformer (2 layers, 8 heads, 512 hidden dim)
- **Target encoder**: Exponential moving average of state encoder (momentum 0.996 → 1.0)
- **Loss**: Cosine embedding loss between predicted and target next-state embeddings

## Quick Start

### Setup

```bash
pip install -r requirements.txt
```

### Full Pipeline

```bash
# 1. Verify environments work
python -m src.benchmarks.verify_envs

# 2. Generate training data (~1 min, CPU)
python -m src.data.generate_trajectories --output data/trajectories.jsonl

# 3. Train (~10 min on T4 GPU)
python -m src.train --config configs/default.yaml --data data/trajectories.jsonl

# 4. Run benchmark
python -m src.benchmarks.run_all --config configs/default.yaml --checkpoint outputs/default/best.pt
```

### Google Colab

The full pipeline runs on a free Colab T4 GPU. See `COLAB_COMMANDS.txt` for copy-paste commands.

## Benchmark Environments

Eight deterministic text-based environments modelling software tasks:

| Environment | Oracle Steps | Description |
|---|---|---|
| DocumentWorkflow | 5 | Open, read, summarise, file, send |
| CodeReview | 6 | Read diff, test, identify issues, comment, request changes, verify |
| EmailTriage | 6 | Scan, read urgent, reply, categorise, forward, archive |
| CustomerSupport | 6 | Greet, identify, check KB, propose, confirm, close |
| IncidentResponse | 7 | Acknowledge, assess, identify, mitigate, root cause, fix, postmortem |
| MeetingPreparation | 5 | Check calendar, gather materials, prepare agenda, set up, send reminders |
| DataPipeline | 5 | Validate, transform, quality check, stage, deploy |
| ResearchTask | 5 | Search, read abstracts, take notes, identify themes, synthesise |

Each step presents 6-7 valid actions (1 oracle + 5-6 distractors).

## Agents

| Agent | Description |
|---|---|
| **Random** | Uniform random action selection |
| **GreedyText** | TF-IDF cosine similarity between action and goal |
| **GPT2-Vanilla** | JEPA architecture with untrained predictor (ablation) |
| **JEPA** | Trained model, scores actions via latent prediction |
| **Oracle** | Always selects optimal action (upper bound) |

## Results (Preliminary)

Training: 500 steps, ~10 min on T4 GPU. Loss: 1.0 → 0.15, cosine similarity: 0.0 → 0.96.

| Environment | Random | GreedyText | GPT2-Vanilla | **JEPA** | Oracle |
|---|---|---|---|---|---|
| CodeReview | 36.5 steps | 40.4 steps | 50.0 (fail) | **15.0 steps** | 6.0 steps |
| DocumentWorkflow | 32.5 steps | 28.6 steps | 50.0 (fail) | **23.0 steps** | 5.0 steps |
| EmailTriage | 35.8 steps | 43.9 steps | 50.0 (fail) | **6.0 steps** | 6.0 steps |

JEPA achieves 100% success rate on these environments. On EmailTriage, it matches oracle performance exactly.

## Configuration

Default config (`configs/default.yaml`):

- Backbone: GPT-2 (768-dim)
- Predictor: 2 transformer layers, 8 heads, 512 hidden
- Training: 500 steps, batch 16, lr 5e-5, AdamW
- EMA: momentum 0.996 → 1.0 over 500 steps

Ablation configs available in `configs/ablation_*.yaml`.

## Project Structure

```
agentic-jepa/
├── configs/                 # YAML configs (default, ablations)
├── src/
│   ├── model.py             # JEPA model (encoders + predictor)
│   ├── train.py             # Training loop
│   ├── plan.py              # Inference-time planning
│   ├── ema.py               # EMA update logic
│   ├── data/
│   │   ├── dataset.py       # PyTorch dataset
│   │   └── generate_trajectories.py
│   └── benchmarks/
│       ├── environments.py  # 8 text environments
│       ├── baselines.py     # Agent implementations
│       ├── metrics.py       # Evaluation metrics
│       ├── run_all.py       # Full benchmark runner
│       └── verify_envs.py   # Environment sanity check
├── tests/                   # Unit tests
├── requirements.txt
└── COLAB_COMMANDS.txt       # Copy-paste commands for Colab
```

## Citation

```bibtex
@article{rodrigogines2026agenticjepa,
  title={Agentic-JEPA: A Self-Supervised World Model for Planning in Text-Based Agent Environments},
  author={Rodrigo-Gin{\'e}s, Francisco-Javier},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

## License

MIT
