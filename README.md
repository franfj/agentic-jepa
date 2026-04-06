# Agentic-JEPA

A self-supervised world model for text-based agent planning, based on the Joint Embedding Predictive Architecture (JEPA).

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

- **State encoder**: Decoder-only LLM backbone (GPT-2, Qwen 2.5, Llama 3.2), last-token pooling
- **Action encoder**: Independent backbone, last-token pooling
- **Predictor**: Lightweight transformer (2 layers, 8 heads, 512 hidden dim)
- **Target encoder**: Exponential moving average of state encoder (momentum 0.996 → 1.0)
- **Loss**: Cosine embedding loss between predicted and target next-state embeddings
- **Multi-step planning**: k-step greedy lookahead rollouts for improved action selection

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
python -m src.data.generate_trajectories --output data/trajectories_v2.jsonl \
    --oracle-episodes 200 --random-episodes 400

# 3. Train (~20 min on T4 GPU)
python -m src.train --config configs/default.yaml --data data/trajectories_v2.jsonl

# 4. Run benchmark with multi-step planning
python -m src.benchmarks.run_all \
    --config configs/default.yaml \
    --checkpoint outputs/default/best.pt \
    --rollout-depths 1 2 3 5

# 5. Run ablation sweep (all variants)
python scripts/run_ablations.py --data data/trajectories_v2.jsonl --ablation all

# 6. Analyze world model quality
python scripts/analyze_world_model.py --checkpoint outputs/default/best.pt

# 7. Generate paper figures
python scripts/generate_figures.py --results results/v2_default.json --output figures/
```

### Google Colab

The full pipeline runs on a free Colab T4 GPU. See `COLAB_COMMANDS.txt` for copy-paste commands.

## Benchmark Environments

13 deterministic text-based environments (8 train, 5 test):

### Training environments (in-distribution)

| Environment | Steps | Description |
|---|---|---|
| DocumentWorkflow | 5 | Open, read, summarise, file, send |
| CodeReview | 6 | Read diff, test, identify issues, comment, request changes, verify |
| EmailTriage | 6 | Scan, read urgent, reply, categorise, forward, archive |
| DataPipeline | 5 | Validate, transform, quality check, stage, deploy |
| ResearchTask | 5 | Search, read abstracts, take notes, identify themes, synthesise |
| BugTriage | 6 | Reproduce, analyze logs, root cause, fix, test, deploy |
| OnboardingProcess | 5 | Create accounts, assign equipment, orientation, workspace, checklist |
| SecurityAudit | 6 | Scope, scan, manual test, classify risks, remediation, report |

### Test environments (out-of-distribution)

| Environment | Steps | Description |
|---|---|---|
| CustomerSupport | 6 | Read ticket, check account, diagnose, fix, verify, close |
| IncidentResponse | 7 | Acknowledge, assess, identify, mitigate, root cause, fix, postmortem |
| MeetingPreparation | 5 | Review agenda, gather data, prepare slides, rehearse, send pre-read |
| ContentPublishing | 5 | Draft, editorial review, add media, SEO, publish |
| ExperimentPipeline | 6 | Hypothesis, dataset, train, evaluate, significance test, write up |

New environments (v2) use **rich natural language action descriptions** instead of identifiers for better semantic generalization.

## Agents

| Agent | Description |
|---|---|
| **Random** | Uniform random action selection |
| **GreedyText** | TF-IDF cosine similarity between action and goal |
| **GPT2-Vanilla** | JEPA architecture with untrained predictor (ablation) |
| **JEPA** | Trained model, scores actions via latent prediction |
| **JEPA(k=N)** | Multi-step planning with N-step greedy lookahead |
| **LLM (GPT-4o-mini)** | Zero-shot LLM baseline with cost/latency tracking |
| **Oracle** | Always selects optimal action (upper bound) |

## Configuration

Default config (`configs/default.yaml`):

- Backbone: GPT-2 (768-dim)
- Predictor: 2 transformer layers, 8 heads, 512 hidden
- Training: 2000 steps, batch 16, lr 5e-5, cosine decay, AdamW
- EMA: momentum 0.996 → 1.0 over 2000 steps

Alternative backbones: `configs/qwen.yaml` (Qwen 2.5-1.5B), `configs/llama.yaml` (Llama 3.2-1B).

Ablation configs: `configs/ablation_ema.yaml`, `configs/ablation_predictor.yaml`.

## Project Structure

```
agentic-jepa/
├── configs/                 # YAML configs (default, ablations, backbones)
├── scripts/
│   ├── run_ablations.py     # Ablation sweep runner
│   ├── generate_figures.py  # Publication figure generation
│   └── analyze_world_model.py  # World model accuracy analysis
├── src/
│   ├── model.py             # JEPA model (encoders + predictor + multi-step)
│   ├── train.py             # Training loop (cosine LR + warmup)
│   ├── plan.py              # Inference-time planning
│   ├── ema.py               # EMA update logic
│   ├── data/
│   │   ├── dataset.py       # PyTorch dataset
│   │   └── generate_trajectories.py
│   └── benchmarks/
│       ├── environments.py  # 13 text environments (8 train + 5 test)
│       ├── baselines.py     # Agent implementations (incl. LLM baseline)
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
