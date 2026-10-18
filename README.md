# Tabular Q-Learning on Gymnasium Taxi

CPSC 4420 — Assignment 2 · Charlie McLaurin · Clemson University, Fall 2026

A from-scratch tabular Q-learning agent for Gymnasium's `Taxi-v4`, with
experiments on learning rate, discount factor, exploration decay, and an
action-noise wrapper for robustness testing.

Video of the trained agent: [`results/taxi_trained_agent.mp4`](results/taxi_trained_agent.mp4)

## Setup

Python 3.10+ is required (tested on 3.12).

```bash
pip install -r requirements.txt
```

## Run

```bash
python experiments.py   # parts C–H: trains every configuration (5 seeds each), writes results/
python h_followup.py    # part H follow-up: noisy training with alpha=0.1, 10k episodes
python record_video.py  # renders the best greedy policy to results/taxi_trained_agent.mp4
```

`experiments.py` takes about 10 minutes on a 12-core laptop (it trains about 140 agents in parallel).

To train a single agent with custom settings:

```python
from qlearning import Config, train, evaluate
agent, returns, steps = train(Config(alpha=0.5, gamma=0.9, epsilon_decay=0.99, episodes=3000))
print(evaluate(agent, episodes=100))   # greedy policy, epsilon = 0
```

All hyperparameters (`alpha`, `gamma`, `epsilon_start`, `epsilon_min`, `epsilon_decay`,
`episodes`, `q_init`, `noise_p`, `seed`) live in the `Config` dataclass in `qlearning.py`.

## Files

| File | Purpose |
|------|---------|
| `qlearning.py` | `Config`, `QLearningAgent` (epsilon-greedy, Q update, terminal handling), `train`, `evaluate`, `NoisyActionWrapper` |
| `experiments.py` | Baseline, alpha/gamma/epsilon-decay sweeps, grid search for the best config, noise experiments |
| `h_followup.py` | Extra noise experiment with a smaller learning rate and longer training |
| `record_video.py` | Renders the trained agent to MP4 |
| `results/` | All plots, `results.json` (every metric), `best_Q.npy` (learned Q-table), the video |

## Results summary

| | Baseline (alpha 0.1, gamma 0.99, decay 0.995) | Best (alpha 0.5, gamma 0.9, decay 0.99) |
|---|---|---|
| Episodes until 100-ep avg reward >= 0 | 1129 | 407 |
| Greedy eval mean reward (100 eps x 5 seeds) | 3.08 | **7.94** |
| Greedy eval success rate | 97.6% | **100%** |
| Mean steps to delivery | 17.4 | **13.1** |

The full write-up is in [`HW2_Charlie_McLaurin.pdf`](HW2_Charlie_McLaurin.pdf).
