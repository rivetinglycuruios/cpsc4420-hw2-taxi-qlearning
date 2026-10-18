"""Part H follow-up: does training under noise work better with a smaller alpha
and more episodes? (The best deterministic config uses alpha = 0.5, which never
settles when the env is stochastic.) Writes results/H_followup.json + plot.

    python h_followup.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import EVAL_EPISODES, N_SEEDS, run_many, smooth
from qlearning import Config, evaluate

PS = [0.10, 0.25]
CFG = Config(alpha=0.1, gamma=0.9, epsilon_decay=0.999, epsilon_min=0.01, episodes=10_000)


def main():
    res = run_many([CFG.with_(noise_p=p, seed=s) for p in PS for s in range(N_SEEDS)])
    out = {"config": vars(CFG), "eval": {}}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for p in PS:
        runs = [x for x in res if x[0].noise_p == p]
        evs = [evaluate(a, EVAL_EPISODES, noise_p=p) for _, a, _, _ in runs]
        out["eval"][str(p)] = {k: float(np.mean([e[k] for e in evs]))
                               for k in ["mean_reward", "std_reward", "success_rate", "mean_steps"]}
        S = np.array([smooth(r) for _, _, r, _ in runs])
        ax.plot(S.mean(0), label=f"p = {p}")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Episode reward (100-ep moving avg)")
    ax.set_title("(H) Follow-up: training under noise with alpha=0.1, 10k episodes")
    ax.set_ylim(-250, 15); ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig("results/H_followup.png", dpi=150); plt.close(fig)
    with open("results/H_followup.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["eval"], indent=1))


if __name__ == "__main__":
    main()
