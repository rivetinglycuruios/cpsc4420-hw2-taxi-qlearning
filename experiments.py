"""Runs every experiment for HW2 (parts C-H) and writes plots + results/results.json.

Each configuration is trained over N_SEEDS independent seeds; curves show the
seed-mean of a moving average, with a shaded band of +/- 1 std across seeds.

    python experiments.py
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qlearning import Config, evaluate, train

N_SEEDS = 5
WINDOW = 100          # moving-average window for reward curves
EVAL_EPISODES = 100   # greedy evaluation episodes (assignment asks for >= 30)
OUT = "results"
BASE = Config(alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01,
              epsilon_decay=0.995, episodes=3000)


def _run(cfg):
    agent, rets, steps = train(cfg)
    return cfg, agent, rets, steps


def run_many(cfgs):
    with ProcessPoolExecutor() as ex:
        return list(ex.map(_run, cfgs))


def smooth(x, w=WINDOW):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def summarize(rets_by_seed):
    """Learning-speed / stability / final-performance metrics across seeds."""
    R = np.array(rets_by_seed)
    S = np.array([smooth(r) for r in R])
    final = R[:, -200:].mean(axis=1)
    # Episodes until the 100-ep moving average first reaches 0 (taxi is net-positive)
    thresh = [int(np.argmax(s >= 0)) if (s >= 0).any() else None for s in S]
    return {
        "final200_mean": float(final.mean()),
        "final200_seed_std": float(final.std()),
        "final200_within_seed_std": float(R[:, -200:].std(axis=1).mean()),
        "episodes_to_ma100_ge_0": thresh,
        "episodes_to_ma100_ge_0_mean": float(np.mean([t for t in thresh if t is not None])) if any(t is not None for t in thresh) else None,
        "auc_mean_reward": float(R.mean()),
    }


def plot_group(groups, title, fname, ylim=(-250, 15)):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, rets in groups.items():
        S = np.array([smooth(r) for r in rets])
        m, sd = S.mean(0), S.std(0)
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, label=label, lw=1.6)
        ax.fill_between(x, m - sd, m + sd, alpha=0.18)
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Episode reward ({WINDOW}-ep moving avg)")
    ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.axhline(0, color="gray", lw=0.6, ls=":")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname), dpi=150)
    plt.close(fig)


def sweep(name, param, values, base=BASE):
    cfgs = [base.with_(**{param: v}, seed=s) for v in values for s in range(N_SEEDS)]
    res = run_many(cfgs)
    groups, stats = {}, {}
    for v in values:
        rets = [r for c, _, r, _ in res if getattr(c, param) == v]
        groups[f"{param} = {v}"] = rets
        stats[str(v)] = summarize(rets)
    return res, groups, stats


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {"baseline_config": vars(BASE), "n_seeds": N_SEEDS}

    # (C) Baseline ---------------------------------------------------------
    base_res = run_many([BASE.with_(seed=s) for s in range(N_SEEDS)])
    base_rets = [r for _, _, r, _ in base_res]
    results["C_baseline"] = summarize(base_rets)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(base_rets[0], color="tab:blue", alpha=0.25, lw=0.6, label="Raw reward (seed 0)")
    ax.plot(np.mean([smooth(r) for r in base_rets], 0), color="tab:blue", lw=2,
            label=f"{WINDOW}-ep moving avg (mean of {N_SEEDS} seeds)")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Episode reward")
    ax.set_title("(C) Baseline: alpha=0.1, gamma=0.99, eps 1.0 -> 0.01 (decay 0.995)")
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "C_baseline.png"), dpi=150); plt.close(fig)
    base_eval = [evaluate(a, EVAL_EPISODES) for _, a, _, _ in base_res]
    results["C_baseline_eval"] = base_eval

    # (D) Learning rate ----------------------------------------------------
    _, g, st = sweep("alpha", "alpha", [0.1, 0.5, 0.9])
    results["D_alpha"] = st
    plot_group(g, "(D) Learning-rate sweep (gamma=0.99, eps-decay=0.995)", "D_alpha.png")

    # (E) Discount factor --------------------------------------------------
    gres, g, st = sweep("gamma", "gamma", [0.5, 0.9, 0.99])
    results["E_gamma"] = st
    results["E_gamma_eval"] = {
        str(v): [evaluate(a, EVAL_EPISODES) for c, a, _, _ in gres if c.gamma == v] for v in [0.5, 0.9, 0.99]}
    plot_group(g, "(E) Discount-factor sweep (alpha=0.1, eps-decay=0.995)", "E_gamma.png")

    # (F) Epsilon decay rate -----------------------------------------------
    _, g, st = sweep("epsilon_decay", "epsilon_decay", [0.99, 0.995, 0.999])
    results["F_epsilon_decay"] = st
    plot_group(g, "(F) Exploration-decay sweep (alpha=0.1, gamma=0.99)", "F_epsilon_decay.png")
    # epsilon schedule plot for reference
    fig, ax = plt.subplots(figsize=(8, 3))
    for d in [0.99, 0.995, 0.999]:
        ax.plot(np.maximum(0.01, d ** np.arange(BASE.episodes)), label=f"decay = {d}")
    ax.set_xlabel("Training episode"); ax.set_ylabel("epsilon"); ax.set_title("Exploration schedules")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "F_epsilon_schedules.png"), dpi=150); plt.close(fig)

    # (G) Best configuration: grid over alpha x gamma x decay ---------------
    grid = list(product([0.1, 0.5, 0.9], [0.9, 0.99], [0.99, 0.995, 0.999]))
    gres = run_many([BASE.with_(alpha=a, gamma=gm, epsilon_decay=d, seed=s)
                     for a, gm, d in grid for s in range(N_SEEDS)])
    table = []
    for a, gm, d in grid:
        rets = [r for c, _, r, _ in gres if (c.alpha, c.gamma, c.epsilon_decay) == (a, gm, d)]
        s = summarize(rets)
        table.append({"alpha": a, "gamma": gm, "epsilon_decay": d, **s})
    # Rank by final training performance, then by area under the curve (learning speed)
    table.sort(key=lambda t: (round(t["final200_mean"], 1), t["auc_mean_reward"]), reverse=True)
    results["G_grid"] = table
    best = table[0]
    best_cfg = BASE.with_(alpha=best["alpha"], gamma=best["gamma"], epsilon_decay=best["epsilon_decay"])
    results["G_best_config"] = vars(best_cfg)
    best_runs = [x for x in gres if (x[0].alpha, x[0].gamma, x[0].epsilon_decay) ==
                 (best_cfg.alpha, best_cfg.gamma, best_cfg.epsilon_decay)]
    best_eval = [evaluate(a, EVAL_EPISODES) for _, a, _, _ in best_runs]
    results["G_best_eval"] = best_eval
    plot_group({"Baseline": base_rets,
                f"Best (alpha={best_cfg.alpha}, gamma={best_cfg.gamma}, decay={best_cfg.epsilon_decay})":
                    [r for _, _, r, _ in best_runs]},
               "(G) Best configuration vs. baseline", "G_best_vs_baseline.png")
    np.save(os.path.join(OUT, "best_Q.npy"), best_runs[0][1].Q)

    # (H) Stochastic environment --------------------------------------------
    ps = [0.0, 0.10, 0.25]
    nres = run_many([best_cfg.with_(noise_p=p, seed=s) for p in ps for s in range(N_SEEDS)])
    plot_group({f"p = {p}": [r for c, _, r, _ in nres if c.noise_p == p] for p in ps},
               "(H) Training under action noise (best config)", "H_noise_training.png")
    h = {"train_stats": {}, "matched_eval": {}, "p0_policy_under_noise": {}}
    p0_agents = [a for c, a, _, _ in nres if c.noise_p == 0.0]
    for p in ps:
        h["train_stats"][str(p)] = summarize([r for c, _, r, _ in nres if c.noise_p == p])
        agents = [a for c, a, _, _ in nres if c.noise_p == p]
        h["matched_eval"][str(p)] = [evaluate(a, EVAL_EPISODES, noise_p=p) for a in agents]
        h["p0_policy_under_noise"][str(p)] = [evaluate(a, EVAL_EPISODES, noise_p=p) for a in p0_agents]
    results["H_noise"] = h
    # bar chart: matched training vs p=0 policy, at each test noise level
    def agg(evs, k): return np.mean([e[k] for e in evs]), np.std([e[k] for e in evs])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(ps)); w = 0.38
    for ax, k, lab in [(axes[0], "mean_reward", "Mean greedy reward"), (axes[1], "success_rate", "Success rate")]:
        m1 = [agg(h["matched_eval"][str(p)], k) for p in ps]
        m0 = [agg(h["p0_policy_under_noise"][str(p)], k) for p in ps]
        ax.bar(x - w/2, [m for m, _ in m1], w, yerr=[s for _, s in m1], capsize=3, label="Trained at test noise p")
        ax.bar(x + w/2, [m for m, _ in m0], w, yerr=[s for _, s in m0], capsize=3, label="Trained at p = 0")
        ax.set_xticks(x); ax.set_xticklabels([f"p = {p}" for p in ps]); ax.set_ylabel(lab)
        ax.set_xlabel("Test-time action-noise level"); ax.grid(alpha=0.3, axis="y")
    axes[0].legend(loc="lower left")
    fig.suptitle(f"(H) Greedy evaluation under action noise ({EVAL_EPISODES} eps x {N_SEEDS} seeds)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "H_noise_eval.png"), dpi=150); plt.close(fig)

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: results[k] for k in ["C_baseline", "D_alpha", "E_gamma", "F_epsilon_decay", "G_best_config"]}, indent=1))


if __name__ == "__main__":
    main()
