"""Tabular Q-learning agent for Gymnasium Taxi-v4 (CPSC-4420 HW2).

Everything tunable lives in Config, so alpha, gamma and the exploration
schedule can be changed in one place (or overridden per experiment).
"""
from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np


@dataclass
class Config:
    alpha: float = 0.1           # learning rate
    gamma: float = 0.99          # discount factor
    epsilon_start: float = 1.0   # initial exploration rate
    epsilon_min: float = 0.01    # exploration floor
    epsilon_decay: float = 0.995 # multiplicative decay applied after every episode
    episodes: int = 3000
    q_init: float = 0.0          # initial value for every Q(s, a)
    noise_p: float = 0.0         # prob. the env swaps the chosen action for a different one (part H)
    seed: int = 0

    def with_(self, **kw):
        return replace(self, **kw)


class NoisyActionWrapper(gym.ActionWrapper):
    """With probability p, replace the agent's action with a *different* valid action."""

    def __init__(self, env, p, seed=0):
        super().__init__(env)
        self.p = p
        self.rng = np.random.default_rng(seed + 10_000)

    def action(self, action):
        if self.p > 0 and self.rng.random() < self.p:
            others = [a for a in range(self.action_space.n) if a != action]
            return int(self.rng.choice(others))
        return action


def make_env(noise_p=0.0, seed=0, render_mode=None):
    env = gym.make("Taxi-v4", render_mode=render_mode)
    if noise_p > 0:
        env = NoisyActionWrapper(env, noise_p, seed)
    return env


class QLearningAgent:
    def __init__(self, n_states, n_actions, cfg: Config):
        self.cfg = cfg
        self.n_actions = n_actions
        self.Q = np.full((n_states, n_actions), cfg.q_init, dtype=np.float64)
        self.epsilon = cfg.epsilon_start
        self.rng = np.random.default_rng(cfg.seed)

    def greedy(self, s):
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))  # random tie-break

    def act(self, s, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        return self.greedy(s)

    def update(self, s, a, r, s2, terminated):
        # Terminal states have no future value; truncation (time limit) is not terminal,
        # so we still bootstrap from s2 in that case.
        target = r if terminated else r + self.cfg.gamma * self.Q[s2].max()
        self.Q[s, a] += self.cfg.alpha * (target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)


def train(cfg: Config):
    """Train one agent. Returns (agent, per-episode returns, per-episode steps)."""
    env = make_env(cfg.noise_p, cfg.seed)
    agent = QLearningAgent(env.observation_space.n, env.action_space.n, cfg)
    returns = np.zeros(cfg.episodes)
    steps = np.zeros(cfg.episodes, dtype=int)
    for ep in range(cfg.episodes):
        s, _ = env.reset(seed=cfg.seed * 100_000 + ep)
        done, total, t = False, 0.0, 0
        while not done:
            a = agent.act(s)
            s2, r, terminated, truncated, _ = env.step(a)
            agent.update(s, a, r, s2, terminated)
            s, total, t = s2, total + r, t + 1
            done = terminated or truncated
        returns[ep], steps[ep] = total, t
        agent.decay_epsilon()
    env.close()
    return agent, returns, steps


def evaluate(agent, episodes=100, noise_p=0.0, seed=999):
    """Run the greedy policy (epsilon = 0). Returns dict of metrics."""
    env = make_env(noise_p, seed)
    rets, lens, success = [], [], []
    for ep in range(episodes):
        s, _ = env.reset(seed=seed * 100_000 + ep)
        done, total, t, terminated = False, 0.0, 0, False
        while not done:
            s, r, terminated, truncated, _ = env.step(agent.act(s, explore=False))
            total, t = total + r, t + 1
            done = terminated or truncated
        rets.append(total)
        lens.append(t)
        success.append(terminated)  # Taxi only terminates on a correct drop-off
    env.close()
    rets, lens = np.array(rets), np.array(lens)
    return {
        "episodes": episodes,
        "mean_reward": float(rets.mean()),
        "std_reward": float(rets.std()),
        "success_rate": float(np.mean(success)),
        "mean_steps": float(lens.mean()),
        "min_reward": float(rets.min()),
        "max_reward": float(rets.max()),
    }
