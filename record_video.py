"""Render the best trained greedy policy to results/taxi_trained_agent.mp4.

Requires results/best_Q.npy (written by experiments.py).

    python record_video.py
"""
import imageio.v2 as imageio
import numpy as np

from qlearning import make_env

N_EPISODES = 5
FPS = 4

Q = np.load("results/best_Q.npy")
env = make_env(render_mode="rgb_array")
frames = []
for ep in range(N_EPISODES):
    s, _ = env.reset(seed=2026 + ep)
    frames.append(env.render())
    done, total, t = False, 0, 0
    while not done:
        s, r, terminated, truncated, _ = env.step(int(np.argmax(Q[s])))
        frames.append(env.render())
        total, t = total + r, t + 1
        done = terminated or truncated
    frames += [frames[-1]] * FPS  # hold the final frame for a second between episodes
    print(f"episode {ep + 1}: reward {total}, steps {t}, delivered {terminated}")
env.close()
imageio.mimsave("results/taxi_trained_agent.mp4", frames, fps=FPS, macro_block_size=1)
print(f"saved results/taxi_trained_agent.mp4 ({len(frames)} frames)")
