"""Entraînement C1 (PPO) ou M2 (MAPPO / CTDE)."""
import argparse, os
import numpy as np
import torch

from sim import make_central_env, make_shared_env

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["c1", "m2"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--zones", type=int, default=4)
ap.add_argument("--budget", type=int, default=3)
ap.add_argument("--steps", type=int, default=300_000)  # pas de temps joints
ap.add_argument("--rollout", type=int, default=1024)   # pas joints par update
a = ap.parse_args()

os.makedirs("results/models", exist_ok=True)
torch.manual_seed(a.seed); np.random.seed(a.seed)

if a.mode == "c1":
    from stable_baselines3 import PPO
    env = make_central_env(a.zones, a.budget, "cible", a.seed)
    model = PPO("MlpPolicy", env, seed=a.seed, verbose=1, n_steps=2048,
                batch_size=256, gamma=0.99, ent_coef=0.01, learning_rate=3e-4)
    model.learn(total_timesteps=a.steps)
    model.save(f"results/models/c1_z{a.zones}_s{a.seed}")
else:
    from marl import MultiAgentCore, MAPPO
    env   = MultiAgentCore(a.zones, a.budget, "cible", a.seed)
    agent = MAPPO(a.zones)

    obs_dict, state = env.reset()
    t, ep_ret, ep_len = 0, 0.0, 0
    while t < a.steps:
        agent.reset_buffer()
        for _ in range(a.rollout):
            acts, logps, zids = agent.act_all(obs_dict)
            next_obs, rews, dones, infos, next_state = env.step(
                {i: acts[i] for i in range(a.zones)}
            )
            r_vec = np.array([rews[i] for i in range(a.zones)], dtype=np.float32)
            done_any = any(dones.values())

            agent.store(
                obs      = np.stack([obs_dict[i] for i in range(a.zones)]),
                zone_ids = np.arange(a.zones),
                actions  = np.array(acts),
                logps    = np.array(logps),
                rewards  = r_vec,
                state    = state,
                done     = done_any,
            )
            ep_ret += float(r_vec.mean()); ep_len += 1
            obs_dict, state = next_obs, next_state
            t += 1

            if done_any:
                print(f"[t={t}] ep_len={ep_len} ep_ret={ep_ret:+.2f} "
                      f"detect={infos[0]['detected']} compromise={infos[0]['compromised']}")
                obs_dict, state = env.reset()
                ep_ret, ep_len = 0.0, 0

        agent.update(last_state=state)

    agent.save(f"results/models/m2_z{a.zones}_s{a.seed}.pt")