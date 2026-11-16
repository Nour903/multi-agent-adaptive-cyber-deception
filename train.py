"""Entraînement PPO. Usage : python train.py --mode c1|m2 --seed 0 [--zones 4] [--steps 300000]"""
import argparse, os
from stable_baselines3 import PPO
from sim import make_central_env, make_shared_env

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["c1", "m2"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--zones", type=int, default=4)
ap.add_argument("--budget", type=int, default=3)
ap.add_argument("--steps", type=int, default=300_000)   # pas de temps du réseau (m2 : x zones sous-pas)
a = ap.parse_args()

mk = make_central_env if a.mode == "c1" else make_shared_env
env = mk(a.zones, a.budget, "cible", a.seed)             # entraînement : profil ciblé uniquement
steps = a.steps * (a.zones if a.mode == "m2" else 1)
model = PPO("MlpPolicy", env, seed=a.seed, verbose=1, n_steps=2048, batch_size=256,
            gamma=0.99, ent_coef=0.01, learning_rate=3e-4)
model.learn(total_timesteps=steps)
os.makedirs("results/models", exist_ok=True)
model.save(f"results/models/{a.mode}_z{a.zones}_s{a.seed}")
