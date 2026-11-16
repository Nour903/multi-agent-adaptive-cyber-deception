"""Évalue baselines (+ modèles C1/M2 s'ils existent) sur tous les profils. Écrit results/rq1.csv"""
import argparse, glob, os
import numpy as np, pandas as pd
from sim import Core, PROFILES, shared_obs
from baselines import random_policy, static_policy, heuristic_policy


def run(policy, n_zones, budget, profile, seed, n_eps):
    core = Core(n_zones, budget, profile, seed=10_000 + seed)   # graines d'évaluation distinctes
    rows = []
    for ep in range(n_eps):
        obs, ret, costs = core.reset(), 0.0, []
        while True:
            obs, r, done, info = core.step(policy(obs, budget))
            ret += r; costs.append(info["cost"])
            if done: break
        rows.append(dict(detected=info["detected"], compromised=info["compromised"],
                         ttd=info["t"] if info["detected"] else np.nan,
                         cost=np.mean(costs), ret=ret))
    return pd.DataFrame(rows).mean().to_dict()


def model_policy(kind, path, n_zones):
    from stable_baselines3 import PPO
    m = PPO.load(path)
    if kind == "c1":
        return lambda obs, b: m.predict(obs.ravel(), deterministic=True)[0]
    return lambda obs, b: m.predict(shared_obs(obs), deterministic=True)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones", type=int, default=4)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--eps", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    out = []
    for prof in PROFILES:
        for s in range(a.seeds):
            pols = {"aleatoire": random_policy(np.random.default_rng(s)),
                    "statique": static_policy(), "heuristique": heuristic_policy()}
            for kind in ("c1", "m2"):                       # modèles entraînés (graine s)
                p = f"results/models/{kind}_z{a.zones}_s{s}.zip"
                if os.path.exists(p): pols[kind.upper()] = model_policy(kind, p, a.zones)
            for name, pol in pols.items():
                out.append(dict(profile=prof, method=name, seed=s,
                                **run(pol, a.zones, a.budget, prof, s, a.eps)))
    df = pd.DataFrame(out); os.makedirs("results", exist_ok=True)
    df.to_csv(f"results/rq1_z{a.zones}_b{a.budget}.csv", index=False)
    g = df.groupby(["profile", "method"])[["detected", "compromised", "ttd", "cost", "ret"]]
    print(pd.concat([g.mean().add_suffix("_moy"), g.std()[["ret"]].add_suffix("_std")], axis=1).round(2).to_string())
