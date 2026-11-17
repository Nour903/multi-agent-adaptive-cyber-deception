"""Évalue baselines (+ modèles C1/M2 s'ils existent) sur tous les profils.
Écrit results/rq1_z{zones}_b{budget}.csv avec une ligne par (profil, méthode, graine).
"""
import argparse, os
import numpy as np
import pandas as pd

from sim import Core, PROFILES
from baselines import random_policy, static_policy, heuristic_policy


# ----------------------------------------------------------------------
# Boucle d'évaluation d'une politique (obs (n,3) -> actions (n,))
# ----------------------------------------------------------------------
def run(policy, n_zones, budget, profile, seed, n_eps):
    core = Core(n_zones, budget, profile, seed=10_000 + seed)  # graines d'éval distinctes
    rows = []
    for ep in range(n_eps):
        obs, ret, costs = core.reset(), 0.0, []
        while True:
            obs, r, done, info = core.step(policy(obs, budget))
            ret += r
            costs.append(info["cost"])
            if done:
                break
        rows.append(dict(
            detected    = float(info["detected"]),
            compromised = float(info["compromised"]),
            ttd         = info["t"] if info["detected"] else np.nan,
            cost        = float(np.mean(costs)),
            ret         = float(ret),
        ))
    return pd.DataFrame(rows).mean().to_dict()


# ----------------------------------------------------------------------
# Chargement des modèles entraînés
# ----------------------------------------------------------------------
def model_policy(kind, path, n_zones):
    """Retourne une politique déterministe obs(n,3) -> actions(n,)."""
    if kind == "c1":
        from stable_baselines3 import PPO
        m = PPO.load(path)
        return lambda obs, b: m.predict(obs.ravel(), deterministic=True)[0]

    # kind == "m2"  ->  MAPPO (exécution décentralisée, argmax)
    import torch
    from marl import MAPPO

    agent = MAPPO(n_zones)
    agent.load(path)
    agent.actor.eval()

    def f(obs, b):
        acts = []
        with torch.no_grad():
            for i in range(n_zones):
                logits = agent.actor(
                    torch.as_tensor(obs[i:i + 1], dtype=torch.float32),
                    torch.as_tensor([i], dtype=torch.long),
                )
                acts.append(int(logits.argmax(-1).item()))
        return np.array(acts, dtype=int)
    return f


# ----------------------------------------------------------------------
# Programme principal
# ----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones",  type=int, default=4)
    ap.add_argument("--budget", type=int, default=3)
    ap.add_argument("--eps",    type=int, default=100)
    ap.add_argument("--seeds",  type=int, default=5)
    a = ap.parse_args()

    out = []
    for prof in PROFILES:
        for s in range(a.seeds):
            # 1) Baselines (toujours disponibles)
            pols = {
                "aleatoire":   random_policy(np.random.default_rng(s)),
                "statique":    static_policy(),
                "heuristique": heuristic_policy(),
            }
            # 2) Modèles entraînés pour la graine s (si présents)
            for kind in ("c1", "m2"):
                ext = "zip" if kind == "c1" else "pt"
                p = f"results/models/{kind}_z{a.zones}_s{s}.{ext}"
                if os.path.exists(p):
                    pols[kind.upper()] = model_policy(kind, p, a.zones)
                else:
                    print(f"  [skip] {kind.upper()} introuvable pour seed={s} ({p})")

            # 3) Évaluation de chaque politique
            for name, pol in pols.items():
                metrics = run(pol, a.zones, a.budget, prof, s, a.eps)
                out.append(dict(profile=prof, method=name, seed=s, **metrics))

    df = pd.DataFrame(out)
    os.makedirs("results", exist_ok=True)
    out_csv = f"results/rq1_z{a.zones}_b{a.budget}.csv"
    df.to_csv(out_csv, index=False)

    # Résumé agrégé (moyenne +/- écart-type de la récompense)
    g = df.groupby(["profile", "method"])[
        ["detected", "compromised", "ttd", "cost", "ret"]
    ]
    summary = pd.concat(
        [g.mean().add_suffix("_moy"), g.std()[["ret"]].add_suffix("_std")],
        axis=1,
    ).round(2)
    print("\n=== Résumé par profil et méthode ===")
    print(summary.to_string())
    print(f"\nRésultats détaillés : {out_csv}")