"""Baselines : fonctions obs (n,3) -> actions (n,)."""
import numpy as np
from sim import MAX_HP


def random_policy(rng):
    return lambda obs, budget: rng.integers(0, 3, size=len(obs))


def static_policy(zones=None):
    """Déploie un leurre dans des zones fixes (par défaut 1..n-1) puis ne bouge plus."""
    def f(obs, budget):
        n = len(obs); zs = zones if zones is not None else list(range(1, n))
        return np.array([1 if (z in zs and obs[z, 0] == 0) else 0 for z in range(n)])
    return f


def heuristic_policy():
    """Place un leurre dans chacune des `budget` zones ayant la plus forte alerte lissée."""
    def f(obs, budget):
        n = len(obs)
        score = obs[:, 2] + 0.01 * np.arange(n)          # départage vers les zones profondes
        want = np.zeros(n, dtype=int); want[np.argsort(-score)[:budget]] = 1
        hp = np.rint(obs[:, 0] * MAX_HP).astype(int)
        return np.where(hp > want, 2, np.where(hp < want, 1, 0))
    return f
