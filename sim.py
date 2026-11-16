"""Simulateur abstrait (numpy pur) + enveloppes Gymnasium pour C1 et M2."""
import numpy as np

MAX_HP = 2        # leurres max par zone
N_REAL = 3        # hôtes réels par zone (dilue les leurres)
T = 50            # horizon (pas de temps)
DWELL = 2         # pas passés dans la dernière zone avant compromission

# alert = proba d'alerte dans la zone de l'attaquant ; evade = discrétion face aux leurres
PROFILES = {
    "cible":   dict(move=0.35, alert=0.50, evade=1.0, direct=True),
    "bruyant": dict(move=0.50, alert=0.90, evade=1.0, direct=False),
    "discret": dict(move=0.15, alert=0.15, evade=0.5, direct=True),
}
FALSE_ALERT = 0.05


class Core:
    """Réseau en chaîne : zone 0 (publique) -> ... -> zone N-1 (cible)."""

    def __init__(self, n_zones=4, budget=3, profile="cible", seed=0):
        self.n, self.budget, self.profile = n_zones, budget, profile
        self.rng = np.random.default_rng(seed)

    def reset(self, profile=None):
        if profile: self.profile = profile
        self.p = PROFILES[self.profile]
        self.hp = np.zeros(self.n, dtype=int)
        self.ema = np.zeros(self.n)
        self.alert = np.zeros(self.n)
        self.pos, self.dwell, self.t = 0, 0, 0
        return self.obs()

    def obs(self):
        """Observation locale par zone : [leurres, alerte, alerte lissée]. Shape (n, 3)."""
        return np.stack([self.hp / MAX_HP, self.alert, self.ema], axis=1).astype(np.float32)

    def step(self, actions):
        """actions[z] : 0 garder, 1 déployer, 2 retirer. Le budget n'est PAS dans l'observation."""
        a = np.asarray(actions)
        for z in range(self.n):                      # retirer d'abord
            if a[z] == 2 and self.hp[z] > 0: self.hp[z] -= 1
        refused = 0
        for z in range(self.n):                      # puis déployer (si budget et place)
            if a[z] == 1:
                if self.hp.sum() < self.budget and self.hp[z] < MAX_HP: self.hp[z] += 1
                else: refused += 1
        self.t += 1
        p, z = self.p, self.pos
        # attaquant : déplacement
        if self.rng.random() < p["move"]:
            if p["direct"]: z = min(z + 1, self.n - 1)
            else: z = min(max(z + (1 if self.rng.random() < 0.5 else -1), 0), self.n - 1)
        if z != self.pos: self.dwell = 0
        self.pos = z
        # interaction avec un leurre
        h = self.hp[z]
        detected = h > 0 and self.rng.random() < p["evade"] * h / (h + N_REAL)
        # alertes bruitées
        self.alert = (self.rng.random(self.n) < FALSE_ALERT).astype(float)
        if self.rng.random() < p["alert"]: self.alert[z] = 1.0
        self.ema = 0.7 * self.ema + 0.3 * self.alert
        # compromission
        compromised = False
        if not detected and z == self.n - 1:
            self.dwell += 1
            compromised = self.dwell >= DWELL
        reward = -0.02 * self.hp.sum()
        if detected: reward += 10 + 0.1 * (T - self.t)
        if compromised: reward -= 10
        timeout = self.t >= T and not detected and not compromised
        done = detected or compromised or timeout
        info = dict(detected=detected, compromised=compromised, timeout=timeout,
                    t=self.t, cost=float(self.hp.sum()), refused=refused)
        return self.obs(), reward, done, info


# ----------------------------------------------------------------------
# Enveloppes Gymnasium (import différé : le simulateur seul n'en a pas besoin)
# ----------------------------------------------------------------------
def make_central_env(n_zones=4, budget=3, profile="cible", seed=0):
    """C1 : une seule observation globale (3n), une action par zone."""
    import gymnasium as gym
    from gymnasium import spaces

    class CentralEnv(gym.Env):
        def __init__(self):
            self.core = Core(n_zones, budget, profile, seed)
            self.observation_space = spaces.Box(0, 1, (3 * n_zones,), np.float32)
            self.action_space = spaces.MultiDiscrete([3] * n_zones)

        def reset(self, seed=None, options=None):
            if seed is not None: self.core.rng = np.random.default_rng(seed)
            return self.core.reset().ravel(), {}

        def step(self, action):
            o, r, d, i = self.core.step(action)
            return o.ravel(), r, (i["detected"] or i["compromised"]), i["timeout"], i
    return CentralEnv()


def make_shared_env(n_zones=4, budget=3, profile="cible", seed=0):
    """M2 : politique partagée. Chaque pas = la décision d'UNE zone (vue locale + id de zone).
    Après les n décisions, l'action jointe est appliquée ; récompense d'équipe au dernier sous-pas."""
    import gymnasium as gym
    from gymnasium import spaces

    class SharedEnv(gym.Env):
        def __init__(self):
            self.core = Core(n_zones, budget, profile, seed)
            self.observation_space = spaces.Box(0, 1, (3 + n_zones,), np.float32)
            self.action_space = spaces.Discrete(3)

        def _o(self):
            return np.concatenate([self.cur[self.k], np.eye(n_zones, dtype=np.float32)[self.k]])

        def reset(self, seed=None, options=None):
            if seed is not None: self.core.rng = np.random.default_rng(seed)
            self.cur, self.k, self.acts = self.core.reset(), 0, []
            return self._o(), {}

        def step(self, action):
            self.acts.append(int(action)); self.k += 1
            if self.k < n_zones:
                return self._o(), 0.0, False, False, {}
            self.cur, r, d, i = self.core.step(self.acts)
            self.k, self.acts = 0, []
            return self._o(), r, (i["detected"] or i["compromised"]), i["timeout"], i
    return SharedEnv()


def shared_obs(cur):
    """Observations (n, 3+n) pour appliquer la politique partagée à chaque zone."""
    n = cur.shape[0]
    return np.concatenate([cur, np.eye(n, dtype=np.float32)], axis=1)
