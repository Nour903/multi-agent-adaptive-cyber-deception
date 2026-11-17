"""MAPPO (CTDE) pour la déception cyber multi-agents.

- Chaque zone = un agent (acteur partagé qui voit obs locale + one-hot zone).
- Un critique centralisé voit l'état global (3n) pendant l'entraînement.
- Exécution décentralisée : à l'éval, seuls les acteurs locaux sont utilisés.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sim import Core


# ----------------------------------------------------------------------
# Environnement multi-agents parallèle (au-dessus du Core)
# ----------------------------------------------------------------------
class MultiAgentCore:
    """Enveloppe multi-agents du simulateur. Récompense d'équipe identique
    à celle du Core, dupliquée par agent (coopératif, Dec-POMDP)."""

    def __init__(self, n_zones=4, budget=3, profile="cible", seed=0):
        self.n = n_zones
        self.core = Core(n_zones, budget, profile, seed)

    def reset(self, profile=None):
        self.core.reset(profile)
        obs = self.core.obs().copy()  # (n, 3)
        return {i: obs[i] for i in range(self.n)}, obs.ravel()

    def step(self, actions_dict):
        joint = np.array([actions_dict[i] for i in range(self.n)], dtype=int)
        new_obs, r, done, info = self.core.step(joint)
        obs_dict  = {i: new_obs[i] for i in range(self.n)}
        rew_dict  = {i: float(r)   for i in range(self.n)}  # récompense d'équipe
        done_dict = {i: bool(done) for i in range(self.n)}
        info_dict = {i: info       for i in range(self.n)}
        return obs_dict, rew_dict, done_dict, info_dict, new_obs.ravel()


# ----------------------------------------------------------------------
# Réseaux
# ----------------------------------------------------------------------
def _mlp(in_dim, out_dim, hidden=64):
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    """Politique partagée : (obs locale 3 + one-hot zone n) -> 3 logits."""

    def __init__(self, n_zones, hidden=64):
        super().__init__()
        self.n = n_zones
        self.net = _mlp(3 + n_zones, 3, hidden)

    def forward(self, obs, zone_ids):
        # obs: (B,3)  zone_ids: (B,)  -> logits: (B,3)
        onehot = F.one_hot(zone_ids, self.n).float()
        return self.net(torch.cat([obs, onehot], dim=-1))

    @torch.no_grad()
    def act(self, obs, zone_id):
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        z = torch.as_tensor([zone_id], dtype=torch.long)
        logits = self.forward(o, z)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item())


class Critic(nn.Module):
    """Critique centralisé : état global (3n) -> V(s)."""

    def __init__(self, n_zones, hidden=128):
        super().__init__()
        self.net = _mlp(3 * n_zones, 1, hidden)

    def forward(self, state):
        return self.net(state).squeeze(-1)


# ----------------------------------------------------------------------
# MAPPO
# ----------------------------------------------------------------------
class MAPPO:
    def __init__(self, n_zones, lr=3e-4, gamma=0.99, lam=0.95,
                 clip=0.2, epochs=4, batch_size=256, ent_coef=0.01, device="cpu"):
        self.n = n_zones
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.batch_size, self.ent_coef = epochs, batch_size, ent_coef
        self.device = torch.device(device)

        self.actor = Actor(n_zones).to(self.device)
        self.critic = Critic(n_zones).to(self.device)
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = []

    # ---------- buffer ----------
    def reset_buffer(self):
        self.buffer = []

    def store(self, obs, zone_ids, actions, logps, rewards, state, done):
        self.buffer.append(dict(
            obs=np.asarray(obs, dtype=np.float32),          # (n,3)
            zone_ids=np.asarray(zone_ids, dtype=np.int64),  # (n,)
            actions=np.asarray(actions, dtype=np.int64),    # (n,)
            logps=np.asarray(logps, dtype=np.float32),      # (n,)
            rewards=np.asarray(rewards, dtype=np.float32),  # (n,) récompense d'équipe
            state=np.asarray(state, dtype=np.float32),      # (3n,)
            done=bool(done),
        ))

    # ---------- GAE (sur récompense d'équipe) ----------
    def _compute_gae(self, last_state):
        T = len(self.buffer)
        rews   = np.array([b["rewards"].mean() for b in self.buffer], dtype=np.float32)  # (T,)
        states = np.array([b["state"] for b in self.buffer],          dtype=np.float32)  # (T,3n)
        dones  = np.array([b["done"] for b in self.buffer],           dtype=np.float32)  # (T,)

        with torch.no_grad():
            s_t = torch.as_tensor(states, device=self.device)
            values = self.critic(s_t).cpu().numpy()                                        # (T,)
            if last_state is not None and not self.buffer[-1]["done"]:
                lv = torch.as_tensor(last_state, dtype=torch.float32,
                                     device=self.device).unsqueeze(0)
                last_val = float(self.critic(lv).item())
            else:
                last_val = 0.0
        v_pad = np.concatenate([values, [last_val]])                                       # (T+1,)

        adv = np.zeros(T, dtype=np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nv = v_pad[t + 1] * (1.0 - dones[t])
            delta = rews[t] + self.gamma * nv - values[t]
            last = delta + self.gamma * self.lam * (1.0 - dones[t]) * last
            adv[t] = last
        returns = adv + values
        return adv, returns

    # ---------- update ----------
    def update(self, last_state=None):
        if len(self.buffer) == 0:
            return
        adv, returns = self._compute_gae(last_state)
        T = len(self.buffer)

        # Aplatir (T x n) -> (T*n,) pour l'acteur
        obs_all  = np.concatenate([b["obs"]      for b in self.buffer], axis=0)
        zone_all = np.concatenate([b["zone_ids"] for b in self.buffer], axis=0)
        act_all  = np.concatenate([b["actions"]  for b in self.buffer], axis=0)
        logp_all = np.concatenate([b["logps"]    for b in self.buffer], axis=0)
        adv_rep  = np.repeat(adv, self.n)
        adv_rep  = (adv_rep - adv_rep.mean()) / (adv_rep.std() + 1e-8)

        obs_t  = torch.as_tensor(obs_all,  device=self.device)
        zone_t = torch.as_tensor(zone_all, device=self.device)
        act_t  = torch.as_tensor(act_all,  device=self.device)
        logp_t = torch.as_tensor(logp_all, device=self.device)
        adv_t  = torch.as_tensor(adv_rep,  dtype=torch.float32, device=self.device)

        states_t = torch.as_tensor(np.stack([b["state"] for b in self.buffer]),
                                   dtype=torch.float32, device=self.device)
        ret_t    = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        N = obs_t.shape[0]
        for _ in range(self.epochs):
            # Acteur
            idx = torch.randperm(N, device=self.device)
            for start in range(0, N, self.batch_size):
                mb = idx[start:start + self.batch_size]
                logits = self.actor(obs_t[mb], zone_t[mb])
                dist = torch.distributions.Categorical(logits=logits)
                new_logp = dist.log_prob(act_t[mb])
                ratio = torch.exp(new_logp - logp_t[mb])
                s1 = ratio * adv_t[mb]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[mb]
                a_loss = -torch.min(s1, s2).mean() - self.ent_coef * dist.entropy().mean()
                self.actor_opt.zero_grad(); a_loss.backward(); self.actor_opt.step()

            # Critique
            idx_s = torch.randperm(T, device=self.device)
            for start in range(0, T, self.batch_size):
                mb = idx_s[start:start + self.batch_size]
                v = self.critic(states_t[mb])
                c_loss = F.mse_loss(v, ret_t[mb])
                self.critic_opt.zero_grad(); c_loss.backward(); self.critic_opt.step()

        self.reset_buffer()

    # ---------- exécution décentralisée ----------
    def act_all(self, obs_dict):
        acts, logps, zids = [], [], []
        for i, o in obs_dict.items():
            a, lp = self.actor.act(o, i)
            acts.append(a); logps.append(lp); zids.append(i)
        return acts, logps, zids

    # ---------- IO ----------
    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])