from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qtrust.config import QTrustConfig


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))
        self.sigma0 = sigma0
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma0 * mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma0 * mu_range)

    def forward(self, x):
        self.weight_eps.normal_()
        self.bias_eps.normal_()
        weight = self.weight_mu + self.weight_sigma * self.weight_eps
        bias = self.bias_mu + self.bias_sigma * self.bias_eps
        return F.linear(x, weight, bias)


class DuelingC51(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, atoms: int, vmin: float, vmax: float, sigma0: float):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.atoms = atoms
        self.vmin = vmin
        self.vmax = vmax

        hidden = 512
        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            NoisyLinear(hidden, hidden, sigma0), nn.ReLU(),
            NoisyLinear(hidden, action_dim * atoms, sigma0),
        )
        self.value = nn.Sequential(
            NoisyLinear(hidden, hidden, sigma0), nn.ReLU(),
            NoisyLinear(hidden, atoms, sigma0),
        )

    def forward(self, x):
        b = x.size(0)
        feat = self.feature(x)
        adv = self.advantage(feat).view(b, self.action_dim, self.atoms)
        val = self.value(feat).view(b, 1, self.atoms)
        q_atoms = val + adv - adv.mean(1, keepdim=True)
        probs = F.softmax(q_atoms, dim=2)
        return probs


@dataclass
class Transition:
    s: np.ndarray
    a: int
    r: float
    s2: np.ndarray
    done: bool


class PrioritizedBuffer:
    def __init__(self, capacity: int, alpha: float):
        self.capacity = capacity
        self.alpha = alpha
        self.storage = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, trans: Transition, priority: float = 1.0):
        if len(self.storage) < self.capacity:
            self.storage.append(trans)
        else:
            self.storage[self.pos] = trans
        self.priorities[self.pos] = max(priority, 1e-6)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float):
        if len(self.storage) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.storage), batch_size, p=probs)
        samples = [self.storage[i] for i in indices]
        total = len(self.storage)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return indices, samples, weights.astype(np.float32)

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = float(p)


class RainbowAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: QTrustConfig):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.vmin = cfg.rl.Vmin
        self.vmax = cfg.rl.Vmax
        self.atoms = cfg.rl.atoms
        self.support = torch.linspace(self.vmin, self.vmax, self.atoms)

        self.gamma = cfg.rl.gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Optional strict determinism if CUDA enabled
        try:
            if self.device.type == "cuda":
                import torch.backends.cudnn as cudnn
                cudnn.benchmark = False
                cudnn.deterministic = True  # type: ignore[attr-defined]
                # reduce nondeterminism from matmul/TF32
                try:
                    import torch.backends.cuda as cuda_backends  # type: ignore
                    cuda_backends.matmul.allow_tf32 = False  # type: ignore[attr-defined]
                except Exception:
                    pass
            # enforce deterministic algorithms where supported
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)  # type: ignore[attr-defined]
            except Exception:
                pass
        except Exception:
            pass

        # Initialize with a compact discrete action space using bucketed src/dst
        self._setup_bucketed_actions(num_shards=cfg.simulation.shards)

        self.replay = PrioritizedBuffer(cfg.rl.replay_capacity, cfg.rl.prioritized_alpha)
        self.beta = cfg.rl.prioritized_beta_start
        self.beta_end = cfg.rl.prioritized_beta_end
        self.learn_step = 0

        # Continuous action will be mapped from discrete bins for each of 3 dims
        self._action_bins = [np.linspace(-0.1, 0.1, 11), None, None]

    def _index_to_action(self, action_idx: int, state: np.ndarray) -> np.ndarray:
        # Decode index into (tau_bin, src_bucket, dst_bucket)
        i_tau = action_idx // (self._num_src_buckets * self._num_dst_buckets)
        rem = action_idx % (self._num_src_buckets * self._num_dst_buckets)
        i_srcb = rem // self._num_dst_buckets
        i_dstb = rem % self._num_dst_buckets

        delta_tau = float(self._tau_bins[i_tau])

        # Map buckets to actual shard indices using observed loads
        S = self.cfg.simulation.shards
        mcross_len = S * S
        shard_load = state[mcross_len + S: mcross_len + 2 * S]
        # Build bucket ranges
        src_start = i_srcb * self._bucket_size
        src_end = min((i_srcb + 1) * self._bucket_size, S)
        dst_start = i_dstb * self._bucket_size
        dst_end = min((i_dstb + 1) * self._bucket_size, S)
        # Choose indices
        src_slice = shard_load[src_start:src_end]
        dst_slice = shard_load[dst_start:dst_end]
        src = int(src_start + int(np.argmax(src_slice))) if len(src_slice) > 0 else 0
        dst = int(dst_start + int(np.argmin(dst_slice))) if len(dst_slice) > 0 else 0
        return np.array([delta_tau, src, dst], dtype=np.float32)

    def _setup_bucketed_actions(self, num_shards: int):
        # Discretize delta_tau into a small number of bins and bucket shards
        self._tau_bins = np.linspace(-0.1, 0.1, num=9)
        self._num_src_buckets = 8
        self._num_dst_buckets = 8
        self._bucket_size = int(np.ceil(num_shards / self._num_src_buckets))
        self.action_dim = len(self._tau_bins) * self._num_src_buckets * self._num_dst_buckets
        self.online = DuelingC51(self.obs_dim, self.action_dim, self.atoms, self.vmin, self.vmax, self.cfg.rl.noisy_sigma0).to(self.device)
        self.target = DuelingC51(self.obs_dim, self.action_dim, self.atoms, self.vmin, self.vmax, self.cfg.rl.noisy_sigma0).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.cfg.rl.learning_rate)

    def select_action(self, state: np.ndarray, safety_filter=None) -> np.ndarray:
        if isinstance(state, np.ndarray) and state.ndim == 1:
            state = state[None, :]
        s = torch.from_numpy(state).float().to(self.device)
        # Ensure support tensor is on the same device as model outputs
        support = self.support.to(self.device)
        with torch.no_grad():
            probs = self.online(s)[0]
            alpha_idx = int(max(0, min(self.atoms - 1, int(self.cfg.rl.cvar_alpha * (self.atoms - 1)))))
            tail_mass = probs[:, : alpha_idx + 1].sum(dim=1)
            tail_ev = (probs[:, : alpha_idx + 1] * support[: alpha_idx + 1]).sum(dim=1) / (tail_mass + 1e-6)
            action_idx = int(torch.argmax(tail_ev).item())
        action = self._index_to_action(action_idx, state[0] if state.ndim == 2 else state)
        if safety_filter is not None:
            action = safety_filter(action)
        return action

    def observe(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray, info):
        # Quantize back to nearest discrete index for replay
        delta_tau_bin = int(np.argmin(np.abs(self._tau_bins - a[0])))
        src_bucket = int(min(self._num_src_buckets - 1, max(0, int(a[1] // self._bucket_size))))
        dst_bucket = int(min(self._num_dst_buckets - 1, max(0, int(a[2] // self._bucket_size))))
        action_idx = delta_tau_bin * (self._num_src_buckets * self._num_dst_buckets) + src_bucket * self._num_dst_buckets + dst_bucket
        tr = Transition(s.astype(np.float32), int(action_idx), float(r), s2.astype(np.float32), False)
        self.replay.add(tr, priority=1.0)

    def learn(self):
        # Allow early learning in minimal runs
        min_samples = max(256, self.cfg.rl.batch_size)
        if len(self.replay.storage) < min_samples:
            return
        indices, samples, weights = self.replay.sample(self.cfg.rl.batch_size, self.beta)
        self.beta = min(self.beta_end, self.beta + (self.beta_end - self.cfg.rl.prioritized_beta_start) / 100000)

        s = torch.from_numpy(np.stack([t.s for t in samples])).float().to(self.device)
        a = torch.tensor([t.a for t in samples]).long().to(self.device)
        r = torch.tensor([t.r for t in samples]).float().to(self.device)
        s2 = torch.from_numpy(np.stack([t.s2 for t in samples])).float().to(self.device)
        w = torch.from_numpy(weights).float().to(self.device)

        support = self.support.to(self.device)
        with torch.no_grad():
            next_probs = self.online(s2)
            next_q = torch.sum(next_probs * support, dim=2)
            next_a = torch.argmax(next_q, dim=1)
            next_dist = self.target(s2)[torch.arange(s2.size(0)), next_a]

            # Distributional projection
            tz = r.unsqueeze(1) + (self.gamma ** self.cfg.rl.n_step) * support
            tz = tz.clamp(self.vmin, self.vmax)
            b = (tz - self.vmin) / (self.vmax - self.vmin) * (self.atoms - 1)
            l = b.floor().long()
            u = b.ceil().long()
            m = torch.zeros_like(next_dist)
            for i in range(self.atoms):
                li = l[:, i]
                ui = u[:, i]
                eq_mask = (li == ui)
                m[eq_mask, li[eq_mask]] += next_dist[eq_mask, i]
                neq_mask = ~eq_mask
                m[neq_mask, li[neq_mask]] += next_dist[neq_mask, i] * (u[neq_mask, i].float() - b[neq_mask, i])
                m[neq_mask, u[neq_mask, i]] += next_dist[neq_mask, i] * (b[neq_mask, i] - l[neq_mask, i].float())

        probs = self.online(s)[torch.arange(s.size(0)), a]
        loss = -torch.sum(m * torch.log(probs + 1e-6), dim=1)
        loss = (loss * w).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()

        self.learn_step += 1
        if self.learn_step % self.cfg.rl.target_update_interval == 0:
            self.target.load_state_dict(self.online.state_dict())


