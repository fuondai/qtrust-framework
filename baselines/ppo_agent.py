from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qtrust.config import QTrustConfig


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, action_bins: Tuple[int, int, int]):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_bins = action_bins
        hidden = 256
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        # Three-headed discrete policy for (tau_bin, src_bucket, dst_bucket)
        self.head_tau = nn.Linear(hidden, action_bins[0])
        self.head_src = nn.Linear(hidden, action_bins[1])
        self.head_dst = nn.Linear(hidden, action_bins[2])
        self.value = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        logits_tau = self.head_tau(h)
        logits_src = self.head_src(h)
        logits_dst = self.head_dst(h)
        value = self.value(h).squeeze(-1)
        return logits_tau, logits_src, logits_dst, value


@dataclass
class Rollout:
    s: np.ndarray
    a: Tuple[int, int, int]
    r: float
    s2: np.ndarray


class PPOAgent:
    """PPO with clipping, GAE, and multiple epochs over mini-batches.

    Discrete factorized policy over (tau_bin, src_bucket, dst_bucket), mirroring Rainbow's discretization.
    """

    def __init__(self, obs_dim: int, action_dim: int, cfg: QTrustConfig):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.S = cfg.simulation.shards
        # Discretize action as in Rainbow agent
        self.tau_bins = np.linspace(-0.1, 0.1, num=9)
        self.num_src_buckets = 8
        self.num_dst_buckets = 8
        self.bucket_size = int(np.ceil(self.S / self.num_src_buckets))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = PolicyNet(obs_dim, (len(self.tau_bins), self.num_src_buckets, self.num_dst_buckets)).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        # PPO hyperparameters (conservative defaults)
        self.clip_ratio = 0.2
        self.entropy_coef = 0.01
        self.value_coef = 0.5
        self.max_grad_norm = 5.0
        self.gamma = float(getattr(cfg.rl, "gamma", 0.99))
        self.lam = 0.95
        self.epochs = 4
        self.minibatch_size = 64
        # rollout storage
        self.rollouts: List[Rollout] = []

    def _index_to_action(self, a_tau: int, a_srcb: int, a_dstb: int, state: np.ndarray) -> np.ndarray:
        delta_tau = float(self.tau_bins[a_tau])
        mcross_len = self.S * self.S
        shard_load = state[mcross_len + self.S: mcross_len + 2 * self.S]
        src_start = a_srcb * self.bucket_size
        dst_start = a_dstb * self.bucket_size
        src_end = min((a_srcb + 1) * self.bucket_size, self.S)
        dst_end = min((a_dstb + 1) * self.bucket_size, self.S)
        src_slice = shard_load[src_start:src_end]
        dst_slice = shard_load[dst_start:dst_end]
        src = int(src_start + int(np.argmax(src_slice))) if len(src_slice) > 0 else 0
        dst = int(dst_start + int(np.argmin(dst_slice))) if len(dst_slice) > 0 else 0
        return np.array([delta_tau, src, dst], dtype=np.float32)

    def _action_buckets_from_state(self, state: np.ndarray) -> Tuple[int, int, int]:
        self.net.eval()
        s = torch.from_numpy(state[None, :]).float().to(self.device)
        with torch.no_grad():
            lt, ls, ld, _ = self.net(s)
            a_tau = int(torch.argmax(lt, dim=1).item())
            a_srcb = int(torch.argmax(ls, dim=1).item())
            a_dstb = int(torch.argmax(ld, dim=1).item())
        return a_tau, a_srcb, a_dstb

    def select_action(self, state: np.ndarray, safety_filter=None) -> np.ndarray:
        self.net.eval()
        s = torch.from_numpy(state[None, :]).float().to(self.device)
        with torch.no_grad():
            lt, ls, ld, _ = self.net(s)
            pt, ps, pd = torch.softmax(lt, dim=1), torch.softmax(ls, dim=1), torch.softmax(ld, dim=1)
            a_tau = int(torch.multinomial(pt[0], 1).item())
            a_srcb = int(torch.multinomial(ps[0], 1).item())
            a_dstb = int(torch.multinomial(pd[0], 1).item())
        action = self._index_to_action(a_tau, a_srcb, a_dstb, state)
        if safety_filter is not None:
            action = safety_filter(action)
        return action

    def observe(self, s: np.ndarray, a: np.ndarray, r: float, s2: np.ndarray, info):
        # Map continuous action back to bins
        a_tau = int(np.argmin(np.abs(self.tau_bins - a[0])))
        a_srcb = int(min(self.num_src_buckets - 1, max(0, int(a[1] // self.bucket_size))))
        a_dstb = int(min(self.num_dst_buckets - 1, max(0, int(a[2] // self.bucket_size))))
        self.rollouts.append(Rollout(s.astype(np.float32), (a_tau, a_srcb, a_dstb), float(r), s2.astype(np.float32)))
        # Truncate to short buffer for fast updates
        if len(self.rollouts) > 256:
            self.rollouts = self.rollouts[-256:]

    def learn(self):
        if not self.rollouts:
            return
        # Build tensors
        batch = self.rollouts
        self.rollouts = []
        s = torch.from_numpy(np.stack([b.s for b in batch])).float().to(self.device)
        s2 = torch.from_numpy(np.stack([b.s2 for b in batch])).float().to(self.device)
        a_tau = torch.tensor([b.a[0] for b in batch], dtype=torch.long, device=self.device)
        a_src = torch.tensor([b.a[1] for b in batch], dtype=torch.long, device=self.device)
        a_dst = torch.tensor([b.a[2] for b in batch], dtype=torch.long, device=self.device)
        r = torch.tensor([b.r for b in batch], dtype=torch.float32, device=self.device)

        # Old policy logits
        self.net.eval()
        with torch.no_grad():
            lt_old, ls_old, ld_old, v_old = self.net(s)
            _, _, _, v_next = self.net(s2)
        pt_old = torch.softmax(lt_old, dim=1)
        ps_old = torch.softmax(ls_old, dim=1)
        pd_old = torch.softmax(ld_old, dim=1)
        logp_old = (
            torch.log(pt_old.gather(1, a_tau[:, None]).squeeze(1) + 1e-8)
            + torch.log(ps_old.gather(1, a_src[:, None]).squeeze(1) + 1e-8)
            + torch.log(pd_old.gather(1, a_dst[:, None]).squeeze(1) + 1e-8)
        )
        # TD(1)-style targets and 1-step GAE using scalar value head
        with torch.no_grad():
            td_target = r + self.gamma * v_next
            delta = td_target - v_old
            adv = delta
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)

        # Optimize for multiple epochs with mini-batches
        N = s.size(0)
        idx = np.arange(N)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, N, self.minibatch_size):
                end = min(start + self.minibatch_size, N)
                mb = idx[start:end]
                mb_s = s[mb]
                mb_at, mb_as, mb_ad = a_tau[mb], a_src[mb], a_dst[mb]
                mb_logp_old = logp_old[mb]
                mb_adv = adv[mb]
                mb_td_target = td_target[mb]

                self.net.train()
                lt, ls, ld, v_pred = self.net(mb_s)
                pt = torch.softmax(lt, dim=1)
                ps = torch.softmax(ls, dim=1)
                pd = torch.softmax(ld, dim=1)
                logp = (
                    torch.log(pt.gather(1, mb_at[:, None]).squeeze(1) + 1e-8)
                    + torch.log(ps.gather(1, mb_as[:, None]).squeeze(1) + 1e-8)
                    + torch.log(pd.gather(1, mb_ad[:, None]).squeeze(1) + 1e-8)
                )
                ratio = torch.exp(logp - mb_logp_old)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * mb_adv
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_v = self.value_coef * ((v_pred - mb_td_target) ** 2).mean()
                # entropy bonus
                ent = (
                    -(pt * torch.log(pt + 1e-8)).sum(dim=1)
                    -(ps * torch.log(ps + 1e-8)).sum(dim=1)
                    -(pd * torch.log(pd + 1e-8)).sum(dim=1)
                ).mean()
                loss = loss_pi + loss_v - self.entropy_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()


