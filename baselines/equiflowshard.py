from __future__ import annotations

import numpy as np

from qtrust.config import QTrustConfig


class EquiFlowShard:
    """EquiFlowShard-like baseline: workload-aware, non-security-aware.

    Heuristic approximation of paper description: use cross-shard traffic matrix M_cross (from obs)
    and current shard queue loads to greedily reassign a validator from a high-pressure source shard
    to a destination shard that maximally reduces estimated cross-shard pressure.

    This baseline is intentionally risk-agnostic and avoids using trust signals. It performs a
    simple two-step selection each decision interval:
      1) Select the source shard with highest cross-out pressure weighted by its current load.
      2) Select a destination shard that minimizes an estimate of incoming cross-shard mass and
         avoids overloaded shards. If estimated improvement is negligible, produce a no-op.
    """

    def __init__(self, obs_dim: int, action_dim: int, cfg: QTrustConfig):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg

    def select_action(self, state: np.ndarray, safety_filter=None) -> np.ndarray:
        S = self.cfg.simulation.shards
        mcross_len = S * S
        # Unpack observation: [M_cross_flat, shard_trust, shard_load]
        mcross = state[: mcross_len].reshape(S, S)
        # shard_trust is ignored to keep this baseline strictly performance-centric
        _shard_trust = state[mcross_len: mcross_len + S]
        shard_load = state[mcross_len + S: mcross_len + 2 * S]

        # Remove diagonal (local tx not contributing to cross-shard pressure)
        off = mcross.copy()
        for i in range(S):
            off[i, i] = 0.0

        # 1) Choose source shard with highest cross-out pressure weighted by its load
        cross_out = off.sum(axis=1)
        src_score = cross_out * (shard_load + 1e-6)
        src = int(np.argmax(src_score))

        # 2) Choose destination shard to reduce pressure: prefer low load and low incoming mass
        cross_in = off.sum(axis=0)
        # Capacity and congestion guards
        med_load = float(np.median(shard_load))
        overload_mask = (shard_load > max(1e-6, med_load * 1.25)).astype(np.float32)
        # Score favors shards that (a) are lightly loaded and (b) currently receive little cross-in
        inv_load = 1.0 / (1e-6 + shard_load)
        inv_in = 1.0 / (1e-6 + cross_in)
        base_score = inv_load * inv_in
        # Penalize overloaded destinations
        score = base_score * (1.0 - 0.75 * overload_mask)
        # Never choose the same shard as destination for reassignment
        score[src] = -np.inf
        dst = int(np.argmax(score))

        # 3) No-op if improvement is negligible (prevents thrashing on steady state)
        # Estimate crude improvement as change in src cross-out if one validator moves
        # (bounded heuristic; avoids unnecessary moves when network is balanced)
        improvement_proxy = float(src_score[src]) - float((cross_out[src] * (shard_load[src] + 1e-6)) * 0.98)
        if not np.isfinite(improvement_proxy) or improvement_proxy < 1e-6:
            action = np.array([0.0, float(src), float(src)], dtype=np.float32)
        else:
            action = np.array([0.0, float(src), float(dst)], dtype=np.float32)

        if safety_filter is not None:
            action = safety_filter(action)
        return action

    def observe(self, *args, **kwargs):
        return

    def learn(self):
        return


