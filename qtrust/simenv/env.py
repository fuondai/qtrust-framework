from __future__ import annotations

import math
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import simpy

from qtrust.config import QTrustConfig
from qtrust.trust.htdcm import HTDCM
from qtrust.metrics.reward import compute_reward
from qtrust.consensus.mad_rapid import MADRapidProtocol
from qtrust.metrics.logging import JsonLogger
from .model_adapter import ModelAdapter


class QTrustSimEnv:
    """Epoch-level decision environment around a continuous SimPy simulation.

    The simulation runs continuously; every decision_interval_seconds we snapshot
    the state and allow the RL/baseline to act.
    """

    def __init__(self, cfg: QTrustConfig, logger: Optional[JsonLogger] = None):
        self.cfg = cfg
        self.logger = logger
        self.env = simpy.Environment()

        self.num_shards = cfg.simulation.shards
        self.validators_per_shard = cfg.simulation.validators_per_shard

        # Cross-shard transaction matrix M_cross (traffic rate between shards) with locality bias
        rng = np.random.default_rng(cfg.seed)
        base = rng.random((self.num_shards, self.num_shards))
        # Increase diagonal mass to reflect local traffic preference
        np.fill_diagonal(base, np.maximum(base.diagonal(), cfg.simulation.base_local_tx_prob))
        # Keep a copy of diagonal bias; off-diagonal used for cross-shard routing distribution
        self._diag_bias = base.diagonal().copy()
        base_off = base.copy()
        np.fill_diagonal(base_off, 0.0)
        denom = float(base_off.sum()) if float(base_off.sum()) > 0.0 else 1.0
        self.M_cross = base_off / denom

        # Trust layer and consensus protocol
        self.trust = HTDCM(cfg)
        self.consensus = MADRapidProtocol(cfg, trust=self.trust, rng=rng, logger=logger)

        # Observation and action spaces
        self.observation_space_dim = self.num_shards * self.num_shards + self.num_shards * 2
        # Actions: [delta_tau, reassign_validator_src, reassign_validator_dst]
        # delta_tau is continuous in [-0.1, 0.1]; reassignment picks two shard indices
        self.action_space_dim = 3

        self._rng = rng
        self._reset_internal()

    def _reset_internal(self):
        self.decision_interval = self.cfg.rl.decision_interval_seconds
        self.total_duration = self.cfg.simulation.duration_hours * 3600

        self.normalized_trust = self.trust.initialize(self.num_shards, self.validators_per_shard)
        self.pending_txs: List[Dict[str, Any]] = []
        self.metrics_history: List[Dict[str, Any]] = []

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.env = simpy.Environment()
        # Start background processes
        self.env.process(self._tx_arrival_process())
        # Optionally FL/SMPC process
        try:
            from qtrust.simenv.fl_controller import FLSMPCController
            self._fl = FLSMPCController(self.cfg, self.trust, self._rng)
            # bind later after agent creation
            self.env.process(self._fl.run(self.env))
        except Exception:
            self._fl = None
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        # Flatten M_cross and append shard avg trust and queue lengths
        mcross_flat = self.M_cross.flatten()
        shard_trust = self.trust.get_shard_trust()
        shard_load = self.consensus.get_shard_queue_loads()
        obs = np.concatenate([mcross_flat, shard_trust, shard_load]).astype(np.float32)
        return obs

    def _tx_arrival_process(self):
        # Poisson process: exponential inter-arrival with rate = tx_arrival_tps
        tps = float(max(self.cfg.simulation.tx_arrival_tps, 1))
        while self.env.now < self.total_duration:
            inter_arrival = float(np.random.exponential(1.0 / tps))
            yield self.env.timeout(inter_arrival)
            # Sample source shard with light load-bias (prefer hotter shards)
            loads = self.consensus.get_shard_queue_loads() + 1e-3
            p_src = (loads / loads.sum()) if loads.sum() > 0 else np.ones(self.num_shards, dtype=float) / float(self.num_shards)
            src = int(self._rng.choice(self.num_shards, p=p_src))
            # With probability proportional to diagonal bias, keep transaction local; else choose cross-shard dst
            if self._rng.random() < float(np.clip(self._diag_bias[min(src, len(self._diag_bias) - 1)], 0.0, 0.95)):
                dst = src
            else:
                row = self.M_cross[src].copy()
                if self.num_shards > 1:
                    # ensure src prob is zero
                    row[src] = 0.0
                s = float(row.sum())
                if s <= 0.0:
                    candidates = [i for i in range(self.num_shards) if i != src]
                    dst = int(self._rng.choice(candidates)) if candidates else src
                else:
                    row /= s
                    dst = int(self._rng.choice(self.num_shards, p=row))
            self.pending_txs.append({"src": int(src), "dst": int(dst), "ts": float(self.env.now)})
            # Process some transactions per interval with consensus
            # Batch size scales modestly with shards to keep runtime bounded
            batch_size = max(50, min(200, 2 * self.num_shards))
            if len(self.pending_txs) >= batch_size:
                batch = self.pending_txs[:batch_size]
                self.pending_txs = self.pending_txs[batch_size:]
                self.consensus.process_batch(batch, now=float(self.env.now))

    def safety_filter(self, action: np.ndarray) -> np.ndarray:
        # Ensure tau within [0.4, 0.95], reassign within shard bounds and locality rules if toggled off
        delta_tau = float(np.clip(action[0], -0.1, 0.1))
        new_tau = float(np.clip(self.trust.threshold_tau + delta_tau, 0.4, 0.95))
        src = int(np.clip(round(action[1]), 0, self.num_shards - 1))
        dst = int(np.clip(round(action[2]), 0, self.num_shards - 1))
        if not self.cfg.simulation.locality_optimization:
            src = dst  # disallow reassignment effectively
        return np.array([new_tau, src, dst], dtype=np.float32)

    def step(self, action: np.ndarray):
        # Apply action via safety filter
        filt = self.safety_filter(action)
        new_tau, src, dst = float(filt[0]), int(filt[1]), int(filt[2])
        self.trust.update_threshold(new_tau)
        if src != dst:
            self.consensus.reassign_validator(src, dst, now=float(self.env.now))

        # Advance simulation for one decision interval
        target_time = min(float(self.env.now) + self.decision_interval, self.total_duration)
        while float(self.env.now) < target_time:
            try:
                self.env.step()  # progress one event
            except Exception:
                break

        # Compute metrics and reward
        shard_trust = self.trust.get_shard_trust()
        throughput, latency, power_w, cross_cost = self.consensus.sample_kpis(now=float(self.env.now))
        tpr, spc, ttd_norm = self.consensus.sample_security_metrics(now=float(self.env.now))

        # Normalize KPIs to [0,1] for reward stability
        thr_norm = float(np.clip(throughput / max(self.cfg.reward.throughput_norm_max_tps, 1e-6), 0.0, 1.0))
        lat_norm = float(np.clip(latency / max(self.cfg.reward.latency_norm_max_s, 1e-6), 0.0, 1.0))
        reward = compute_reward(
            cfg=self.cfg,
            T_norm=thr_norm,
            L_norm=lat_norm,
            P=float(1.0 - spc),  # penalty proxy from specificity
            C_cross=float(cross_cost),
        )

        info = {
            "time": float(self.env.now),
            "throughput": float(throughput),
            "latency": float(latency),
            # 'power' is a model proxy combining message count and latency; expose a separate explicit proxy
            "power": power_w,
            "power_proxy": float(self.consensus._msg_ema),
            "cross_cost": cross_cost,
            "tpr": tpr,
            "spc": spc,
            "ttd_norm": ttd_norm,
        }
        self.metrics_history.append(info)
        if self.logger:
            self.logger.log({"type": "metrics", **info})

        return self._get_obs(), reward, info

    # Agent integration hooks for FL/SMPC canary
    def bind_agent(self, agent) -> None:
        # create adapter and bind to FL controller for local vector extraction
        try:
            if self._fl is not None:
                self._fl.bind_agent(agent)
        except Exception:
            pass
        # store adapter to evaluate canary candidates when available
        try:
            self._model_adapter = ModelAdapter.from_agent(agent)
        except Exception:
            self._model_adapter = None

    def canary_try_promote(self) -> None:
        if self._fl is None or getattr(self._fl, "_last_candidate", None) is None:
            return
        if self._model_adapter is None:
            return
        # build a small validation set from recent observations
        recent = self.metrics_history[-50:] if len(self.metrics_history) > 0 else []
        # generate synthetic states around current obs for stability
        states: list[np.ndarray] = []
        if recent:
            obs0 = self._get_obs()
            states.append(obs0)
            for _ in range(min(9, len(recent))):
                jitter = np.random.normal(scale=0.01, size=obs0.shape).astype(np.float32)
                states.append(np.clip(obs0 + jitter, 0.0, 1e9).astype(np.float32))
        else:
            states = [self._get_obs()]
        cand = self._fl._last_candidate
        try:
            accepted = self._model_adapter.apply_candidate(cand, states)
            if accepted:
                # reset candidate marker
                self._fl._last_candidate = None
        except Exception:
            pass

    def close(self):
        if self.logger:
            self.logger.close()

    # Introspection helpers for agent
    @property
    def action_low(self) -> np.ndarray:
        return np.array([-0.1, 0.0, 0.0], dtype=np.float32)

    @property
    def action_high(self) -> np.ndarray:
        return np.array([0.1, float(self.num_shards - 1), float(self.num_shards - 1)], dtype=np.float32)


