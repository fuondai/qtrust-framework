from __future__ import annotations

from typing import Optional, List

import numpy as np
import simpy

from qtrust.config import QTrustConfig
from fl_smpc.sss import split_secret, reconstruct_secret, split_vector, reconstruct_vector
from .model_adapter import ModelAdapter


class FLSMPCController:
    def __init__(self, cfg: QTrustConfig, trust, rng: np.random.Generator):
        self.cfg = cfg
        self.trust = trust
        self.rng = rng
        self.adapter: Optional[ModelAdapter] = None
        self.global_quant_scale: float = 2048.0  # fixed-point scale
        self.global_prime: int = cfg.fl_smpc.sss_prime
        self._last_candidate: Optional[np.ndarray] = None

    def bind_agent(self, agent) -> None:
        self.adapter = ModelAdapter.from_agent(agent)

    def _local_model_vector(self) -> np.ndarray:
        assert self.adapter is not None
        return self.adapter.get_local_vector()

    def _quant_local(self, vec: np.ndarray) -> np.ndarray:
        assert self.adapter is not None
        q = self.adapter.quantize(vec, self.global_quant_scale, self.global_prime)
        # shift into [0, prime) for SSS
        return (q % self.global_prime).astype(int)

    def run(self, env: simpy.Environment):
        while True:
            yield env.timeout(self.cfg.fl_smpc.round_interval_minutes * 60)
            # Select committee at validator level based on aggregated validator trust
            vtrust = self.trust.get_validator_trust()  # shape [shards, validators]
            # Build flat index of (shard, validator) pairs above threshold, sorted by trust desc
            eligible: list[tuple[int, int, float]] = []
            for s in range(vtrust.shape[0]):
                for v in range(vtrust.shape[1]):
                    t = float(vtrust[s, v])
                    if t >= float(self.cfg.fl_smpc.committee_threshold):
                        eligible.append((s, v, t))
            if len(eligible) < self.cfg.fl_smpc.sss_threshold:
                continue
            if self.adapter is None:
                continue
            # Take top-N validators by trust up to committee_size
            eligible.sort(key=lambda x: x[2], reverse=True)
            # ensure we pick at most committee_size but at least threshold if available
            chosen = eligible[: int(min(self.cfg.fl_smpc.committee_size, len(eligible)))]
            if len(chosen) < int(self.cfg.fl_smpc.sss_threshold):
                continue
            # Build quantized local model vectors for chosen validators (model is agent-global)
            qlocals: List[np.ndarray] = []
            for _s, _v, _t in chosen:
                try:
                    v = self._local_model_vector()
                    # optional per-validator perturbation to emulate heterogeneity (disabled by default)
                    try:
                        sigma = float(getattr(self.cfg.fl_smpc, "local_perturb_sigma", 0.0))
                    except Exception:
                        sigma = 0.0
                    if sigma > 0.0:
                        noise = np.random.normal(loc=0.0, scale=sigma, size=v.shape).astype(np.float32)
                        v = (v + noise).astype(np.float32)
                    qv = self._quant_local(v)
                    qlocals.append(qv)
                except Exception:
                    continue
            if not qlocals:
                continue
            # Secure sum of quantized model vectors via SSS per component (no proxy, full vector)
            prime = self.global_prime
            model_dim = int(qlocals[0].shape[0])
            # Split each local vector into per-component shares and sum shares component-wise modulo prime
            shares_sum_vec: list[list[tuple[int, int]]] | None = None
            for qv in qlocals:
                vec_shares = split_vector(qv.tolist(), self.cfg.fl_smpc.sss_shares, self.cfg.fl_smpc.sss_threshold, prime)
                if shares_sum_vec is None:
                    shares_sum_vec = [[(x, y) for (x, y) in comp] for comp in vec_shares]
                else:
                    for i, comp in enumerate(vec_shares):
                        shares_sum_vec[i] = [(x, (y + comp[j][1]) % prime) for j, (x, y) in enumerate(shares_sum_vec[i])]
            assert shares_sum_vec is not None
            # Reconstruct from any threshold subset of shares (use first threshold indices)
            chosen_idx = list(range(self.cfg.fl_smpc.sss_threshold))
            qagg_vec = reconstruct_vector([[shares_sum_vec[i][j] for j in chosen_idx] for i in range(model_dim)], prime)
            qagg = np.array(qagg_vec, dtype=np.int64)
            # Dequantize to float model vector
            agg_float = self.adapter.dequantize(qagg.astype(np.int64), self.global_quant_scale, prime)
            # Average over participating validators to prevent magnitude blow-up
            num_participants = max(1, len(chosen))
            agg_float = (agg_float / float(num_participants)).astype(np.float32)
            # Optional DP noise: per-parameter Gaussian noise (zero-mean, sigma from config)
            if bool(self.cfg.fl_smpc.dp_enabled) and float(self.cfg.fl_smpc.dp_sigma) > 0.0:
                try:
                    noise = np.random.normal(loc=0.0, scale=float(self.cfg.fl_smpc.dp_sigma), size=agg_float.shape).astype(np.float32)
                    agg_float = (agg_float + noise).astype(np.float32)
                except Exception:
                    pass
            # Prepare candidate by small step toward aggregated vector (to reduce oscillation)
            current = self._local_model_vector()
            candidate = (0.9 * current + 0.1 * agg_float).astype(np.float32)
            self._last_candidate = candidate
            # Estimate per-node data exchange (bytes) for this round
            try:
                approx_model_bytes = int(self.cfg.fl_smpc.approx_model_bytes)
                num_participants = len(chosen)
                # Each participant sends n_shares shares per component; approximate by model bytes
                per_node_bytes = approx_model_bytes * int(self.cfg.fl_smpc.sss_shares)
                # Log via adapter if it supports side-channel metrics (optional)
                _ = per_node_bytes, num_participants  # placeholders for future logging
            except Exception:
                pass


