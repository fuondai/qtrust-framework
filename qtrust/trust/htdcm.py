from __future__ import annotations

from typing import List, Tuple

import numpy as np

from qtrust.config import QTrustConfig


class HTDCM:
    """Hierarchical Trust-based Coordination Mechanism (HTDCM).

    Implements a six-dimensional per-validator trust vector with adaptive weighted
    geometric mean aggregation and temporal smoothing. Dimensions align with the
    paper's definitions:
      1) Performance and Stability (T_perf): inverse latency and throughput stability
      2) Behavioral Consistency (T_cons): stability of recent vs long-term action distribution
      3) Security Compliance (T_sec): discrete compliance incidents (1=no incidents)
      4) Economic Stake (T_econ): normalized stake associated with validator
      5) Network Participation (T_net): uptime/availability
      6) Cross-Shard Interaction History (T_cross): success rate in atomic commits

    Adaptive weights are inversely proportional to EWMA variance of each dimension
    to down-weight noisy metrics. Aggregation uses a non-compensatory weighted
    geometric mean and an EWMA temporal smoother. A grace region around the
    threshold prevents flapping.
    """

    def __init__(self, cfg: QTrustConfig):
        self.cfg = cfg
        self.threshold_tau = cfg.trust.threshold_tau
        self.ewma_alpha = cfg.trust.ewma_alpha
        self.decay_lambda = cfg.trust.decay_lambda
        self.grace_period_seconds = cfg.trust.grace_period_seconds

        self.num_shards = cfg.simulation.shards
        self.validators_per_shard = cfg.simulation.validators_per_shard

        self.dim = 6
        self._rng = np.random.default_rng(cfg.seed)

        # Per-validator trust dimensions and stats
        self._vals: np.ndarray
        self._ewma_var: np.ndarray
        self._uptime: np.ndarray
        self._stake: np.ndarray
        self._commit_success: np.ndarray
        self._commit_total: np.ndarray
        self._cons_hist_short: np.ndarray
        self._cons_hist_long: np.ndarray
        self._sec_incidents: np.ndarray

    def initialize(self, num_shards: int, validators_per_shard: int) -> np.ndarray:
        self.num_shards = num_shards
        self.validators_per_shard = validators_per_shard

        # Initialize raw stats
        self._uptime = np.ones((num_shards, validators_per_shard), dtype=np.float32)
        self._stake = self._rng.uniform(0.5, 1.0, size=(num_shards, validators_per_shard)).astype(np.float32)
        self._commit_success = np.zeros((num_shards, validators_per_shard), dtype=np.int32)
        self._commit_total = np.zeros((num_shards, validators_per_shard), dtype=np.int32)
        self._cons_hist_short = np.ones((num_shards, validators_per_shard, 3), dtype=np.float32) * (1.0 / 3.0)
        self._cons_hist_long = np.ones((num_shards, validators_per_shard, 3), dtype=np.float32) * (1.0 / 3.0)
        self._sec_incidents = np.zeros((num_shards, validators_per_shard), dtype=np.int32)

        # Trust vector per validator across six dimensions (values in (0,1])
        self._vals = np.clip(
            self._rng.uniform(0.65, 0.85, size=(num_shards, validators_per_shard, self.dim)).astype(np.float32),
            1e-6,
            1.0,
        )
        # Start with modest variance estimates per dimension for adaptive weighting
        self._ewma_var = np.ones(self.dim, dtype=np.float32) * 0.05
        # Initialize scalar EWMA over aggregated per-validator trust
        w0 = self._compute_weights()
        log_vals0 = np.log(self._vals)
        weighted_log0 = (log_vals0 * w0.reshape(1, 1, -1)).sum(axis=2)
        per_validator0 = np.exp(weighted_log0)
        self._ewma_agg = per_validator0.astype(np.float32)
        # Track last time a validator fell below threshold to implement grace period acceptance
        self._last_below_tau_at = np.full((num_shards, validators_per_shard), -1e18, dtype=float)
        return self.aggregate_shard_trust()

    def update_threshold(self, new_tau: float) -> None:
        self.threshold_tau = float(np.clip(new_tau, 0.0, 1.0))

    def get_shard_trust(self) -> np.ndarray:
        return self.aggregate_shard_trust()

    def get_validator_trust(self, shard: int | None = None) -> np.ndarray:
        """Returns per-validator aggregated trust in [0,1].
        If shard is None, returns shape [num_shards, validators_per_shard].
        """
        agg = self._aggregate_per_validator()
        if shard is None:
            return agg
        return agg[shard]

    def _compute_weights(self) -> np.ndarray:
        inv_var = 1.0 / np.maximum(self._ewma_var, 1e-6)
        w = inv_var / inv_var.sum()
        return w.astype(np.float32)

    def _aggregate_per_validator(self) -> np.ndarray:
        # Weighted geometric mean per validator across six dimensions
        w = self._compute_weights()
        vals = np.clip(self._vals, 1e-6, 1.0)
        log_vals = np.log(vals)
        weighted_log = (log_vals * w.reshape(1, 1, -1)).sum(axis=2)
        per_validator = np.exp(weighted_log)
        # Temporal smoothing using scalar EWMA of aggregated trust per validator
        self._ewma_agg = (1.0 - self.decay_lambda) * self._ewma_agg + self.decay_lambda * per_validator
        return np.clip(self._ewma_agg, 1e-6, 1.0).astype(np.float32)

    def aggregate_shard_trust(self) -> np.ndarray:
        per_validator = self._aggregate_per_validator()
        return per_validator.mean(axis=1).astype(np.float32)

    def _update_variance_weights(self, new_metrics: np.ndarray) -> None:
        # Update EWMA of per-dimension variance based on current observation spread
        var = new_metrics.var(axis=0)
        self._ewma_var = (1.0 - self.ewma_alpha) * self._ewma_var + self.ewma_alpha * var

    def observe_event(
        self,
        shard_indices: List[int],
        coordinator: Tuple[int, int] | None,
        participants: List[Tuple[int, int]],
        latency_s: float,
        success: bool,
        is_slow_poisoning: bool,
        is_collusion: bool,
        now: float,
        adv_involved: bool,
    ) -> None:
        """Update per-validator trust dimensions based on an observed cross-shard event.

        shard_indices: shards involved (e.g., [src, dst])
        coordinator: (shard, validator_idx) of coordinator if any
        participants: list of (shard, validator_idx) participating
        latency_s: end-to-end latency for the event
        success: commit success flag
        is_slow_poisoning: if event is under slow-poisoning conditions
        is_collusion: if collusion scenario is active
        """
        # Normalize latency to (0,1] with a conservative cap
        L = float(np.clip(latency_s / max(1e-3, 1.0), 0.0, 5.0))  # 0..5
        t_perf = float(np.clip(1.0 / (1.0 + L), 0.0, 1.0))

        # Participants to update include coordinator and shard-local participants
        touched: List[Tuple[int, int]] = []
        if coordinator is not None:
            touched.append(coordinator)
        touched.extend(participants)

        # Build per-validator metric deltas
        for (s, v) in touched:
            # Performance & Stability: higher when latency low and success
            # Stronger penalty under adversarial involvement or failed outcome
            if success:
                perf = t_perf
            else:
                perf = t_perf * (0.4 if adv_involved or is_slow_poisoning else 0.7)
            # Behavioral Consistency: update short vs long hist of (prepare, commit, abort)
            # Simplify: success->commit, else abort
            idx = 1 if success else 2
            self._cons_hist_short[s, v, idx] = 0.9 * self._cons_hist_short[s, v, idx] + 0.1
            self._cons_hist_short[s, v] /= self._cons_hist_short[s, v].sum()
            self._cons_hist_long[s, v, idx] = 0.99 * self._cons_hist_long[s, v, idx] + 0.01
            self._cons_hist_long[s, v] /= self._cons_hist_long[s, v].sum()
            # KL divergence proxy: 1 / (1 + KL) in (0,1]
            p = np.clip(self._cons_hist_short[s, v], 1e-6, 1.0)
            q = np.clip(self._cons_hist_long[s, v], 1e-6, 1.0)
            kl = float(np.sum(p * (np.log(p) - np.log(q))))
            t_cons = float(np.clip(1.0 / (1.0 + kl), 0.0, 1.0))

            # Security Compliance: increment incidents on collusion or adversarial involvement
            if is_collusion or (adv_involved and (not success or is_slow_poisoning)):
                self._sec_incidents[s, v] += 1
            t_sec = float(np.clip(1.0 - 0.05 * self._sec_incidents[s, v], 0.0, 1.0))

            # Economic Stake: static per run (normalized 0.5..1.0) but can decay on incidents
            if not success:
                self._stake[s, v] = float(max(0.1, self._stake[s, v] * 0.999))
            t_econ = float(np.clip(self._stake[s, v], 0.0, 1.0))

            # Network Participation: uptime decays; stronger when adversary involved or failure
            if is_slow_poisoning or not success:
                decay = 0.01 if adv_involved else 0.002
                self._uptime[s, v] = float(max(0.0, self._uptime[s, v] - decay))
            else:
                self._uptime[s, v] = float(min(1.0, self._uptime[s, v] + 0.001))
            t_net = float(np.clip(self._uptime[s, v], 0.0, 1.0))

            # Cross-shard Interaction History: success ratio with Laplace smoothing
            self._commit_total[s, v] += 1
            self._commit_success[s, v] += 1 if success else 0
            success_rate = (self._commit_success[s, v] + 1) / (self._commit_total[s, v] + 2)
            t_cross = float(np.clip(success_rate, 0.0, 1.0))

            new_metrics = np.array([perf, t_cons, t_sec, t_econ, t_net, t_cross], dtype=np.float32)
            # Non-compensatory update: geometric interpolation toward new metrics
            current = np.clip(self._vals[s, v], 1e-6, 1.0)
            log_cur = np.log(current)
            log_new = np.log(np.clip(new_metrics, 1e-6, 1.0))
            step = 0.2 if adv_involved or (is_slow_poisoning and not success) else 0.1
            step = float(np.clip(step, 0.05, 0.3))
            self._vals[s, v] = np.exp((1.0 - step) * log_cur + step * log_new)

        # Update global variance weights
        # Construct pseudo-observation matrix across dims from touched validators
        if touched:
            obs = np.stack([self._vals[s, v] for (s, v) in touched], axis=0)
            self._update_variance_weights(obs)

        # Update last-below-threshold timestamps for grace logic at validator level
        try:
            agg = self._aggregate_per_validator()
            below = agg < float(self.threshold_tau)
            self._last_below_tau_at[below] = float(now)
        except Exception:
            pass

    def eligible_validator_mask(self, now: float, tau: float, grace_period_s: float) -> np.ndarray:
        """Return [shards, validators] boolean mask of validators eligible by tau or within grace window."""
        try:
            agg = self._aggregate_per_validator()
            below = agg < float(tau)
            within_grace = (now - self._last_below_tau_at) <= float(grace_period_s)
            return (~below) | within_grace
        except Exception:
            # Fallback: all eligible
            return np.ones((self.num_shards, self.validators_per_shard), dtype=bool)

    # Optional: allow consensus layer to request reassignment hooks
    def reassign_validator(self, src_shard: int, dst_shard: int) -> None:
        if src_shard == dst_shard:
            return
        # Swap top validator between shards to improve locality (simple heuristic)
        vt_src = self.get_validator_trust(src_shard)
        vt_dst = self.get_validator_trust(dst_shard)
        i_src = int(np.argmax(vt_src))
        i_dst = int(np.argmax(vt_dst))
        # Swap stats across all maintained arrays
        for arr in [self._vals, self._ewma_agg, self._uptime, self._stake, self._commit_success, self._commit_total, self._cons_hist_short, self._cons_hist_long, self._sec_incidents]:
            try:
                a = arr
                if isinstance(a, np.ndarray) and a.ndim == 3 and a.shape[2] == self.dim:
                    tmp = a[src_shard, i_src, :].copy()
                    a[src_shard, i_src, :] = a[dst_shard, i_dst, :]
                    a[dst_shard, i_dst, :] = tmp
                elif isinstance(a, np.ndarray) and a.ndim == 3:
                    tmp = a[src_shard, i_src, :].copy()
                    a[src_shard, i_src, :] = a[dst_shard, i_dst, :]
                    a[dst_shard, i_dst, :] = tmp
                elif isinstance(a, np.ndarray) and a.ndim == 2:
                    tmp = a[src_shard, i_src].copy()
                    a[src_shard, i_src] = a[dst_shard, i_dst]
                    a[dst_shard, i_dst] = tmp
            except Exception:
                continue



