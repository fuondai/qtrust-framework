from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import math

from qtrust.config import QTrustConfig
from qtrust.metrics.logging import JsonLogger


class MADRapidProtocol:
    """MAD-RAPID-style cross-shard commit with trust-gated coordinator selection.

    Implements a deterministic two-phase commit across source/destination shards,
    selecting a coordinator validator from a high-trust pool. Integrates with
    HTDCM to update per-validator trust based on observed outcomes.
    """

    def __init__(self, cfg: QTrustConfig, trust, rng: np.random.Generator, logger: Optional[JsonLogger] = None):
        self.cfg = cfg
        self.trust = trust
        self.rng = rng
        self.logger = logger
        self.num_shards = cfg.simulation.shards
        self.validators_per_shard = cfg.simulation.validators_per_shard

        self._queues = [0 for _ in range(self.num_shards)]
        self._latency_s = 0.1
        self._throughput = 0.0
        self._power_w = 0.0
        self._cross_cost = 0.0
        # Track simple per-round message counts to drive a more meaningful power proxy
        self._msg_ema = 0.0
        # Detection metrics computed from events rather than fixed proxies
        self._tpr = 0.0
        self._spc = 1.0
        self._ttd_norm = 1.0
        # Track when a shard last dipped below the gate threshold to apply grace period
        self._last_below_tau_at = np.full(self.num_shards, -1e18, dtype=float)

        # Attack state
        self._slow_poisoning_enabled = cfg.attacks.slow_poisoning.enabled
        self._sp_warmup = cfg.attacks.slow_poisoning.warmup_minutes * 60
        self._sp_delay_mean = cfg.attacks.slow_poisoning.delay_ms_mean / 1000.0
        self._sp_drop_prob = cfg.attacks.slow_poisoning.drop_prob

        self._collusion_enabled = cfg.attacks.collusion_sybil.enabled
        self._collusion_start = cfg.attacks.collusion_sybil.start_minutes * 60
        self._collusion_strength = cfg.attacks.collusion_sybil.manipulation_strength
        # Adversary model at validator granularity
        adv_validators = max(0, int(self.num_shards * self.validators_per_shard * self.cfg.simulation.adversary_ratio))
        all_pairs = [(s, v) for s in range(self.num_shards) for v in range(self.validators_per_shard)]
        self._adv_nodes = set()
        if adv_validators > 0 and len(all_pairs) > 0:
            sel_idx = self.rng.choice(len(all_pairs), size=min(adv_validators, len(all_pairs)), replace=False).tolist()
            for i in sel_idx:
                self._adv_nodes.add(tuple(all_pairs[i]))

    def get_shard_queue_loads(self) -> np.ndarray:
        q = np.array(self._queues, dtype=float)
        maxq = max(q.max(), 1.0)
        return (q / maxq).astype(np.float32)

    def _select_coordinator_validator(self, shard_candidates: np.ndarray, now: float) -> Tuple[int, int]:
        """Select (shard, validator_idx) as coordinator from eligible validators.
        We rank shards by trust, then within a shard pick the highest-trust validator.
        """
        # Shard-level gate
        shard_trust = self.trust.get_shard_trust()
        grace = float(self.cfg.trust.grace_period_seconds)
        gate = float(self.cfg.consensus.trust_gate_threshold)
        # Eligible if above gate or still within grace window since last dip below (shard-level)
        eligible_mask = (shard_trust >= gate) | ((now - self._last_below_tau_at) <= grace)
        eligible_shards = np.where(eligible_mask)[0]
        if len(eligible_shards) == 0:
            shard = int(shard_candidates[self.rng.integers(0, len(shard_candidates))])
        else:
            topk = min(self.cfg.consensus.coordinator_pool_size, len(eligible_shards))
            top_idx = eligible_shards[np.argsort(shard_trust[eligible_shards])][-topk:]
            shard = int(self.rng.choice(top_idx))
        # Within shard, randomly choose among top-k highest-trust validators eligible by validator-level grace mask
        vtrust = self.trust.get_validator_trust(shard)
        # validator-level eligibility
        try:
            vmask = self.trust.eligible_validator_mask(now=now, tau=gate, grace_period_s=grace)[shard]
        except Exception:
            vmask = np.ones_like(vtrust, dtype=bool)
        vtrust_eff = vtrust.copy()
        vtrust_eff[~vmask] = -np.inf
        # Select a random validator from top-k to mitigate targeting/censorship risk
        k = max(1, min(int(self.cfg.consensus.coordinator_pool_size), vtrust_eff.size))
        # argsort ascending; take last k as top-k
        top_idx = np.argsort(vtrust_eff)[-k:]
        # Guard against all -inf (no eligible): fall back to uniform among all
        if not np.isfinite(vtrust_eff[top_idx]).any():
            v_idx = int(self.rng.integers(0, vtrust_eff.size))
        else:
            # Filter to finite
            finite_idx = [int(i) for i in top_idx if np.isfinite(vtrust_eff[int(i)])]
            if not finite_idx:
                v_idx = int(self.rng.integers(0, vtrust_eff.size))
            else:
                v_idx = int(self.rng.choice(finite_idx))
        return shard, v_idx

    def _two_phase_commit(self, src: int, dst: int, now: float) -> Tuple[bool, float, Tuple[int, int], List[Tuple[int, int]]]:
        """Execute a simplified two-phase commit and return (success, latency, coord, participants)."""
        # Adaptive consensus: choose base latency based on average trust and add network jitter
        shard_trust = self.trust.get_shard_trust()
        avg_trust = float(0.5 * (shard_trust[src] + shard_trust[dst]))
        # Update grace window tracking for shards currently below gate
        gate = float(self.cfg.consensus.trust_gate_threshold)
        below = shard_trust < gate
        if np.any(below):
            self._last_below_tau_at[below] = float(now)
        if self.cfg.consensus.enable_adaptive and avg_trust >= self.cfg.consensus.fastbft_trust_threshold:
            base_ms = self.cfg.consensus.fastbft_base_latency_ms
        else:
            base_ms = self.cfg.consensus.pbft_base_latency_ms
        # Log-normal WAN latency jitter from config (mean, std in ms)
        mu = float(self.cfg.simulation.latency_log_normal_ms.get("mean", 100))
        sigma = float(self.cfg.simulation.latency_log_normal_ms.get("std", 50))
        # sample log-normal properly
        # convert desired mean/std to log-space parameters
        # mean = exp(m + s^2/2), var = (exp(s^2)-1)exp(2m+s^2)
        try:
            var = sigma ** 2
            s2 = math.log(1 + var / (mu ** 2)) if mu > 0 else 0.0
            m = math.log(max(mu, 1e-6)) - 0.5 * s2
            jitter_ms = float(self.rng.lognormal(mean=m, sigma=math.sqrt(max(s2, 1e-12))))
        except Exception:
            jitter_ms = float(self.cfg.simulation.wan_latency_ms)
        l = (max(base_ms, 1) / 1000.0) + max(0.0, jitter_ms) / 1000.0
        # If latency exceeds MAD-RAPID timeout, treat as timeout (drop)
        try:
            if (l * 1000.0) > float(self.cfg.consensus.mad_rapid_timeouts_ms):
                # simulate timeout-induced abort
                l = float(self.cfg.consensus.mad_rapid_timeouts_ms) / 1000.0
                # mark as dropped later
                timeout_drop = True
            else:
                timeout_drop = False
        except Exception:
            timeout_drop = False
        participants = []
        # Coordinator selection over candidate shards {src,dst}
        coord = self._select_coordinator_validator(np.array([src, dst], dtype=int), now=now)

        # Apply slow poisoning if enabled; check adversarial coordinator/participants
        is_sp = self._slow_poisoning_enabled and now >= self._sp_warmup
        dropped = False
        if is_sp and self.rng.random() < self._sp_drop_prob:
            dropped = True
            self._ttd_norm = min(1.0, self._ttd_norm + 0.02)
            self._tpr = max(0.0, self._tpr - 0.01)
        else:
            if is_sp:
                l += abs(self.rng.normal(loc=self._sp_delay_mean, scale=self._sp_delay_mean * 0.25))

        # Participants: pick k validators from src and dst shards to reflect quorum work
        k = max(1, min(3, self.validators_per_shard // 16))
        vt_src = self.trust.get_validator_trust(src)
        vt_dst = self.trust.get_validator_trust(dst)
        v_src_idx = list(np.argsort(vt_src)[-k:])
        v_dst_idx = list(np.argsort(vt_dst)[-k:])
        for i in v_src_idx:
            participants.append((src, int(i)))
        for i in v_dst_idx:
            participants.append((dst, int(i)))

        # If adversarial nodes are involved and SP active, increase failure chance
        if is_sp and ((coord in self._adv_nodes) or any(p in self._adv_nodes for p in participants)):
            if self.rng.random() < min(1.0, self._sp_drop_prob * 1.5):
                dropped = True
        # Apply timeout drop if any
        if timeout_drop:
            dropped = True
        success = not dropped
        return success, float(l), coord, participants

    def process_batch(self, batch: List[Dict[str, Any]], now: float):
        for tx in batch:
            self._queues[tx["src"]] += 1
            self._queues[tx["dst"]] += 1

        for tx in batch:
            src = int(tx["src"])
            dst = int(tx["dst"])
            success, l, coord, participants = self._two_phase_commit(src, dst, now)

            cross = 1.0 if src != dst else 0.0
            self._latency_s = 0.9 * self._latency_s + 0.1 * l
            self._cross_cost = 0.9 * self._cross_cost + 0.1 * cross
            self._throughput = 0.9 * self._throughput + 0.1 * (len(batch) / max(l, 1e-3))
            # power model (proxy): account for 2PC messages and participants touched
            num_participants = max(1, len(participants))
            msgs = 2 * num_participants + 2  # prepare+commit per participant + coord msgs
            self._msg_ema = 0.9 * self._msg_ema + 0.1 * float(msgs)
            # Convert messages to an approximate power figure with gentle scaling
            # Incorporate WAN jitter impact as additional cost via latency multiplier
            lat_factor = 1.0 + min(1.0, float(l))
            self._power_w = 0.9 * self._power_w + 0.1 * (2.0 + 0.02 * self._msg_ema * lat_factor)

            # Update trust using observed event
            is_sp = self._slow_poisoning_enabled and now >= self._sp_warmup
            is_col = self._collusion_enabled and now >= self._collusion_start and (len(self._adv_nodes) > 0)
            try:
                # mark adversarial involvement if any touched node is adversary
                adv_involved = any((coord in self._adv_nodes) or (p in self._adv_nodes) for p in participants)
                self.trust.observe_event(
                    shard_indices=[src, dst],
                    coordinator=coord,
                    participants=participants,
                    latency_s=l,
                    success=success,
                    is_slow_poisoning=is_sp,
                    is_collusion=is_col,
                    now=float(now),
                    adv_involved=bool(adv_involved),
                )
            except Exception:
                pass

            # Update detection metrics from trust vs threshold (compute confusion elements per tx)
            # A simple detector: validators with aggregated trust < tau are labeled malicious
            tau = float(self.cfg.trust.threshold_tau)
            vt = self.trust.get_validator_trust()
            # Define ground-truth adversarial set if enabled
            has_adv = len(self._adv_nodes) > 0
            if has_adv:
                # Build boolean arrays for detection and truth over touched validators
                touched = [coord] + participants
                detected_mal = 0
                true_mal = 0
                true_benign = 0
                correct_detect = 0
                for (s, v) in touched:
                    is_truth_mal = ((s, v) in self._adv_nodes)
                    is_detect_mal = bool(vt[s, v] < tau)
                    detected_mal += int(is_detect_mal)
                    true_mal += int(is_truth_mal)
                    true_benign += int(not is_truth_mal)
                    correct_detect += int((is_detect_mal and is_truth_mal) or ((not is_detect_mal) and (not is_truth_mal)))
                # Update online averages: TPR, SPC, and normalized time-to-detect (proxy: failures increase ttd)
                tp = int(sum(1 for (s, v) in touched if ((s, v) in self._adv_nodes) and (vt[s, v] < tau)))
                fn = int(sum(1 for (s, v) in touched if ((s, v) in self._adv_nodes) and not (vt[s, v] < tau)))
                tn = int(sum(1 for (s, v) in touched if ((s, v) not in self._adv_nodes) and not (vt[s, v] < tau)))
                fp = int(sum(1 for (s, v) in touched if ((s, v) not in self._adv_nodes) and (vt[s, v] < tau)))
                tpr = float(tp / max(tp + fn, 1))
                spc = float(tn / max(tn + fp, 1))
                # time-to-detect proxy: if any malicious undetected and under slow poisoning, increase
                ttd = float(self._ttd_norm)
                if is_sp and fn > 0:
                    ttd = min(1.0, ttd + 0.02)
                else:
                    ttd = max(0.0, ttd - 0.01)
                # EMA update
                self._tpr = 0.9 * self._tpr + 0.1 * tpr
                self._spc = 0.9 * self._spc + 0.1 * spc
                self._ttd_norm = ttd

            if self.logger:
                self.logger.log({
                    "type": "tx",
                    "time": float(now),
                    "src": src,
                    "dst": dst,
                    "latency": float(l),
                    "success": bool(success),
                })

        for tx in batch:
            self._queues[tx["src"]] = max(self._queues[tx["src"]] - 1, 0)
            self._queues[tx["dst"]] = max(self._queues[tx["dst"]] - 1, 0)

    def reassign_validator(self, src: int, dst: int, now: float):
        if src == dst:
            return
        # Delegate reassignment to trust layer (swap representative validators)
        try:
            self.trust.reassign_validator(src, dst)
        except Exception:
            # Fallback: no-op if trust layer cannot reassign
            pass
        # Reduce cross-shard cost slightly to reflect improved locality (proxy)
        self._cross_cost = max(0.0, self._cross_cost - 0.02)

    def sample_kpis(self, now: float):
        return float(self._throughput), float(self._latency_s), float(self._power_w), float(self._cross_cost)

    def sample_security_metrics(self, now: float):
        # Recompute global detection metrics at sampling time using all validators
        try:
            vt = self.trust.get_validator_trust()
            tau = float(self.cfg.trust.threshold_tau)
            pred_mal = vt < tau
            # Build adversary mask matrix
            adv_mask = np.zeros_like(vt, dtype=bool)
            if len(self._adv_nodes) > 0:
                for (s, v) in self._adv_nodes:
                    if 0 <= s < adv_mask.shape[0] and 0 <= v < adv_mask.shape[1]:
                        adv_mask[s, v] = True
            tp = int(np.logical_and(pred_mal, adv_mask).sum())
            fn = int(np.logical_and(~pred_mal, adv_mask).sum())
            tn = int(np.logical_and(~pred_mal, ~adv_mask).sum())
            fp = int(np.logical_and(pred_mal, ~adv_mask).sum())
            tpr = float(tp / max(tp + fn, 1))
            spc_now = float(tn / max(tn + fp, 1))
            # Smooth metrics
            self._tpr = 0.9 * self._tpr + 0.1 * tpr
            self._spc = 0.9 * self._spc + 0.1 * spc_now
            # Update time-to-detect proxy: increase if any malicious remains undetected under slow-poisoning
            if (self._slow_poisoning_enabled and now >= self._sp_warmup) and (fn > 0):
                self._ttd_norm = min(1.0, self._ttd_norm + 0.02)
            else:
                self._ttd_norm = max(0.0, self._ttd_norm - 0.01)
        except Exception:
            pass

        # Apply collusion pressure effect on specificity/time-to-detect if enabled
        if self._collusion_enabled and now >= self._collusion_start and len(self._adv_nodes) > 0:
            spc = max(0.0, min(1.0, self._spc - 0.05 * self._collusion_strength))
            ttd = min(1.0, self._ttd_norm + 0.05 * self._collusion_strength)
        else:
            spc = self._spc
            ttd = self._ttd_norm
        return float(self._tpr), float(spc), float(ttd)


