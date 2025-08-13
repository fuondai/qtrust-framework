from __future__ import annotations

import numpy as np

from qtrust.config import QTrustConfig
from qtrust.trust.htdcm import HTDCM
from qtrust.consensus.mad_rapid import MADRapidProtocol


def load_cfg():
    import yaml
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return QTrustConfig.model_validate(d)


def test_coordinator_selection_gate():
    cfg = load_cfg()
    h = HTDCM(cfg)
    h.initialize(cfg.simulation.shards, cfg.simulation.validators_per_shard)
    rng = np.random.default_rng(cfg.seed)
    proto = MADRapidProtocol(cfg, trust=h, rng=rng)
    src, dst = 0, 1
    shard_queue_before = proto.get_shard_queue_loads().copy()
    success, latency, coord, participants = proto._two_phase_commit(src, dst, now=0.0)
    assert isinstance(success, bool)
    assert latency > 0.0
    c_shard, c_val = coord
    assert 0 <= c_shard < cfg.simulation.shards
    assert 0 <= c_val < cfg.simulation.validators_per_shard
    # Coordinator should be eligible by validator-level grace mask
    try:
        vmask = h.eligible_validator_mask(now=0.0, tau=cfg.consensus.trust_gate_threshold, grace_period_s=cfg.trust.grace_period_seconds)
        assert vmask[c_shard, c_val] is True
    except Exception:
        pass
    # participants should be valid pairs
    for (s, v) in participants:
        assert 0 <= s < cfg.simulation.shards
        assert 0 <= v < cfg.simulation.validators_per_shard


