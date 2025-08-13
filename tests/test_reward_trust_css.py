from __future__ import annotations

import json

from qtrust.config import QTrustConfig
from qtrust.metrics.reward import compute_reward
from qtrust.metrics.security import composite_security_score
from qtrust.trust.htdcm import HTDCM


def load_cfg():
    import yaml
    with open("configs/default.yaml", "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return QTrustConfig.model_validate(d)


def test_reward_equation():
    cfg = load_cfg()
    # Inputs are normalized per design; example normalized values
    r = compute_reward(cfg, T_norm=0.8, L_norm=0.2, P=0.1, C_cross=0.3)
    # Expected: 1*0.8 + 1*(0.8) -1*0.1 -1*0.3 = 1.2
    assert abs(r - 1.2) < 1e-6


def test_css_equation():
    cfg = load_cfg()
    css = composite_security_score(cfg, tpr=0.9, spc=0.95, ttd_norm=0.2)
    # 0.5*0.9 + 0.3*0.95 - 0.2*(0.2) = 0.45 + 0.285 - 0.04 = 0.695
    assert abs(css - 0.695) < 1e-6


def test_htdcm_aggregation_shape():
    cfg = load_cfg()
    h = HTDCM(cfg)
    tr = h.initialize(cfg.simulation.shards, cfg.simulation.validators_per_shard)
    assert tr.shape[0] == cfg.simulation.shards
    assert tr.min() > 0.0 and tr.max() <= 1.0
    # per-validator trust available and within bounds
    vt = h.get_validator_trust()
    assert vt.shape == (cfg.simulation.shards, cfg.simulation.validators_per_shard)
    assert vt.min() > 0.0 and vt.max() <= 1.0


def test_htdcm_non_compensatory_zero_dimension():
    cfg = load_cfg()
    h = HTDCM(cfg)
    h.initialize(cfg.simulation.shards, cfg.simulation.validators_per_shard)
    # Force one validator's one dimension to near zero and ensure aggregate plunges
    s, v = 0, 0
    # directly manipulate protected state for test
    h._vals[s, v, 0] = 1e-9  # type: ignore[attr-defined]
    vt = h.get_validator_trust()
    assert vt[s, v] < 0.2


def test_htdcm_validator_grace_mask():
    cfg = load_cfg()
    h = HTDCM(cfg)
    h.initialize(cfg.simulation.shards, cfg.simulation.validators_per_shard)
    now = 0.0
    tau = cfg.trust.threshold_tau
    mask0 = h.eligible_validator_mask(now=now, tau=tau, grace_period_s=cfg.trust.grace_period_seconds)
    assert mask0.shape == (cfg.simulation.shards, cfg.simulation.validators_per_shard)
    # Simulate drop below tau and check grace window keeps eligibility briefly
    vt = h.get_validator_trust()
    s, v = 0, 0
    h._ewma_agg[s, v] = tau * 0.5  # force below
    h._last_below_tau_at[s, v] = now  # type: ignore[attr-defined]
    mask1 = h.eligible_validator_mask(now=now + cfg.trust.grace_period_seconds / 2, tau=tau, grace_period_s=cfg.trust.grace_period_seconds)
    assert mask1[s, v] is True


