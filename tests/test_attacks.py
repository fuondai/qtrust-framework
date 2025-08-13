from __future__ import annotations

import numpy as np

from qtrust.config import QTrustConfig
from qtrust.simenv.env import QTrustSimEnv


def load_cfg():
    import yaml

    with open("configs/minimal.yaml", "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return QTrustConfig.model_validate(d)


def _run_steps(env: QTrustSimEnv, steps: int = 5):
    state = env.reset(seed=env.cfg.seed)
    class Dummy:
        def select_action(self, s, safety_filter=None):
            a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return safety_filter(a) if safety_filter is not None else a
        def observe(self, *args, **kwargs):
            return
        def learn(self):
            return
    agent = Dummy()
    env.bind_agent(agent)
    obs = state
    for _ in range(steps):
        action = agent.select_action(obs, safety_filter=env.safety_filter)
        obs, r, info = env.step(action)
    env.close()
    return env.metrics_history


def test_slow_poisoning_effects():
    cfg = load_cfg()
    cfg.attacks.slow_poisoning.enabled = True
    cfg.attacks.slow_poisoning.warmup_minutes = 0
    # Make attack harsher for smoke to ensure detector reacts
    cfg.attacks.slow_poisoning.drop_prob = 0.15
    env = QTrustSimEnv(cfg)
    hist = _run_steps(env, steps=10)
    # Expect ttd_norm to increase under slow poisoning
    ttd_vals = [h["ttd_norm"] for h in hist if "ttd_norm" in h]
    assert len(ttd_vals) > 0
    assert ttd_vals[-1] >= ttd_vals[0]


def test_collusion_sybil_effects():
    cfg = load_cfg()
    cfg.attacks.collusion_sybil.enabled = True
    cfg.attacks.collusion_sybil.start_minutes = 0
    cfg.simulation.adversary_ratio = 0.2
    # Lower detection threshold slightly for smoke tests to observe CSS change
    cfg.trust.threshold_tau = 0.65
    env = QTrustSimEnv(cfg)
    hist = _run_steps(env, steps=10)
    # Expect specificity to degrade under collusion pressure
    spc_vals = [h["spc"] for h in hist if "spc" in h]
    assert len(spc_vals) > 0
    assert spc_vals[-1] <= spc_vals[0]


def test_mad_rapid_timeouts_and_css_dynamics():
    cfg = load_cfg()
    # Enable slow poisoning from start and shrink MAD-RAPID timeout to trigger drops
    cfg.attacks.slow_poisoning.enabled = True
    cfg.attacks.slow_poisoning.warmup_minutes = 0
    cfg.attacks.slow_poisoning.drop_prob = 0.25
    cfg.consensus.mad_rapid_timeouts_ms = 50
    cfg.trust.threshold_tau = 0.7
    # Introduce adversaries to make TPR/SPC meaningful
    cfg.simulation.adversary_ratio = 0.2
    env = QTrustSimEnv(cfg)
    hist = _run_steps(env, steps=20)
    # Expect some latencies capped around timeout and CSS to reflect dynamics (not fixed value)
    lats = [h["latency"] for h in hist if "latency" in h]
    assert len(lats) > 0
    assert max(lats) >= cfg.consensus.mad_rapid_timeouts_ms / 1000.0 * 0.9
    # TPR should react over time (EMA), not remain constant zeros
    tprs = [h.get("tpr", 0.0) for h in hist]
    assert any(abs(tprs[i] - tprs[i-1]) > 1e-9 for i in range(1, len(tprs)))
    # CSS computed in summary; here we just ensure SPC not always 1 under attack
    spcs = [h.get("spc", 1.0) for h in hist]
    assert min(spcs) < 1.0


