from __future__ import annotations

import random

from fl_smpc.sss import split_secret, reconstruct_secret, split_vector, reconstruct_vector
from qtrust.config import QTrustConfig
from qtrust.simenv.env import QTrustSimEnv


def test_sss_scalar_roundtrip():
    prime = 2_147_483_647
    secret = 123456789
    shares = split_secret(secret, n_shares=5, threshold=3, prime=prime)
    # pick any 3 shares
    chosen = random.sample(shares, 3)
    rec = reconstruct_secret(chosen, prime)
    assert rec == secret


def test_sss_vector_roundtrip():
    prime = 2_147_483_647
    vec = [5, 42, 1234, 999_999, 7]
    shares_vec = split_vector(vec, n_shares=5, threshold=3, prime=prime)
    # choose any 3 indices
    idx = [0, 1, 2]
    chosen = [[comp[i] for i in idx] for comp in shares_vec]
    rec_vec = reconstruct_vector(chosen, prime)
    assert rec_vec == vec


def test_flsmpc_canary_roundtrip_integration():
    # Minimal integration: run a short env with rainbow agent bound so FL controller can extract vectors
    import yaml
    with open("configs/minimal_rainbow.yaml", "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    cfg = QTrustConfig.model_validate(d)
    # shorten FL interval for test
    cfg.fl_smpc.round_interval_minutes = 1
    env = QTrustSimEnv(cfg)
    # bind a dummy agent exposing ModelAdapter-compatible interface by running runner reset path
    state = env.reset(seed=cfg.seed)
    # create a minimal agent with same API as Rainbow (no learning needed)
    class Dummy:
        def select_action(self, s, safety_filter=None):
            import numpy as np
            a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            return safety_filter(a) if safety_filter is not None else a
        def observe(self, *args, **kwargs):
            return
        def learn(self):
            return
    agent = Dummy()
    env.bind_agent(agent)
    # step a few times to allow FL controller to produce a candidate (controller runs in background process)
    obs = state
    for _ in range(5):
        obs, r, info = env.step(agent.select_action(obs, safety_filter=env.safety_filter))
    # Try canary promotion (no-op if no candidate yet, but should not error)
    env.canary_try_promote()
    env.close()


