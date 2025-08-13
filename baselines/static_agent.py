from __future__ import annotations

import numpy as np

from qtrust.config import QTrustConfig


class StaticAgent:
    """Non-adaptive static baseline.

    Produces no-op control: keep tau unchanged and perform no reassignment.
    Used to emulate a static PBFT-style baseline with fixed configuration.
    """

    def __init__(self, obs_dim: int, action_dim: int, cfg: QTrustConfig):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.cfg = cfg

    def select_action(self, state: np.ndarray, safety_filter=None) -> np.ndarray:
        # delta_tau = 0, src = dst (no reassignment)
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        if safety_filter is not None:
            action = safety_filter(action)
        # force src == dst to ensure no reassignment
        action[1] = action[2]
        return action

    def observe(self, *args, **kwargs):
        return

    def learn(self):
        return


