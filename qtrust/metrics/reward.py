from __future__ import annotations

from qtrust.config import QTrustConfig


def compute_reward(cfg: QTrustConfig, T_norm: float, L_norm: float, P: float, C_cross: float) -> float:
    """Eq. (1): R_t = w_tau T_norm + w_lambda (1 - L_norm) - w_p P(s_t,a_t) - w_c C_cross.

    Inputs are assumed normalized to [0,1].
    """
    w = cfg.reward
    return (
        w.w_tau * T_norm
        + w.w_lambda * (1.0 - L_norm)
        - w.w_p * P
        - w.w_c * C_cross
    )


