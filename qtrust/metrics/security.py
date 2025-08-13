from __future__ import annotations

from typing import Dict

from qtrust.config import QTrustConfig


def composite_security_score(cfg: QTrustConfig, tpr: float, spc: float, ttd_norm: float) -> float:
    """Eq. (5): CSS = w_tpr * TPR + w_spc * SPC - w_ttd * TTD_norm
    where ttd_norm in [0,1] is normalized time-to-detect.
    """
    w = cfg.css
    return w.w_tpr * tpr + w.w_spc * spc - w.w_ttd * ttd_norm


