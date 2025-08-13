from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
try:
    from scipy.stats import shapiro
except Exception:  # pragma: no cover
    shapiro = None  # type: ignore

from qtrust.metrics.security import composite_security_score
from qtrust.config import QTrustConfig


def _load_events(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return [r for r in rows if isinstance(r, dict) and r.get("type") == "metrics"]


def _bootstrap_ci(values: np.ndarray, iters: int = 1000, alpha: float = 0.05):
    rng = np.random.default_rng(1337)
    n = len(values)
    stats = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        stats.append(np.mean(values[idx]))
    lo = np.percentile(stats, 100 * (alpha / 2))
    hi = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def compute_summary_metrics(run_dir: Path) -> Dict[str, Any]:
    events_path = run_dir / "events.jsonl"
    rows = _load_events(events_path)
    if not rows:
        return {}

    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in [
        "throughput", "latency", "power", "power_proxy", "cross_cost", "tpr", "spc", "ttd_norm"
    ]}
    # Derived metrics for paper alignment (minimal): cross-shard ratio estimate
    # Approximated as EMA mean of cross_cost in [0,1]
    arr["cross_ratio"] = arr["cross_cost"]
    # Provide explicit proxy labels for clarity in paper/code mapping
    arr["power_proxy_labeled"] = arr["power_proxy"]
    arr["cross_cost_proxy"] = arr["cross_cost"]

    # CSS per record then aggregate
    # Load config used
    with open(run_dir / "resolved_config.json", "r", encoding="utf-8") as f:
        cfg = QTrustConfig.model_validate(json.load(f))

    css_vec = np.array([composite_security_score(cfg, tpr, spc, ttd)
                        for tpr, spc, ttd in zip(arr["tpr"], arr["spc"], arr["ttd_norm"])])
    # guard: if no adversaries (TPR all zeros, SPC all ones), report CSS as 1 - w_ttd rather than 0.1 for clearer baseline
    if np.all(arr["tpr"] == 0.0) and np.all(arr["spc"] == 1.0):
        css_vec = np.full_like(css_vec, cfg.css.w_spc - cfg.css.w_ttd * float(np.mean(arr["ttd_norm"])) + cfg.css.w_tpr * 0.0)

    summary = {}
    for k in arr:
        lo, hi = _bootstrap_ci(arr[k])
        if shapiro is not None and len(arr[k]) >= 3:
            normality_p = shapiro(arr[k]).pvalue
        else:
            normality_p = np.nan
        summary[k] = {
            "mean": float(arr[k].mean()),
            "std": float(arr[k].std(ddof=1)) if len(arr[k]) > 1 else 0.0,
            "ci95": [lo, hi],
            "normality_p": float(normality_p),
        }

    lo, hi = _bootstrap_ci(css_vec)
    summary["css"] = {"mean": float(css_vec.mean()), "std": float(css_vec.std(ddof=1)), "ci95": [lo, hi]}

    return summary


