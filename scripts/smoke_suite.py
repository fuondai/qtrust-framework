from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import copy
import yaml


def _ensure_dirs(artifacts: Path, results: Path) -> None:
    artifacts.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)


def _override_cfg(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    d = copy.deepcopy(base)
    # shallow merge for simplicity in smoke runs
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k].update(v)
        else:
            d[k] = v
    return d


def _run_one(cfg_dict: Dict[str, Any], run_name: str, artifacts: Path) -> Path:
    from qtrust.config import QTrustConfig
    from qtrust.runner import run_experiment

    # tighten runtime for smoke: override duration to ~10-15s
    cfg_local = copy.deepcopy(cfg_dict)
    cfg_local.setdefault("simulation", {})
    cfg_local["simulation"]["duration_hours"] = 0.004  # ~14.4 seconds

    qcfg = QTrustConfig.model_validate(cfg_local)

    run_dir = artifacts / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(qcfg.model_dump(), f, indent=2)

    run_experiment(qcfg, output_dir=run_dir)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a minimal smoke suite across algorithms and attacks")
    parser.add_argument("--base_config", type=str, default="configs/minimal.yaml")
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--results", type=str, default="results")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    results = Path(args.results)
    _ensure_dirs(artifacts, results)

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    scenarios = [
        ("smoke_rainbow_normal", {"experiment_name": "smoke_rainbow_normal", "rl": {"algo": "rainbow"}, "attacks": {"slow_poisoning": {"enabled": False}, "collusion_sybil": {"enabled": False}}}),
        ("smoke_ppo_normal", {"experiment_name": "smoke_ppo_normal", "rl": {"algo": "ppo"}, "attacks": {"slow_poisoning": {"enabled": False}, "collusion_sybil": {"enabled": False}}}),
        ("smoke_equiflowshard_normal", {"experiment_name": "smoke_equiflowshard_normal", "rl": {"algo": "equiflowshard"}, "attacks": {"slow_poisoning": {"enabled": False}, "collusion_sybil": {"enabled": False}}}),
        ("smoke_rainbow_attack", {"experiment_name": "smoke_rainbow_attack", "rl": {"algo": "rainbow"}, "simulation": {"adversary_ratio": 0.2}, "attacks": {"slow_poisoning": {"enabled": True, "warmup_minutes": 0, "drop_prob": 0.15}, "collusion_sybil": {"enabled": False}}}),
    ]

    run_dirs = []
    for name, upd in scenarios:
        cfg = _override_cfg(base_cfg, upd)
        rd = _run_one(cfg, name, artifacts)
        run_dirs.append(rd)

    # Aggregate summaries into results/metrics.json
    from qtrust.metrics.evaluation import compute_summary_metrics

    metrics_out = results / "metrics.json"
    all_summaries: Dict[str, Any] = {}
    for rd in run_dirs:
        summary = compute_summary_metrics(rd)
        if summary:
            all_summaries[rd.name] = summary
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Wrote {metrics_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


