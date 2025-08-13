from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _override_cfg(d: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    import copy

    cfg = copy.deepcopy(d)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def _run_experiment(cfg_dict: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    from qtrust.config import QTrustConfig
    from qtrust.runner import run_experiment

    qcfg = QTrustConfig.model_validate(cfg_dict)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(qcfg.model_dump(), f, indent=2)
    run_experiment(qcfg, output_dir=run_dir)
    # Load summary
    with (run_dir / "summary_metrics.json").open("r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary


def _ensure_dirs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def gen_fig2_scaling(base_cfg: Dict[str, Any], artifacts: Path, out: Path, shards_list: List[int]) -> None:
    import matplotlib.pyplot as plt

    base_cfg = _override_cfg(base_cfg, {"rl": {"algo": "rainbow"}})
    # Allow longer duration via env var or default modest duration for stability
    import os
    dur = float(os.environ.get("QTRUST_FIG_DURATION_HOURS", "0.1"))
    base_cfg = _override_cfg(base_cfg, {"simulation": {"duration_hours": dur}})

    systems = [
        ("QTrust", {"rl": {"algo": "rainbow"}}),
        ("Simple Adaptive", {"rl": {"algo": "equiflowshard"}}),
        ("Static PBFT", {"rl": {"algo": "static"}, "simulation": {"locality_optimization": False}}),
    ]

    rows: List[Tuple[str, int, float, float]] = []
    for name, upd in systems:
        for s in shards_list:
            cfg = _override_cfg(base_cfg, {"simulation": {"shards": s, "validators_per_shard": max(8, min(64, s * 8))}})
            cfg = _override_cfg(cfg, upd)
            run_dir = artifacts / f"fig2_scaling_{name.replace(' ', '_').lower()}_{s}sh"
            summary = _run_experiment(cfg, run_dir)
            thr = float(summary.get("throughput", {}).get("mean", 0.0))
            rows.append((name, s, thr, 0.0))

    # Compute scaling efficiency per system relative to its own min-shard throughput
    import collections
    by_sys: Dict[str, List[Tuple[int, float]]] = collections.defaultdict(list)
    for name, s, thr, _ in rows:
        by_sys[name].append((s, thr))
    for name in by_sys:
        by_sys[name].sort(key=lambda x: x[0])
    eff_rows: List[Tuple[str, int, float]] = []
    for name, arr in by_sys.items():
        base_s, base_thr = arr[0]
        for s, thr in arr:
            eff = (thr / max(1e-6, base_thr)) / (s / max(1, base_s)) * 100.0
            eff_rows.append((name, s, eff))

    # Save CSV
    csv_path = out / "fig2_scaling.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("system,shards,efficiency_percent\n")
        for name, s, eff in eff_rows:
            f.write(f"{name},{s},{eff:.2f}\n")

    # Plot
    # Plot each system
    import pandas as pd
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(6, 3))
    for name in df["system"].unique():
        sub = df[df["system"] == name].sort_values("shards")
        plt.plot(sub["shards"], sub["efficiency_percent"], marker="o", label=name)
    plt.xlabel("Number of Shards")
    plt.ylabel("Scaling Efficiency (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    fig_path = out / "fig2_scaling.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def gen_table_vi(base_cfg: Dict[str, Any], artifacts: Path, out: Path) -> None:
    # Systems: Rainbow, PPO, EquiFlowShard; Normal and Attack (slow poisoning + adversaries)
    systems = [
        ("QTrust (Ours)", {"rl": {"algo": "rainbow"}}),
        ("PPO-based RL Agent", {"rl": {"algo": "ppo"}}),
        ("EquiFlowShard", {"rl": {"algo": "equiflowshard"}}),
        ("Static PBFT", {"rl": {"algo": "static"}, "simulation": {"locality_optimization": False}}),
    ]

    def run_one(name: str, upd: Dict[str, Any], run_name: str) -> Dict[str, float]:
        cfg = _override_cfg(base_cfg, upd)
        # allow override of duration for stability
        import os
        dur = float(os.environ.get("QTRUST_TABLE_DURATION_HOURS", "0.1"))
        cfg = _override_cfg(cfg, {"simulation": {"duration_hours": dur}})
        run_dir = artifacts / run_name
        summary = _run_experiment(cfg, run_dir)
        # Include failure classification: throughput < thr_min or latency > lat_max
        thr_min = 1e-3
        lat_max = 10.0
        res = {
            "throughput": float(summary.get("throughput", {}).get("mean", 0.0)),
            "latency": float(summary.get("latency", {}).get("mean", 0.0)),
            "power": float(summary.get("power", {}).get("mean", 0.0)),
            "css": float(summary.get("css", {}).get("mean", 0.0)),
            "failure": (float(summary.get("throughput", {}).get("mean", 0.0)) <= thr_min) or (float(summary.get("latency", {}).get("mean", 0.0)) >= lat_max),
        }
        return res

    rows: List[Dict[str, Any]] = []
    for sys_name, upd in systems:
        normal_upd = _override_cfg(upd, {"simulation": {"adversary_ratio": 0.0}, "attacks": {"slow_poisoning": {"enabled": False}, "collusion_sybil": {"enabled": False}}})
        attack_upd = _override_cfg(upd, {"simulation": {"adversary_ratio": 0.25}, "attacks": {"slow_poisoning": {"enabled": True, "warmup_minutes": 0}, "collusion_sybil": {"enabled": False}}})
        normal = run_one(sys_name, normal_upd, f"table_vi_{sys_name}_normal".replace(" ", "_"))
        attack = run_one(sys_name, attack_upd, f"table_vi_{sys_name}_attack".replace(" ", "_"))
        degradation = 0.0
        if normal["throughput"] > 0:
            degradation = (normal["throughput"] - attack["throughput"]) / normal["throughput"] * 100.0
        rows.append({
            "system": sys_name,
            "thr_normal": normal["throughput"],
            "lat_normal": normal["latency"],
            "power_normal": normal["power"],
            "thr_attack": attack["throughput"],
            "degradation_percent": -degradation,
            "css_attack": attack["css"],
            "failure_attack": attack["failure"],
        })

    out_csv = out / "table_vi.csv"
    import csv

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["system", "thr_normal", "lat_normal", "power_normal", "thr_attack", "degradation_percent", "css_attack", "failure_attack"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def gen_table_viii(out: Path) -> None:
    # Communication cost O(k^2) for participants k with shards N ~ mapping
    data = [
        {"N": 4, "k": 2, "O(k^2)": 4},
        {"N": 8, "k": 3, "O(k^2)": 9},
        {"N": 16, "k": 4, "O(k^2)": 16},
        {"N": 32, "k": 5, "O(k^2)": 25},
        {"N": 64, "k": 6, "O(k^2)": 36},
    ]
    import csv

    with (out / "table_viii.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["N", "k", "O(k^2)"])
        w.writeheader()
        for row in data:
            w.writerow(row)


def gen_table_ix(base_cfg: Dict[str, Any], artifacts: Path, out: Path, shards_list: List[int]) -> None:
    # Cross-shard traffic (%) approximated from cross_ratio mean * 100
    rows: List[Dict[str, Any]] = []
    import os
    dur = float(os.environ.get("QTRUST_TABLE_DURATION_HOURS", "0.1"))
    base_cfg = _override_cfg(base_cfg, {"rl": {"algo": "rainbow"}, "simulation": {"duration_hours": dur}})
    for N in shards_list:
        cfg = _override_cfg(base_cfg, {"simulation": {"shards": N, "validators_per_shard": max(8, min(64, N * 8))}})
        run_dir = artifacts / f"table_ix_{N}sh"
        summary = _run_experiment(cfg, run_dir)
        cross_ratio = float(summary.get("cross_ratio", {}).get("mean", 0.0))
        rows.append({"N": N, "cross_shard_percent": 100.0 * cross_ratio})
    import csv

    with (out / "table_ix.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["N", "cross_shard_percent"])
        w.writeheader()
        for row in rows:
            w.writerow(row)


def gen_table_x_ablation(base_cfg: Dict[str, Any], artifacts: Path, out: Path) -> None:
    # Compare locality optimization on vs off under rainbow
    base = _override_cfg(base_cfg, {"rl": {"algo": "rainbow"}, "simulation": {"shards": 64, "validators_per_shard": 64, "duration_hours": 0.01}})
    loc_on = _override_cfg(base, {"simulation": {"locality_optimization": True}})
    loc_off = _override_cfg(base, {"simulation": {"locality_optimization": False}})

    def run(cfg: Dict[str, Any], name: str) -> Dict[str, float]:
        summary = _run_experiment(cfg, artifacts / name)
        return {
            "cross_ratio": float(summary.get("cross_ratio", {}).get("mean", 0.0)),
            "throughput": float(summary.get("throughput", {}).get("mean", 0.0)),
            "latency": float(summary.get("latency", {}).get("mean", 0.0)),
        }

    a = run(loc_off, "table_x_no_locality")
    b = run(loc_on, "table_x_full_system")

    import csv

    with (out / "table_x.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "no_locality", "full_system", "improvement_percent"])
        w.writeheader()
        # Cross-shard TX Volume proxy: cross_ratio
        imp_cross = (a["cross_ratio"] - b["cross_ratio"]) / max(1e-6, a["cross_ratio"]) * 100.0
        w.writerow({"metric": "cross_ratio", "no_locality": a["cross_ratio"], "full_system": b["cross_ratio"], "improvement_percent": imp_cross})
        # Throughput
        imp_thr = (b["throughput"] - a["throughput"]) / max(1e-6, a["throughput"]) * 100.0
        w.writerow({"metric": "throughput", "no_locality": a["throughput"], "full_system": b["throughput"], "improvement_percent": imp_thr})
        # Latency (lower better)
        imp_lat = (a["latency"] - b["latency"]) / max(1e-6, a["latency"]) * 100.0
        w.writerow({"metric": "latency", "no_locality": a["latency"], "full_system": b["latency"], "improvement_percent": imp_lat})


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper figures and tables")
    parser.add_argument("--asset", type=str, required=True, choices=["fig2", "table_vi", "table_viii", "table_ix", "table_x"])
    parser.add_argument("--base_config", type=str, default="configs/minimal.yaml")
    parser.add_argument("--artifacts", type=str, default="artifacts")
    parser.add_argument("--out", type=str, default="results/paper_assets")
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        base_cfg = json.load(f) if args.base_config.endswith(".json") else __import__("yaml").safe_load(f)

    artifacts = Path(args.artifacts)
    out = Path(args.out)
    _ensure_dirs(artifacts)
    _ensure_dirs(out)

    if args.asset == "fig2":
        # support up to 64 shards for paper-scale figure; can adjust via env var
        import os
        shards_list = os.environ.get("QTRUST_FIG_SHARDS", "4,8,16,32,64")
        shards = [int(s) for s in shards_list.split(",") if s.strip()]
        gen_fig2_scaling(base_cfg, artifacts, out, shards)
    elif args.asset == "table_vi":
        gen_table_vi(base_cfg, artifacts, out)
    elif args.asset == "table_viii":
        gen_table_viii(out)
    elif args.asset == "table_ix":
        import os
        shards_list = os.environ.get("QTRUST_TABLE_SHARDS", "4,8,16,32,64")
        shards = [int(s) for s in shards_list.split(",") if s.strip()]
        gen_table_ix(base_cfg, artifacts, out, shards)
    elif args.asset == "table_x":
        gen_table_x_ablation(base_cfg, artifacts, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


