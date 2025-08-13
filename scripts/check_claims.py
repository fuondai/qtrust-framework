from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    # Key claims we can automatically check against configs and results
    # Example: ensure minimal configs exist and CSS weights match paper
    "css_weights": {"w_tpr": 0.5, "w_spc": 0.3, "w_ttd": 0.2},
    # Minimal smoke runs must exist (accept either historical names or new smoke_* runs)
    "required_runs_prefix": [
        "qtrust_minimal",
        "qtrust_minimal_rainbow",
        "smoke_rainbow_normal",
        "smoke_ppo_normal",
        "smoke_equiflowshard_normal",
        "smoke_rainbow_attack",
    ],
    # Validate presence of key metrics with reasonable ranges in smoke runs
    "metric_ranges": {
        "throughput": (0.0, 1e6),
        "latency": (0.0, 10.0),
        "css": (-1.0, 1.0),
        "power": (0.0, 1e6),
        "power_proxy": (0.0, 1e6),
    },
    # Expected trends for smoke runs (weak checks; paper-scale runs should use longer durations):
    # - CSS under attack is lower than normal (CSS_normal > CSS_attack)
    # - Throughput degrades and latency increases under attack
    # - TPR increases and TTD_norm increases under attack
    "trend_pairs": [
        ("smoke_rainbow_normal", "smoke_rainbow_attack", "css", ">"),
        ("smoke_rainbow_normal", "smoke_rainbow_attack", "throughput", ">"),
        ("smoke_rainbow_normal", "smoke_rainbow_attack", "latency", "<"),
        # Additional security trend checks
        ("smoke_rainbow_attack", "smoke_rainbow_normal", "tpr", ">"),
        ("smoke_rainbow_attack", "smoke_rainbow_normal", "ttd_norm", ">"),
    ],
}


def check_config(cfg: dict) -> list[str]:
    errs = []
    css = cfg.get("css", {})
    for k, v in EXPECTED["css_weights"].items():
        if abs(float(css.get(k, -1)) - float(v)) > 1e-9:
            errs.append(f"CSS weight {k} mismatch: got {css.get(k)}, expected {v}")
    # FL/SMPC sanity
    fl = cfg.get("fl_smpc", {})
    if fl:
        shares = int(fl.get("sss_shares", 0))
        thr = int(fl.get("sss_threshold", 0))
        csize = int(fl.get("committee_size", 0))
        mbytes = int(fl.get("approx_model_bytes", 0))
        if not (shares >= 2 and thr >= 2 and shares >= thr):
            errs.append(f"FL/SMPC shares/threshold invalid: shares={shares}, threshold={thr}")
        if not (csize >= thr):
            errs.append(f"FL/SMPC committee_size ({csize}) must be >= threshold ({thr})")
        if not (mbytes > 0):
            errs.append("FL/SMPC approx_model_bytes must be positive")
        # Optional heterogeneity flag bounds
        if "local_perturb_sigma" in fl:
            try:
                sig = float(fl.get("local_perturb_sigma", 0.0))
                if sig < 0.0 or sig > 1.0:
                    errs.append("FL/SMPC local_perturb_sigma must be in [0,1]")
            except Exception:
                errs.append("FL/SMPC local_perturb_sigma must be numeric")
    # Required artifacts presence (TLA specs)
    from pathlib import Path
    tla_dir = Path("tla")
    if not (tla_dir / "MADRapid.tla").exists() or not (tla_dir / "QTrustHTDCM.tla").exists():
        errs.append("TLA+ specs missing under tla/")
    return errs


def check_results(metrics_json: Path) -> list[str]:
    errs = []
    if not metrics_json.exists():
        errs.append("metrics.json missing")
        return errs
    with metrics_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        errs.append("no run summaries found in metrics.json")
    # Ensure minimal runs are present
    present_prefixes = {p for p in EXPECTED["required_runs_prefix"] if any(run_id.startswith(p) for run_id in data.keys())}
    # only emit a single warning if none of the expected prefixes are present
    if not present_prefixes:
        errs.append("missing required smoke runs; expected at least one of: " + ", ".join(EXPECTED["required_runs_prefix"]))

    # Check key metrics exist and are within reasonable ranges
    for run_id, summary in data.items():
        for key, (lo, hi) in EXPECTED["metric_ranges"].items():
            stats = summary.get(key, {})
            mean = stats.get("mean")
            if mean is None:
                errs.append(f"{run_id}: missing {key} in summary")
                continue
            try:
                mv = float(mean)
            except Exception:
                errs.append(f"{run_id}: {key}.mean not numeric")
                continue
            if not (lo <= mv <= hi):
                errs.append(f"{run_id}: {key}.mean out of range [{lo},{hi}]: {mv}")
        # additional sanity: power should be non-negative
        pstats = summary.get("power", {})
        if isinstance(pstats, dict) and isinstance(pstats.get("mean"), (int, float)):
            if pstats["mean"] < 0:
                errs.append(f"{run_id}: power.mean must be non-negative")
    # Trend checks (best-effort; ignore if runs missing)
    runs = data.keys()
    def mean_of(run: str, key: str) -> float | None:
        s = data.get(run, {}).get(key, {})
        return s.get("mean") if isinstance(s, dict) else None
    eps = 1e-6
    for a, b, key, rel in EXPECTED.get("trend_pairs", []):
        if a in runs and b in runs:
            ma = mean_of(a, key)
            mb = mean_of(b, key)
            if isinstance(ma, (int, float)) and isinstance(mb, (int, float)):
                if rel == ">" and not (ma > mb + eps):
                    errs.append(f"trend failed: {a}.{key} ({ma}) not >= {b}.{key} ({mb})")
                if rel == "<" and not (ma < mb - eps):
                    errs.append(f"trend failed: {a}.{key} ({ma}) not <= {b}.{key} ({mb})")
    return errs


def main() -> int:
    parser = argparse.ArgumentParser(description="Check code and result claims against expectations")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--metrics", type=str, default="results/metrics.json")
    args = parser.parse_args()

    import yaml

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    errs = []
    errs += check_config(cfg)
    errs += check_results(Path(args.metrics))

    if errs:
        for e in errs:
            print(f"FAIL: {e}")
        return 1
    print("PASS: all checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


