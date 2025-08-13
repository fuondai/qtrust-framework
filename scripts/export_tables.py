from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Export tables from results into CSVs")
    parser.add_argument("results", type=str)
    parser.add_argument("out", type=str)
    args = parser.parse_args()

    results_dir = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics_json = results_dir / "metrics.json"
    if not metrics_json.exists():
        print("metrics.json not found; run eval first")
        return 1

    with metrics_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for run_id, summary in data.items():
        row = {"run": run_id}
        for k, stats in summary.items():
            if isinstance(stats, dict) and "mean" in stats:
                row[f"{k}_mean"] = stats["mean"]
                row[f"{k}_std"] = stats.get("std", 0.0)
        # convenience: if both power and power_proxy exist, keep both
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = out / "summary_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


