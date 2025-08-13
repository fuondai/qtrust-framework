from __future__ import annotations

import argparse
import json
from pathlib import Path

from qtrust.metrics.evaluation import compute_summary_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate run directories to results")
    parser.add_argument("artifacts", type=str, help="Artifacts directory containing runs")
    parser.add_argument("out", type=str, help="Output directory for results")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    metrics_out = out / "metrics.json"
    all_summaries = {}
    for run_dir in sorted(artifacts.glob("*")):
        if not run_dir.is_dir():
            continue
        summary = compute_summary_metrics(run_dir)
        if summary:
            all_summaries[run_dir.name] = summary
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Wrote {metrics_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


