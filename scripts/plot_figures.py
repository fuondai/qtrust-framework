from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot key figures from results")
    parser.add_argument("artifacts", type=str)
    parser.add_argument("out", type=str)
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Simple plot: throughput over time for first run with events.jsonl
    for run in sorted(artifacts.glob("*")):
        ev = run / "events.jsonl"
        if not ev.exists():
            continue
        times, thr = [], []
        with ev.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if row.get("type") == "metrics":
                    times.append(float(row["time"]))
                    thr.append(float(row["throughput"]))
        if times:
            plt.figure(figsize=(6, 3))
            plt.plot(times, thr, label=run.name)
            plt.xlabel("time (s)")
            plt.ylabel("throughput")
            plt.legend()
            fig_path = out / f"throughput_{run.name}.png"
            plt.tight_layout()
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"Saved {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


