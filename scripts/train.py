from __future__ import annotations

import argparse
from pathlib import Path

from qtrust.config import QTrustConfig
from qtrust.runner import run_experiment
import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="Train/run a QTrust experiment")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="artifacts")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    qcfg = QTrustConfig.model_validate(cfg)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    run_experiment(qcfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


