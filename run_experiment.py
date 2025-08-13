import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path
from typing import Optional

import yaml
import numpy as np


def sha_short() -> str:
    data = f"{time.time_ns()}-{os.getpid()}-{np.random.randint(0, 1_000_000)}".encode()
    return hashlib.sha256(data).hexdigest()[:8]


def main():
    parser = argparse.ArgumentParser(description="Run QTrust experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=str, default="artifacts")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = int(args.seed)

    from qtrust.config import QTrustConfig
    from qtrust.runner import run_experiment

    qcfg = QTrustConfig.model_validate(cfg)

    np.random.seed(qcfg.seed)
    try:
        import torch
        torch.manual_seed(qcfg.seed)
    except Exception:
        pass

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    run_id = f"{qcfg.experiment_name}-{sha_short()}"
    (output / run_id).mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")

    with open(output / run_id / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(qcfg.model_dump(), f, indent=2)

    run_experiment(qcfg, output_dir=output / run_id)


if __name__ == "__main__":
    main()


