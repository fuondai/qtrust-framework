from __future__ import annotations

import subprocess
from pathlib import Path


def run_tlc(module: str, params: dict[str, int], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = out_dir / f"{module}.cfg"
    with cfg.open("w", encoding="utf-8") as f:
        for k, v in params.items():
            f.write(f"CONSTANT {k} = {v}\n")
    tla_path = Path(__file__).parent / f"{module}.tla"
    cmd = [
        "java", "-cp", "tlc2.jar", "tlc2.TLC",
        str(tla_path),
        "-deadlock",
        "-workers", "1",
        "-config", str(cfg),
    ]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("TLC not found; ensure tlc2.jar is on the classpath")
        return 1


if __name__ == "__main__":
    base = Path(__file__).parent / "out"
    run_tlc("QTrustHTDCM", {"Shards": 3, "ValidatorsPerShard": 4}, base)
    run_tlc("MADRapid", {"Shards": 3}, base)


