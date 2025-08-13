Quick Reproduction Guide

Prerequisites: Python 3.10+; optional GPU for Rainbow DQN.

- Install runtime and dev requirements using pip.
- Run the smoke reproduction on minimal configs.
- Aggregate results and optionally run TLA+ model checks on HTDCM/MAD-RAPID safety properties.

Commands (run from repo root):
- pip install -r requirements.txt
- pip install -r requirements-dev.txt
- make smoke
- python scripts/check_claims.py --config configs/default.yaml --metrics results/metrics.json

TLA+ verification:
- Install TLA+ Tools (TLC) and ensure `tlc2.jar` is available.
- cd tla && python run_tlc.py
- Reports and counterexamples (if any) are saved under tla/out/


