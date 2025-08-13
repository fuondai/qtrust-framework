QTrust Reproduction (Code + Experiments)

This repository implements the QTrust framework and baselines end-to-end, with a SimPy-based simulator, RL agents (Rainbow DQN with CVaR risk-sensitive policy), trust layer (HTDCM), MAD-RAPID-style cross-shard protocol with trust-gated coordinator selection, federated learning with SSS-style secure aggregation, metrics, statistics, ablations, and plotting.

Quickstart:
1. Create a Python 3.10+ environment
2. pip install -r requirements.txt && pip install -r requirements-dev.txt
3. Smoke reproduce (Windows-friendly):
   - python run_experiment.py --config configs/minimal.yaml --output artifacts
   - python run_experiment.py --config configs/minimal_rainbow.yaml --output artifacts
   - python -m scripts.eval artifacts results
   - python -m scripts.plot_figures artifacts results/figures
   - python -m scripts.export_tables results results/tables
   - python scripts/check_claims.py --config configs/default.yaml --metrics results/metrics.json
   - make smoke  # one-shot minimal suite across algos/attacks
4. CI (recommended): add GitHub Actions workflow `.github/workflows/ci.yml` (pytest + ruff/black/isort + smoke + paper assets). Example below.
4. Full run:
   - python run_experiment.py --config configs/default.yaml --output artifacts

See docs/REPRODUCE.md for details.

Reproducibility:
- The experiment runner seeds Python, NumPy, and PyTorch (CPU/CUDA) to improve determinism; deterministic backends are enabled where supported.
- Generated outputs (`artifacts/`, `results/`) are ignored by VCS; paper assets are written to `results/paper_assets/`.

Note on versioning: A `.gitignore` is included to exclude run artifacts (`artifacts/`, `results/`), caches, local environments, IDE settings, and TLA+ outputs.

Components implemented end-to-end (with explicit proxy labels where applicable):
- Simulator (SimPy) with cross-shard traffic, WAN latency, bandwidth; per-tx two-phase commit.
- MAD-RAPID cross-shard protocol with trust-gated coordinator selection at shard and validator levels, and trust updates.
- HTDCM trust layer: six-dimensional per-validator trust vector; adaptive weighted geometric mean; temporal EWMA smoothing.
- RL agents: Rainbow DQN (distributional) and a functional minimal PPO baseline (discrete factorized policy) for parity and ablation.
- FL/SMPC: validator-level committee selection by aggregated validator trust (with per-validator grace), SSS-based secure aggregation of full model parameter vectors (shares/threshold/committee size configurable), canary validation and application.
- Metrics: reward (Eq. 1), composite security score (Eq. 5), summary with bootstrap CI, and plotting/tables. Power is reported alongside an explicit power_proxy (message-rate proxy). Cross-shard communication intensity is exposed as cross_cost and cross_cost_proxy for clarity.

Paper-scale assets:

- To generate Figure 2 up to 64 shards with longer duration, set environment variables:

  - QTRUST_FIG_SHARDS="4,8,16,32,64"
 - QTRUST_FIG_DURATION_HOURS=0.5  # increase for tighter confidence intervals and clearer trends

- For Tables VI/IX duration:

  - QTRUST_TABLE_SHARDS="4,8,16,32,64"
 - QTRUST_TABLE_DURATION_HOURS=0.5  # increase for attack scenarios to show CSS/TPR changes

Federated learning DP (optional): set in configs under fl_smpc:

- dp_enabled: true
- dp_sigma: 0.01  # Gaussian noise std per-parameter (post-aggregation)
- CSS follows Eq. (5) exactly: `w_tpr*TPR + w_spc*SPC - w_ttd*TTD_norm`.
- Coordinator selection applies a shard-level grace window based on `trust.grace_period_seconds`.
- Secure aggregation averages over participants to avoid magnitude blow-up; you may emulate local heterogeneity via `fl_smpc.local_perturb_sigma` (default 0.0) in configs for robustness tests.
- Reproduction scripts and checks that verify key configurations and presence/ranges of metrics; unit tests covering equations and protocol hooks.

Documentation (and reproducibility/proxies):
- Power (W) is an analytic proxy derived from message counts and latency factors in the two-phase commit; it is not a hardware power measurement. If you need calibrated power, integrate hardware counters/telemetry and adjust. We also report `power_proxy` explicitly.
- Cross-shard communication cost is a proxy (`cross_cost_proxy`) designed to correlate with communication intensity; for absolute comparisons across systems, calibrate against a network model or real deployments. `cross_ratio` in summaries is derived directly from `cross_cost_proxy`.
- FL/SMPC measurements currently estimate per-round data volume from configured `approx_model_bytes` and shares/threshold. End-to-end network latency/bandwidth effects are not simulated; numbers in the paper (e.g., ~55 MB / <450 ms) are scenario-dependent and should be validated independently in a distributed setting.
 - Smoke artifacts in this repo are minimal-duration runs; for paper-scale results (e.g., 64 shards, long duration), re-run with `configs/default.yaml` or set `QTRUST_*` env vars in scripts to increase duration. Security metrics (TPR/SPC/TTD/CSS) are more informative under longer runs and stronger adversary settings.

Independent validation suggestions:
- Add a network emulator (e.g., ns-3/Mininet) or extend the simulator with explicit link models to validate communication overheads.
- For power, add a secondary metric from a learned regression calibrated on real hardware traces to sanity-check the proxy.


Formal specification:
- See tla/ for TLA+ specs and `tla/run_tlc.py` to run TLC model checker on HTDCM/MAD-RAPID invariants.
 - Note: HTDCM TLA+ is an abstract model enforcing safety invariants (eligibility, EWMA bounds) rather than exact numeric weighted geometric mean. This matches the paper’s stated scope (protocol-level safety, not semantic content of inputs).

Tests:
- All unit/integration tests under tests/ (pytest)
- Smoke suite orchestrated via scripts/smoke_suite.py

GitHub Actions CI example (save as `.github/workflows/ci.yml`):

```yaml
name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install deps
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Lint
        run: |
          ruff check .
          black --check .
          isort --check-only .
      - name: Tests
        run: pytest -q
      - name: Smoke + assets + checks
        run: |
          python -m scripts.smoke_suite --base_config configs/minimal.yaml --artifacts artifacts --results results
          python scripts/check_claims.py --config configs/default.yaml --metrics results/metrics.json
          python -m scripts.paper_assets --asset fig2 --base_config configs/minimal.yaml --artifacts artifacts --out results/paper_assets
          python -m scripts.paper_assets --asset table_vi --base_config configs/minimal.yaml --artifacts artifacts --out results/paper_assets
```

License:
- MIT (see `LICENSE`).

