from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    shards: int = Field(..., gt=0)
    validators_per_shard: int = Field(..., gt=0)
    duration_hours: float = Field(..., gt=0.0)
    tx_arrival_tps: int = Field(..., gt=0)
    latency_log_normal_ms: dict
    wan_latency_ms: int = 150
    bandwidth_mbps: int = 100
    adversary_ratio: float = Field(0.0, ge=0.0, le=0.5)
    network_bandwidth_gbps: float = 1.0
    locality_optimization: bool = True
    # Base probability that a transaction remains intra-shard (diagonal of M_cross)
    base_local_tx_prob: float = Field(0.2, ge=0.0, le=0.95)


class AttacksConfig(BaseModel):
    class SlowPoisoning(BaseModel):
        enabled: bool = False
        warmup_minutes: int = 60
        delay_ms_mean: float = 40.0
        drop_prob: float = 0.02

    class CollusionSybil(BaseModel):
        enabled: bool = False
        start_minutes: int = 90
        manipulation_strength: float = 0.2

    slow_poisoning: SlowPoisoning = SlowPoisoning()
    collusion_sybil: CollusionSybil = CollusionSybil()


class RLConfig(BaseModel):
    algo: str = "rainbow"
    gamma: float = 0.99
    target_update_interval: int = 10000
    replay_capacity: int = 1_000_000
    batch_size: int = 32
    learning_rate: float = 6.25e-5
    n_step: int = 3
    prioritized_alpha: float = 0.5
    prioritized_beta_start: float = 0.4
    prioritized_beta_end: float = 1.0
    noisy_sigma0: float = 0.5
    Vmin: float = -10.0
    Vmax: float = 10.0
    atoms: int = 51
    update_every: int = 4
    cvar_alpha: float = 0.1
    decision_interval_seconds: int = 30


class RewardConfig(BaseModel):
    w_tau: float = 1.0
    w_lambda: float = 1.0
    w_p: float = 1.0
    w_c: float = 1.0
    # Normalization constants to keep inputs in [0,1] for Eq. (1)
    throughput_norm_max_tps: float = 5000.0
    latency_norm_max_s: float = 2.0


class TrustConfig(BaseModel):
    decay_lambda: float = 0.01
    threshold_tau: float = 0.7
    ewma_alpha: float = 0.2
    grace_period_seconds: int = 120


class CSSConfig(BaseModel):
    w_tpr: float = 0.5
    w_spc: float = 0.3
    w_ttd: float = 0.2


class ConsensusConfig(BaseModel):
    coordinator_pool_size: int = 5
    trust_gate_threshold: float = 0.75
    mad_rapid_timeouts_ms: int = 500
    # Adaptive consensus (minimal simulation): switch latency model by trust
    enable_adaptive: bool = True
    fastbft_trust_threshold: float = 0.8
    pbft_base_latency_ms: int = 30
    fastbft_base_latency_ms: int = 10


class FLSMPCConfig(BaseModel):
    round_interval_minutes: int = 15
    committee_threshold: float = 0.75
    sss_prime: int = 2_147_483_647
    sss_shares: int = 5
    sss_threshold: int = 3
    # Number of validators to include in the FL/SMPC committee when available
    committee_size: int = 21
    # Approximate model size in bytes used to estimate FL round data exchange
    approx_model_bytes: int = 2_621_440  # ~2.5MB
    # Optional DP noise injection for federated aggregation (per-parameter Gaussian noise)
    dp_enabled: bool = False
    dp_sigma: float = 0.0


class LoggingConfig(BaseModel):
    level: str = "INFO"
    out_dir: str = "artifacts"
    jsonl: bool = True


class QTrustConfig(BaseModel):
    experiment_name: str = "qtrust_main"
    seed: int = 1337
    simulation: SimulationConfig
    attacks: AttacksConfig = AttacksConfig()
    rl: RLConfig = RLConfig()
    reward: RewardConfig = RewardConfig()
    trust: TrustConfig = TrustConfig()
    css: CSSConfig = CSSConfig()
    consensus: ConsensusConfig = ConsensusConfig()
    fl_smpc: FLSMPCConfig = FLSMPCConfig()
    logging: LoggingConfig = LoggingConfig()


