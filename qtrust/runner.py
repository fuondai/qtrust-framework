from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np

from qtrust.config import QTrustConfig
from qtrust.simenv.env import QTrustSimEnv
from qtrust.metrics.logging import JsonLogger
from qtrust.metrics.evaluation import compute_summary_metrics


def run_experiment(cfg: QTrustConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Strengthen reproducibility via explicit seeding across libraries
    try:
        import random as _py_random  # noqa: F401
        _py_random.seed(cfg.seed)
    except Exception:
        pass
    try:
        np.random.seed(cfg.seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(cfg.seed)  # type: ignore[attr-defined]
    except Exception:
        pass

    logger = JsonLogger(output_dir / "events.jsonl")

    env = QTrustSimEnv(cfg, logger=logger)

    if cfg.rl.algo.lower() == "rainbow":
        from qtrust.rl.rainbow_agent import RainbowAgent
        agent = RainbowAgent(env.observation_space_dim, env.action_space_dim, cfg)
    elif cfg.rl.algo.lower() == "ppo":
        from baselines.ppo_agent import PPOAgent
        agent = PPOAgent(env.observation_space_dim, env.action_space_dim, cfg)
    elif cfg.rl.algo.lower() == "equiflowshard":
        from baselines.equiflowshard import EquiFlowShard
        agent = EquiFlowShard(env.observation_space_dim, env.action_space_dim, cfg)
    elif cfg.rl.algo.lower() == "static":
        from baselines.static_agent import StaticAgent
        agent = StaticAgent(env.observation_space_dim, env.action_space_dim, cfg)
    else:
        raise ValueError(f"Unknown RL algo: {cfg.rl.algo}")

    num_decisions = int((cfg.simulation.duration_hours * 3600) / cfg.rl.decision_interval_seconds)

    state = env.reset(seed=cfg.seed)
    # bind agent to FL/SMPC controller for full-implementation aggregation/canary
    try:
        env.bind_agent(agent)
    except Exception:
        pass
    for step_idx in range(num_decisions):
        action = agent.select_action(state, safety_filter=env.safety_filter)
        next_state, reward, info = env.step(action)
        agent.observe(state, action, reward, next_state, info)
        if (step_idx + 1) % cfg.rl.update_every == 0:
            agent.learn()
        state = next_state
        # opportunistically try canary promotion
        try:
            env.canary_try_promote()
        except Exception:
            pass

    env.close()

    # Aggregate & persist metrics
    summary = compute_summary_metrics(output_dir)
    with open(output_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        import json
        json.dump(summary, f, indent=2)


