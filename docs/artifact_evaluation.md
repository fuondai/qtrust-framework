# Artifact Evaluation Guide

This document provides detailed instructions for evaluating the QTrust Blockchain Sharding Framework artifact as part of an academic review process.

## Overview

The QTrust artifact is designed to support the claims made in our paper

This guide will help reviewers:

1. Set up the evaluation environment
2. Reproduce key results from the paper
3. Explore the system's capabilities
4. Validate the claims made in the paper

## Hardware Requirements

### Minimum Configuration (Demo & Basic Validation)

- 4+ CPU cores
- 8+ GB RAM
- 10+ GB free disk space

### Recommended Configuration (Full Reproduction)

- 32+ CPU cores
- 64+ GB RAM
- 100+ GB SSD storage
- 10 Gbps network (for multi-machine deployment)

### CPU-Only Fallback

All components can run in CPU-only mode, though performance will be reduced compared to GPU-accelerated execution.

## Quick Start Guide

### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/qtrust/qtrust.git
cd qtrust

# Set up the environment (CPU-only mode)
./scripts/setup_environment.sh
```

### Step 2: Verify Installation

```bash
# Run a quick verification test
python scripts/run_demo.py --verify
```

This should complete without errors and display basic system functionality.

### Step 3: Run Demo

```bash
# Run the small-scale demo
python scripts/run_demo.py
```

This will:

- Initialize a 4-node network
- Process sample transactions
- Display performance metrics
- Generate visualization charts

## Reproducing Key Results

### Throughput Evaluation

To reproduce the throughput results from Figure 5 in the paper:

```bash
# Run the throughput benchmark
python scripts/run_benchmark.py --config configs/throughput_benchmark_config.json
```

Expected outcome:

- Average throughput: ~12,400 TPS
- Peak throughput: ~13,120 TPS
- Results will be saved to `benchmark_results/throughput_benchmark/`

### Byzantine Fault Tolerance

To reproduce the Byzantine fault tolerance results from Figure 7 in the paper:

```bash
# Run the Byzantine fault tolerance benchmark
python scripts/run_benchmark.py --config configs/byzantine_benchmark_config.json
```

Expected outcome:

- Byzantine detection rate: ~99.9%
- False positive rate: ~1.0%
- Results will be saved to `benchmark_results/byzantine_benchmark/`

### Cross-Shard Transaction Performance

To reproduce the cross-shard transaction results from Figure 9 in the paper:

```bash
# Run the cross-shard transaction benchmark
python scripts/run_benchmark.py --config configs/cross_shard_benchmark_config.json
```

Expected outcome:

- Cross-shard transaction latency: ~3.5 ms
- Cross-shard overhead: ~8.5%
- Results will be saved to `benchmark_results/cross_shard_benchmark/`

## Using Pre-collected Results

If your hardware is insufficient for full reproduction, you can use the pre-collected results:

```bash
# Generate visualizations from pre-collected results
python scripts/generate_visuals.py --use-paper-results
```

This will generate the same charts as in the paper using our pre-collected data.

## Validating Specific Claims

### Claim 1: Throughput Scalability

To validate that throughput scales with the number of shards:

```bash
# Run the scaling benchmark
python scripts/run_benchmark.py --config configs/scaling_benchmark_config.json
```

This will run benchmarks with 4, 8, 16, 32, and 64 shards and show the throughput scaling.

### Claim 2: Trust Propagation Effectiveness

To validate the effectiveness of trust propagation:

```bash
# Run the trust impact benchmark
python scripts/run_benchmark.py --config configs/trust_impact_benchmark_config.json
```

This will compare performance with and without trust propagation enabled.

### Claim 3: Rainbow DQN Effectiveness

To validate the effectiveness of the Rainbow DQN agent:

```bash
# Run the agent comparison benchmark
python scripts/run_benchmark.py --config configs/agent_comparison_config.json
```

This will compare performance with the Rainbow DQN agent versus baseline routing algorithms.

## Exploring the Codebase

To understand the implementation details:

### Core Components

- **Trust Propagation**: `qtrust/trust/trust_propagation.py`
- **Rainbow DQN Agent**: `qtrust/agents/rainbow_agent.py`
- **MAD-RAPID Routing**: `qtrust/routing/mad_rapid.py`
- **Adaptive Consensus**: `qtrust/consensus/dynamic_consensus.py`

### Benchmark Implementation

- **Benchmark Runner**: `qtrust/benchmark/benchmark_runner.py`
- **Metrics Collection**: `qtrust/benchmark/metrics_collector.py`
- **Transaction Generator**: `qtrust/benchmark/transaction_generator.py`

## Customizing the Evaluation

You can customize the evaluation by modifying the configuration files:

```bash
# Copy a configuration file
cp configs/default_config.json configs/custom_config.json

# Edit the configuration
nano configs/custom_config.json

# Run with custom configuration
python scripts/run_benchmark.py --config configs/custom_config.json
```

Key parameters to experiment with:

- `num_shards`: Number of shards (4-64)
- `num_nodes`: Number of nodes (20-200)
- `byzantine_ratio`: Percentage of Byzantine nodes (0-30%)
- `cross_shard_ratio`: Percentage of cross-shard transactions (0-50%)

## Troubleshooting

### Common Issues

#### Memory Errors

If you encounter memory errors:

```
MemoryError: Unable to allocate array with shape (X, Y)
```

Try reducing the scale:

```bash
python scripts/run_benchmark.py --nodes 50 --shards 16
```

#### Import Errors

If you encounter import errors:

```
ModuleNotFoundError: No module named 'src.agents'
```

Ensure you're running from the root directory and all dependencies are installed:

```bash
cd qtrust
./scripts/setup_environment.sh
```

#### GPU-related Errors

If you encounter GPU-related errors:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Run in CPU-only mode:

```bash
python scripts/run_benchmark.py --cpu-only
```

### Getting Help

If you encounter issues not covered here:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Contact the artifact evaluation support team at ae-support@qtrust.org

## Expected Time Requirements

| Task                    | Estimated Time | Hardware    |
| ----------------------- | -------------- | ----------- |
| Environment setup       | 10-15 minutes  | Any         |
| Demo run                | 5 minutes      | Minimum     |
| Single benchmark        | 15-30 minutes  | Recommended |
| Full paper reproduction | 3-4 hours      | Recommended |
| Full exploration        | 1-2 days       | Recommended |

## Evaluation Checklist

- [ ] Environment setup completed successfully
- [ ] Demo runs without errors
- [ ] Throughput benchmark reproduces paper results
- [ ] Byzantine fault tolerance benchmark reproduces paper results
- [ ] Cross-shard transaction benchmark reproduces paper results
- [ ] Scaling behavior matches paper claims
- [ ] Trust propagation effectiveness validated
- [ ] Rainbow DQN effectiveness validated
- [ ] Code structure and quality assessed

## Conclusion

This artifact provides comprehensive support for the claims made in our paper. By following this guide, reviewers should be able to validate our results and explore the system's capabilities. We welcome feedback on both the artifact and the evaluation process.
