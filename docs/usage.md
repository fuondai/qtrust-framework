# Usage Guide

This document provides detailed instructions for installing, configuring, and running the QTrust Blockchain Sharding Framework.

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Operating System**: Linux (recommended), macOS, or Windows
- **Hardware**:
  - Minimum: 4 CPU cores, 8 GB RAM, 10 GB storage
  - Recommended: 8+ CPU cores, 16+ GB RAM, 50+ GB SSD storage
  - Full paper reproduction: 32+ CPU cores, 64+ GB RAM, 100+ GB SSD storage

### Step 1: Clone the Repository

```bash
git clone https://github.com/qtrust/qtrust.git
cd qtrust
```

### Step 2: Set Up the Environment

For CPU-only setup (recommended for most users):

```bash
./scripts/setup_environment.sh
```

For GPU-accelerated setup (optional, requires CUDA-compatible GPU):

```bash
./scripts/setup_environment.sh --gpu
```

This script will:
- Install all required dependencies
- Create necessary directories
- Check system resources
- Configure the environment for optimal performance

### Step 3: Verify Installation

To verify that QTrust is installed correctly:

```bash
python scripts/run_demo.py --verify
```

This will run a quick verification test to ensure all components are functioning properly.

## Running QTrust

### Quick Demo

To run a simple demonstration with 4 nodes in a single region:

```bash
python scripts/run_demo.py
```

This will:
- Initialize a small network with 4 nodes
- Process a set of sample transactions
- Display performance metrics
- Generate visualization charts in the `demo_results` directory

### Full Benchmark

To run the complete benchmark suite:

```bash
python scripts/run_benchmark.py
```

Options:
- `--config CONFIG_FILE`: Specify a custom configuration file
- `--output-dir OUTPUT_DIR`: Specify a custom output directory
- `--nodes NODES`: Specify the number of nodes (default: 200)
- `--shards SHARDS`: Specify the number of shards (default: 64)
- `--regions REGIONS`: Specify the number of regions (default: 5)
- `--duration SECONDS`: Specify the benchmark duration in seconds (default: 300)

Example with custom parameters:

```bash
python scripts/run_benchmark.py --nodes 100 --shards 32 --regions 3 --duration 180
```

### Generating Visualizations

To generate visualization charts from benchmark results:

```bash
python scripts/generate_visuals.py
```

Options:
- `--results-dir RESULTS_DIR`: Specify the results directory
- `--output-dir OUTPUT_DIR`: Specify the output directory for charts
- `--format FORMAT`: Specify the output format (png, pdf, svg)

## Configuration

### Main Configuration Files

QTrust uses JSON configuration files located in the `configs` directory:

- `default_config.json`: Default configuration for general use
- `scaling_benchmark_config.json`: Configuration for scaling benchmarks
- `cross_shard_benchmark_config.json`: Configuration for cross-shard transaction benchmarks
- `trust_impact_benchmark_config.json`: Configuration for trust mechanism benchmarks

### Custom Configuration

To create a custom configuration:

1. Copy one of the existing configuration files:
   ```bash
   cp configs/default_config.json configs/my_custom_config.json
   ```

2. Edit the configuration file to adjust parameters:
   ```bash
   nano configs/my_custom_config.json
   ```

3. Run QTrust with your custom configuration:
   ```bash
   python scripts/run_benchmark.py --config configs/my_custom_config.json
   ```

### Key Configuration Parameters

#### Shard Cluster Configuration

```json
"shard_cluster": {
    "num_shards": 64,
    "num_clusters": 5,
    "shards_per_cluster": 12,
    "topology": "hierarchical",
    "coordinator_selection": "trust_based",
    "rebalance_interval": 300,
    "min_validators_per_shard": 3,
    "max_validators_per_shard": 12
}
```

#### Trust Manager Configuration

```json
"trust_manager": {
    "base_threshold": 0.7,
    "dynamic_threshold_enabled": true,
    "threshold_adjustment_rate": 0.05,
    "min_threshold": 0.5,
    "max_threshold": 0.95,
    "trust_decay_rate": 0.01,
    "trust_boost_rate": 0.05,
    "trust_dimensions": [
        "transaction_validation",
        "block_proposal",
        "response_time",
        "uptime",
        "resource_contribution"
    ]
}
```

#### Router Configuration

```json
"router": {
    "routing_algorithm": "optimized_mad_rapid",
    "path_selection_strategy": "multi_objective",
    "congestion_threshold": 0.75,
    "congestion_ratio": 0.4,
    "path_cache_enabled": true,
    "path_cache_size": 20000,
    "path_cache_ttl": 300
}
```

#### Rainbow DQN Configuration

```json
"rainbow_dqn": {
    "enabled": true,
    "use_mock": true,
    "learning_rate": 0.001,
    "discount_factor": 0.99,
    "replay_buffer_size": 10000,
    "batch_size": 64,
    "target_update_frequency": 1000,
    "exploration_rate": 0.1
}
```

## Advanced Usage

### Running in CPU-only Mode

QTrust can run in CPU-only mode by using mock implementations of GPU-dependent components:

```bash
python scripts/run_benchmark.py --cpu-only
```

This will use the `MockRainbowAgent` instead of the full `RainbowAgent` implementation.

### Distributed Deployment

For large-scale deployments across multiple machines:

1. Configure the network settings in `configs/geo_distributed_config.json`
2. Start the coordinator node:
   ```bash
   python scripts/run_distributed.py --coordinator
   ```
3. Start worker nodes on each machine:
   ```bash
   python scripts/run_distributed.py --worker --coordinator-ip IP_ADDRESS
   ```

### Custom Transaction Workloads

To generate custom transaction workloads:

```bash
python scripts/generate_workload.py --output workloads/custom_workload.json --tx-count 100000 --cross-shard-ratio 0.3 --contract-ratio 0.2
```

Then run the benchmark with the custom workload:

```bash
python scripts/run_benchmark.py --workload workloads/custom_workload.json
```

## Troubleshooting

### Common Issues

#### Import Errors

If you encounter import errors:

```
ModuleNotFoundError: No module named 'src.agents'
```

Ensure you're running from the root directory of the repository and that all dependencies are installed:

```bash
cd qtrust
./scripts/setup_environment.sh
```

#### Memory Errors

If you encounter memory errors:

```
MemoryError: Unable to allocate array with shape (X, Y)
```

Reduce the scale of the benchmark:

```bash
python scripts/run_benchmark.py --nodes 50 --shards 16
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

If you encounter issues not covered in this guide:

1. Check the [Troubleshooting Guide](troubleshooting.md) for more detailed solutions
2. Open an issue on GitHub with detailed information about your problem
3. Contact the authors at qtrust-authors@example.com

## Next Steps

- [Benchmark Methodology](benchmark.md): Learn about the benchmarking methodology
- [Architecture](architecture.md): Explore the system architecture
- [Developer Guide](developer_guide.md): Learn how to contribute to QTrust
- [Artifact Evaluation](artifact_evaluation.md): Instructions for artifact evaluation
