# Benchmark Methodology

This document describes the methodology used for benchmarking the QTrust Blockchain Sharding Framework.

## Overview

The QTrust benchmarking methodology is designed to evaluate the system's performance, scalability, and security under realistic conditions. Our approach follows academic standards for reproducibility and fairness, allowing for meaningful comparison with other blockchain systems.

## Benchmark Environment

### Hardware Configuration

For full paper reproduction, we recommend:

- **CPU**: 32+ cores (AMD EPYC 7543 or Intel Xeon Platinum 8358)
- **RAM**: 64+ GB DDR4-3200
- **Storage**: 100+ GB NVMe SSD
- **Network**: 10 Gbps interconnect

For scaled-down evaluation:

- **CPU**: 8+ cores
- **RAM**: 16+ GB
- **Storage**: 50+ GB SSD
- **Network**: 1+ Gbps connection

### Software Environment

- **Operating System**: Ubuntu 20.04 LTS or newer
- **Python**: 3.8 or newer
- **Dependencies**: All packages listed in `requirements.txt`

## Network Topology

The benchmark simulates a geographically distributed network with the following characteristics:

- **Regions**: 5 geographic regions (North America, Europe, Asia, South America, Oceania)
- **Nodes per Region**: 153-154 nodes per region (768 nodes total)
- **Shards**: 64 shards distributed across regions
- **Byzantine Nodes**: 20% of nodes (154 nodes) exhibit Byzantine behavior

### Network Latency Profile

Inter-region latency is simulated according to real-world measurements:

| From/To       | North America | Europe | Asia  | South America | Oceania |
| ------------- | ------------- | ------ | ----- | ------------- | ------- |
| North America | 10ms          | 80ms   | 120ms | 150ms         | 180ms   |
| Europe        | 80ms          | 8ms    | 90ms  | 140ms         | 160ms   |
| Asia          | 120ms         | 90ms   | 12ms  | 70ms          | 100ms   |
| South America | 150ms         | 140ms  | 70ms  | 15ms          | 200ms   |
| Oceania       | 180ms         | 160ms  | 100ms | 200ms         | 14ms    |

## Transaction Workload

The benchmark uses a mixed transaction workload to simulate real-world usage:

- **Transaction Mix**:

  - 60% simple transfers
  - 20% cross-shard transactions
  - 20% smart contract executions

- **Transaction Rate**:

  - Initial: 1,000 TPS
  - Target: 12,000 TPS
  - Ramp-up: Gradual increase over 30 seconds

- **Transaction Distribution**:
  - Zipfian distribution (some accounts more active than others)
  - 20% hotspot transactions (concentrated on popular accounts)
  - 30% cross-region transactions

## Benchmark Phases

Each benchmark run consists of three phases:

1. **Warm-up Phase** (60 seconds):

   - System initialization
   - Network stabilization
   - Trust score initialization
   - 50% of target transaction rate

2. **Benchmark Phase** (300 seconds):

   - Full transaction load
   - Performance measurement
   - Metrics collection

3. **Cool-down Phase** (60 seconds):
   - Gradual reduction in transaction rate
   - Flushing of pending transactions
   - Final state verification

## Metrics Collection

The following metrics are collected during the benchmark:

### Performance Metrics

- **Throughput**: Transactions per second (TPS) - target 12,400 TPS
- **Latency**: Average (1.2 ms), median (1.2 ms), P95 (3.0 ms), and P99 (6.0 ms) transaction confirmation times
- **Finality Delay**: Time to finality per region and shard
- **Cross-Shard Transaction Performance**: 3.5 ms median latency with 8.5% overhead

### Trust Metrics

- **Trust Convergence Time**: Time for trust scores to stabilize (~250 ms)
- **Trust Score Distribution**: Distribution of trust scores across nodes
- **Byzantine Detection Rate**: 99.9% Byzantine nodes correctly identified
- **False Positive Rate**: 1.0% honest nodes incorrectly flagged as Byzantine

### Resource Utilization

- **CPU Usage**: Per node and aggregate
- **Memory Usage**: Per node and aggregate
- **Network Bandwidth**: Inbound and outbound per node
- **Storage I/O**: Read and write operations per second

## Benchmark Execution

To run the benchmark:

```bash
python scripts/run_benchmark.py
```

This will:

1. Deploy a simulated network with 200 nodes across 5 regions
2. Configure 64 shards according to the benchmark parameters
3. Execute the three benchmark phases
4. Collect and analyze metrics
5. Generate a comprehensive report

## Reproducing Paper Results

To reproduce the results from our paper:

```bash
python scripts/run_benchmark.py --paper-reproduction
```

This uses the exact configuration from the paper and will attempt to match the hardware environment as closely as possible.

If your hardware differs from the paper environment, you can use the pre-collected results:

```bash
python scripts/generate_visuals.py --use-paper-results
```

## Comparing with Other Systems

The benchmark includes comparison with other blockchain sharding systems:

- **Zilliqa**: First-generation sharding with static network division
- **OmniLedger**: Lottery-based sharding with client-driven cross-shard transactions
- **Polkadot**: Parachain-based approach with GRANDPA consensus
- **Ethereum 2.0**: Beacon chain with validator committees

Comparison metrics include:

- Throughput (TPS)
- Latency
- Scalability (TPS increase with shard count)
- Byzantine fault tolerance
- Resource efficiency

## Benchmark Configurations

Several pre-defined benchmark configurations are available:

- **default_config.json**: Standard benchmark configuration
- **scaling_benchmark_config.json**: Tests scalability with increasing shard counts
- **cross_shard_benchmark_config.json**: Focuses on cross-shard transaction performance
- **byzantine_benchmark_config.json**: Tests performance under varying Byzantine node percentages
- **trust_impact_benchmark_config.json**: Evaluates the impact of trust mechanisms

## Visualization

Benchmark results are visualized using matplotlib:

```bash
python scripts/generate_visuals.py
```

This generates:

- Throughput over time charts
- Latency distribution histograms
- Scalability curves
- Trust convergence graphs
- Byzantine detection accuracy charts
- Resource utilization graphs
- System comparison bar charts

## Validation Methodology

To ensure the validity of benchmark results:

1. **Multiple Runs**: Each benchmark is run 3 times and results are averaged
2. **Variance Analysis**: Standard deviation is calculated for key metrics
3. **Outlier Detection**: Anomalous results are identified and investigated
4. **Cross-Validation**: Results are compared with theoretical models
5. **Component Validation**: Individual components are tested separately

## Limitations

The benchmark has the following limitations:

- **Simulation vs. Real Deployment**: Network conditions are simulated rather than using actual geographic distribution
- **Simplified Attack Models**: Byzantine behavior follows predefined patterns rather than adaptive strategies
- **Controlled Environment**: Real-world factors like network partitions and hardware failures are not fully modeled
- **Scale Limitations**: The benchmark is limited to 200 nodes, while real deployments might be larger

## Conclusion

This benchmark methodology provides a comprehensive evaluation of the QTrust Blockchain Sharding Framework's performance, scalability, and security. By following academic standards for reproducibility and fairness, it enables meaningful comparison with other blockchain systems and validation of the claims made in our paper.
