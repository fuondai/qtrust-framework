# QTrust Performance Testing Methodology

This document outlines the rigorous methodology used to evaluate QTrust's performance across various configurations and scenarios.

## Testing Environment

### Hardware Specifications

All benchmark tests were conducted on the following hardware configurations:

**Simulation Environment:**

- CPU: AMD EPYC 7763 (64-core)
- Memory: 512 GB DDR4-3200
- Storage: 2 TB NVMe SSD
- Network: Simulated with realistic latency models

**Controlled Testnet:**

- Instance type: AWS EC2 c5.4xlarge
- vCPUs: 16
- Memory: 32 GB
- Network performance: Up to 10 Gbps
- Regions: us-east-1, us-west-2, eu-central-1

**Full-Scale Deployment:**

- Instance type: AWS EC2 c5n.9xlarge (validators) and c5.2xlarge (nodes)
- vCPUs: 36 (validators), 8 (nodes)
- Memory: 96 GB (validators), 16 GB (nodes)
- Network performance: 25 Gbps (validators), 10 Gbps (nodes)
- Regions: us-east-1, us-west-2, eu-central-1, ap-southeast-1, sa-east-1, ap-northeast-1

### Software Configuration

- Operating System: Ubuntu 20.04 LTS
- Python: 3.10.0
- QTrust Framework: Version 3.0.0
- Network Stack: Custom implementation based on gRPC
- Database: RocksDB 6.27.3

## Benchmark Methodology

### Test Configurations

We tested QTrust across three different configurations:

1. **Small-Scale:**

   - 16 shards with 192 nodes
   - 12 validators per shard
   - 3 geographic regions
   - 30% cross-shard transactions

2. **Medium-Scale:**

   - 32 shards with 384 nodes
   - 12 validators per shard
   - 4 geographic regions
   - 25% cross-shard transactions

3. **Large-Scale:**
   - 64 shards with 768 nodes
   - 12 validators per shard
   - 6 geographic regions
   - 20% cross-shard transactions

### Testing Protocol

Each benchmark was conducted following this procedure:

1. **Environment Setup:**

   - Deployment of nodes according to configuration
   - Network topology configuration with realistic latency simulation
   - Initialization of all nodes and shards

2. **Warmup Phase:**

   - 2-minute warmup period to stabilize the network
   - Gradually increasing transaction load

3. **Measurement Phase:**

   - 10-minute full-load test phase
   - Constant transaction generation rate
   - Metrics collection at 0.5-second intervals

4. **Cooldown Phase:**

   - 1-minute cooldown period
   - Gradual reduction of transaction load
   - Final state verification

5. **Repetition:**
   - Each test was repeated 10 times
   - Results were averaged to ensure statistical significance
   - Standard deviation was calculated and verified to be within 5% of the mean

### Transaction Types

The benchmark included a mix of transaction types:

- **Simple Transfers:** 70% - Basic value transfers within a single shard
- **Cross-Shard Transfers:** 20% - Value transfers across different shards
- **Smart Contract Calls:** 10% - Simple and complex contract interactions

### Byzantine Fault Testing

To evaluate Byzantine fault tolerance:

1. **Controlled Byzantine Nodes:**

   - Random selection of nodes to exhibit Byzantine behavior
   - Various byzantion behavior patterns including:
     - Equivocation (double-signing)
     - Transaction withholding
     - Invalid block proposals
     - Selective message delivery

2. **Metrics Collected:**
   - Detection rate (percentage of Byzantine actions identified)
   - False positive rate (legitimate nodes incorrectly flagged)
   - Recovery time (time to restore normal operation)
   - Network overhead of detection mechanisms

## Measurement Tools

### Custom Instrumentation

We developed specialized tools for accurate measurement:

1. **QTrust Performance Monitor:**

   - Distributed monitoring agents on each node
   - Nanosecond precision timestamps
   - Low-overhead operations (<0.1% CPU impact)

2. **Network Analyzer:**

   - Packet-level monitoring
   - Bandwidth utilization tracking
   - Latency measurements between all node pairs

3. **Transaction Generator:**
   - Configurable transaction mix and rate
   - Deterministic transaction sequences for reproducibility
   - Real-time adjustment capabilities

### Metrics Calculation

Key performance metrics were calculated as follows:

1. **Transactions Per Second (TPS):**

   ```
   TPS = Total Successfully Processed Transactions / Test Duration (seconds)
   ```

2. **Latency:**

   ```
   Transaction Latency = Confirmation Time - Submission Time
   ```

   - Percentiles (p50, p95, p99) calculated across all transactions

3. **Cross-Shard Performance:**

   ```
   Cross-Shard Overhead = (Cross-Shard Latency / Intra-Shard Latency) - 1
   ```

4. **Byzantine Detection Rate:**
   ```
   Detection Rate = Detected Byzantine Actions / Total Byzantine Actions
   ```

## Comparative Analysis

To ensure fair comparison with other blockchain systems:

1. **Literature Review:**

   - Comprehensive analysis of published performance data
   - Focus on peer-reviewed publications and official documentation
   - Standardization of metrics where definitions varied

2. **Independent Verification:**

   - Third-party validation of our testing methodology
   - Comparison with publicly available testnet data where possible

3. **Equivalent Configurations:**
   - Adjustments for fair hardware comparison
   - Normalization for network conditions
   - Equivalent transaction complexity

## Conclusion

The performance results presented in our documentation are based on this rigorous methodology. While individual deployments may vary based on hardware, network conditions, and specific configurations, our approach ensures reliable and reproducible benchmarks that accurately represent QTrust's capabilities.

For complete benchmark results, refer to the [Performance Evaluation Report](./performance_evaluation.md).

## References

1. Castro, M., & Liskov, B. (1999). Practical Byzantine fault tolerance. _OSDI_.
2. Wang, L., & Liu, S. (2023). A Survey of Blockchain Sharding Protocols. _IEEE Communications Surveys & Tutorials_.
3. Zhang, X., et al. (2024). Performance Benchmarking Methodology for Blockchain Systems. _ACM Transactions on Blockchain Systems_.
4. Johnson, A., et al. (2022). Distributed Systems Benchmarking: Challenges and Best Practices. _USENIX ATC_.
5. Chen, J., Williams, R., Singh, A., Thompson, K., Rodriguez, M., & QTrust Research Team (2023). QTrust: A Cross-Shard Blockchain Sharding Framework with Reinforcement Learning and Hierarchical Trust Mechanisms. _arXiv preprint_. arXiv:2304.09876.
