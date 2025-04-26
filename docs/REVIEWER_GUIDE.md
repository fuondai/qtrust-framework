# QTrust: Reviewer Documentation

## Overview

QTrust is a cross-shard blockchain sharding framework that integrates reinforcement learning and hierarchical trust mechanisms. This document provides comprehensive instructions for reviewers to evaluate the implementation and reproduce the results presented in the paper.

## Repository Structure

```
QTrust_GitHub/
├── benchmark/                # Performance benchmarking tools
│   ├── mock_small_scale_benchmark.py  # PyTorch-free benchmark implementation
│   └── small_scale_benchmark.py       # Original benchmark implementation
├── benchmark_results/        # Results from benchmark runs
├── charts/                   # Visualization scripts and charts
├── docs/                     # Additional documentation
├── documentation/            # Reviewer-focused documentation
├── qtrust/                   # Core implementation
│   ├── agents/               # Reinforcement learning agents
│   ├── consensus/            # Consensus protocols
│   ├── federated/            # Federated learning implementation
│   ├── mocks/                # PyTorch-free implementations for testing
│   ├── routing/              # Cross-shard routing algorithms
│   ├── trust/                # Trust mechanisms
│   ├── implementation_switch.py  # Switch between PyTorch and mock implementations
│   └── qtrust_framework.py   # Main framework implementation
├── tests/                    # Test suite
│   ├── integration/          # Integration tests
│   └── unit/                 # Unit tests
├── README.md                 # Project overview
├── requirements.txt          # Dependencies
└── setup.py                  # Installation script
```

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository (if not already done):
   ```bash
   git clone <repository-url>
   cd QTrust_GitHub
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Running Tests

QTrust includes a comprehensive test suite to verify the correctness of the implementation. The tests can be run with or without PyTorch.

### Running Tests Without PyTorch

For environments where PyTorch is not available or to verify the core functionality without deep learning dependencies:

```bash
python -m pytest tests/unit/test_mock_*.py -v
```

### Running All Tests (Requires PyTorch)

To run the complete test suite (requires PyTorch):

```bash
python run_tests.sh
```

Or using pytest directly:

```bash
python -m pytest tests/
```

## Running Benchmarks

QTrust includes benchmarks to evaluate the performance of the framework under different conditions.

### Small-Scale Benchmark (Without PyTorch)

For quick evaluation without PyTorch dependencies:

```bash
python benchmark/mock_small_scale_benchmark.py --nodes 32 --shards 8 --duration 30 --warmup 5 --cooldown 5
```

Parameters:
- `--nodes`: Total number of nodes in the network
- `--shards`: Number of shards to create
- `--duration`: Duration of the benchmark in seconds
- `--warmup`: Warmup period in seconds
- `--cooldown`: Cooldown period in seconds

### Full Benchmark (Requires PyTorch)

For comprehensive performance evaluation (requires PyTorch):

```bash
python benchmark/small_scale_benchmark.py --nodes 64 --shards 16 --duration 60 --warmup 10 --cooldown 10
```

## Reproducing Paper Results

To reproduce the results presented in the paper:

1. Run the full benchmark suite:
   ```bash
   python benchmark/run_all_benchmarks.py
   ```

2. Generate the performance charts:
   ```bash
   python charts/generate_charts.py
   ```

3. View the results in the `benchmark_results` directory.

## Implementation Details

### PyTorch and Mock Implementations

QTrust provides two implementations:

1. **PyTorch Implementation**: The full implementation using PyTorch for reinforcement learning and federated learning components.

2. **Mock Implementation**: A PyTorch-free implementation that mimics the behavior of the full implementation for testing and evaluation purposes.

To switch between implementations, use the `implementation_switch.py` module:

```python
from qtrust.implementation_switch import set_use_pytorch

# Use PyTorch implementation
set_use_pytorch(True)

# Use mock implementation (no PyTorch required)
set_use_pytorch(False)
```

### Key Components

1. **Rainbow DQN Agent**: Implements the Rainbow DQN algorithm for dynamic shard management.

2. **Hierarchical Trust Determination and Consensus Mechanism (HTDCM)**: Manages trust relationships between nodes and shards.

3. **Adaptive Consensus**: Dynamically selects the most appropriate consensus protocol based on network conditions.

4. **MAD-RAPID Router**: Optimizes cross-shard transaction routing.

5. **Privacy-Preserving Federated Learning**: Enables secure model updates across shards.

## Evaluation Metrics

The benchmark results include the following metrics:

1. **Throughput**: Transactions per second (TPS)
2. **Latency**: Average transaction confirmation time
3. **Trust Score**: Average trust score across nodes
4. **Byzantine Detection Rate**: Percentage of Byzantine nodes correctly identified
5. **False Positive Rate**: Percentage of honest nodes incorrectly flagged as Byzantine

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'torch'**:
   - This is expected if PyTorch is not installed
   - Use the mock implementation by setting `set_use_pytorch(False)`
   - Or install PyTorch: `pip install torch`

2. **Memory Issues**:
   - Reduce the number of nodes and shards in the benchmark
   - Use the mock implementation which requires less memory

3. **Performance Issues**:
   - Ensure no other resource-intensive processes are running
   - Try running on a machine with more CPU cores

### Getting Help

If you encounter any issues not covered here, please contact the authors at [contact information].

## Citation

If you use QTrust in your research, please cite:

```
@article{qtrust2025,
  title={QTrust: A Cross-Shard Blockchain Sharding Framework with Reinforcement Learning and Hierarchical Trust Mechanisms},
  author={[Authors]},
  journal={[Journal]},
  year={2025}
}
```

## License

This project is licensed under [License] - see the LICENSE file for details.
