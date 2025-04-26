# QTrust: Quick Start Guide for Reviewers

This quick start guide provides essential commands to get you started with evaluating the QTrust framework. For more detailed information, please refer to the comprehensive [REVIEWER_GUIDE.md](./REVIEWER_GUIDE.md).

## 1. Installation

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## 2. Running Tests Without PyTorch

```bash
# Run all mock implementation tests (no PyTorch required)
python -m pytest tests/unit/test_mock_*.py -v
```

## 3. Running Small-Scale Benchmark

```bash
# Run the PyTorch-free benchmark
python benchmark/mock_small_scale_benchmark.py --nodes 32 --shards 8 --duration 30
```

## 4. Viewing Results

Benchmark results are stored in the `benchmark_results/small_scale/` directory:
- `benchmark_summary.txt`: Summary of performance metrics
- `throughput.csv`: Detailed throughput data
- `latency.csv`: Detailed latency data
- `trust_scores.csv`: Trust scores for each node

## 5. Implementation Switching

To switch between PyTorch and mock implementations in your code:

```python
from qtrust.implementation_switch import set_use_pytorch

# Use PyTorch-free implementation
set_use_pytorch(False)

# Import QTrust framework
from qtrust.qtrust_framework import QTrustFramework

# Create and use the framework
framework = QTrustFramework()
```

## 6. Key Metrics to Evaluate

1. **Throughput**: Should exceed 100,000 TPS in small-scale benchmark
2. **Latency**: Should be under 20ms average
3. **Byzantine Detection Rate**: Should be 100% for simulated attacks
4. **Trust Convergence**: Trust scores should stabilize within the benchmark duration

For any issues, please refer to the Troubleshooting section in the comprehensive guide.
