# Developer Guide

This document provides guidelines and information for developers who want to contribute to or extend the QTrust Blockchain Sharding Framework.

## Development Environment Setup

### Prerequisites

- **Python**: 3.8 or higher
- **Git**: Latest version
- **IDE**: We recommend PyCharm or Visual Studio Code with Python extensions

### Setting Up for Development

1. Clone the repository:
   ```bash
   git clone https://github.com/qtrust/qtrust.git
   cd qtrust
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies in development mode:
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Code Structure

The QTrust codebase is organized into the following modules:

### Core Modules

- **qtrust/agents/**: Rainbow DQN agent implementation
  - `rainbow_agent.py`: Full Rainbow DQN implementation
  - `mock_rainbow_agent.py`: CPU-only mock implementation
  - `adaptive_rainbow.py`: Adaptive version with dynamic parameters

- **qtrust/trust/**: Trust propagation and computation
  - `trust_propagation.py`: Trust propagation algorithms
  - `trust_vector.py`: Multi-dimensional trust vector implementation
  - `trust_computation.py`: HTDCM trust computation

- **qtrust/consensus/**: Consensus mechanisms
  - `dynamic_consensus.py`: Adaptive consensus protocol
  - `committee_selection.py`: Trust-based committee selection
  - `byzantine_detection.py`: Byzantine fault detection

- **qtrust/routing/**: Transaction routing
  - `mad_rapid.py`: MAD-RAPID routing algorithm
  - `congestion_prediction.py`: Network congestion prediction
  - `path_optimization.py`: Path optimization algorithms

- **qtrust/federated/**: Federated learning components
  - `federated_model.py`: Federated learning model
  - `privacy_preserving.py`: Privacy-preserving mechanisms
  - `model_aggregation.py`: Model aggregation algorithms

### Support Modules

- **qtrust/simulation/**: Network simulation
  - `network_simulation.py`: Network topology and latency simulation
  - `node_simulation.py`: Node behavior simulation
  - `byzantine_simulation.py`: Byzantine behavior simulation

- **qtrust/benchmark/**: Benchmarking tools
  - `benchmark_runner.py`: Benchmark execution
  - `metrics_collector.py`: Performance metrics collection
  - `transaction_generator.py`: Transaction workload generation

- **qtrust/utils/**: Utility functions
  - `crypto_utils.py`: Cryptographic utilities
  - `network_utils.py`: Networking utilities
  - `data_structures.py`: Common data structures

## Coding Standards

QTrust follows these coding standards:

### Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) for Python code style
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Use [flake8](https://flake8.pycqa.org/) for linting

### Documentation

- Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Document all public classes and methods
- Include examples in docstrings where appropriate
- Keep documentation up-to-date with code changes

### Testing

- Write unit tests for all new functionality
- Maintain at least 80% code coverage
- Use pytest for testing
- Place tests in the `tests/` directory with the same structure as the code

## Development Workflow

### Feature Development

1. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Implement your changes, following the coding standards

3. Write tests for your changes

4. Run the tests locally:
   ```bash
   pytest
   ```

5. Submit a pull request to the `main` branch

### Code Review Process

1. All pull requests require at least one review
2. Automated tests must pass
3. Code style checks must pass
4. Documentation must be updated

### Continuous Integration

QTrust uses GitHub Actions for continuous integration:

- **Code Style**: Black, isort, and flake8
- **Unit Tests**: pytest with coverage reporting
- **Integration Tests**: End-to-end tests with simulated network
- **Documentation**: Build and verify documentation

## Extending QTrust

### Adding New Trust Metrics

To add a new trust metric:

1. Extend the `TrustVector` class in `qtrust/trust/trust_vector.py`
2. Add the new dimension to the trust dimensions list in configuration
3. Implement the computation logic in `qtrust/trust/trust_computation.py`
4. Update the trust propagation mechanism if necessary
5. Add tests for the new metric

### Implementing New Consensus Mechanisms

To add a new consensus algorithm:

1. Create a new class in `qtrust/consensus/` that implements the `ConsensusProtocol` interface
2. Add the new algorithm to the `ConsensusFactory` in `qtrust/consensus/dynamic_consensus.py`
3. Implement the selection logic in the adaptive consensus mechanism
4. Add tests for the new consensus algorithm

### Creating Custom Benchmarks

To create a new benchmark scenario:

1. Create a new configuration file in `configs/`
2. Extend the `BenchmarkScenario` class in `qtrust/benchmark/benchmark_simulation.py`
3. Implement the specific logic for your benchmark
4. Add visualization support in `scripts/generate_visuals.py`

## Performance Optimization

When optimizing QTrust performance:

1. Profile the code to identify bottlenecks:
   ```bash
   python -m cProfile -o profile.pstats scripts/run_benchmark.py
   python -m pstats profile.pstats
   ```

2. Focus on these common bottlenecks:
   - Trust computation and propagation
   - Cross-shard transaction routing
   - Byzantine detection algorithms
   - Network simulation overhead

3. Consider these optimization strategies:
   - Caching frequently computed values
   - Parallelizing independent operations
   - Optimizing data structures for common operations
   - Reducing network message size and frequency

## Debugging

For effective debugging:

1. Enable detailed logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. Use the built-in debugging tools:
   ```bash
   python scripts/debug_tool.py --component trust --verbose
   ```

3. Visualize system state:
   ```bash
   python scripts/visualize_state.py --snapshot snapshots/latest.json
   ```

4. Common issues and solutions:
   - Trust convergence issues: Check propagation parameters
   - Routing inefficiencies: Examine congestion thresholds
   - Byzantine detection false positives: Adjust detection sensitivity
   - Performance degradation: Look for resource leaks or inefficient algorithms

## Release Process

QTrust follows semantic versioning (MAJOR.MINOR.PATCH):

1. Update version number in `qtrust/__init__.py`
2. Update CHANGELOG.md with the changes
3. Create a new release branch: `release/vX.Y.Z`
4. Run the full test suite and fix any issues
5. Tag the release: `git tag vX.Y.Z`
6. Push the tag: `git push origin vX.Y.Z`
7. Create a GitHub release with release notes

## Getting Help

If you need assistance with development:

1. Check the existing documentation
2. Look for similar issues in the GitHub issue tracker
3. Ask questions in the developer discussion forum
4. Contact the core development team at qtrust-dev@example.com

## Future Development Roadmap

Planned future enhancements:

1. **Enhanced Privacy**: Zero-knowledge proofs for private transactions
2. **Adaptive Sharding**: Dynamic shard count based on network load
3. **Cross-Chain Integration**: Interoperability with other blockchain systems
4. **Advanced Smart Contracts**: Support for more complex contract execution
5. **Mobile Node Support**: Lightweight client implementation for mobile devices

## Contributing

We welcome contributions to QTrust! Please see our [Contributing Guide](../CONTRIBUTING.md) for details on how to submit pull requests, report issues, and suggest improvements.
