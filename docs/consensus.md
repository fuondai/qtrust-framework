# Adaptive Consensus

This document details the Adaptive Consensus Selection mechanism in the QTrust framework.

## Overview

The Adaptive Consensus module dynamically selects the most appropriate consensus protocol based on network conditions, security requirements, and transaction patterns.

## Supported Consensus Protocols

The framework supports three main consensus protocols with standardized interfaces:

1. **PBFT (Practical Byzantine Fault Tolerance)**

   - High security with 3f+1 nodes required
   - Good performance for small to medium shard sizes
   - High message complexity

2. **DPoS (Delegated Proof of Stake)**

   - Delegated validators with trust-weighted selection
   - Good performance for medium to large shards
   - Moderate message complexity

3. **PoA (Proof of Authority)**
   - Authorized validators only
   - Very efficient for trusted environments
   - Low message complexity

## Bayesian Decision Tree

The protocol selection uses a Bayesian decision tree that considers:

```python
def select_consensus_protocol(self, shard_state):
    # Extract relevant features
    shard_size = shard_state.get_size()
    byzantine_ratio = shard_state.get_byzantine_ratio()
    transaction_complexity = shard_state.get_transaction_complexity()
    network_latency = shard_state.get_network_latency()

    # Calculate conditional probabilities
    probabilities = {}
    for protocol in self.supported_protocols:
        probabilities[protocol] = self._calculate_protocol_probability(
            protocol, shard_size, byzantine_ratio,
            transaction_complexity, network_latency
        )

    # Select protocol with highest probability
    selected_protocol = max(probabilities, key=probabilities.get)
    return selected_protocol
```

### Decision Factors

1. **Network Conditions**

   - Latency between nodes
   - Bandwidth availability
   - Network partition probability

2. **Security Risks**

   - Byzantine node ratio
   - Trust scores from HTDCM
   - Historical attack patterns

3. **Transaction Complexity**

   - Cross-shard ratio
   - Smart contract execution
   - Transaction size

4. **Shard Size**
   - Number of nodes
   - Geographic distribution
   - Computational resources

## Protocol Transition Mechanism

The framework implements smooth transitions between consensus protocols:

```python
def transition_to_protocol(self, new_protocol):
    # Save current state
    current_state = self.current_protocol.get_state()

    # Initialize new protocol with current state
    self.new_protocol = self.protocol_factory.create(new_protocol)
    self.new_protocol.initialize(current_state)

    # Handle pending transactions
    pending_txs = self.current_protocol.get_pending_transactions()
    self.new_protocol.add_pending_transactions(pending_txs)

    # Switch protocols
    old_protocol = self.current_protocol
    self.current_protocol = self.new_protocol

    # Clean up old protocol
    old_protocol.shutdown()
```

### Transition Steps

1. **State Transfer**

   - Consensus state is captured
   - Pending transactions are preserved
   - Validator set is maintained

2. **Protocol Initialization**

   - New protocol is initialized with previous state
   - Configuration parameters are adjusted

3. **Handover**
   - Nodes are notified of protocol change
   - New protocol takes over transaction processing

## Performance Feedback Loop

The system learns from protocol performance:

```python
def update_protocol_performance(self, protocol, metrics):
    # Update performance history
    self.performance_history[protocol].append(metrics)

    # Calculate recent performance
    recent_performance = self._calculate_recent_performance(protocol)

    # Update Bayesian priors
    self._update_bayesian_priors(protocol, recent_performance)
```

### Metrics Tracked

1. **Throughput**

   - Transactions per second
   - Block production rate

2. **Latency**

   - Transaction confirmation time
   - Block finalization time

3. **Resource Usage**

   - CPU utilization
   - Memory consumption
   - Network bandwidth

4. **Security Incidents**
   - Double-spending attempts
   - Byzantine behavior detected

## Integration with Other Components

The Adaptive Consensus module integrates with:

1. **HTDCM**: Uses trust scores to assess Byzantine risk
2. **Rainbow DQN**: Provides feedback for reinforcement learning
3. **MAD-RAPID**: Coordinates cross-shard consensus
4. **Monitoring**: Collects performance metrics for adaptation

## Configuration

The module supports flexible configuration:

```python
config = {
    "default_protocol": "pbft",
    "adaptation_interval": 100,  # blocks
    "min_transition_interval": 500,  # blocks
    "performance_history_length": 10,
    "bayesian_prior_strength": 0.7
}

adaptive_consensus = AdaptiveConsensus(config)
```

## Protocol-Specific Parameters

Each protocol has specific parameters that can be tuned:

```python
protocol_params = {
    "pbft": {
        "timeout": 5000,  # ms
        "view_change_timeout": 10000  # ms
    },
    "hotstuff": {
        "block_interval": 3000,  # ms
        "view_timeout": 9000  # ms
    },
    "tendermint": {
        "propose_timeout": 3000,  # ms
        "prevote_timeout": 1000,  # ms
        "precommit_timeout": 1000  # ms
    },
    "raft": {
        "election_timeout": 1500,  # ms
        "heartbeat_interval": 500  # ms
    },
    "poa": {
        "block_period": 5000,  # ms
        "authority_set_size": 5
    }
}
```
