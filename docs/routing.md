# MAD-RAPID Protocol

This document details the enhanced MAD-RAPID protocol implementation in the QTrust framework.

## Overview

The MAD-RAPID (Multi-Agent Directed Routing with Adaptive Path and Intelligent Distribution) protocol provides optimized routing for cross-shard transactions in the QTrust blockchain sharding framework.

## Key Features

### Adaptive Path Selection

The protocol dynamically selects optimal paths for cross-shard transactions based on:

```python
def select_optimal_path(self, source_shard, target_shard, transaction):
    # Get all possible paths
    possible_paths = self.topology_manager.get_paths(source_shard, target_shard)
    
    # Calculate path scores
    path_scores = {}
    for path in possible_paths:
        path_scores[path] = self._calculate_path_score(path, transaction)
    
    # Select path with highest score
    return max(path_scores, key=path_scores.get)
```

Path scoring considers:
- Current congestion levels
- Historical performance
- Trust scores from HTDCM
- Path length and latency

### Congestion-Aware Routing

The protocol monitors network congestion and adjusts routing decisions:

```python
def _calculate_congestion_level(self, path):
    congestion_level = 0
    for node in path:
        # Get node's current queue size
        queue_size = self.node_stats[node].get_queue_size()
        # Get node's processing rate
        processing_rate = self.node_stats[node].get_processing_rate()
        # Calculate node congestion
        node_congestion = queue_size / processing_rate if processing_rate > 0 else float('inf')
        # Accumulate congestion
        congestion_level += node_congestion
    
    return congestion_level / len(path)
```

The system maintains a congestion map that is updated in real-time:

```python
def update_congestion_map(self):
    for shard_id in self.shards:
        for node_id in self.shards[shard_id].get_nodes():
            # Get latest metrics
            metrics = self.monitoring.get_node_metrics(node_id)
            # Update node stats
            self.node_stats[node_id].update(metrics)
    
    # Update path congestion estimates
    self._update_path_congestion_estimates()
```

### Cross-Shard Transaction Manager

The protocol includes a dedicated manager for cross-shard transactions:

```python
class CrossShardTransactionManager:
    def __init__(self, config):
        self.config = config
        self.pending_transactions = {}
        self.transaction_paths = {}
        self.transaction_states = {}
        self.timeout_manager = TimeoutManager()
    
    def process_transaction(self, transaction):
        # Identify shards involved
        involved_shards = self._identify_involved_shards(transaction)
        
        if len(involved_shards) == 1:
            # Single-shard transaction
            return self._process_single_shard_transaction(transaction, involved_shards[0])
        else:
            # Cross-shard transaction
            return self._process_cross_shard_transaction(transaction, involved_shards)
```

#### Transaction Splitting

Complex transactions are split for parallel processing:

```python
def _split_transaction(self, transaction, involved_shards):
    sub_transactions = []
    
    # Analyze transaction dependencies
    dependency_graph = self._build_dependency_graph(transaction)
    
    # Identify independent components
    independent_components = self._find_independent_components(dependency_graph)
    
    # Create sub-transactions
    for component in independent_components:
        sub_tx = self._create_sub_transaction(transaction, component)
        sub_transactions.append(sub_tx)
    
    return sub_transactions
```

#### Atomic Commitment

The protocol ensures atomicity for cross-shard transactions:

```python
def _ensure_atomicity(self, transaction_id, sub_transaction_results):
    # Check if all sub-transactions succeeded
    all_succeeded = all(result.success for result in sub_transaction_results)
    
    if all_succeeded:
        # Commit all sub-transactions
        self._commit_transaction(transaction_id)
        return True
    else:
        # Abort all sub-transactions
        self._abort_transaction(transaction_id)
        return False
```

## Reinforcement Learning Integration

The routing system uses reinforcement learning to optimize decisions:

```python
def update_routing_policy(self, state, action, reward, next_state):
    # Convert state to tensor
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
    
    # Get current Q-value
    current_q = self.policy_network(state_tensor).gather(1, torch.LongTensor([[action]]))
    
    # Get next Q-value
    next_q = self.target_network(next_state_tensor).max(1)[0].detach()
    
    # Calculate target Q-value
    target_q = reward + (self.gamma * next_q)
    
    # Calculate loss
    loss = F.smooth_l1_loss(current_q, target_q.unsqueeze(1))
    
    # Optimize
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

### Performance Metrics

The system maintains historical performance metrics:

```python
def _update_performance_metrics(self, path, transaction, result):
    # Calculate latency
    latency = result.completion_time - transaction.submission_time
    
    # Update path metrics
    self.path_metrics[path].update_latency(latency)
    self.path_metrics[path].update_success_rate(result.success)
    self.path_metrics[path].update_throughput(transaction.size)
    
    # Update prediction accuracy
    predicted_latency = self.path_metrics[path].get_predicted_latency(transaction)
    prediction_error = abs(latency - predicted_latency)
    self.path_metrics[path].update_prediction_accuracy(prediction_error)
```

## Adaptive Timeouts

The protocol adjusts timeouts based on expected latency:

```python
def _calculate_timeout(self, path, transaction):
    # Get base timeout
    base_timeout = self.config.get('base_timeout', 5000)  # ms
    
    # Get predicted latency for this path and transaction
    predicted_latency = self.path_metrics[path].get_predicted_latency(transaction)
    
    # Add safety margin (2x predicted latency)
    timeout = base_timeout + (2 * predicted_latency)
    
    # Cap at maximum timeout
    max_timeout = self.config.get('max_timeout', 30000)  # ms
    return min(timeout, max_timeout)
```

## Integration with Other Components

The MAD-RAPID protocol integrates with:

1. **HTDCM**: Uses trust scores to evaluate path security
2. **Adaptive Consensus**: Coordinates with consensus protocols for cross-shard transactions
3. **Rainbow DQN**: Provides feedback for reinforcement learning
4. **Monitoring**: Collects performance metrics for adaptation

## Configuration

The protocol supports flexible configuration:

```python
config = {
    "routing_update_interval": 100,  # blocks
    "path_history_length": 1000,
    "congestion_threshold": 0.8,
    "base_timeout": 5000,  # ms
    "max_timeout": 30000,  # ms
    "min_trust_score": 0.6
}

mad_rapid = MADRAPIDRouter(config)
```
