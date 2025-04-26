# API Reference

## QTrust Framework API Documentation

This comprehensive API reference details the classes, methods, and interfaces available in the QTrust framework. The documentation follows standard academic conventions and provides detailed information about parameters, return types, exceptions, and usage examples.

## Table of Contents

- [QTrustFramework](#qtrustframework)
- [Rainbow DQN Agent](#rainbow-dqn-agent)
- [Hierarchical Trust Mechanism](#hierarchical-trust-mechanism)
- [Adaptive Consensus](#adaptive-consensus)
- [MAD-RAPID Router](#mad-rapid-router)
- [Privacy-Preserving Federated Learning](#privacy-preserving-federated-learning)

## QTrustFramework

The `QTrustFramework` class serves as the primary interface for interacting with the QTrust blockchain sharding system.

### Constructor

```python
QTrustFramework(config: Dict[str, Any])
```

**Parameters:**
- `config`: Configuration dictionary with the following parameters:
  - `num_shards` (int): Number of shards in the network
  - `nodes_per_shard` (int): Number of nodes per shard
  - `state_dim` (int, optional): State dimension for RL agents. Default: 64
  - `action_dim` (int, optional): Action dimension for RL agents. Default: 8
  - `trust_threshold` (float, optional): Minimum trust score for nodes. Default: 0.7
  - `consensus_update_frequency` (int, optional): Frequency of consensus updates. Default: 100
  - `routing_optimization_frequency` (int, optional): Frequency of routing optimizations. Default: 50
  - `federated_learning_frequency` (int, optional): Frequency of federated learning rounds. Default: 200
  - `use_pytorch` (bool, optional): Whether to use PyTorch implementations. Default: True

**Returns:**
- An initialized `QTrustFramework` instance

**Exceptions:**
- `ValueError`: If configuration parameters are invalid
- `RuntimeError`: If initialization fails

### Methods

#### start

```python
start() -> bool
```

Initializes and starts the QTrust network.

**Returns:**
- `bool`: True if the network started successfully, False otherwise

**Exceptions:**
- `RuntimeError`: If the network fails to start

#### submit_transaction

```python
submit_transaction(transaction: Dict[str, Any]) -> str
```

Submits a transaction to the network.

**Parameters:**
- `transaction`: Transaction dictionary with the following fields:
  - `source_shard` (str): Source shard identifier
  - `dest_shard` (str): Destination shard identifier
  - `sender` (str): Sender account identifier
  - `receiver` (str): Receiver account identifier
  - `amount` (float): Transaction amount
  - `fee` (float, optional): Transaction fee. Default: 0.1
  - `nonce` (int, optional): Transaction nonce for replay protection
  - `timestamp` (int, optional): Transaction timestamp
  - `smart_contract` (bool, optional): Whether this is a smart contract transaction
  - `contract_data` (Dict, optional): Smart contract data if applicable

**Returns:**
- `str`: Transaction hash/identifier

**Exceptions:**
- `ValueError`: If transaction parameters are invalid
- `RuntimeError`: If transaction submission fails

#### get_transaction_status

```python
get_transaction_status(tx_hash: str) -> Dict[str, Any]
```

Retrieves the status of a transaction.

**Parameters:**
- `tx_hash` (str): Transaction hash/identifier

**Returns:**
- `Dict[str, Any]`: Transaction status information including:
  - `status` (str): One of "pending", "confirmed", "rejected"
  - `confirmations` (int): Number of confirmations
  - `timestamp` (int): Confirmation timestamp if confirmed
  - `block_number` (int): Block number if confirmed
  - `error` (str, optional): Error message if rejected

**Exceptions:**
- `ValueError`: If transaction hash is invalid
- `KeyError`: If transaction is not found

#### evaluate_trust

```python
evaluate_trust(node_id: str) -> float
```

Evaluates the trust score of a specific node.

**Parameters:**
- `node_id` (str): Node identifier

**Returns:**
- `float`: Trust score between 0.0 and 1.0

**Exceptions:**
- `ValueError`: If node identifier is invalid
- `KeyError`: If node is not found

#### get_network_metrics

```python
get_network_metrics() -> Dict[str, Any]
```

Retrieves current network performance metrics.

**Returns:**
- `Dict[str, Any]`: Network metrics including:
  - `throughput` (float): Current throughput in TPS
  - `latency` (float): Average transaction latency in ms
  - `active_nodes` (int): Number of active nodes
  - `byzantine_nodes` (int): Number of detected Byzantine nodes
  - `cross_shard_tx_count` (int): Number of cross-shard transactions
  - `shard_distribution` (Dict[str, int]): Distribution of nodes across shards

**Exceptions:**
- `RuntimeError`: If metrics collection fails

#### shutdown

```python
shutdown() -> bool
```

Gracefully shuts down the QTrust network.

**Returns:**
- `bool`: True if shutdown was successful, False otherwise

**Exceptions:**
- `RuntimeError`: If shutdown fails

## Rainbow DQN Agent

The `RainbowDQNAgent` class implements a state-of-the-art reinforcement learning agent for dynamic shard allocation and management.

### Constructor

```python
RainbowDQNAgent(state_dim: int, action_dim: int, config: Dict[str, Any])
```

**Parameters:**
- `state_dim` (int): Dimension of the state space
- `action_dim` (int): Dimension of the action space
- `config`: Configuration dictionary with the following parameters:
  - `learning_rate` (float, optional): Learning rate. Default: 0.0001
  - `gamma` (float, optional): Discount factor. Default: 0.99
  - `buffer_size` (int, optional): Replay buffer size. Default: 10000
  - `batch_size` (int, optional): Batch size for training. Default: 64
  - `target_update_frequency` (int, optional): Frequency of target network updates. Default: 100
  - `double_dqn` (bool, optional): Whether to use Double DQN. Default: True
  - `dueling_dqn` (bool, optional): Whether to use Dueling DQN. Default: True
  - `noisy_nets` (bool, optional): Whether to use Noisy Networks. Default: True
  - `prioritized_replay` (bool, optional): Whether to use Prioritized Experience Replay. Default: True
  - `n_step` (int, optional): Number of steps for multi-step learning. Default: 3
  - `num_atoms` (int, optional): Number of atoms for distributional RL. Default: 51
  - `v_min` (float, optional): Minimum value for distributional RL. Default: -10.0
  - `v_max` (float, optional): Maximum value for distributional RL. Default: 10.0

**Returns:**
- An initialized `RainbowDQNAgent` instance

**Exceptions:**
- `ValueError`: If parameters are invalid

### Methods

#### select_action

```python
select_action(state: np.ndarray, epsilon: float = 0.0) -> int
```

Selects an action based on the current state.

**Parameters:**
- `state` (np.ndarray): Current state
- `epsilon` (float, optional): Exploration rate. Default: 0.0

**Returns:**
- `int`: Selected action

#### update

```python
update(state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float
```

Updates the agent based on a transition.

**Parameters:**
- `state` (np.ndarray): Current state
- `action` (int): Action taken
- `reward` (float): Reward received
- `next_state` (np.ndarray): Next state
- `done` (bool): Whether the episode is done

**Returns:**
- `float`: Loss value

#### save

```python
save(path: str) -> None
```

Saves the agent's model to disk.

**Parameters:**
- `path` (str): Path to save the model

#### load

```python
load(path: str) -> None
```

Loads the agent's model from disk.

**Parameters:**
- `path` (str): Path to load the model from

**Exceptions:**
- `FileNotFoundError`: If the model file is not found
- `RuntimeError`: If loading fails

## Hierarchical Trust Mechanism

The `HTDCM` class implements the Hierarchical Trust-based Data Center Mechanism for quantifiable trust metrics.

### Constructor

```python
HTDCM(config: Dict[str, Any] = None)
```

**Parameters:**
- `config` (Dict[str, Any], optional): Configuration dictionary with the following parameters:
  - `trust_threshold` (float, optional): Minimum trust score for nodes. Default: 0.7
  - `trust_decay_factor` (float, optional): Rate at which trust decays over time. Default: 0.99
  - `trust_update_weight` (float, optional): Weight for new trust observations. Default: 0.1
  - `byzantine_detection_threshold` (float, optional): Threshold for Byzantine detection. Default: 0.3
  - `global_trust_weight` (float, optional): Weight for global trust in the hierarchy. Default: 0.3
  - `inter_shard_trust_weight` (float, optional): Weight for inter-shard trust. Default: 0.3
  - `intra_shard_trust_weight` (float, optional): Weight for intra-shard trust. Default: 0.4

**Returns:**
- An initialized `HTDCM` instance

### Methods

#### add_node

```python
add_node(node_id: str, shard_id: str) -> bool
```

Adds a node to the trust management system.

**Parameters:**
- `node_id` (str): Node identifier
- `shard_id` (str): Shard identifier

**Returns:**
- `bool`: True if node was added successfully, False otherwise

#### add_shard

```python
add_shard(shard_id: str) -> bool
```

Adds a shard to the trust management system.

**Parameters:**
- `shard_id` (str): Shard identifier

**Returns:**
- `bool`: True if shard was added successfully, False otherwise

#### update_trust

```python
update_trust(node_id: str, behavior_score: float) -> float
```

Updates the trust score of a node based on observed behavior.

**Parameters:**
- `node_id` (str): Node identifier
- `behavior_score` (float): Observed behavior score between 0.0 and 1.0

**Returns:**
- `float`: Updated trust score

**Exceptions:**
- `ValueError`: If behavior score is outside the valid range
- `KeyError`: If node is not found

#### get_trust

```python
get_trust(node_id: str) -> float
```

Retrieves the current trust score of a node.

**Parameters:**
- `node_id` (str): Node identifier

**Returns:**
- `float`: Trust score between 0.0 and 1.0

**Exceptions:**
- `KeyError`: If node is not found

#### is_byzantine

```python
is_byzantine(node_id: str) -> bool
```

Determines if a node is considered Byzantine based on its trust score.

**Parameters:**
- `node_id` (str): Node identifier

**Returns:**
- `bool`: True if node is considered Byzantine, False otherwise

**Exceptions:**
- `KeyError`: If node is not found

## Adaptive Consensus

The `AdaptiveConsensus` class implements dynamic consensus mechanism selection based on network conditions.

### Constructor

```python
AdaptiveConsensus(config: Dict[str, Any] = None)
```

**Parameters:**
- `config` (Dict[str, Any], optional): Configuration dictionary with the following parameters:
  - `available_consensus` (List[str], optional): List of available consensus mechanisms. Default: ["pbft", "dpos", "poa"]
  - `default_consensus` (str, optional): Default consensus mechanism. Default: "pbft"
  - `adaptation_frequency` (int, optional): Frequency of adaptation in blocks. Default: 100
  - `performance_weight` (float, optional): Weight for performance in adaptation. Default: 0.4
  - `security_weight` (float, optional): Weight for security in adaptation. Default: 0.4
  - `decentralization_weight` (float, optional): Weight for decentralization in adaptation. Default: 0.2

**Returns:**
- An initialized `AdaptiveConsensus` instance

### Methods

#### select_consensus

```python
select_consensus(shard_id: str, metrics: Dict[str, Any]) -> str
```

Selects the optimal consensus mechanism based on current metrics.

**Parameters:**
- `shard_id` (str): Shard identifier
- `metrics` (Dict[str, Any]): Current metrics including:
  - `node_count` (int): Number of nodes in the shard
  - `byzantine_ratio` (float): Ratio of Byzantine nodes
  - `transaction_volume` (int): Current transaction volume
  - `network_latency` (float): Current network latency
  - `trust_distribution` (Dict[str, float]): Distribution of trust scores

**Returns:**
- `str`: Selected consensus mechanism

#### get_consensus_parameters

```python
get_consensus_parameters(consensus_type: str, shard_id: str) -> Dict[str, Any]
```

Retrieves optimal parameters for a specific consensus mechanism.

**Parameters:**
- `consensus_type` (str): Consensus mechanism type
- `shard_id` (str): Shard identifier

**Returns:**
- `Dict[str, Any]`: Consensus parameters

**Exceptions:**
- `ValueError`: If consensus type is invalid

## MAD-RAPID Router

The `MADRAPIDRouter` class implements the Multi-Agent Dynamic Routing with Adaptive Path Identification and Decision algorithm for cross-shard transaction optimization.

### Constructor

```python
MADRAPIDRouter(config: Dict[str, Any] = None)
```

**Parameters:**
- `config` (Dict[str, Any], optional): Configuration dictionary with the following parameters:
  - `path_cache_size` (int, optional): Size of the path cache. Default: 1000
  - `path_cache_ttl` (int, optional): Time-to-live for cached paths in seconds. Default: 60
  - `max_path_length` (int, optional): Maximum path length. Default: 5
  - `latency_weight` (float, optional): Weight for latency in path selection. Default: 0.4
  - `trust_weight` (float, optional): Weight for trust in path selection. Default: 0.4
  - `load_weight` (float, optional): Weight for load in path selection. Default: 0.2
  - `optimization_frequency` (int, optional): Frequency of path optimization. Default: 50

**Returns:**
- An initialized `MADRAPIDRouter` instance

### Methods

#### find_optimal_path

```python
find_optimal_path(source_shard: str, dest_shard: str, trust_manager: HTDCM) -> List[str]
```

Finds the optimal path for a cross-shard transaction.

**Parameters:**
- `source_shard` (str): Source shard identifier
- `dest_shard` (str): Destination shard identifier
- `trust_manager` (HTDCM): Trust manager instance

**Returns:**
- `List[str]`: Optimal path as a list of shard identifiers

**Exceptions:**
- `ValueError`: If source or destination shard is invalid
- `RuntimeError`: If no path can be found

#### update_network_topology

```python
update_network_topology(topology: Dict[str, List[str]]) -> None
```

Updates the network topology.

**Parameters:**
- `topology` (Dict[str, List[str]]): Network topology as an adjacency list

#### get_path_metrics

```python
get_path_metrics(path: List[str]) -> Dict[str, float]
```

Retrieves metrics for a specific path.

**Parameters:**
- `path` (List[str]): Path as a list of shard identifiers

**Returns:**
- `Dict[str, float]`: Path metrics including:
  - `latency` (float): Estimated latency in ms
  - `trust_score` (float): Aggregate trust score
  - `load` (float): Current load factor

## Privacy-Preserving Federated Learning

The `PrivacyPreservingFL` class implements secure federated learning with privacy guarantees.

### Constructor

```python
PrivacyPreservingFL(num_clients: int, dp_params: Dict[str, float] = None)
```

**Parameters:**
- `num_clients` (int): Number of participating clients
- `dp_params` (Dict[str, float], optional): Differential privacy parameters:
  - `epsilon` (float, optional): Privacy budget. Default: 1.0
  - `delta` (float, optional): Probability of privacy breach. Default: 1e-5
  - `clip_norm` (float, optional): Maximum L2 norm for gradient clipping. Default: 1.0

**Returns:**
- An initialized `PrivacyPreservingFL` instance

### Methods

#### federated_learning_round

```python
federated_learning_round(client_ids: List[int], data: Dict[int, np.ndarray], 
                        labels: Dict[int, np.ndarray], epochs: int = 1, 
                        batch_size: int = 32, secure: bool = True) -> Dict[str, np.ndarray]
```

Performs one round of federated learning.

**Parameters:**
- `client_ids` (List[int]): List of participating client IDs
- `data` (Dict[int, np.ndarray]): Dictionary of client data
- `labels` (Dict[int, np.ndarray]): Dictionary of client labels
- `epochs` (int, optional): Number of training epochs. Default: 1
- `batch_size` (int, optional): Batch size. Default: 32
- `secure` (bool, optional): Whether to use secure aggregation. Default: True

**Returns:**
- `Dict[str, np.ndarray]`: Updated global model

**Exceptions:**
- `ValueError`: If parameters are invalid

#### evaluate

```python
evaluate(data: np.ndarray, labels: np.ndarray) -> Dict[str, float]
```

Evaluates the global model on test data.

**Parameters:**
- `data` (np.ndarray): Test data
- `labels` (np.ndarray): Test labels

**Returns:**
- `Dict[str, float]`: Evaluation metrics including:
  - `accuracy` (float): Model accuracy
  - `loss` (float): Model loss
  - `f1_score` (float): F1 score

#### secure_aggregate

```python
secure_aggregate(models: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]
```

Performs secure aggregation of client models.

**Parameters:**
- `models` (List[Dict[str, np.ndarray]]): List of client models

**Returns:**
- `Dict[str, np.ndarray]`: Securely aggregated model

**Exceptions:**
- `ValueError`: If no models to aggregate
