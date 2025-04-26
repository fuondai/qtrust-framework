"""
QTrust Blockchain Sharding Framework main implementation.

This module provides the real implementation of the QTrust framework that integrates
all components: Rainbow DQN, HTDCM, Adaptive Consensus, MAD-RAPID, and Privacy-Preserving
Federated Learning.
"""

import os
import sys
import logging
import threading
import time
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional, Set, Union
import copy
import random

# Import the implementation switch module
from qtrust.implementation_switch import (
    get_rainbow_agent,
    get_adaptive_rainbow_agent,
    get_privacy_preserving_fl,
    set_use_pytorch,
    get_use_pytorch,
)

# Import real implementations
from qtrust.consensus.adaptive_consensus import AdaptiveConsensusSelector
from qtrust.routing.mad_rapid import MADRAPIDRouter
from qtrust.trust.htdcm import HTDCM

logger = logging.getLogger(__name__)


class Shard:
    """
    Represents a shard in the blockchain network.
    """

    def __init__(self, shard_id: str, config: Dict[str, Any] = None):
        """
        Initialize a shard.

        Args:
            shard_id: Unique identifier for the shard
            config: Configuration parameters
        """
        self.shard_id = shard_id

        # Default configuration
        self.config = {
            "max_nodes": 64,
            "min_nodes": 4,
            "max_transactions_per_block": 1000,
            "block_time": 5.0,  # seconds
            "consensus_protocol": "PoW",
            "state_size": 1024,  # KB
        }

        # Update configuration if provided
        if config:
            self.config.update(config)

        # Node management
        self.nodes = set()
        self.node_stats = {}

        # Transaction management
        self.transaction_pool = {}
        self.processed_transactions = set()

        # Block management
        self.blocks = []
        self.current_block_height = 0
        self.current_state_root = "0" * 64

        # Lock for thread safety
        self.lock = threading.RLock()

        logger.info(f"Initialized shard {shard_id}")

    def add_node(self, node_id: str) -> bool:
        """
        Add a node to the shard.

        Args:
            node_id: Node identifier

        Returns:
            Whether the node was added successfully
        """
        with self.lock:
            if len(self.nodes) >= self.config["max_nodes"]:
                logger.warning(
                    f"Cannot add node {node_id} to shard {self.shard_id}: maximum nodes reached"
                )
                return False

            if node_id in self.nodes:
                logger.warning(
                    f"Node {node_id} already exists in shard {self.shard_id}"
                )
                return False

            self.nodes.add(node_id)
            self.node_stats[node_id] = {
                "joined_at": time.time(),
                "transactions_processed": 0,
                "blocks_produced": 0,
                "last_active": time.time(),
            }

            logger.info(f"Added node {node_id} to shard {self.shard_id}")
            return True

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the shard.

        Args:
            node_id: Node identifier

        Returns:
            Whether the node was removed successfully
        """
        with self.lock:
            if node_id not in self.nodes:
                logger.warning(
                    f"Node {node_id} does not exist in shard {self.shard_id}"
                )
                return False

            if len(self.nodes) <= self.config["min_nodes"]:
                logger.warning(
                    f"Cannot remove node {node_id} from shard {self.shard_id}: minimum nodes reached"
                )
                return False

            self.nodes.remove(node_id)
            if node_id in self.node_stats:
                del self.node_stats[node_id]

            logger.info(f"Removed node {node_id} from shard {self.shard_id}")
            return True

    def get_nodes(self) -> Set[str]:
        """
        Get all nodes in the shard.

        Returns:
            Set of node identifiers
        """
        with self.lock:
            return set(self.nodes)

    def add_transaction(self, tx_id: str, tx_data: Dict[str, Any]) -> bool:
        """
        Add a transaction to the pool.

        Args:
            tx_id: Transaction identifier
            tx_data: Transaction data

        Returns:
            Whether the transaction was added successfully
        """
        with self.lock:
            if tx_id in self.transaction_pool or tx_id in self.processed_transactions:
                logger.warning(
                    f"Transaction {tx_id} already exists in shard {self.shard_id}"
                )
                return False

            self.transaction_pool[tx_id] = {
                "data": tx_data,
                "timestamp": time.time(),
                "status": "pending",
            }

            return True

    def process_transaction(self, tx_id: str) -> Dict[str, Any]:
        """
        Process a transaction.

        Args:
            tx_id: Transaction identifier

        Returns:
            Transaction result
        """
        with self.lock:
            if tx_id not in self.transaction_pool:
                return {
                    "success": False,
                    "error": f"Transaction {tx_id} not found in shard {self.shard_id}",
                }

            tx = self.transaction_pool[tx_id]

            # Mark as processed
            tx["status"] = "processed"
            self.processed_transactions.add(tx_id)
            del self.transaction_pool[tx_id]

            # Update node stats
            for node_id in self.nodes:
                if node_id in self.node_stats:
                    self.node_stats[node_id]["transactions_processed"] += 1
                    self.node_stats[node_id]["last_active"] = time.time()

            return {
                "success": True,
                "shard_id": self.shard_id,
                "block_height": self.current_block_height,
                "timestamp": time.time(),
            }

    def create_block(self) -> Dict[str, Any]:
        """
        Create a new block.

        Returns:
            Block data
        """
        with self.lock:
            # Get transactions to include
            txs = list(self.transaction_pool.keys())[
                : self.config["max_transactions_per_block"]
            ]

            # Create block
            block = {
                "shard_id": self.shard_id,
                "height": self.current_block_height + 1,
                "timestamp": time.time(),
                "transactions": txs,
                "previous_hash": self.current_state_root,
                "state_root": "0" * 64,  # Placeholder
            }

            # Process transactions
            for tx_id in txs:
                self.process_transaction(tx_id)

            # Update state
            self.current_block_height += 1
            self.current_state_root = block["state_root"]
            self.blocks.append(block)

            return block

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get shard metrics.

        Returns:
            Metrics dictionary
        """
        with self.lock:
            return {
                "shard_id": self.shard_id,
                "node_count": len(self.nodes),
                "transaction_pool_size": len(self.transaction_pool),
                "block_height": self.current_block_height,
                "state_root": self.current_state_root,
                "consensus_protocol": self.config["consensus_protocol"],
            }


class ShardManager:
    """
    Manages shards in the QTrust framework.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the shard manager.

        Args:
            config: Configuration parameters
        """
        # Default configuration
        self.config: Dict[str, Any] = {
            "initial_shards": 4,
            "max_shards": 64,
            "min_shards": 1,
            "target_shard_size": 32,  # nodes per shard
            "state_dim": 64,
            "action_dim": 8,
        }

        # Update configuration if provided
        if config:
            self.config.update(config)

        # Initialize shard management
        self.shards: Dict[str, Shard] = {}
        self.node_to_shard: Dict[str, str] = {}
        
        # Setup the RL agents
        state_dim = int(self.config["state_dim"])
        action_dim = int(self.config["action_dim"])
        
        # Get agent implementation based on switch module
        if get_use_pytorch():
            self.agent = get_adaptive_rainbow_agent(state_dim, action_dim, self.config)
        else:
            self.agent = get_rainbow_agent(state_dim, action_dim, self.config)
            
        # Initialize shards
        self._initialize_shards()
        
        logger.info("ShardManager initialized with {} shards".format(len(self.shards)))

    def _initialize_shards(self):
        """Initialize shards based on configuration."""
        initial_shards = int(self.config["initial_shards"])
        for i in range(initial_shards):
            shard_id = f"shard_{i}"
            self.shards[shard_id] = Shard(shard_id, self.config)

    def add_shard(self) -> Optional[str]:
        """
        Add a new shard.

        Returns:
            Shard ID or None if maximum shards reached
        """
        with self.lock:
            if len(self.shards) >= self.config["max_shards"]:
                logger.warning(
                    f"Cannot add shard: maximum shards reached ({self.config['max_shards']})"
                )
                return None

            # Generate new shard ID
            shard_id = f"shard_{len(self.shards)}"
            while shard_id in self.shards:
                shard_id = f"shard_{int(shard_id.split('_')[1]) + 1}"

            # Create new shard
            self.shards[shard_id] = Shard(shard_id)

            logger.info(f"Added new shard: {shard_id}")
            return shard_id

    def remove_shard(self, shard_id: str) -> bool:
        """
        Remove a shard.

        Args:
            shard_id: Shard identifier

        Returns:
            Whether the shard was removed successfully
        """
        with self.lock:
            if shard_id not in self.shards:
                logger.warning(f"Shard {shard_id} does not exist")
                return False

            if len(self.shards) <= self.config["min_shards"]:
                logger.warning(
                    f"Cannot remove shard {shard_id}: minimum shards reached ({self.config['min_shards']})"
                )
                return False

            # Get nodes in the shard
            nodes = self.shards[shard_id].get_nodes()

            # Remove shard
            del self.shards[shard_id]

            # Update node-to-shard mapping
            for node_id in nodes:
                if (
                    node_id in self.node_to_shard
                    and self.node_to_shard[node_id] == shard_id
                ):
                    del self.node_to_shard[node_id]

            logger.info(f"Removed shard: {shard_id}")
            return True

    def add_node(self, node_id: str, shard_id: Optional[str] = None) -> bool:
        """
        Add a node to a shard.

        Args:
            node_id: Node identifier
            shard_id: Shard identifier (if None, assign to least loaded shard)

        Returns:
            Whether the node was added successfully
        """
        with self.lock:
            if node_id in self.node_to_shard:
                logger.warning(
                    f"Node {node_id} already exists in shard {self.node_to_shard[node_id]}"
                )
                return False

            # If shard_id not specified, assign to least loaded shard
            if shard_id is None:
                shard_id = self._get_least_loaded_shard()

            # Check if shard exists
            if shard_id not in self.shards:
                logger.warning(f"Shard {shard_id} does not exist")
                return False

            # Add node to shard
            if not self.shards[shard_id].add_node(node_id):
                return False

            # Update node-to-shard mapping
            self.node_to_shard[node_id] = shard_id

            return True

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from its shard.

        Args:
            node_id: Node identifier

        Returns:
            Whether the node was removed successfully
        """
        with self.lock:
            if node_id not in self.node_to_shard:
                logger.warning(f"Node {node_id} does not exist in any shard")
                return False

            shard_id = self.node_to_shard[node_id]

            # Remove node from shard
            if not self.shards[shard_id].remove_node(node_id):
                return False

            # Update node-to-shard mapping
            del self.node_to_shard[node_id]

            return True

    def _get_least_loaded_shard(self) -> str:
        """
        Get the least loaded shard.

        Returns:
            Shard ID
        """
        min_load = float("inf")
        min_shard = None

        for shard_id, shard in self.shards.items():
            load = len(shard.get_nodes())
            if load < min_load:
                min_load = load
                min_shard = shard_id

        return min_shard

    def rebalance_shards(self) -> Dict[str, Any]:
        """
        Rebalance nodes across shards.

        Returns:
            Rebalance results
        """
        with self.lock:
            # Get current distribution
            distribution = {}
            for shard_id, shard in self.shards.items():
                distribution[shard_id] = len(shard.get_nodes())

            # Check if rebalance is needed
            if not distribution:
                return {"rebalanced": False, "reason": "No shards"}

            avg_nodes = sum(distribution.values()) / len(distribution)
            max_imbalance = max(
                abs(nodes - avg_nodes) / avg_nodes if avg_nodes > 0 else 0
                for nodes in distribution.values()
            )

            if max_imbalance <= self.config["rebalance_threshold"]:
                return {
                    "rebalanced": False,
                    "reason": "Imbalance below threshold",
                    "imbalance": max_imbalance,
                }

            # Rebalance
            moves = 0

            # Sort shards by load
            sorted_shards = sorted(distribution.items(), key=lambda x: x[1])

            # Move nodes from most loaded to least loaded
            while sorted_shards[-1][1] - sorted_shards[0][1] > 1:
                source_shard = sorted_shards[-1][0]
                target_shard = sorted_shards[0][0]

                # Find a node to move
                nodes = list(self.shards[source_shard].get_nodes())
                if not nodes:
                    break

                node_to_move = nodes[0]

                # Move node
                if self.shards[source_shard].remove_node(node_to_move) and self.shards[
                    target_shard
                ].add_node(node_to_move):
                    # Update node-to-shard mapping
                    self.node_to_shard[node_to_move] = target_shard

                    # Update distribution
                    sorted_shards[-1] = (source_shard, sorted_shards[-1][1] - 1)
                    sorted_shards[0] = (target_shard, sorted_shards[0][1] + 1)

                    # Re-sort
                    sorted_shards.sort(key=lambda x: x[1])

                    moves += 1
                else:
                    break

            return {
                "rebalanced": True,
                "moves": moves,
                "imbalance_before": max_imbalance,
                "imbalance_after": max(
                    abs(nodes - avg_nodes) / avg_nodes if avg_nodes > 0 else 0
                    for shard_id, nodes in sorted_shards
                ),
            }

    def get_shard_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics for all shards.

        Returns:
            Dictionary of shard metrics
        """
        with self.lock:
            metrics = {}
            for shard_id, shard in self.shards.items():
                metrics[shard_id] = shard.get_metrics()
            return metrics

    def get_node_distribution(self) -> Dict[str, int]:
        """
        Get node distribution across shards.

        Returns:
            Dictionary mapping shard IDs to node counts
        """
        with self.lock:
            distribution = {}
            for shard_id, shard in self.shards.items():
                distribution[shard_id] = len(shard.get_nodes())
            return distribution

    def get_state(self) -> np.ndarray:
        """
        Get the current state representation for RL.

        Returns:
            State representation as a numpy array
        """
        state_dim = int(self.config["state_dim"])
        state = np.zeros(state_dim, dtype=np.float32)
        
        # Basic features (first 8 dimensions)
        num_shards = len(self.shards)
        max_shards = int(self.config["max_shards"])
        ratio_shards = num_shards / max_shards if max_shards > 0 else 0
        
        # Fill the state vector
        if len(state) > 0:
            state[0] = ratio_shards
        
        if len(state) > 1:
            # Calculate load balance metric (variance in nodes per shard)
            nodes_per_shard = [len(shard.nodes) for shard in self.shards.values()]
            balance = np.var(nodes_per_shard) if nodes_per_shard else 0
            state[1] = balance / 100.0  # Normalize
        
        # More features can be added based on transaction patterns, network topology, etc.
        
        return state

    def get_action(self, state: np.ndarray) -> int:
        """
        Get the action to take based on the current state.

        Args:
            state: Current state

        Returns:
            Action index
        """
        with self.lock:
            # Use Rainbow DQN agent to get action
            return self.agent.act(state)

    def update_agent(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Update the Rainbow DQN agent.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        with self.lock:
            # Update Rainbow DQN agent
            self.agent.update(state, action, reward, next_state, done)


class QTrustFramework:
    """
    Main class for the QTrust blockchain sharding framework.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the QTrust framework.

        Args:
            config: Configuration parameters
        """
        # Set up logger
        self.logger = logging
        
        # Default configuration
        self.config: Dict[str, Any] = {
            "initial_shards": 4,
            "max_shards": 64,
            "min_shards": 1,
            "target_shard_size": 32,  # nodes per shard
            "rebalance_threshold": 0.3,  # trigger rebalance if imbalance > 30%
            "rebalance_frequency": 100,  # blocks
            "consensus_update_frequency": 50,  # blocks
            "routing_optimization_frequency": 20,  # blocks
            "federated_learning_frequency": 200,  # blocks
            "state_dim": 64,
            "action_dim": 8,
            "use_pytorch": False,  # Use NumPy implementation by default
        } 
        
        # Update configuration if provided
        if config:
            self.config.update(config)
            
        # Set PyTorch usage mode
        set_use_pytorch(bool(self.config["use_pytorch"]))
        
        # Initialize components
        self._init_components()
        
        # Set up metrics tracking
        self.metrics = {
            "tps": [],
            "latency": [],
            "cross_shard_txs": [],
            "byzantine_detection": []
        }
        
        # Threading lock
        self.lock = threading.RLock()
        
        # Running state
        self.running = False
        
        # Background threads
        self.background_threads: Dict[str, threading.Thread] = {}
        
        self.logger.info("QTrust framework initialized")

    def _init_components(self):
        """Initialize all framework components."""
        # Create shard manager
        self.shard_manager = ShardManager(self.config)
        
        # Initialize core components
        self.consensus_selector = AdaptiveConsensusSelector(self.config)
        self.router = MADRAPIDRouter(self.config)
        self.trust_manager = HTDCM(self.config)
        
        # Initialize federated learning manager if enabled
        if self.config.get("enable_federated_learning", True):
            federated_config = self.config.get("federated_learning", {})
            self.federated_manager = get_privacy_preserving_fl(federated_config)
        else:
            self.federated_manager = None
            
        # Register components with each other
        if hasattr(self.consensus_selector, "register_trust_provider"):
            self.consensus_selector.register_trust_provider(self.trust_manager)
            
        if hasattr(self.router, "register_trust_provider"):
            self.router.register_trust_provider(self.trust_manager)
            
        # Initialized time tracking
        self.last_rebalance = time.time()
        self.last_consensus_update = time.time()
        self.last_routing_optimization = time.time()
        self.last_federated_learning = time.time()

    def start(self):
        """Start the QTrust framework."""
        if hasattr(self, 'running') and self.running:
            self.logger.warning("QTrust framework already running")
            return
            
        self.running = True
        
        # Start the consensus selector
        if hasattr(self.consensus_selector, 'start'):
            self.consensus_selector.start()
            
        # Start the router
        if hasattr(self.router, 'start'):
            self.router.start()
            
        # Start the trust manager
        if hasattr(self.trust_manager, 'start'):
            self.trust_manager.start()
            
        # Start federated learning if enabled
        if self.federated_manager and hasattr(self.federated_manager, 'start'):
            self.federated_manager.start()
            
        self.logger.info("QTrust framework started")

    def stop(self):
        """
        Stop the QTrust framework.
        
        This method stops any background processes or threads started by the framework.
        """
        with self.lock:
            if not self.running:
                logger.warning("QTrustFramework is not running")
                return
                
            # Stop any background processes or threads here
            logger.info("Stopping QTrustFramework")
            self.running = False

    def add_node(self, node_id: str, shard_id: Optional[str] = None) -> bool:
        """
        Add a node to the network.

        Args:
            node_id: Node identifier
            shard_id: Shard identifier (if None, assign to least loaded shard)

        Returns:
            Whether the node was added successfully
        """
        with self.lock:
            return self.shard_manager.add_node(node_id, shard_id)

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the network.

        Args:
            node_id: Node identifier

        Returns:
            Whether the node was removed successfully
        """
        with self.lock:
            return self.shard_manager.remove_node(node_id)

    def add_shard(self) -> Optional[str]:
        """
        Add a new shard.

        Returns:
            Shard ID or None if maximum shards reached
        """
        with self.lock:
            return self.shard_manager.add_shard()

    def remove_shard(self, shard_id: str) -> bool:
        """
        Remove a shard.

        Args:
            shard_id: Shard identifier

        Returns:
            Whether the shard was removed successfully
        """
        with self.lock:
            return self.shard_manager.remove_shard(shard_id)

    def update(self, network_state):
        """
        Update the framework with the latest network state.
        
        Args:
            network_state: Current state of the network (can be dict or numpy array)
            
        Returns:
            Action to take
        """
        with self.lock:
            # Increment step counter
            self.steps += 1

            # Handle numpy array input from benchmark
            if hasattr(network_state, 'shape') and not hasattr(network_state, 'get'):
                # For benchmark, network_state is a numpy array with metrics
                # network_state[0] = success rate
                # network_state[1] = cross-shard ratio
                # network_state[2] = smart contract ratio
                
                # Update global metrics based on array values
                self.metrics['success_rate'] = float(network_state[0])
                self.metrics['cross_shard_ratio'] = float(network_state[1])
                self.metrics['smart_contract_ratio'] = float(network_state[2])
                
                # Determine action to take
                action = {
                    "rebalance": self.steps % self.config["rebalance_frequency"] == 0,
                    "update_consensus": self.steps % self.config["consensus_update_frequency"] == 0,
                    "optimize_routing": self.steps % self.config["routing_optimization_frequency"] == 0,
                    "update_federated_learning": self.steps % self.config["federated_learning_frequency"] == 0,
                }
                
                return action
                
            # Handle dictionary input (normal operation)
            # Update trust scores
            for node_id, trust in network_state.get("trust_scores", {}).items():
                self.trust_manager.update_trust(node_id, trust)

            # Update shard loads
            for shard_id, load in network_state.get("shard_loads", {}).items():
                # In a real implementation, we would update shard loads here
                pass

            # Update link states
            for link, state in network_state.get("links", {}).items():
                source, dest = link.split("-")
                self.router.update_link(
                    source, dest, load=state.get("load"), latency=state.get("latency")
                )

            # Determine action to take
            action = {
                "rebalance": self.steps % self.config["rebalance_frequency"] == 0,
                "update_consensus": self.steps
                % self.config["consensus_update_frequency"]
                == 0,
                "optimize_routing": self.steps
                % self.config["routing_optimization_frequency"]
                == 0,
                "update_federated_learning": self.steps
                % self.config["federated_learning_frequency"]
                == 0,
            }

            # Update consensus protocol if needed
            if self.steps % self.config["consensus_update_frequency"] == 0:
                # Get network conditions
                network_conditions = {
                    "node_count": len(self.shard_manager.node_to_shard),
                    "shard_count": len(self.shard_manager.shards),
                    "cross_shard_ratio": self.metrics["cross_shard_ratio"],
                    "smart_contract_ratio": self.metrics["smart_contract_ratio"],
                    "success_rate": self.metrics["success_rate"],
                }

                # Update consensus protocol
                for shard_id, shard in self.shard_manager.shards.items():
                    shard_size = len(shard.get_nodes())
                    protocol = self.consensus_selector.select_protocol(
                        shard_size=shard_size,
                        transaction_complexity=self.metrics["smart_contract_ratio"],
                        network_conditions=network_conditions,
                    )
                    shard.config["consensus_protocol"] = protocol

            # Optimize routing if needed
            if self.steps % self.config["routing_optimization_frequency"] == 0:
                self.router.optimize_routing()
