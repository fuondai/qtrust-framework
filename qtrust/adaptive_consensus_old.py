"""
Implementation of AdaptiveConsensus using a simplified Bayesian approach.
"""

import logging
import threading
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import random

logger = logging.getLogger(__name__)


class AdaptiveConsensus:
    """
    Adaptive Consensus implementation that dynamically selects the most appropriate
    consensus protocol based on network conditions, security requirements, and transaction patterns.

    This implementation uses a simplified Bayesian approach to select protocols.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Adaptive Consensus module.

        Args:
            config: Configuration parameters
        """
        # Default configuration
        self.config = {
            "default_protocol": "pbft",
            "adaptation_interval": 100,  # blocks
            "min_transition_interval": 500,  # blocks
            "performance_history_length": 10,
            "bayesian_prior_strength": 0.7,
            "supported_protocols": ["PBFT", "HotStuff", "Tendermint", "Raft", "PoA"],
        }

        # Protocol-specific parameters
        self.protocol_params = {
            "pbft": {"timeout": 5000, "view_change_timeout": 10000},  # ms  # ms
            "hotstuff": {"block_interval": 3000, "view_timeout": 9000},  # ms  # ms
            "tendermint": {
                "propose_timeout": 3000,  # ms
                "prevote_timeout": 1000,  # ms
                "precommit_timeout": 1000,  # ms
            },
            "raft": {"election_timeout": 1500, "heartbeat_interval": 500},  # ms  # ms
            "poa": {"block_period": 5000, "authority_set_size": 5},  # ms
        }

        # Update configuration if provided
        if config:
            self.config.update(config)

        # Initialize state
        self.current_protocol = self.config.get("default_protocol", "pbft").upper()
        self.last_adaptation_block = 0
        self.blocks_since_adaptation = 0

        # Network conditions
        self.network_conditions = {
            "node_count": 16,
            "byzantine_ratio": 0.1,
            "average_latency": 100,
            "transaction_complexity": "medium",
            "network_partition": False,
            "cross_shard_ratio": 0.2,
            "geographic_distribution": "medium",
        }

        # Performance metrics
        self.metrics = {
            "throughput": [],
            "latency": [],
            "finality_time": [],
            "resource_usage": [],
        }

        # Performance history for each protocol
        self.performance_history = {
            protocol: [] for protocol in self.config["supported_protocols"]
        }

        # Bayesian priors (initial probabilities for each protocol)
        self.bayesian_priors = self._initialize_bayesian_priors()

        # Adaptation history
        self.adaptation_history = []

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.update_thread = None

        logger.info(
            f"Initialized AdaptiveConsensus with protocol {self.current_protocol}"
        )

    def _initialize_bayesian_priors(self) -> Dict[str, float]:
        """
        Initialize Bayesian priors for each protocol.

        Returns:
            Dictionary of protocol priors
        """
        protocols = self.config["supported_protocols"]
        # Equal probability for all protocols initially
        priors = {protocol: 1.0 / len(protocols) for protocol in protocols}

        # Slightly favor the default protocol
        default = self.config.get("default_protocol", "pbft").upper()
        if default in priors:
            boost = 0.1
            priors[default] += boost

            # Normalize to ensure sum is 1.0
            total = sum(priors.values())
            for protocol in priors:
                priors[protocol] /= total

        return priors

    def start(self):
        """
        Start the adaptive consensus module.
        """
        if self.running:
            return

        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

        logger.info("Started AdaptiveConsensus update loop")

    def stop(self):
        """
        Stop the adaptive consensus module.
        """
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
            self.update_thread = None

        logger.info("Stopped AdaptiveConsensus update loop")

    def _update_loop(self):
        """Background thread for periodic adaptation."""
        while self.running:
            try:
                self.update_network_conditions()

                # Check if adaptation is needed
                if self.blocks_since_adaptation >= self.config["adaptation_interval"]:
                    self.adapt_consensus_protocol()

                # Sleep for a while
                time.sleep(5.0)
            except Exception as e:
                logger.error(f"Error in AdaptiveConsensus update loop: {e}")
                time.sleep(10.0)  # Sleep longer on error

    def update_network_conditions(self, conditions: Dict[str, Any] = None):
        """
        Update network conditions based on monitoring data.

        Args:
            conditions: New network conditions to set
        """
        with self.lock:
            if conditions:
                self.network_conditions.update(conditions)
            else:
                # Simulate changing network conditions for testing
                self._simulate_network_changes()

    def _simulate_network_changes(self):
        """Simulate network condition changes for testing."""
        # Randomly adjust node count
        self.network_conditions["node_count"] = max(
            4, self.network_conditions["node_count"] + random.randint(-2, 2)
        )

        # Adjust byzantine ratio (0.0 to 0.33)
        self.network_conditions["byzantine_ratio"] = max(
            0.0,
            min(
                0.33,
                self.network_conditions["byzantine_ratio"]
                + random.uniform(-0.02, 0.02),
            ),
        )

        # Adjust average latency (10ms to 500ms)
        self.network_conditions["average_latency"] = max(
            10,
            min(
                500,
                self.network_conditions["average_latency"] + random.randint(-20, 20),
            ),
        )

        # Adjust transaction complexity
        complexity_options = ["low", "medium", "high"]
        current_index = complexity_options.index(
            self.network_conditions["transaction_complexity"]
        )
        new_index = max(
            0, min(len(complexity_options) - 1, current_index + random.randint(-1, 1))
        )
        self.network_conditions["transaction_complexity"] = complexity_options[
            new_index
        ]

        # Small chance of network partition
        self.network_conditions["network_partition"] = random.random() < 0.05

        # Adjust cross-shard ratio (0.0 to 0.8)
        self.network_conditions["cross_shard_ratio"] = max(
            0.0,
            min(
                0.8,
                self.network_conditions["cross_shard_ratio"]
                + random.uniform(-0.05, 0.05),
            ),
        )

    def adapt_consensus_protocol(self):
        """
        Adapt the consensus protocol based on current conditions.
        """
        with self.lock:
            # Select the best protocol using Bayesian decision
            selected_protocol = self._select_consensus_protocol()

            # If different from current, transition to new protocol
            if selected_protocol != self.current_protocol:
                self._transition_to_protocol(selected_protocol)

            # Reset adaptation counter
            self.blocks_since_adaptation = 0
            self.last_adaptation_block += self.config["adaptation_interval"]

    def _select_consensus_protocol(self) -> str:
        """
        Select the most appropriate consensus protocol using Bayesian decision.

        Returns:
            Selected protocol name
        """
        # Extract relevant features
        shard_size = self.network_conditions["node_count"]
        byzantine_ratio = self.network_conditions["byzantine_ratio"]
        transaction_complexity = self.network_conditions["transaction_complexity"]
        network_latency = self.network_conditions["average_latency"]
        network_partition = self.network_conditions["network_partition"]
        cross_shard_ratio = self.network_conditions["cross_shard_ratio"]

        # Calculate conditional probabilities for each protocol
        probabilities = {}
        for protocol in self.config["supported_protocols"]:
            probabilities[protocol] = self._calculate_protocol_probability(
                protocol,
                shard_size,
                byzantine_ratio,
                transaction_complexity,
                network_latency,
                network_partition,
                cross_shard_ratio,
            )

        # Apply Bayesian priors
        for protocol in probabilities:
            probabilities[protocol] *= self.bayesian_priors.get(protocol, 0.2)

        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            for protocol in probabilities:
                probabilities[protocol] /= total_prob

        # Select protocol with highest probability
        selected_protocol = max(probabilities, key=probabilities.get)

        logger.info(f"Protocol selection probabilities: {probabilities}")
        logger.info(f"Selected protocol: {selected_protocol}")

        return selected_protocol

    def _calculate_protocol_probability(
        self,
        protocol: str,
        shard_size: int,
        byzantine_ratio: float,
        transaction_complexity: str,
        network_latency: float,
        network_partition: bool,
        cross_shard_ratio: float,
    ) -> float:
        """
        Calculate the conditional probability for a protocol given network conditions.

        Args:
            protocol: Protocol name
            shard_size: Number of nodes in the shard
            byzantine_ratio: Ratio of potentially Byzantine nodes
            transaction_complexity: Complexity level (low, medium, high)
            network_latency: Average network latency in ms
            network_partition: Whether network partition is detected
            cross_shard_ratio: Ratio of cross-shard transactions

        Returns:
            Conditional probability
        """
        # Base probability
        probability = 1.0

        # Protocol-specific factors
        if protocol == "PBFT":
            # PBFT works well with small to medium shard sizes
            if shard_size <= 20:
                probability *= 1.0
            elif shard_size <= 50:
                probability *= 0.7
            else:
                probability *= 0.3

            # PBFT handles Byzantine nodes well
            if byzantine_ratio > 0.2:
                probability *= 1.0
            else:
                probability *= 0.8

            # PBFT has high message complexity, so penalize for high latency
            if network_latency > 200:
                probability *= 0.6

            # PBFT handles network partitions well
            if network_partition:
                probability *= 1.0

        elif protocol == "HOTSTUFF":
            # HotStuff works well with medium to large shard sizes
            if shard_size <= 20:
                probability *= 0.7
            elif shard_size <= 100:
                probability *= 1.0
            else:
                probability *= 0.8

            # HotStuff handles Byzantine nodes well
            if byzantine_ratio > 0.2:
                probability *= 1.0
            else:
                probability *= 0.9

            # HotStuff has linear message complexity, good for high latency
            if network_latency > 200:
                probability *= 0.9

            # HotStuff handles network partitions well
            if network_partition:
                probability *= 0.9

        elif protocol == "TENDERMINT":
            # Tendermint works well with medium shard sizes
            if shard_size <= 30:
                probability *= 0.8
            elif shard_size <= 80:
                probability *= 1.0
            else:
                probability *= 0.6

            # Tendermint handles Byzantine nodes well
            if byzantine_ratio > 0.2:
                probability *= 0.9
            else:
                probability *= 0.8

            # Tendermint has moderate message complexity
            if network_latency > 200:
                probability *= 0.7

            # Tendermint handles network partitions moderately
            if network_partition:
                probability *= 0.8

        elif protocol == "RAFT":
            # Raft works well with any shard size but not Byzantine
            if byzantine_ratio > 0.1:
                probability *= 0.3
            else:
                probability *= 1.0

            # Raft has low message complexity, great for high latency
            if network_latency > 200:
                probability *= 1.0

            # Raft doesn't handle network partitions well
            if network_partition:
                probability *= 0.3

        elif protocol == "POA":
            # PoA works with any shard size but requires trusted nodes
            if byzantine_ratio > 0.05:
                probability *= 0.2
            else:
                probability *= 1.0

            # PoA has very low message complexity
            if network_latency > 200:
                probability *= 1.0

            # PoA doesn't handle network partitions well
            if network_partition:
                probability *= 0.4

        # Transaction complexity factors
        complexity_factor = {
            "low": {
                "PBFT": 0.9,
                "HOTSTUFF": 0.8,
                "TENDERMINT": 0.9,
                "RAFT": 1.0,
                "POA": 1.0,
            },
            "medium": {
                "PBFT": 1.0,
                "HOTSTUFF": 1.0,
                "TENDERMINT": 1.0,
                "RAFT": 0.9,
                "POA": 0.8,
            },
            "high": {
                "PBFT": 0.7,
                "HOTSTUFF": 0.9,
                "TENDERMINT": 0.8,
                "RAFT": 0.7,
                "POA": 0.6,
            },
        }

        probability *= complexity_factor.get(transaction_complexity, {}).get(
            protocol, 0.8
        )

        # Cross-shard transaction factors
        if cross_shard_ratio > 0.5:
            cross_shard_factor = {
                "PBFT": 0.7,
                "HOTSTUFF": 0.9,
                "TENDERMINT": 0.8,
                "RAFT": 0.6,
                "POA": 0.5,
            }
            probability *= cross_shard_factor.get(protocol, 0.7)

        return probability

    def _transition_to_protocol(self, new_protocol: str):
        """
        Transition from current protocol to a new one.

        Args:
            new_protocol: New protocol to transition to
        """
        old_protocol = self.current_protocol

        # In a real implementation, this would handle state transfer
        # For now, we just update the protocol name
        self.current_protocol = new_protocol

        # Record adaptation
        adaptation_event = {
            "old_protocol": old_protocol,
            "new_protocol": new_protocol,
            "network_conditions": self.network_conditions.copy(),
            "block_number": self.last_adaptation_block + self.blocks_since_adaptation,
        }

        self.adaptation_history.append(adaptation_event)

        logger.info(
            f"Transitioned from {old_protocol} to {new_protocol} at block {adaptation_event['block_number']}"
        )

    def update_protocol_performance(self, protocol: str, metrics: Dict[str, float]):
        """
        Update performance metrics for a protocol.

        Args:
            protocol: Protocol name
            metrics: Performance metrics
        """
        with self.lock:
            # Add to performance history
            if protocol in self.performance_history:
                self.performance_history[protocol].append(metrics)

                # Limit history length
                max_length = self.config.get("performance_history_length", 10)
                if len(self.performance_history[protocol]) > max_length:
                    self.performance_history[protocol] = self.performance_history[
                        protocol
                    ][-max_length:]

                # Update Bayesian priors based on performance
                self._update_bayesian_priors(protocol)

    def _update_bayesian_priors(self, protocol: str):
        """
        Update Bayesian priors based on protocol performance.

        Args:
            protocol: Protocol that was updated
        """
        if (
            protocol not in self.performance_history
            or not self.performance_history[protocol]
        ):
            return

        # Calculate recent performance score (higher is better)
        recent_metrics = self.performance_history[protocol][-1]

        # Simple scoring: throughput / latency (higher is better)
        throughput = recent_metrics.get("throughput", 0)
        latency = max(1, recent_metrics.get("latency", 100))  # Avoid division by zero
        resource_usage = recent_metrics.get("resource_usage", 50)

        # Performance score: higher throughput, lower latency, lower resource usage is better
        performance_score = (throughput / latency) * (100 / max(1, resource_usage))

        # Boost prior for this protocol
        prior_strength = self.config.get("bayesian_prior_strength", 0.7)
        boost_factor = 0.1 * performance_score / 10.0  # Normalize to ~0.1 range

        # Update prior
        self.bayesian_priors[protocol] = (
            prior_strength * self.bayesian_priors[protocol]
            + (1 - prior_strength) * boost_factor
        )

        # Normalize priors
        total = sum(self.bayesian_priors.values())
        for p in self.bayesian_priors:
            self.bayesian_priors[p] /= total

    def on_new_block(self):
        """
        Called when a new block is produced.
        """
        with self.lock:
            self.blocks_since_adaptation += 1

    def get_current_protocol(self) -> str:
        """
        Get the current consensus protocol.

        Returns:
            Current protocol name
        """
        with self.lock:
            return self.current_protocol

    def get_protocol_params(self, protocol: str = None) -> Dict[str, Any]:
        """
        Get parameters for a specific protocol.

        Args:
            protocol: Protocol name, or None for current protocol

        Returns:
            Protocol parameters
        """
        if protocol is None:
            protocol = self.current_protocol.lower()
        else:
            protocol = protocol.lower()

        return self.protocol_params.get(protocol, {})

    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get performance metrics.

        Returns:
            Dictionary of metrics
        """
        with self.lock:
            return self.metrics.copy()

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """
        Get adaptation history.

        Returns:
            List of adaptation events
        """
        with self.lock:
            return self.adaptation_history.copy()

    def get_bayesian_priors(self) -> Dict[str, float]:
        """
        Get current Bayesian priors.

        Returns:
            Dictionary of protocol priors
        """
        with self.lock:
            return self.bayesian_priors.copy()
