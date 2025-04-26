"""
Mock implementation of AdaptiveConsensus for testing without pgmpy dependency.
"""

import random
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class AdaptiveConsensus:
    """
    Mock implementation of Adaptive Consensus for testing.

    This class provides a simplified implementation that mimics the behavior
    of the actual Adaptive Consensus without requiring pgmpy.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the mock Adaptive Consensus.

        Args:
            config: Configuration parameters
        """
        # Default configuration
        self.config = {
            "adaptation_frequency": 100,
            "min_nodes": 4,
            "max_latency": 500,
            "consensus_protocols": ["PBFT", "Raft", "HotStuff", "Tendermint"],
        }

        # Update configuration if provided
        if config:
            self.config.update(config)

        # Initialize state
        self.current_protocol = self.config["consensus_protocols"][0]
        self.network_conditions = {
            "node_count": 16,
            "average_latency": 100,
            "transaction_load": "medium",
            "network_partition": False,
        }

        # Performance metrics
        self.metrics = {"throughput": [], "latency": [], "finality_time": []}

        # Adaptation history
        self.adaptation_history = []

        logger.info(
            f"Initialized MockAdaptiveConsensus with protocol {self.current_protocol}"
        )

    def update(self) -> None:
        """Update consensus protocol based on network conditions."""
        # In a real implementation, this would use Bayesian Network for decision making
        # For testing, we just randomly select a protocol

        # Mock network condition changes
        self._update_network_conditions()

        # Decide whether to adapt
        if self._should_adapt():
            old_protocol = self.current_protocol

            # Select new protocol
            available_protocols = [
                p
                for p in self.config["consensus_protocols"]
                if p != self.current_protocol
            ]
            self.current_protocol = random.choice(available_protocols)

            # Record adaptation
            self.adaptation_history.append(
                {
                    "old_protocol": old_protocol,
                    "new_protocol": self.current_protocol,
                    "network_conditions": self.network_conditions.copy(),
                }
            )

            logger.info(
                f"Adapted consensus protocol from {old_protocol} to {self.current_protocol}"
            )

    def _update_network_conditions(self) -> None:
        """Update network conditions."""
        # Randomly change network conditions
        self.network_conditions["node_count"] = max(
            self.config["min_nodes"],
            self.network_conditions["node_count"] + random.randint(-2, 2),
        )

        self.network_conditions["average_latency"] = max(
            10,
            min(
                self.config["max_latency"],
                self.network_conditions["average_latency"] + random.randint(-20, 20),
            ),
        )

        load_options = ["low", "medium", "high"]
        current_load_index = load_options.index(
            self.network_conditions["transaction_load"]
        )
        new_load_index = max(
            0, min(len(load_options) - 1, current_load_index + random.randint(-1, 1))
        )
        self.network_conditions["transaction_load"] = load_options[new_load_index]

        # Small chance of network partition
        self.network_conditions["network_partition"] = random.random() < 0.05

    def _should_adapt(self) -> bool:
        """Decide whether to adapt the consensus protocol."""
        # In a real implementation, this would use more sophisticated logic
        # For testing, we just use a simple heuristic

        # High latency suggests need for a different protocol
        if self.network_conditions["average_latency"] > 300:
            return True

        # Network partition requires a BFT protocol
        if self.network_conditions[
            "network_partition"
        ] and self.current_protocol not in ["PBFT", "HotStuff"]:
            return True

        # High load might require a more scalable protocol
        if self.network_conditions[
            "transaction_load"
        ] == "high" and self.current_protocol not in ["HotStuff", "Tendermint"]:
            return True

        # Random chance of adaptation for testing
        return random.random() < 0.2

    def get_current_protocol(self) -> str:
        """
        Get the current consensus protocol.

        Returns:
            Current protocol name
        """
        return self.current_protocol

    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get performance metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics

    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """
        Get adaptation history.

        Returns:
            List of adaptation events
        """
        return self.adaptation_history
