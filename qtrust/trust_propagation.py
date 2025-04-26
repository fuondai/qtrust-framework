#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Trust Propagation
This module implements hierarchical trust-driven consensus mechanisms (HTDCM)
with enhanced trust propagation and Sybil resistance.
"""

import time
import threading
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from collections import defaultdict


class TrustVector:
    """
    Implements a multi-dimensional weighted trust vector.
    """

    # Trust dimensions
    DIMENSION_VALIDATION = 0  # Transaction validation correctness
    DIMENSION_LATENCY = 1  # Response time reliability
    DIMENSION_AVAILABILITY = 2  # Node uptime and responsiveness
    DIMENSION_CONSISTENCY = 3  # Consistency of behavior
    DIMENSION_THROUGHPUT = 4  # Transaction processing capacity

    # Number of dimensions
    NUM_DIMENSIONS = 5

    def __init__(self, initial_values: Optional[List[float]] = None):
        """
        Initialize the trust vector.

        Args:
            initial_values: Initial trust values for each dimension
        """
        if initial_values and len(initial_values) == self.NUM_DIMENSIONS:
            self.values = initial_values.copy()
        else:
            # Default to neutral trust (0.5) in all dimensions
            self.values = [0.5] * self.NUM_DIMENSIONS

        # Ensure values are in valid range [0, 1]
        for i in range(len(self.values)):
            self.values[i] = max(0.0, min(1.0, self.values[i]))

    def get_dimension(self, dimension: int) -> float:
        """
        Get trust value for a specific dimension.

        Args:
            dimension: Trust dimension index

        Returns:
            Trust value in range [0, 1]
        """
        if 0 <= dimension < len(self.values):
            return self.values[dimension]
        return 0.5  # Default to neutral trust

    def set_dimension(self, dimension: int, value: float):
        """
        Set trust value for a specific dimension.

        Args:
            dimension: Trust dimension index
            value: Trust value in range [0, 1]
        """
        if 0 <= dimension < len(self.values):
            self.values[dimension] = max(0.0, min(1.0, value))

    def update_dimension(self, dimension: int, value: float, weight: float = 0.1):
        """
        Update trust value for a specific dimension.

        Args:
            dimension: Trust dimension index
            value: New observation in range [0, 1]
            weight: Weight of the new observation (learning rate)
        """
        if 0 <= dimension < len(self.values):
            current = self.values[dimension]
            updated = current * (1 - weight) + value * weight
            self.values[dimension] = max(0.0, min(1.0, updated))

    def get_aggregate_trust(self, weights: Optional[List[float]] = None) -> float:
        """
        Get aggregate trust value across all dimensions.

        Args:
            weights: Optional weights for each dimension

        Returns:
            Aggregate trust value in range [0, 1]
        """
        if not weights:
            # Default to equal weights
            weights = [1.0 / self.NUM_DIMENSIONS] * self.NUM_DIMENSIONS

        # Ensure weights are normalized
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / self.NUM_DIMENSIONS] * self.NUM_DIMENSIONS

        # Calculate weighted sum
        return sum(
            self.values[i] * normalized_weights[i]
            for i in range(min(len(self.values), len(normalized_weights)))
        )

    def combine(self, other: "TrustVector", weight: float = 0.5) -> "TrustVector":
        """
        Combine with another trust vector.

        Args:
            other: Other trust vector
            weight: Weight of the other vector (0.5 means equal weight)

        Returns:
            New combined trust vector
        """
        result = TrustVector()
        for i in range(self.NUM_DIMENSIONS):
            result.values[i] = self.values[i] * (1 - weight) + other.values[i] * weight
        return result

    def to_dict(self) -> Dict[str, float]:
        """
        Convert to dictionary.

        Returns:
            Dictionary with dimension names and values
        """
        dimension_names = [
            "validation",
            "latency",
            "availability",
            "consistency",
            "throughput",
        ]

        return {
            dimension_names[i]: self.values[i]
            for i in range(min(len(dimension_names), len(self.values)))
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "TrustVector":
        """
        Create from dictionary.

        Args:
            data: Dictionary with dimension names and values

        Returns:
            New trust vector
        """
        dimension_names = [
            "validation",
            "latency",
            "availability",
            "consistency",
            "throughput",
        ]

        values = [0.5] * cls.NUM_DIMENSIONS
        for i, name in enumerate(dimension_names):
            if name in data:
                values[i] = data[name]

        return cls(values)


class TrustPropagation:
    """
    Implements gossip-based trust propagation with TTL and expiration.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the trust propagation system.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Default configuration
        self.ttl_max = self.config.get(
            "ttl_max", 5
        )  # Maximum time-to-live for trust messages
        self.expiration_time = self.config.get(
            "expiration_time", 3600
        )  # Trust expiration in seconds
        self.gossip_interval = self.config.get(
            "gossip_interval", 10.0
        )  # Gossip interval in seconds
        self.gossip_fanout = self.config.get(
            "gossip_fanout", 3
        )  # Number of peers to gossip to
        self.sybil_threshold = self.config.get(
            "sybil_threshold", 0.3
        )  # Threshold for Sybil detection
        self.trust_threshold = self.config.get(
            "trust_threshold", 0.6
        )  # Minimum trust threshold
        self.adaptive_threshold = self.config.get(
            "adaptive_threshold", True
        )  # Enable adaptive threshold

        # Trust data
        self.direct_trust = {}  # node_id -> TrustVector (direct observations)
        self.indirect_trust = (
            {}
        )  # node_id -> {observer_id -> (TrustVector, timestamp, ttl)}
        self.aggregate_trust = (
            {}
        )  # node_id -> TrustVector (combined direct and indirect)

        # Trust update timestamps
        self.last_update = {}  # node_id -> timestamp

        # Sybil detection
        self.trust_graph = defaultdict(set)  # node_id -> set of trusted nodes
        self.sybil_suspects = set()  # Set of suspected Sybil nodes

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.gossip_thread = None
        self.expiration_thread = None

    def start(self):
        """
        Start the trust propagation system.
        """
        if self.running:
            return

        self.running = True

        # Start gossip thread
        self.gossip_thread = threading.Thread(target=self._gossip_loop)
        self.gossip_thread.daemon = True
        self.gossip_thread.start()

        # Start expiration thread
        self.expiration_thread = threading.Thread(target=self._expiration_loop)
        self.expiration_thread.daemon = True
        self.expiration_thread.start()

    def stop(self):
        """
        Stop the trust propagation system.
        """
        self.running = False

        if self.gossip_thread:
            self.gossip_thread.join(timeout=2.0)
            self.gossip_thread = None

        if self.expiration_thread:
            self.expiration_thread.join(timeout=2.0)
            self.expiration_thread = None

    def update_direct_trust(
        self, node_id: str, dimension: int, value: float, weight: float = 0.1
    ):
        """
        Update direct trust for a node based on observation.

        Args:
            node_id: Node identifier
            dimension: Trust dimension index
            value: Observed value in range [0, 1]
            weight: Weight of the observation
        """
        with self.lock:
            if node_id not in self.direct_trust:
                self.direct_trust[node_id] = TrustVector()

            self.direct_trust[node_id].update_dimension(dimension, value, weight)
            self.last_update[node_id] = time.time()

            # Update aggregate trust
            self._update_aggregate_trust(node_id)

            # Update trust graph
            self._update_trust_graph(node_id)

    def add_indirect_trust(
        self, node_id: str, observer_id: str, trust_vector: TrustVector, ttl: int = None
    ):
        """
        Add indirect trust observation from another node.

        Args:
            node_id: Target node identifier
            observer_id: Observer node identifier
            trust_vector: Trust vector from observer
            ttl: Time-to-live for this trust information
        """
        if ttl is None:
            ttl = self.ttl_max

        if ttl <= 0:
            return  # Expired TTL

        with self.lock:
            if node_id not in self.indirect_trust:
                self.indirect_trust[node_id] = {}

            self.indirect_trust[node_id][observer_id] = (trust_vector, time.time(), ttl)

            # Update aggregate trust
            self._update_aggregate_trust(node_id)

    def get_trust(self, node_id: str) -> Optional[TrustVector]:
        """
        Get aggregate trust for a node.

        Args:
            node_id: Node identifier

        Returns:
            Trust vector or None if node not known
        """
        with self.lock:
            return self.aggregate_trust.get(node_id)

    def get_trust_value(
        self, node_id: str, weights: Optional[List[float]] = None
    ) -> float:
        """
        Get aggregate trust value for a node.

        Args:
            node_id: Node identifier
            weights: Optional dimension weights

        Returns:
            Trust value in range [0, 1] or 0.0 if node not known
        """
        trust_vector = self.get_trust(node_id)
        if trust_vector:
            return trust_vector.get_aggregate_trust(weights)
        return 0.0

    def is_trusted(self, node_id: str) -> bool:
        """
        Check if a node is trusted.

        Args:
            node_id: Node identifier

        Returns:
            True if node is trusted, False otherwise
        """
        with self.lock:
            # Check if node is a suspected Sybil
            if node_id in self.sybil_suspects:
                return False

            # Get trust value
            trust_value = self.get_trust_value(node_id)

            # Compare with threshold
            return trust_value >= self.get_trust_threshold()

    def get_trusted_nodes(self) -> List[str]:
        """
        Get list of trusted nodes.

        Returns:
            List of trusted node identifiers
        """
        with self.lock:
            return [
                node_id for node_id in self.aggregate_trust if self.is_trusted(node_id)
            ]

    def get_trust_threshold(self) -> float:
        """
        Get the current trust threshold.

        Returns:
            Trust threshold value
        """
        if not self.adaptive_threshold:
            return self.trust_threshold

        # Implement adaptive threshold based on Byzantine density
        with self.lock:
            if not self.aggregate_trust:
                return self.trust_threshold

            # Calculate average trust
            avg_trust = sum(
                self.get_trust_value(node_id) for node_id in self.aggregate_trust
            ) / len(self.aggregate_trust)

            # Adjust threshold based on average trust
            # Lower average trust means higher Byzantine density, so increase threshold
            adjustment = (0.5 - avg_trust) * 0.5  # Scale adjustment

            # Ensure threshold stays in reasonable range
            return max(0.4, min(0.8, self.trust_threshold + adjustment))

    def get_sybil_suspects(self) -> Set[str]:
        """
        Get set of suspected Sybil nodes.

        Returns:
            Set of suspected Sybil node identifiers
        """
        with self.lock:
            return self.sybil_suspects.copy()

    def _gossip_loop(self):
        """
        Background thread for periodic trust gossip.
        """
        while self.running:
            try:
                self._perform_gossip()
                time.sleep(self.gossip_interval)
            except Exception as e:
                print(f"Error in trust gossip: {e}")
                time.sleep(1.0)  # Shorter interval on error

    def _expiration_loop(self):
        """
        Background thread for trust expiration.
        """
        while self.running:
            try:
                self._expire_old_trust()
                time.sleep(10.0)  # Check every 10 seconds
            except Exception as e:
                print(f"Error in trust expiration: {e}")
                time.sleep(1.0)  # Shorter interval on error

    def _perform_gossip(self):
        """
        Perform trust gossip to peers.
        """
        with self.lock:
            # Get list of peers (in a real implementation, this would be from the network layer)
            peers = list(self.aggregate_trust.keys())

            if not peers:
                return

            # Select random subset of peers to gossip to
            gossip_peers = random.sample(peers, min(self.gossip_fanout, len(peers)))

            # For each peer, send our direct trust observations with decremented TTL
            for peer in gossip_peers:
                for node_id, trust_vector in self.direct_trust.items():
                    # In a real implementation, this would send a network message
                    # For simulation, we directly call add_indirect_trust
                    # This assumes the peer is also an instance of TrustPropagation
                    # peer_instance.add_indirect_trust(node_id, self_id, trust_vector, self.ttl_max - 1)
                    pass

    def _expire_old_trust(self):
        """
        Expire old trust information.
        """
        current_time = time.time()

        with self.lock:
            # Expire indirect trust
            for node_id in list(self.indirect_trust.keys()):
                for observer_id in list(self.indirect_trust[node_id].keys()):
                    _, timestamp, _ = self.indirect_trust[node_id][observer_id]
                    if current_time - timestamp > self.expiration_time:
                        del self.indirect_trust[node_id][observer_id]

                # Remove empty entries
                if not self.indirect_trust[node_id]:
                    del self.indirect_trust[node_id]

            # Expire direct trust (reduce confidence over time)
            for node_id in list(self.direct_trust.keys()):
                last_update_time = self.last_update.get(node_id, 0)
                if current_time - last_update_time > self.expiration_time:
                    # Gradually move trust towards neutral (0.5)
                    for dimension in range(TrustVector.NUM_DIMENSIONS):
                        current = self.direct_trust[node_id].get_dimension(dimension)
                        self.direct_trust[node_id].set_dimension(
                            dimension, current * 0.9 + 0.5 * 0.1
                        )

            # Update aggregate trust for affected nodes
            for node_id in set(
                list(self.direct_trust.keys()) + list(self.indirect_trust.keys())
            ):
                self._update_aggregate_trust(node_id)

    def _update_aggregate_trust(self, node_id: str):
        """
        Update aggregate trust for a node.

        Args:
            node_id: Node identifier
        """
        # Start with direct trust if available
        if node_id in self.direct_trust:
            aggregate = self.direct_trust[node_id]
            direct_weight = 0.7  # Direct observations have higher weight
        else:
            aggregate = TrustVector()  # Default to neutral trust
            direct_weight = 0.0

        # Add indirect trust
        if node_id in self.indirect_trust and self.indirect_trust[node_id]:
            # Calculate weighted average of indirect trust
            indirect_vectors = []
            indirect_weights = []

            for observer_id, (trust_vector, _, ttl) in self.indirect_trust[
                node_id
            ].items():
                # Weight by observer's trust and TTL
                observer_trust = self.get_trust_value(observer_id)
                ttl_factor = ttl / self.ttl_max
                weight = observer_trust * ttl_factor

                indirect_vectors.append(trust_vector)
                indirect_weights.append(weight)

            # Normalize weights
            total_weight = sum(indirect_weights)
            if total_weight > 0:
                normalized_weights = [w / total_weight for w in indirect_weights]
            else:
                normalized_weights = [1.0 / len(indirect_weights)] * len(
                    indirect_weights
                )

            # Combine indirect trust vectors
            indirect_aggregate = TrustVector()
            for i, vector in enumerate(indirect_vectors):
                indirect_aggregate = indirect_aggregate.combine(
                    vector, normalized_weights[i]
                )

            # Combine direct and indirect trust
            if direct_weight > 0:
                aggregate = aggregate.combine(indirect_aggregate, 1.0 - direct_weight)
            else:
                aggregate = indirect_aggregate

        # Update aggregate trust
        self.aggregate_trust[node_id] = aggregate

    def _update_trust_graph(self, node_id: str):
        """
        Update the trust graph for Sybil detection.

        Args:
            node_id: Node identifier
        """
        # Clear existing edges for this node
        self.trust_graph[node_id] = set()

        # Add edges to trusted nodes
        for other_id in self.aggregate_trust:
            if (
                other_id != node_id
                and self.get_trust_value(other_id) >= self.get_trust_threshold()
            ):
                self.trust_graph[node_id].add(other_id)

        # Run Sybil detection
        self._detect_sybils()

    def _detect_sybils(self):
        """
        Detect Sybil nodes using clustering-based detection.
        """
        # Reset Sybil suspects
        self.sybil_suspects = set()

        if len(self.trust_graph) < 3:
            return  # Not enough nodes for meaningful detection

        # Implement a simple clustering-based Sybil detection
        # In a real implementation, this would use more sophisticated algorithms

        # Calculate trust density for each node
        trust_density = {}
        for node_id in self.trust_graph:
            # Count how many other nodes trust this node
            trusted_by = sum(
                1
                for other_id in self.trust_graph
                if node_id in self.trust_graph[other_id]
            )
            trust_density[node_id] = trusted_by / max(1, len(self.trust_graph) - 1)

        # Identify nodes with suspiciously low trust density
        avg_density = sum(trust_density.values()) / max(1, len(trust_density))
        for node_id, density in trust_density.items():
            if density < avg_density * self.sybil_threshold:
                self.sybil_suspects.add(node_id)
