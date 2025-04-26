#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Shard Clustering
This module implements hierarchical shard clustering and management.
"""

import os
import time
import threading
import random
import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, Union
import logging
import networkx as nx
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("qtrust_clustering.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class ShardCluster:
    """
    Represents a shard cluster in the blockchain network.
    """

    def __init__(self, cluster_id: str, capacity: int = 10):
        """
        Initialize a shard cluster.

        Args:
            cluster_id: Unique identifier for the cluster
            capacity: Maximum number of nodes in the cluster
        """
        self.cluster_id = cluster_id
        self.capacity = capacity
        self.nodes = set()
        self.active = False
        self.creation_time = time.time()
        self.last_rebalance_time = self.creation_time

        # Metrics
        self.metrics = {
            "transaction_count": 0,
            "cross_shard_ratio": 0.0,
            "avg_latency": 0.0,
            "throughput": 0.0,
            "max_throughput": 10000.0,
            "max_latency": 5.0,
        }

        # Load factor weights
        self.load_weights = {"throughput": 0.5, "cross_shard": 0.3, "latency": 0.2}

        logger.info(f"Created ShardCluster {cluster_id} with capacity {capacity}")

    def add_node(self, node_id: str) -> bool:
        """
        Add a node to the cluster.

        Args:
            node_id: Node identifier

        Returns:
            True if node was added or already exists, False if cluster is full
        """
        if node_id in self.nodes:
            return True

        if len(self.nodes) >= self.capacity:
            return False

        self.nodes.add(node_id)
        logger.info(f"Added node {node_id} to cluster {self.cluster_id}")
        return True

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the cluster.

        Args:
            node_id: Node identifier

        Returns:
            True if node was removed, False if node was not in cluster
        """
        if node_id not in self.nodes:
            return False

        self.nodes.remove(node_id)
        logger.info(f"Removed node {node_id} from cluster {self.cluster_id}")
        return True

    def get_node_count(self) -> int:
        """
        Get the number of nodes in the cluster.

        Returns:
            Number of nodes
        """
        return len(self.nodes)

    def is_active(self) -> bool:
        """
        Check if the cluster is active.

        Returns:
            True if active, False otherwise
        """
        return self.active

    def activate(self):
        """
        Activate the cluster.
        """
        self.active = True
        logger.info(f"Activated cluster {self.cluster_id}")

    def deactivate(self):
        """
        Deactivate the cluster.
        """
        self.active = False
        logger.info(f"Deactivated cluster {self.cluster_id}")

    def update_metrics(self, metrics: Dict[str, Any]):
        """
        Update cluster metrics.

        Args:
            metrics: Dictionary of metrics to update
        """
        for key, value in metrics.items():
            self.metrics[key] = value

        logger.info(f"Updated metrics for cluster {self.cluster_id}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get cluster metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()

    def get_load_factor(self) -> float:
        """
        Calculate the load factor of the cluster.

        Returns:
            Load factor (0.0 to 1.0)
        """
        # Normalize metrics
        throughput_ratio = min(
            1.0, self.metrics["throughput"] / self.metrics["max_throughput"]
        )
        cross_shard_ratio = self.metrics["cross_shard_ratio"]
        latency_ratio = min(
            1.0, self.metrics["avg_latency"] / self.metrics["max_latency"]
        )

        # Calculate weighted load factor
        load_factor = (
            self.load_weights["throughput"] * throughput_ratio
            + self.load_weights["cross_shard"] * cross_shard_ratio
            + self.load_weights["latency"] * latency_ratio
        )

        return load_factor


class ShardManager:
    """
    Manages shard clusters in the blockchain network.
    """

    def __init__(
        self,
        initial_shards: int = 3,
        nodes_per_shard: int = 10,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize the shard manager.

        Args:
            initial_shards: Number of initial shards
            nodes_per_shard: Number of nodes per shard
            config: Configuration dictionary
        """
        self.config = config or {}
        self.initial_shards = initial_shards
        self.nodes_per_shard = nodes_per_shard

        # Clusters
        self.clusters = {}  # cluster_id -> ShardCluster

        # Node assignments
        self.node_assignments = {}  # node_id -> cluster_id

        # Rebalance settings
        self.rebalance_threshold = self.config.get("rebalance_threshold", 0.3)
        self.rebalance_interval = self.config.get(
            "rebalance_interval", 300.0
        )  # 5 minutes

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.rebalance_thread = None

        logger.info(f"Initialized ShardManager with {initial_shards} initial shards")

    def initialize(self):
        """
        Initialize the shard manager.
        """
        with self.lock:
            # Create initial clusters
            for i in range(self.initial_shards):
                cluster_id = f"shard_{i}"
                cluster = ShardCluster(cluster_id, capacity=self.nodes_per_shard)
                cluster.activate()
                self.clusters[cluster_id] = cluster

            # Start rebalance thread
            self.running = True
            self.rebalance_thread = threading.Thread(target=self._rebalance_loop)
            self.rebalance_thread.daemon = True
            self.rebalance_thread.start()

            logger.info(f"Initialized ShardManager with {len(self.clusters)} clusters")

    def shutdown(self):
        """
        Shut down the shard manager.
        """
        self.running = False

        if self.rebalance_thread:
            self.rebalance_thread.join(timeout=5.0)
            self.rebalance_thread = None

        logger.info("Shut down ShardManager")

    def add_node(self, node_id: str) -> Optional[str]:
        """
        Add a node to the network.

        Args:
            node_id: Node identifier

        Returns:
            Cluster ID the node was assigned to, or None if assignment failed
        """
        with self.lock:
            # Check if node is already assigned
            if node_id in self.node_assignments:
                return self.node_assignments[node_id]

            # Find least loaded cluster with available capacity
            available_clusters = [
                cluster
                for cluster in self.clusters.values()
                if cluster.get_node_count() < cluster.capacity
            ]

            if available_clusters:
                # Sort by load factor
                available_clusters.sort(key=lambda c: c.get_load_factor())

                # Assign to least loaded cluster
                cluster = available_clusters[0]
                if cluster.add_node(node_id):
                    self.node_assignments[node_id] = cluster.cluster_id
                    return cluster.cluster_id

            # No available clusters, create a new one
            new_cluster_id = f"shard_{len(self.clusters)}"
            new_cluster = ShardCluster(new_cluster_id, capacity=self.nodes_per_shard)
            new_cluster.activate()

            if new_cluster.add_node(node_id):
                self.clusters[new_cluster_id] = new_cluster
                self.node_assignments[node_id] = new_cluster_id
                logger.info(f"Created new cluster {new_cluster_id} for node {node_id}")
                return new_cluster_id

            return None

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the network.

        Args:
            node_id: Node identifier

        Returns:
            True if node was removed, False otherwise
        """
        with self.lock:
            # Check if node is assigned
            if node_id not in self.node_assignments:
                return False

            # Get cluster
            cluster_id = self.node_assignments[node_id]
            cluster = self.clusters.get(cluster_id)

            if not cluster:
                # Cluster doesn't exist, just remove assignment
                del self.node_assignments[node_id]
                return True

            # Remove from cluster
            if cluster.remove_node(node_id):
                del self.node_assignments[node_id]

                # If cluster is empty, consider removing it
                if (
                    cluster.get_node_count() == 0
                    and len(self.clusters) > self.initial_shards
                ):
                    del self.clusters[cluster_id]
                    logger.info(f"Removed empty cluster {cluster_id}")

                return True

            return False

    def get_node_assignment(self, node_id: str) -> Optional[str]:
        """
        Get the cluster assignment for a node.

        Args:
            node_id: Node identifier

        Returns:
            Cluster ID the node is assigned to, or None if not assigned
        """
        with self.lock:
            return self.node_assignments.get(node_id)

    def get_cluster(self, cluster_id: str) -> Optional[ShardCluster]:
        """
        Get a cluster by ID.

        Args:
            cluster_id: Cluster identifier

        Returns:
            ShardCluster instance, or None if not found
        """
        with self.lock:
            return self.clusters.get(cluster_id)

    def get_all_clusters(self) -> List[ShardCluster]:
        """
        Get all clusters.

        Returns:
            List of ShardCluster instances
        """
        with self.lock:
            return list(self.clusters.values())

    def update_cluster_metrics(self, cluster_id: str, metrics: Dict[str, Any]) -> bool:
        """
        Update metrics for a cluster.

        Args:
            cluster_id: Cluster identifier
            metrics: Dictionary of metrics to update

        Returns:
            True if metrics were updated, False if cluster not found
        """
        with self.lock:
            cluster = self.clusters.get(cluster_id)

            if not cluster:
                return False

            cluster.update_metrics(metrics)
            return True

    def get_network_state(self) -> Dict[str, Any]:
        """
        Get the current state of the network.

        Returns:
            Dictionary with network state information
        """
        with self.lock:
            # Count nodes
            node_count = len(self.node_assignments)

            # Get cluster information
            cluster_info = []
            for cluster_id, cluster in self.clusters.items():
                cluster_info.append(
                    {
                        "id": cluster_id,
                        "node_count": cluster.get_node_count(),
                        "active": cluster.is_active(),
                        "load_factor": cluster.get_load_factor(),
                        "metrics": cluster.get_metrics(),
                    }
                )

            return {
                "node_count": node_count,
                "cluster_count": len(self.clusters),
                "clusters": cluster_info,
                "node_assignments": self.node_assignments.copy(),
            }

    def _rebalance_loop(self):
        """
        Background thread for periodic rebalancing.
        """
        last_rebalance = time.time()

        while self.running:
            try:
                current_time = time.time()

                # Check if it's time to rebalance
                if current_time - last_rebalance >= self.rebalance_interval:
                    # Perform rebalance
                    result = self.rebalance_clusters()
                    result.wait(timeout=30.0)

                    if result.success:
                        last_rebalance = current_time

                time.sleep(10.0)

            except Exception as e:
                logger.error(f"Error in rebalance loop: {e}")
                time.sleep(60.0)  # Longer interval on error

    def rebalance_clusters(self) -> "RebalanceResult":
        """
        Rebalance clusters based on load factors.

        Returns:
            RebalanceResult object
        """
        result = RebalanceResult()

        # Start rebalance in background thread
        thread = threading.Thread(target=self._do_rebalance, args=(result,))
        thread.daemon = True
        thread.start()

        return result

    def _do_rebalance(self, result: "RebalanceResult"):
        """
        Perform cluster rebalancing.

        Args:
            result: RebalanceResult to update with results
        """
        try:
            with self.lock:
                # Get clusters
                clusters = list(self.clusters.values())

                if len(clusters) < 2:
                    # Not enough clusters to rebalance
                    result.set_result(
                        False,
                        {
                            "rebalanced": False,
                            "reason": "Not enough clusters",
                            "moves": [],
                        },
                    )
                    return

                # Calculate load factors
                load_factors = [
                    (cluster, cluster.get_load_factor()) for cluster in clusters
                ]

                # Sort by load factor
                load_factors.sort(key=lambda x: x[1])

                # Get most and least loaded clusters
                least_loaded = load_factors[0][0]
                most_loaded = load_factors[-1][0]

                least_load = load_factors[0][1]
                most_load = load_factors[-1][1]

                # Check if rebalance is needed
                if most_load - least_load < self.rebalance_threshold:
                    # Load difference is below threshold
                    result.set_result(
                        True,
                        {
                            "rebalanced": False,
                            "reason": "Load difference below threshold",
                            "moves": [],
                        },
                    )
                    return

                # Check if least loaded cluster has capacity
                if least_loaded.get_node_count() >= least_loaded.capacity:
                    # No capacity to move nodes
                    result.set_result(
                        True,
                        {
                            "rebalanced": False,
                            "reason": "No capacity in least loaded cluster",
                            "moves": [],
                        },
                    )
                    return

                # Calculate how many nodes to move
                available_capacity = (
                    least_loaded.capacity - least_loaded.get_node_count()
                )
                nodes_to_move = min(
                    available_capacity, most_loaded.get_node_count() // 2
                )

                if nodes_to_move == 0:
                    # No nodes to move
                    result.set_result(
                        True,
                        {
                            "rebalanced": False,
                            "reason": "No nodes to move",
                            "moves": [],
                        },
                    )
                    return

                # Select nodes to move
                nodes_in_most_loaded = list(most_loaded.nodes)
                nodes_to_move = random.sample(nodes_in_most_loaded, nodes_to_move)

                # Move nodes
                moves = []
                for node_id in nodes_to_move:
                    # Remove from most loaded
                    if most_loaded.remove_node(node_id):
                        # Add to least loaded
                        if least_loaded.add_node(node_id):
                            # Update assignment
                            self.node_assignments[node_id] = least_loaded.cluster_id

                            # Record move
                            moves.append(
                                {
                                    "node_id": node_id,
                                    "from": most_loaded.cluster_id,
                                    "to": least_loaded.cluster_id,
                                }
                            )

                # Update rebalance time
                most_loaded.last_rebalance_time = time.time()
                least_loaded.last_rebalance_time = time.time()

                # Set result
                result.set_result(
                    True,
                    {
                        "rebalanced": len(moves) > 0,
                        "moves": moves,
                        "from_cluster": most_loaded.cluster_id,
                        "to_cluster": least_loaded.cluster_id,
                        "from_load": most_load,
                        "to_load": least_load,
                    },
                )

                logger.info(
                    f"Rebalanced clusters: moved {len(moves)} nodes from {most_loaded.cluster_id} to {least_loaded.cluster_id}"
                )

        except Exception as e:
            logger.error(f"Error in rebalance: {e}")
            result.set_result(
                False, {"rebalanced": False, "reason": f"Error: {str(e)}", "moves": []}
            )


class RebalanceResult:
    """
    Result of a cluster rebalance operation.
    """

    def __init__(self):
        """
        Initialize rebalance result.
        """
        self.event = threading.Event()
        self.success = False
        self.result = None

    def set_result(self, success: bool, result: Dict[str, Any]):
        """
        Set the result of the rebalance operation.

        Args:
            success: Whether the operation was successful
            result: Result data
        """
        self.success = success
        self.result = result
        self.event.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the rebalance operation to complete.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if the operation completed, False if timeout occurred
        """
        return self.event.wait(timeout)


class HierarchicalShardCluster:
    """
    Implements hierarchical shard clustering for optimized cross-shard communication.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize hierarchical shard cluster.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Default configuration
        self.max_clusters = self.config.get("max_clusters", 100)
        self.min_similarity = self.config.get("min_similarity", 0.7)
        self.update_interval = self.config.get("update_interval", 60.0)  # seconds

        # Shard data
        self.shards = {}  # shard_id -> features
        self.shard_clusters = {}  # cluster_id -> set(shard_ids)
        self.shard_to_cluster = {}  # shard_id -> cluster_id

        # Communication graph
        self.communication_graph = nx.Graph()

        # Transaction history
        self.transaction_history = defaultdict(
            float
        )  # (shard_id1, shard_id2) -> weight

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.update_thread = None

        logger.info("Initialized HierarchicalShardCluster")

    def start(self):
        """
        Start hierarchical shard clustering.
        """
        if self.running:
            return

        self.running = True

        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

        logger.info("Started HierarchicalShardCluster")

    def stop(self):
        """
        Stop hierarchical shard clustering.
        """
        self.running = False

        if self.update_thread:
            self.update_thread.join(timeout=5.0)
            self.update_thread = None

        logger.info("Stopped HierarchicalShardCluster")

    def add_shard(self, shard_id: str, features: np.ndarray):
        """
        Add a shard to the cluster.

        Args:
            shard_id: Shard identifier
            features: Feature vector for the shard
        """
        with self.lock:
            self.shards[shard_id] = features
            self.communication_graph.add_node(shard_id)

            # Assign to initial cluster
            cluster_id = f"cluster_{len(self.shard_clusters)}"

            if cluster_id not in self.shard_clusters:
                self.shard_clusters[cluster_id] = set()

            self.shard_clusters[cluster_id].add(shard_id)
            self.shard_to_cluster[shard_id] = cluster_id

            logger.info(f"Added shard {shard_id} to cluster {cluster_id}")

    def remove_shard(self, shard_id: str):
        """
        Remove a shard from the cluster.

        Args:
            shard_id: Shard identifier
        """
        with self.lock:
            if shard_id not in self.shards:
                return

            # Remove from features
            del self.shards[shard_id]

            # Remove from communication graph
            if self.communication_graph.has_node(shard_id):
                self.communication_graph.remove_node(shard_id)

            # Remove from cluster
            cluster_id = self.shard_to_cluster.get(shard_id)
            if cluster_id and cluster_id in self.shard_clusters:
                self.shard_clusters[cluster_id].discard(shard_id)

                # Remove empty cluster
                if not self.shard_clusters[cluster_id]:
                    del self.shard_clusters[cluster_id]

            # Remove from mapping
            if shard_id in self.shard_to_cluster:
                del self.shard_to_cluster[shard_id]

            # Remove from transaction history
            keys_to_remove = []
            for key in self.transaction_history:
                if shard_id in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self.transaction_history[key]

            logger.info(f"Removed shard {shard_id}")

    def record_transaction(self, shard_id1: str, shard_id2: str, weight: float = 1.0):
        """
        Record a transaction between two shards.

        Args:
            shard_id1: First shard identifier
            shard_id2: Second shard identifier
            weight: Transaction weight
        """
        if shard_id1 == shard_id2:
            return

        with self.lock:
            # Ensure consistent ordering
            if shard_id1 > shard_id2:
                shard_id1, shard_id2 = shard_id2, shard_id1

            # Update transaction history
            key = (shard_id1, shard_id2)
            self.transaction_history[key] += weight

            # Update communication graph
            if self.communication_graph.has_edge(shard_id1, shard_id2):
                # Update edge weight
                current_weight = self.communication_graph[shard_id1][shard_id2][
                    "weight"
                ]
                new_weight = current_weight + weight
                self.communication_graph[shard_id1][shard_id2]["weight"] = new_weight
            else:
                # Add new edge
                self.communication_graph.add_edge(shard_id1, shard_id2, weight=weight)

    def update_communication_metrics(
        self, shard_id1: str, shard_id2: str, latency: float, utilization: float
    ):
        """
        Update communication metrics between two shards.

        Args:
            shard_id1: First shard identifier
            shard_id2: Second shard identifier
            latency: Communication latency
            utilization: Link utilization
        """
        if shard_id1 == shard_id2:
            return

        with self.lock:
            # Ensure consistent ordering
            if shard_id1 > shard_id2:
                shard_id1, shard_id2 = shard_id2, shard_id1

            # Calculate weight (inverse of latency)
            weight = 1.0 / max(0.1, latency)

            # Update communication graph
            if self.communication_graph.has_edge(shard_id1, shard_id2):
                # Update edge attributes
                self.communication_graph[shard_id1][shard_id2]["latency"] = latency
                self.communication_graph[shard_id1][shard_id2][
                    "utilization"
                ] = utilization
                self.communication_graph[shard_id1][shard_id2]["weight"] = weight
            else:
                # Add new edge
                self.communication_graph.add_edge(
                    shard_id1,
                    shard_id2,
                    weight=weight,
                    latency=latency,
                    utilization=utilization,
                )

    def get_optimal_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """
        Get the optimal path between two shards.

        Args:
            source_id: Source shard identifier
            target_id: Target shard identifier

        Returns:
            List of shard identifiers representing the path, or None if no path exists
        """
        with self.lock:
            if not self.communication_graph.has_node(
                source_id
            ) or not self.communication_graph.has_node(target_id):
                return None

            try:
                # Find shortest path
                path = nx.shortest_path(
                    self.communication_graph,
                    source=source_id,
                    target=target_id,
                    weight="weight",
                    method="dijkstra",
                )

                return path

            except nx.NetworkXNoPath:
                # No path exists
                return None

    def _update_loop(self):
        """
        Background thread for periodic clustering updates.
        """
        while self.running:
            try:
                # Update clusters
                self._update_clusters()

                # Sleep until next update
                time.sleep(self.update_interval)

            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                time.sleep(10.0)  # Shorter interval on error

    def _update_clusters(self):
        """
        Update shard clusters based on communication patterns.
        """
        with self.lock:
            # Check if we have enough shards
            if len(self.shards) < 2:
                return

            # Build similarity matrix
            shard_ids = list(self.shards.keys())
            n_shards = len(shard_ids)

            similarity = np.zeros((n_shards, n_shards))

            for i in range(n_shards):
                for j in range(i + 1, n_shards):
                    shard_id1 = shard_ids[i]
                    shard_id2 = shard_ids[j]

                    # Calculate similarity based on communication
                    if shard_id1 > shard_id2:
                        shard_id1, shard_id2 = shard_id2, shard_id1

                    key = (shard_id1, shard_id2)
                    comm_weight = self.transaction_history.get(key, 0.0)

                    # Calculate feature similarity
                    feature1 = self.shards[shard_id1]
                    feature2 = self.shards[shard_id2]

                    feature_sim = self._calculate_similarity(feature1, feature2)

                    # Combined similarity (70% communication, 30% features)
                    combined_sim = (
                        0.7 * min(1.0, comm_weight / 10.0) + 0.3 * feature_sim
                    )

                    similarity[i, j] = combined_sim
                    similarity[j, i] = combined_sim

            # Perform clustering
            clusters = self._cluster_shards(shard_ids, similarity)

            # Update cluster assignments
            self.shard_clusters = {}
            self.shard_to_cluster = {}

            for cluster_id, shard_ids in clusters.items():
                self.shard_clusters[cluster_id] = set(shard_ids)

                for shard_id in shard_ids:
                    self.shard_to_cluster[shard_id] = cluster_id

            logger.info(
                f"Updated clusters: {len(self.shard_clusters)} clusters for {len(self.shards)} shards"
            )

    def _calculate_similarity(
        self, feature1: np.ndarray, feature2: np.ndarray
    ) -> float:
        """
        Calculate similarity between two feature vectors.

        Args:
            feature1: First feature vector
            feature2: Second feature vector

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Cosine similarity
        dot_product = np.dot(feature1, feature2)
        norm1 = np.linalg.norm(feature1)
        norm2 = np.linalg.norm(feature2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _cluster_shards(
        self, shard_ids: List[str], similarity: np.ndarray
    ) -> Dict[str, List[str]]:
        """
        Cluster shards based on similarity matrix.

        Args:
            shard_ids: List of shard identifiers
            similarity: Similarity matrix

        Returns:
            Dictionary mapping cluster IDs to lists of shard IDs
        """
        n_shards = len(shard_ids)

        # Initialize each shard as its own cluster
        clusters = {f"cluster_{i}": [shard_id] for i, shard_id in enumerate(shard_ids)}

        # Hierarchical clustering
        while len(clusters) > 1 and len(clusters) > self.max_clusters:
            # Find most similar clusters
            max_sim = -1.0
            merge_pair = None

            cluster_ids = list(clusters.keys())

            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    cluster1 = clusters[cluster_ids[i]]
                    cluster2 = clusters[cluster_ids[j]]

                    # Calculate average similarity between clusters
                    total_sim = 0.0
                    count = 0

                    for shard1 in cluster1:
                        idx1 = shard_ids.index(shard1)

                        for shard2 in cluster2:
                            idx2 = shard_ids.index(shard2)

                            total_sim += similarity[idx1, idx2]
                            count += 1

                    avg_sim = total_sim / max(1, count)

                    if avg_sim > max_sim:
                        max_sim = avg_sim
                        merge_pair = (cluster_ids[i], cluster_ids[j])

            # Check if similarity is above threshold
            if max_sim < self.min_similarity:
                break

            # Merge clusters
            if merge_pair:
                cluster1, cluster2 = merge_pair

                # Create new cluster
                new_cluster_id = f"cluster_{int(time.time() * 1000) % 1000000}"
                clusters[new_cluster_id] = clusters[cluster1] + clusters[cluster2]

                # Remove old clusters
                del clusters[cluster1]
                del clusters[cluster2]

        return clusters
