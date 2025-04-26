#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Topology Manager
This module implements network topology optimization for efficient routing.
"""

import time
import random
import heapq
import threading
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable


class TopologyManager:
    """
    Manages network topology for optimized routing based on latency measurements.
    Implements topology-aware routing to reduce communication overhead.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the topology manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Default configuration
        self.update_interval = self.config.get(
            "topology_update_interval", 60.0
        )  # seconds
        self.ping_timeout = self.config.get("ping_timeout", 2.0)  # seconds
        self.ping_retries = self.config.get("ping_retries", 3)
        self.latency_weight = self.config.get("latency_weight", 0.7)
        self.bandwidth_weight = self.config.get("bandwidth_weight", 0.2)
        self.reliability_weight = self.config.get("reliability_weight", 0.1)
        self.max_hops = self.config.get("max_hops", 5)
        self.cluster_size = self.config.get("cluster_size", 8)

        # Topology data structures
        self.nodes = {}  # node_id -> node_info
        self.latency_matrix = {}  # (node_id1, node_id2) -> latency
        self.bandwidth_matrix = {}  # (node_id1, node_id2) -> bandwidth
        self.reliability_matrix = {}  # (node_id1, node_id2) -> reliability
        self.routing_table = {}  # (source, target) -> next_hop
        self.clusters = {}  # cluster_id -> [node_ids]
        self.cluster_coordinators = {}  # cluster_id -> coordinator_node_id
        self.node_to_cluster = {}  # node_id -> cluster_id

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.update_thread = None

    def start(self):
        """
        Start the topology manager.
        """
        if self.running:
            return

        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

    def stop(self):
        """
        Stop the topology manager.
        """
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5.0)
            self.update_thread = None

    def _update_loop(self):
        """
        Background thread for periodic topology updates.
        """
        while self.running:
            try:
                self.update_topology()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in topology update: {e}")
                time.sleep(5.0)  # Shorter interval on error

    def add_node(self, node_id: str, node_info: Dict[str, Any]):
        """
        Add a node to the topology.

        Args:
            node_id: Unique identifier for the node
            node_info: Node information including location, capabilities, etc.
        """
        with self.lock:
            self.nodes[node_id] = node_info

            # Initialize matrices for the new node
            for existing_id in self.nodes:
                if existing_id != node_id:
                    self.latency_matrix[(node_id, existing_id)] = float("inf")
                    self.latency_matrix[(existing_id, node_id)] = float("inf")
                    self.bandwidth_matrix[(node_id, existing_id)] = 0.0
                    self.bandwidth_matrix[(existing_id, node_id)] = 0.0
                    self.reliability_matrix[(node_id, existing_id)] = 0.0
                    self.reliability_matrix[(existing_id, node_id)] = 0.0

            # Assign to a cluster
            self._assign_node_to_cluster(node_id)

            # Update routing table
            self._update_routing_table()

    def remove_node(self, node_id: str):
        """
        Remove a node from the topology.

        Args:
            node_id: Unique identifier for the node
        """
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]

                # Remove from matrices
                keys_to_remove = []
                for key in self.latency_matrix:
                    if node_id in key:
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    if key in self.latency_matrix:
                        del self.latency_matrix[key]
                    if key in self.bandwidth_matrix:
                        del self.bandwidth_matrix[key]
                    if key in self.reliability_matrix:
                        del self.reliability_matrix[key]

                # Remove from routing table
                keys_to_remove = []
                for key in self.routing_table:
                    if node_id in key or self.routing_table[key] == node_id:
                        keys_to_remove.append(key)

                for key in keys_to_remove:
                    if key in self.routing_table:
                        del self.routing_table[key]

                # Remove from cluster
                if node_id in self.node_to_cluster:
                    cluster_id = self.node_to_cluster[node_id]
                    if (
                        cluster_id in self.clusters
                        and node_id in self.clusters[cluster_id]
                    ):
                        self.clusters[cluster_id].remove(node_id)

                    # If this was a coordinator, elect a new one
                    if self.cluster_coordinators.get(cluster_id) == node_id:
                        self._elect_cluster_coordinator(cluster_id)

                    del self.node_to_cluster[node_id]

                # Update routing table
                self._update_routing_table()

    def update_link_metrics(
        self,
        node_id1: str,
        node_id2: str,
        latency: float,
        bandwidth: float,
        reliability: float,
    ):
        """
        Update metrics for a link between two nodes.

        Args:
            node_id1: First node ID
            node_id2: Second node ID
            latency: Measured latency in milliseconds
            bandwidth: Measured bandwidth in Mbps
            reliability: Reliability score (0-1)
        """
        with self.lock:
            if node_id1 in self.nodes and node_id2 in self.nodes:
                self.latency_matrix[(node_id1, node_id2)] = latency
                self.latency_matrix[(node_id2, node_id1)] = latency
                self.bandwidth_matrix[(node_id1, node_id2)] = bandwidth
                self.bandwidth_matrix[(node_id2, node_id1)] = bandwidth
                self.reliability_matrix[(node_id1, node_id2)] = reliability
                self.reliability_matrix[(node_id2, node_id1)] = reliability

                # Update routing table if significant change
                self._update_routing_table()

    def get_next_hop(self, source: str, target: str) -> Optional[str]:
        """
        Get the next hop for routing from source to target.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Next hop node ID or None if no route exists
        """
        with self.lock:
            return self.routing_table.get((source, target))

    def get_path(self, source: str, target: str) -> List[str]:
        """
        Get the full path from source to target.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            List of node IDs representing the path
        """
        with self.lock:
            if source == target:
                return [source]

            path = [source]
            current = source

            # Prevent infinite loops
            visited = {source}

            for _ in range(self.max_hops):
                next_hop = self.routing_table.get((current, target))
                if not next_hop:
                    break

                if next_hop in visited:
                    # Loop detected
                    break

                path.append(next_hop)
                visited.add(next_hop)

                if next_hop == target:
                    return path

                current = next_hop

            # Could not find complete path
            return []

    def get_cluster_for_node(self, node_id: str) -> Optional[str]:
        """
        Get the cluster ID for a node.

        Args:
            node_id: Node ID

        Returns:
            Cluster ID or None if node not found
        """
        with self.lock:
            return self.node_to_cluster.get(node_id)

    def get_cluster_coordinator(self, cluster_id: str) -> Optional[str]:
        """
        Get the coordinator node for a cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            Coordinator node ID or None if cluster not found
        """
        with self.lock:
            return self.cluster_coordinators.get(cluster_id)

    def get_nodes_in_cluster(self, cluster_id: str) -> List[str]:
        """
        Get all nodes in a cluster.

        Args:
            cluster_id: Cluster ID

        Returns:
            List of node IDs in the cluster
        """
        with self.lock:
            return self.clusters.get(cluster_id, []).copy()

    def get_all_clusters(self) -> Dict[str, List[str]]:
        """
        Get all clusters.

        Returns:
            Dictionary mapping cluster IDs to lists of node IDs
        """
        with self.lock:
            return {k: v.copy() for k, v in self.clusters.items()}

    def update_topology(self):
        """
        Update the entire topology based on current measurements.
        """
        with self.lock:
            # Measure latency between nodes
            self._measure_latency()

            # Measure bandwidth between nodes
            self._measure_bandwidth()

            # Measure reliability between nodes
            self._measure_reliability()

            # Update clusters
            self._update_clusters()

            # Update routing table
            self._update_routing_table()

    def _measure_latency(self):
        """
        Measure latency between nodes.
        In a real implementation, this would use actual network measurements.
        Here we simulate it with random values.
        """
        # In a real implementation, this would perform actual ping measurements
        # For simulation, we'll use random values with some structure

        # Get all node pairs
        node_ids = list(self.nodes.keys())

        for i, node_id1 in enumerate(node_ids):
            for node_id2 in node_ids[i + 1 :]:
                # Check if nodes are in the same cluster
                same_cluster = self.node_to_cluster.get(
                    node_id1
                ) == self.node_to_cluster.get(node_id2)

                # Base latency depends on whether nodes are in the same cluster
                base_latency = 10.0 if same_cluster else 50.0

                # Add some randomness
                latency = base_latency * (1.0 + 0.2 * random.random())

                # Update latency matrix
                self.latency_matrix[(node_id1, node_id2)] = latency
                self.latency_matrix[(node_id2, node_id1)] = latency

    def _measure_bandwidth(self):
        """
        Measure bandwidth between nodes.
        In a real implementation, this would use actual network measurements.
        Here we simulate it with random values.
        """
        # In a real implementation, this would perform actual bandwidth tests
        # For simulation, we'll use random values with some structure

        # Get all node pairs
        node_ids = list(self.nodes.keys())

        for i, node_id1 in enumerate(node_ids):
            for node_id2 in node_ids[i + 1 :]:
                # Check if nodes are in the same cluster
                same_cluster = self.node_to_cluster.get(
                    node_id1
                ) == self.node_to_cluster.get(node_id2)

                # Base bandwidth depends on whether nodes are in the same cluster
                base_bandwidth = 1000.0 if same_cluster else 500.0

                # Add some randomness
                bandwidth = base_bandwidth * (1.0 + 0.2 * random.random())

                # Update bandwidth matrix
                self.bandwidth_matrix[(node_id1, node_id2)] = bandwidth
                self.bandwidth_matrix[(node_id2, node_id1)] = bandwidth

    def _measure_reliability(self):
        """
        Measure reliability between nodes.
        In a real implementation, this would use actual network measurements.
        Here we simulate it with random values.
        """
        # In a real implementation, this would analyze packet loss, etc.
        # For simulation, we'll use random values with some structure

        # Get all node pairs
        node_ids = list(self.nodes.keys())

        for i, node_id1 in enumerate(node_ids):
            for node_id2 in node_ids[i + 1 :]:
                # Check if nodes are in the same cluster
                same_cluster = self.node_to_cluster.get(
                    node_id1
                ) == self.node_to_cluster.get(node_id2)

                # Base reliability depends on whether nodes are in the same cluster
                base_reliability = 0.99 if same_cluster else 0.95

                # Add some randomness
                reliability = min(
                    1.0, base_reliability * (1.0 + 0.05 * random.random())
                )

                # Update reliability matrix
                self.reliability_matrix[(node_id1, node_id2)] = reliability
                self.reliability_matrix[(node_id2, node_id1)] = reliability

    def _update_clusters(self):
        """
        Update cluster assignments based on latency measurements.
        """
        # If we have very few nodes, just put them all in one cluster
        if len(self.nodes) <= self.cluster_size:
            cluster_id = "cluster_0"
            self.clusters = {cluster_id: list(self.nodes.keys())}
            self.node_to_cluster = {node_id: cluster_id for node_id in self.nodes}
            self._elect_cluster_coordinator(cluster_id)
            return

        # Use a clustering algorithm based on latency
        # Here we'll use a simple greedy approach

        # Start with empty clusters
        self.clusters = {}
        self.node_to_cluster = {}

        # Get all nodes
        node_ids = list(self.nodes.keys())

        # Sort node pairs by latency
        node_pairs = []
        for i, node_id1 in enumerate(node_ids):
            for node_id2 in node_ids[i + 1 :]:
                latency = self.latency_matrix.get((node_id1, node_id2), float("inf"))
                node_pairs.append((latency, node_id1, node_id2))

        node_pairs.sort()  # Sort by latency

        # Create clusters by connecting low-latency pairs
        cluster_counter = 0

        for _, node_id1, node_id2 in node_pairs:
            # If both nodes already have clusters, skip
            if node_id1 in self.node_to_cluster and node_id2 in self.node_to_cluster:
                continue

            # If one node has a cluster, try to add the other
            if node_id1 in self.node_to_cluster:
                cluster_id = self.node_to_cluster[node_id1]
                if len(self.clusters[cluster_id]) < self.cluster_size:
                    self.clusters[cluster_id].append(node_id2)
                    self.node_to_cluster[node_id2] = cluster_id
                continue

            if node_id2 in self.node_to_cluster:
                cluster_id = self.node_to_cluster[node_id2]
                if len(self.clusters[cluster_id]) < self.cluster_size:
                    self.clusters[cluster_id].append(node_id1)
                    self.node_to_cluster[node_id1] = cluster_id
                continue

            # Both nodes don't have clusters, create a new one
            cluster_id = f"cluster_{cluster_counter}"
            cluster_counter += 1
            self.clusters[cluster_id] = [node_id1, node_id2]
            self.node_to_cluster[node_id1] = cluster_id
            self.node_to_cluster[node_id2] = cluster_id

        # Assign any remaining nodes to the smallest clusters
        for node_id in node_ids:
            if node_id not in self.node_to_cluster:
                # Find the smallest cluster
                smallest_cluster = min(self.clusters.items(), key=lambda x: len(x[1]))
                cluster_id = smallest_cluster[0]
                self.clusters[cluster_id].append(node_id)
                self.node_to_cluster[node_id] = cluster_id

        # Elect coordinators for each cluster
        for cluster_id in self.clusters:
            self._elect_cluster_coordinator(cluster_id)

    def _assign_node_to_cluster(self, node_id: str):
        """
        Assign a node to the most appropriate cluster.

        Args:
            node_id: Node ID to assign
        """
        # If no clusters exist, create the first one
        if not self.clusters:
            cluster_id = "cluster_0"
            self.clusters[cluster_id] = [node_id]
            self.node_to_cluster[node_id] = cluster_id
            self.cluster_coordinators[cluster_id] = node_id
            return

        # Find the cluster with the lowest average latency to this node
        best_cluster = None
        best_latency = float("inf")

        for cluster_id, nodes in self.clusters.items():
            if len(nodes) >= self.cluster_size:
                continue  # Cluster is full

            # Calculate average latency to nodes in this cluster
            latencies = []
            for existing_node in nodes:
                latency = self.latency_matrix.get(
                    (node_id, existing_node), float("inf")
                )
                if latency < float("inf"):
                    latencies.append(latency)

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                if avg_latency < best_latency:
                    best_latency = avg_latency
                    best_cluster = cluster_id

        # If we found a suitable cluster, add the node to it
        if best_cluster:
            self.clusters[best_cluster].append(node_id)
            self.node_to_cluster[node_id] = best_cluster
            return

        # Otherwise, create a new cluster
        cluster_id = f"cluster_{len(self.clusters)}"
        self.clusters[cluster_id] = [node_id]
        self.node_to_cluster[node_id] = cluster_id
        self.cluster_coordinators[cluster_id] = node_id

    def _elect_cluster_coordinator(self, cluster_id: str):
        """
        Elect a coordinator for a cluster.

        Args:
            cluster_id: Cluster ID
        """
        if cluster_id not in self.clusters or not self.clusters[cluster_id]:
            return

        # Find the node with the lowest average latency to all other nodes in the cluster
        nodes = self.clusters[cluster_id]
        best_node = None
        best_avg_latency = float("inf")

        for node_id in nodes:
            latencies = []
            for other_node in nodes:
                if other_node != node_id:
                    latency = self.latency_matrix.get(
                        (node_id, other_node), float("inf")
                    )
                    if latency < float("inf"):
                        latencies.append(latency)

            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                if avg_latency < best_avg_latency:
                    best_avg_latency = avg_latency
                    best_node = node_id

        # If we found a suitable coordinator, set it
        if best_node:
            self.cluster_coordinators[cluster_id] = best_node
        else:
            # Default to the first node
            self.cluster_coordinators[cluster_id] = nodes[0]

    def _update_routing_table(self):
        """
        Update the routing table based on current topology.
        Uses a modified Dijkstra's algorithm that considers latency, bandwidth, and reliability.
        """
        # Clear the current routing table
        self.routing_table = {}

        # For each node, compute shortest paths to all other nodes
        for source in self.nodes:
            self._compute_routes_from_source(source)

    def _compute_routes_from_source(self, source: str):
        """
        Compute routes from a source node to all other nodes.

        Args:
            source: Source node ID
        """
        # Initialize distances and previous nodes
        distances = {node_id: float("inf") for node_id in self.nodes}
        previous = {node_id: None for node_id in self.nodes}
        distances[source] = 0

        # Priority queue for Dijkstra's algorithm
        pq = [(0, source)]

        # Set of visited nodes
        visited = set()

        while pq:
            # Get the node with the smallest distance
            current_distance, current = heapq.heappop(pq)

            # If we've already processed this node, skip it
            if current in visited:
                continue

            # Mark as visited
            visited.add(current)

            # If we've visited all nodes, we're done
            if len(visited) == len(self.nodes):
                break

            # Check all neighbors
            for neighbor in self.nodes:
                if neighbor == current or neighbor in visited:
                    continue

                # Calculate the edge weight
                latency = self.latency_matrix.get((current, neighbor), float("inf"))
                bandwidth = self.bandwidth_matrix.get((current, neighbor), 0.0)
                reliability = self.reliability_matrix.get((current, neighbor), 0.0)

                # Skip if there's no connection
                if latency == float("inf") or bandwidth == 0.0 or reliability == 0.0:
                    continue

                # Calculate the combined weight
                # Lower is better for latency, higher is better for bandwidth and reliability
                # Normalize and combine with weights
                normalized_latency = min(1.0, latency / 1000.0)  # Normalize to 0-1
                normalized_bandwidth = min(1.0, bandwidth / 10000.0)  # Normalize to 0-1

                # Combined weight (lower is better)
                weight = (
                    self.latency_weight * normalized_latency
                    - self.bandwidth_weight * normalized_bandwidth
                    - self.reliability_weight * reliability
                )

                # Ensure weight is positive
                weight = max(0.001, weight)

                # Calculate the new distance
                new_distance = distances[current] + weight

                # If this path is better, update
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_distance, neighbor))

        # Build the routing table
        for target in self.nodes:
            if target == source:
                continue

            # If there's no path, skip
            if previous[target] is None:
                continue

            # Find the first hop in the path
            current = target
            while previous[current] != source:
                if previous[current] is None:
                    break
                current = previous[current]

            # If we found a valid path, add to routing table
            if previous[current] == source:
                self.routing_table[(source, target)] = current

    def get_topology_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current topology.

        Returns:
            Dictionary with topology statistics
        """
        with self.lock:
            stats = {
                "num_nodes": len(self.nodes),
                "num_clusters": len(self.clusters),
                "avg_cluster_size": 0,
                "avg_latency": 0,
                "avg_bandwidth": 0,
                "avg_reliability": 0,
                "avg_path_length": 0,
            }

            # Calculate average cluster size
            if self.clusters:
                stats["avg_cluster_size"] = sum(
                    len(nodes) for nodes in self.clusters.values()
                ) / len(self.clusters)

            # Calculate average latency, bandwidth, and reliability
            latencies = [
                lat for lat in self.latency_matrix.values() if lat < float("inf")
            ]
            bandwidths = [bw for bw in self.bandwidth_matrix.values() if bw > 0]
            reliabilities = [rel for rel in self.reliability_matrix.values() if rel > 0]

            if latencies:
                stats["avg_latency"] = sum(latencies) / len(latencies)
            if bandwidths:
                stats["avg_bandwidth"] = sum(bandwidths) / len(bandwidths)
            if reliabilities:
                stats["avg_reliability"] = sum(reliabilities) / len(reliabilities)

            # Calculate average path length
            path_lengths = []
            for source in self.nodes:
                for target in self.nodes:
                    if source != target:
                        path = self.get_path(source, target)
                        if path:
                            path_lengths.append(
                                len(path) - 1
                            )  # -1 because we count hops, not nodes

            if path_lengths:
                stats["avg_path_length"] = sum(path_lengths) / len(path_lengths)

            return stats

    def visualize_topology(self, filename: str = "topology.png"):
        """
        Visualize the current topology.

        Args:
            filename: Output filename
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx

            # Create a graph
            G = nx.Graph()

            # Add nodes
            for node_id in self.nodes:
                cluster_id = self.node_to_cluster.get(node_id)
                is_coordinator = self.cluster_coordinators.get(cluster_id) == node_id
                G.add_node(node_id, cluster=cluster_id, coordinator=is_coordinator)

            # Add edges
            for (node1, node2), latency in self.latency_matrix.items():
                if latency < float("inf"):
                    G.add_edge(
                        node1, node2, weight=1.0 / latency
                    )  # Inverse latency as weight

            # Set up colors for clusters
            colors = {}
            for i, cluster_id in enumerate(self.clusters):
                colors[cluster_id] = f"C{i % 10}"

            # Get node colors
            node_colors = [
                colors.get(self.node_to_cluster.get(node_id), "gray")
                for node_id in G.nodes()
            ]

            # Get node sizes (larger for coordinators)
            node_sizes = [
                300 if G.nodes[node_id].get("coordinator", False) else 100
                for node_id in G.nodes()
            ]

            # Create the plot
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(
                G, pos, with_labels=True, node_color=node_colors, node_size=node_sizes
            )

            # Add edge labels (latency)
            edge_labels = {
                (node1, node2): f"{self.latency_matrix.get((node1, node2), 0):.1f}ms"
                for (node1, node2) in G.edges()
            }
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

            # Add a legend
            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=10,
                    label=f"Cluster {cluster_id}",
                )
                for cluster_id, color in colors.items()
            ]
            plt.legend(handles=legend_elements, loc="upper right")

            plt.title("QTrust Network Topology")
            plt.savefig(filename)
            plt.close()

            return True
        except Exception as e:
            print(f"Error visualizing topology: {e}")
            return False
