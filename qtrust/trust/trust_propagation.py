#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Trust Propagation Optimizer
This module implements optimized trust propagation with gossip-based algorithms.
"""

import time
import threading
import random
import math
import heapq
from typing import Dict, List, Tuple, Set, Optional, Any, Callable

from ..common.async_utils import AsyncProcessor, AsyncEvent, AsyncCache


class TrustPropagationOptimizer:
    """
    Implements optimized trust propagation with gossip-based algorithms.
    Reduces trust propagation overhead and improves convergence time.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the trust propagation optimizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Default configuration
        self.fanout = self.config.get("fanout", 3)  # Number of peers to propagate to
        self.propagation_interval = self.config.get(
            "propagation_interval", 5.0
        )  # seconds
        self.propagation_ttl = self.config.get("propagation_ttl", 3)  # Max hops
        self.significance_threshold = self.config.get(
            "significance_threshold", 0.05
        )  # Min change to propagate
        self.cache_ttl = self.config.get("cache_ttl", 300.0)  # 5 minutes
        self.batch_size = self.config.get("batch_size", 10)  # Updates per batch
        self.compression_threshold = self.config.get(
            "compression_threshold", 0.8
        )  # Similarity for compression

        # Node data structures
        self.nodes = {}  # node_id -> node_info
        self.peers = {}  # node_id -> [peer_ids]
        self.trust_scores = {}  # (node_id1, node_id2) -> trust_score
        self.trust_updates = []  # List of pending updates
        self.propagation_history = {}  # update_id -> set(node_ids)

        # Propagation cache
        self.propagation_cache = AsyncCache(max_size=10000, ttl=self.cache_ttl)

        # Async processor for propagation
        self.async_processor = AsyncProcessor(num_workers=4)

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.propagation_thread = None

    def start(self):
        """
        Start the trust propagation optimizer.
        """
        if self.running:
            return

        self.running = True
        self.async_processor.start()

        self.propagation_thread = threading.Thread(target=self._propagation_loop)
        self.propagation_thread.daemon = True
        self.propagation_thread.start()

    def stop(self):
        """
        Stop the trust propagation optimizer.
        """
        self.running = False

        if self.propagation_thread:
            self.propagation_thread.join(timeout=5.0)
            self.propagation_thread = None

        self.async_processor.stop()

    def _propagation_loop(self):
        """
        Background thread for periodic trust propagation.
        """
        while self.running:
            try:
                self._process_trust_updates()
                time.sleep(self.propagation_interval)
            except Exception as e:
                print(f"Error in trust propagation: {e}")
                time.sleep(5.0)  # Shorter interval on error

    def add_node(self, node_id: str, node_info: Dict[str, Any]):
        """
        Add a node to the trust network.

        Args:
            node_id: Unique identifier for the node
            node_info: Node information including location, capacity, etc.
        """
        with self.lock:
            self.nodes[node_id] = node_info
            self.peers[node_id] = []

    def remove_node(self, node_id: str):
        """
        Remove a node from the trust network.

        Args:
            node_id: Unique identifier for the node
        """
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]

            if node_id in self.peers:
                del self.peers[node_id]

            # Remove from peer lists
            for peer_id, peer_list in self.peers.items():
                if node_id in peer_list:
                    peer_list.remove(node_id)

            # Remove trust relationships
            keys_to_remove = []
            for key in self.trust_scores:
                if node_id in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                if key in self.trust_scores:
                    del self.trust_scores[key]

    def add_peer(self, node_id: str, peer_id: str):
        """
        Add a peer relationship between nodes.

        Args:
            node_id: First node ID
            peer_id: Second node ID
        """
        with self.lock:
            if node_id not in self.nodes or peer_id not in self.nodes:
                return

            if node_id not in self.peers:
                self.peers[node_id] = []

            if peer_id not in self.peers:
                self.peers[peer_id] = []

            if peer_id not in self.peers[node_id]:
                self.peers[node_id].append(peer_id)

            if node_id not in self.peers[peer_id]:
                self.peers[peer_id].append(node_id)

    def remove_peer(self, node_id: str, peer_id: str):
        """
        Remove a peer relationship between nodes.

        Args:
            node_id: First node ID
            peer_id: Second node ID
        """
        with self.lock:
            if node_id in self.peers and peer_id in self.peers[node_id]:
                self.peers[node_id].remove(peer_id)

            if peer_id in self.peers and node_id in self.peers[peer_id]:
                self.peers[peer_id].remove(node_id)

    def update_trust_score(
        self,
        source_id: str,
        target_id: str,
        score: float,
        timestamp: Optional[float] = None,
        ttl: Optional[int] = None,
    ):
        """
        Update a trust score and queue it for propagation.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            score: Trust score (0-1)
            timestamp: Optional timestamp (defaults to current time)
            ttl: Optional time-to-live for propagation (defaults to propagation_ttl)
        """
        with self.lock:
            if source_id not in self.nodes or target_id not in self.nodes:
                return

            # Check if this is a significant update
            old_score = self.trust_scores.get((source_id, target_id), 0.0)
            if abs(score - old_score) < self.significance_threshold:
                return  # Not significant enough to propagate

            # Update local score
            self.trust_scores[(source_id, target_id)] = score

            # Create update record
            update_id = f"{source_id}:{target_id}:{time.time()}"
            update = {
                "id": update_id,
                "source_id": source_id,
                "target_id": target_id,
                "score": score,
                "timestamp": timestamp or time.time(),
                "ttl": ttl or self.propagation_ttl,
                "origin": source_id,
            }

            # Add to update queue
            self.trust_updates.append(update)

            # Initialize propagation history
            self.propagation_history[update_id] = {source_id}

    def get_trust_score(self, source_id: str, target_id: str) -> float:
        """
        Get the trust score from source to target.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            Trust score (0-1) or 0.0 if not found
        """
        with self.lock:
            return self.trust_scores.get((source_id, target_id), 0.0)

    def _process_trust_updates(self):
        """
        Process pending trust updates and propagate them.
        """
        with self.lock:
            if not self.trust_updates:
                return

            # Sort updates by timestamp (oldest first)
            self.trust_updates.sort(key=lambda u: u["timestamp"])

            # Process updates in batches
            batch = self.trust_updates[: self.batch_size]
            self.trust_updates = self.trust_updates[self.batch_size :]

            # Group updates by source-target pair
            grouped_updates = {}
            for update in batch:
                key = (update["source_id"], update["target_id"])
                if (
                    key not in grouped_updates
                    or update["timestamp"] > grouped_updates[key]["timestamp"]
                ):
                    grouped_updates[key] = update

            # Propagate each update
            for update in grouped_updates.values():
                self._propagate_update(update)

    def _propagate_update(self, update: Dict[str, Any]):
        """
        Propagate a trust update to peers.

        Args:
            update: Trust update record
        """
        source_id = update["source_id"]
        update_id = update["id"]
        ttl = update["ttl"]

        if ttl <= 0:
            return  # TTL expired

        # Decrement TTL for next hop
        update["ttl"] = ttl - 1

        # Select peers for propagation
        selected_peers = self._select_propagation_peers(source_id, update_id)

        # Propagate to selected peers
        for peer_id in selected_peers:
            # In a real implementation, this would send to other nodes
            # For simulation, we'll just update our local state
            self._receive_update(peer_id, update)

    def _select_propagation_peers(self, node_id: str, update_id: str) -> List[str]:
        """
        Select peers for propagating an update.

        Args:
            node_id: Node ID
            update_id: Update ID

        Returns:
            List of selected peer IDs
        """
        if node_id not in self.peers:
            return []

        # Get peers that haven't received this update yet
        eligible_peers = []
        for peer_id in self.peers[node_id]:
            if (
                update_id not in self.propagation_history
                or peer_id not in self.propagation_history[update_id]
            ):
                eligible_peers.append(peer_id)

        if not eligible_peers:
            return []

        # If we have fewer eligible peers than fanout, use all of them
        if len(eligible_peers) <= self.fanout:
            return eligible_peers

        # Otherwise, select fanout peers randomly
        return random.sample(eligible_peers, self.fanout)

    def _receive_update(self, node_id: str, update: Dict[str, Any]):
        """
        Process a received trust update.

        Args:
            node_id: Receiving node ID
            update: Trust update record
        """
        update_id = update["id"]
        source_id = update["source_id"]
        target_id = update["target_id"]
        score = update["score"]

        # Mark as received
        if update_id not in self.propagation_history:
            self.propagation_history[update_id] = set()

        self.propagation_history[update_id].add(node_id)

        # Update local score
        self.trust_scores[(source_id, target_id)] = score

        # Add to update queue for further propagation
        self.trust_updates.append(update.copy())

    def compress_updates(self, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compress a list of trust updates to reduce bandwidth.

        Args:
            updates: List of trust updates

        Returns:
            Compressed list of trust updates
        """
        if not updates:
            return []

        # Group updates by source-target pair
        grouped_updates = {}
        for update in updates:
            key = (update["source_id"], update["target_id"])
            if key not in grouped_updates:
                grouped_updates[key] = []
            grouped_updates[key].append(update)

        # Compress each group
        compressed_updates = []
        for key, group in grouped_updates.items():
            if len(group) == 1:
                compressed_updates.append(group[0])
                continue

            # Sort by timestamp
            group.sort(key=lambda u: u["timestamp"])

            # Check if we can compress
            scores = [u["score"] for u in group]
            timestamps = [u["timestamp"] for u in group]

            # Calculate similarity
            score_range = max(scores) - min(scores)
            if score_range < self.compression_threshold:
                # Compress to a single update with the latest timestamp and average score
                avg_score = sum(scores) / len(scores)
                latest_timestamp = max(timestamps)
                min_ttl = min(u["ttl"] for u in group)

                compressed = group[-1].copy()
                compressed["score"] = avg_score
                compressed["timestamp"] = latest_timestamp
                compressed["ttl"] = min_ttl
                compressed["compressed"] = True
                compressed["compressed_count"] = len(group)

                compressed_updates.append(compressed)
            else:
                # Can't compress, add all updates
                compressed_updates.extend(group)

        return compressed_updates

    def optimize_propagation_paths(self):
        """
        Optimize the propagation paths based on network topology.
        """
        with self.lock:
            # Build a graph of the network
            graph = {}
            for node_id, peers in self.peers.items():
                graph[node_id] = {}
                for peer_id in peers:
                    # In a real implementation, this would use actual latency measurements
                    # For simulation, we'll use a placeholder
                    graph[node_id][peer_id] = 1.0

            # Calculate shortest paths
            shortest_paths = {}
            for node_id in self.nodes:
                shortest_paths[node_id] = self._calculate_shortest_paths(graph, node_id)

            # Optimize peer connections
            for node_id in self.nodes:
                self._optimize_peer_connections(node_id, shortest_paths[node_id])

    def _calculate_shortest_paths(
        self, graph: Dict[str, Dict[str, float]], start_node: str
    ) -> Dict[str, Tuple[float, List[str]]]:
        """
        Calculate shortest paths from start_node to all other nodes.

        Args:
            graph: Network graph
            start_node: Starting node ID

        Returns:
            Dictionary mapping node IDs to (distance, path) tuples
        """
        # Dijkstra's algorithm
        distances = {node: float("infinity") for node in graph}
        distances[start_node] = 0
        previous = {node: None for node in graph}
        priority_queue = [(0, start_node)]

        while priority_queue:
            current_distance, current_node = heapq.heappop(priority_queue)

            if current_distance > distances[current_node]:
                continue

            for neighbor, weight in graph[current_node].items():
                distance = current_distance + weight

                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous[neighbor] = current_node
                    heapq.heappush(priority_queue, (distance, neighbor))

        # Reconstruct paths
        paths = {}
        for node in graph:
            if node == start_node:
                paths[node] = (0, [node])
                continue

            if distances[node] == float("infinity"):
                paths[node] = (float("infinity"), [])
                continue

            path = []
            current = node
            while current:
                path.append(current)
                current = previous[current]

            path.reverse()
            paths[node] = (distances[node], path)

        return paths

    def _optimize_peer_connections(
        self, node_id: str, shortest_paths: Dict[str, Tuple[float, List[str]]]
    ):
        """
        Optimize peer connections for a node.

        Args:
            node_id: Node ID
            shortest_paths: Shortest paths from the node to all other nodes
        """
        # Find nodes that are more than 2 hops away
        distant_nodes = []
        for target_id, (distance, path) in shortest_paths.items():
            if distance > 2 and path:
                distant_nodes.append((target_id, distance, path))

        # Sort by distance (furthest first)
        distant_nodes.sort(key=lambda x: x[1], reverse=True)

        # Add direct connections to the furthest nodes (up to fanout)
        for target_id, distance, path in distant_nodes[: self.fanout]:
            self.add_peer(node_id, target_id)

    def get_propagation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about trust propagation.

        Returns:
            Dictionary with propagation statistics
        """
        with self.lock:
            stats = {
                "num_nodes": len(self.nodes),
                "num_trust_relationships": len(self.trust_scores),
                "pending_updates": len(self.trust_updates),
                "avg_propagation_coverage": 0.0,
                "avg_peers_per_node": 0.0,
            }

            # Calculate average propagation coverage
            if self.propagation_history:
                coverage_ratios = []
                for node_set in self.propagation_history.values():
                    if self.nodes:
                        coverage_ratios.append(len(node_set) / len(self.nodes))

                if coverage_ratios:
                    stats["avg_propagation_coverage"] = sum(coverage_ratios) / len(
                        coverage_ratios
                    )

            # Calculate average peers per node
            if self.peers:
                peer_counts = [len(peers) for peers in self.peers.values()]
                if peer_counts:
                    stats["avg_peers_per_node"] = sum(peer_counts) / len(peer_counts)

            return stats
