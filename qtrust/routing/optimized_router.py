#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Optimized Cross-Shard Routing
This module implements optimized cross-shard routing with reduced overhead.
"""

import time
import threading
import random
import heapq
import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("qtrust_router.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


class AsyncCache:
    """
    Thread-safe cache with TTL support.
    """

    def __init__(self, max_size: int = 1000, ttl: float = 300.0):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of items in the cache
            ttl: Time-to-live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.lock = threading.RLock()

    def get(self, key: Any) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        with self.lock:
            if key not in self.cache:
                return None

            # Check if expired
            timestamp = self.timestamps.get(key, 0)
            if time.time() - timestamp > self.ttl:
                self.remove(key)
                return None

            return self.cache[key]

    def set(self, key: Any, value: Any):
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_oldest()

            self.cache[key] = value
            self.timestamps[key] = time.time()

    def remove(self, key: Any):
        """
        Remove a value from the cache.

        Args:
            key: Cache key
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]

            if key in self.timestamps:
                del self.timestamps[key]

    def clear(self):
        """
        Clear the cache.
        """
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()

    def _evict_oldest(self):
        """
        Evict the oldest item from the cache.
        """
        if not self.timestamps:
            return

        oldest_key = min(self.timestamps, key=self.timestamps.get)
        self.remove(oldest_key)


class AsyncEvent:
    """
    Thread-safe event with callback support.
    """

    def __init__(self):
        """
        Initialize the event.
        """
        self.event = threading.Event()
        self.callbacks = []
        self.lock = threading.RLock()

    def set(self):
        """
        Set the event and trigger callbacks.
        """
        with self.lock:
            self.event.set()

            # Trigger callbacks
            for callback in self.callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Error in event callback: {e}")

    def clear(self):
        """
        Clear the event.
        """
        self.event.clear()

    def is_set(self) -> bool:
        """
        Check if the event is set.

        Returns:
            True if the event is set, False otherwise
        """
        return self.event.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the event to be set.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if the event was set, False if timeout occurred
        """
        return self.event.wait(timeout)

    def add_callback(self, callback: Callable[[], None]):
        """
        Add a callback to be triggered when the event is set.

        Args:
            callback: Callback function
        """
        with self.lock:
            self.callbacks.append(callback)

    def remove_callback(self, callback: Callable[[], None]):
        """
        Remove a callback.

        Args:
            callback: Callback function
        """
        with self.lock:
            if callback in self.callbacks:
                self.callbacks.remove(callback)


class AsyncProcessor:
    """
    Thread pool for asynchronous processing.
    """

    def __init__(self, num_workers: int = 4):
        """
        Initialize the processor.

        Args:
            num_workers: Number of worker threads
        """
        self.num_workers = num_workers
        self.queue = []
        self.queue_lock = threading.RLock()
        self.queue_not_empty = threading.Condition(self.queue_lock)
        self.workers = []
        self.running = False

    def start(self):
        """
        Start the processor.
        """
        if self.running:
            return

        self.running = True

        # Start worker threads
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def stop(self):
        """
        Stop the processor.
        """
        self.running = False

        # Wake up workers
        with self.queue_not_empty:
            self.queue_not_empty.notify_all()

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=1.0)

        self.workers = []

    def submit(self, func: Callable, *args, **kwargs) -> AsyncEvent:
        """
        Submit a task for processing.

        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Event that will be set when the task is complete
        """
        event = AsyncEvent()

        with self.queue_lock:
            self.queue.append((func, args, kwargs, event))
            self.queue_not_empty.notify()

        return event

    def _worker_loop(self):
        """
        Main worker loop.
        """
        while self.running:
            task = None

            with self.queue_lock:
                while self.running and not self.queue:
                    self.queue_not_empty.wait(timeout=1.0)

                if self.queue:
                    task = self.queue.pop(0)

            if task:
                func, args, kwargs, event = task

                try:
                    result = func(*args, **kwargs)
                    event.set()
                except Exception as e:
                    logger.error(f"Error in worker task: {e}")
                    event.set()  # Set event even on error


class OptimizedCrossShardRouter:
    """
    Implements optimized cross-shard routing with reduced overhead.
    Simplifies congestion prediction and optimizes routing table structure.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the optimized cross-shard router.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Default configuration
        self.route_cache_size = self.config.get("route_cache_size", 10000)
        self.route_cache_ttl = self.config.get("route_cache_ttl", 300.0)  # 5 minutes
        self.congestion_window = self.config.get("congestion_window", 60.0)  # 1 minute
        self.path_selection_algorithm = self.config.get(
            "path_selection_algorithm", "weighted"
        )
        self.max_path_length = self.config.get("max_path_length", 5)
        self.update_interval = self.config.get("update_interval", 10.0)  # seconds

        # Routing data structures
        self.shards = {}  # shard_id -> shard_info
        self.links = {}  # (shard_id1, shard_id2) -> link_info
        self.routing_table = {}  # (source_id, target_id) -> [paths]
        self.congestion_metrics = {}  # shard_id -> congestion_level
        self.link_metrics = {}  # (shard_id1, shard_id2) -> metrics
        self.recent_transactions = []  # List of recent cross-shard transactions

        # Route cache
        self.route_cache = AsyncCache(
            max_size=self.route_cache_size, ttl=self.route_cache_ttl
        )

        # Import HierarchicalShardCluster here to avoid circular imports
        from ..shard_clustering import HierarchicalShardCluster

        # Hierarchical shard clustering for optimized routing
        self.shard_cluster = HierarchicalShardCluster(config)

        # Async processor for route calculations
        self.async_processor = AsyncProcessor(num_workers=4)

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.update_thread = None

        logger.info("Initialized OptimizedCrossShardRouter")

    def start(self):
        """
        Start the optimized cross-shard router.
        """
        if self.running:
            return

        self.running = True
        self.async_processor.start()

        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

        logger.info("Started OptimizedCrossShardRouter")

    def stop(self):
        """
        Stop the optimized cross-shard router.
        """
        self.running = False

        if self.update_thread:
            self.update_thread.join(timeout=5.0)
            self.update_thread = None

        self.async_processor.stop()

        logger.info("Stopped OptimizedCrossShardRouter")

    def _update_loop(self):
        """
        Background thread for periodic routing updates.
        """
        while self.running:
            try:
                self.update_routing_tables()
                self.update_congestion_metrics()
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in routing update: {e}")
                time.sleep(5.0)  # Shorter interval on error

    def add_shard(self, shard_id: str, shard_info: Dict[str, Any]):
        """
        Add a shard to the routing network.

        Args:
            shard_id: Unique identifier for the shard
            shard_info: Shard information including location, capacity, etc.
        """
        with self.lock:
            self.shards[shard_id] = shard_info
            self.congestion_metrics[shard_id] = 0.0

            # Add to shard cluster
            features = np.array(
                [
                    shard_info.get("capacity", 1000.0),
                    shard_info.get("reliability", 0.99),
                    shard_info.get("geographic_zone", 0),
                    shard_info.get("node_count", 10),
                ]
            )
            self.shard_cluster.add_shard(shard_id, features)

            # Invalidate affected routes
            self._invalidate_routes_for_shard(shard_id)

            logger.info(f"Added shard {shard_id} to routing network")

    def remove_shard(self, shard_id: str):
        """
        Remove a shard from the routing network.

        Args:
            shard_id: Unique identifier for the shard
        """
        with self.lock:
            if shard_id in self.shards:
                del self.shards[shard_id]

            if shard_id in self.congestion_metrics:
                del self.congestion_metrics[shard_id]

            # Remove from shard cluster
            self.shard_cluster.remove_shard(shard_id)

            # Remove links
            links_to_remove = []
            for link_key in self.links:
                if shard_id in link_key:
                    links_to_remove.append(link_key)

            for link_key in links_to_remove:
                if link_key in self.links:
                    del self.links[link_key]
                if link_key in self.link_metrics:
                    del self.link_metrics[link_key]

            # Invalidate affected routes
            self._invalidate_routes_for_shard(shard_id)

            logger.info(f"Removed shard {shard_id} from routing network")

    def add_link(self, shard_id1: str, shard_id2: str, link_info: Dict[str, Any]):
        """
        Add a link between two shards.

        Args:
            shard_id1: First shard ID
            shard_id2: Second shard ID
            link_info: Link information including latency, bandwidth, etc.
        """
        with self.lock:
            if shard_id1 not in self.shards or shard_id2 not in self.shards:
                return

            self.links[(shard_id1, shard_id2)] = link_info
            self.links[(shard_id2, shard_id1)] = link_info

            # Initialize link metrics
            if (shard_id1, shard_id2) not in self.link_metrics:
                self.link_metrics[(shard_id1, shard_id2)] = {
                    "latency": link_info.get("latency", 100.0),
                    "bandwidth": link_info.get("bandwidth", 1.0),
                    "congestion": 0.0,
                    "reliability": 1.0,
                    "utilization": 0.0,
                    "last_updated": time.time(),
                }

            if (shard_id2, shard_id1) not in self.link_metrics:
                self.link_metrics[(shard_id2, shard_id1)] = self.link_metrics[
                    (shard_id1, shard_id2)
                ].copy()

            # Update shard cluster with transaction information
            self.shard_cluster.record_transaction(shard_id1, shard_id2, 1.0)

            # Invalidate affected routes
            self._invalidate_routes_for_link(shard_id1, shard_id2)

            logger.info(f"Added link between shards {shard_id1} and {shard_id2}")

    def update_link_metrics(
        self, shard_id1: str, shard_id2: str, metrics: Dict[str, Any]
    ):
        """
        Update metrics for a link between two shards.

        Args:
            shard_id1: First shard ID
            shard_id2: Second shard ID
            metrics: Updated metrics including latency, bandwidth, etc.
        """
        with self.lock:
            if (shard_id1, shard_id2) not in self.links:
                return

            # Update link metrics
            if (shard_id1, shard_id2) in self.link_metrics:
                for key, value in metrics.items():
                    self.link_metrics[(shard_id1, shard_id2)][key] = value

                self.link_metrics[(shard_id1, shard_id2)]["last_updated"] = time.time()

            # Mirror for reverse direction
            if (shard_id2, shard_id1) in self.link_metrics:
                for key, value in metrics.items():
                    self.link_metrics[(shard_id2, shard_id1)][key] = value

                self.link_metrics[(shard_id2, shard_id1)]["last_updated"] = time.time()

            # Update shard cluster with transaction information if latency changed
            if "latency" in metrics:
                # Lower latency means higher transaction weight
                weight = 10.0 / max(1.0, metrics["latency"])
                self.shard_cluster.record_transaction(shard_id1, shard_id2, weight)

            # Invalidate affected routes
            self._invalidate_routes_for_link(shard_id1, shard_id2)

    def update_congestion_metrics(self):
        """
        Update congestion metrics for all shards and links.
        """
        with self.lock:
            # Get recent transactions (last congestion_window seconds)
            current_time = time.time()
            recent_time = current_time - self.congestion_window

            recent_txs = [
                tx for tx in self.recent_transactions if tx["timestamp"] >= recent_time
            ]

            # Count transactions per shard and link
            shard_counts = {}
            link_counts = {}

            for tx in recent_txs:
                path = tx.get("path", [])

                for shard_id in path:
                    shard_counts[shard_id] = shard_counts.get(shard_id, 0) + 1

                for i in range(len(path) - 1):
                    link_key = (path[i], path[i + 1])
                    link_counts[link_key] = link_counts.get(link_key, 0) + 1

            # Update congestion metrics
            for shard_id in self.shards:
                count = shard_counts.get(shard_id, 0)
                capacity = self.shards[shard_id].get("capacity", 1000.0)

                # Calculate congestion level (0-1)
                congestion = min(1.0, count / capacity) if capacity > 0 else 0.0

                # Smooth with previous value (70% new, 30% old)
                old_congestion = self.congestion_metrics.get(shard_id, 0.0)
                self.congestion_metrics[shard_id] = (
                    0.7 * congestion + 0.3 * old_congestion
                )

            # Update link congestion metrics
            for link_key, count in link_counts.items():
                if link_key in self.link_metrics:
                    bandwidth = self.links.get(link_key, {}).get("bandwidth", 100.0)

                    # Calculate congestion level (0-1)
                    congestion = min(1.0, count / bandwidth) if bandwidth > 0 else 0.0

                    # Smooth with previous value (70% new, 30% old)
                    old_congestion = self.link_metrics[link_key].get("congestion", 0.0)
                    self.link_metrics[link_key]["congestion"] = (
                        0.7 * congestion + 0.3 * old_congestion
                    )

                    # Update utilization
                    self.link_metrics[link_key]["utilization"] = count

            # Trim recent transactions list
            self.recent_transactions = recent_txs

    def update_routing_tables(self):
        """
        Update routing tables for all shard pairs.
        """
        # This is a potentially expensive operation, so we'll do it incrementally
        # Each update cycle, we'll update a subset of the routing table

        with self.lock:
            # Get all shard pairs
            shard_pairs = []
            for source_id in self.shards:
                for target_id in self.shards:
                    if source_id != target_id:
                        shard_pairs.append((source_id, target_id))

            # Shuffle to ensure even coverage over time
            random.shuffle(shard_pairs)

            # Update a subset of pairs (10% or at least 10)
            update_count = max(10, len(shard_pairs) // 10)
            pairs_to_update = shard_pairs[:update_count]

            for source_id, target_id in pairs_to_update:
                self._update_route(source_id, target_id)

    def _update_route(self, source_id: str, target_id: str):
        """
        Update the routing table for a specific source-target pair.

        Args:
            source_id: Source shard ID
            target_id: Target shard ID
        """
        if source_id not in self.shards or target_id not in self.shards:
            return

        # Get optimal path from shard cluster
        cluster_path = self.shard_cluster.get_optimal_path(source_id, target_id)

        # Calculate additional paths
        paths = self._calculate_paths(source_id, target_id)

        # Add cluster path if not already in paths
        if cluster_path and cluster_path not in paths:
            paths.append(cluster_path)

        # Update routing table
        self.routing_table[(source_id, target_id)] = paths

        # Invalidate cache
        self.route_cache.remove((source_id, target_id))

    def _calculate_paths(self, source_id: str, target_id: str) -> List[List[str]]:
        """
        Calculate multiple paths between source and target shards.

        Args:
            source_id: Source shard ID
            target_id: Target shard ID

        Returns:
            List of paths, where each path is a list of shard IDs
        """
        # Build a graph of the network
        graph = {}
        for shard_id in self.shards:
            graph[shard_id] = {}

        for (s1, s2), link_info in self.links.items():
            if s1 in graph and s2 in graph:
                # Calculate edge weight based on metrics
                metrics = self.link_metrics.get((s1, s2), {})
                latency = metrics.get("latency", 100.0)
                congestion = metrics.get("congestion", 0.0)
                reliability = metrics.get("reliability", 1.0)

                # Weight formula: latency * (1 + congestion) / reliability
                weight = latency * (1.0 + congestion) / max(0.1, reliability)

                graph[s1][s2] = weight
                graph[s2][s1] = weight

        # Find k-shortest paths
        paths = self._k_shortest_paths(graph, source_id, target_id, k=3)

        return paths

    def _k_shortest_paths(
        self, graph: Dict[str, Dict[str, float]], source: str, target: str, k: int
    ) -> List[List[str]]:
        """
        Find k-shortest paths between source and target.

        Args:
            graph: Network graph
            source: Source node ID
            target: Target node ID
            k: Number of paths to find

        Returns:
            List of paths, where each path is a list of node IDs
        """
        if source not in graph or target not in graph:
            return []

        # Use a modified version of Yen's algorithm
        A = []  # List of shortest paths
        B = []  # Candidate paths

        # Find the shortest path
        shortest_path = self._shortest_path(graph, source, target)
        if not shortest_path:
            return []

        A.append(shortest_path)

        # Find k-1 more paths
        for i in range(1, k):
            # For each node in the previous path except the last
            prev_path = A[i - 1]

            for j in range(len(prev_path) - 1):
                # Spur node is the j-th node in the previous path
                spur_node = prev_path[j]

                # Root path is the path from source to spur node
                root_path = prev_path[: j + 1]

                # Remove links that are part of previous paths with the same root
                removed_links = []
                for path in A:
                    if len(path) > j and path[: j + 1] == root_path:
                        if j + 1 < len(path):
                            link = (path[j], path[j + 1])
                            if link not in removed_links:
                                removed_links.append(link)

                # Remove the links from the graph temporarily
                for u, v in removed_links:
                    if u in graph and v in graph[u]:
                        saved_weight = graph[u][v]
                        del graph[u][v]
                        removed_links.append((u, v, saved_weight))

                # Find the shortest path from spur node to target
                spur_path = self._shortest_path(graph, spur_node, target)

                # Restore the removed links
                for u, v, saved_weight in removed_links:
                    graph[u][v] = saved_weight

                if spur_path:
                    # Complete path is root_path + spur_path
                    candidate_path = root_path[:-1] + spur_path

                    # Add to candidates if not already there
                    if candidate_path not in B and candidate_path not in A:
                        B.append(candidate_path)

            if not B:
                break

            # Find the shortest path in B
            B.sort(key=lambda path: self._path_cost(graph, path))
            A.append(B[0])
            B.pop(0)

        return A

    def _shortest_path(
        self, graph: Dict[str, Dict[str, float]], source: str, target: str
    ) -> List[str]:
        """
        Find the shortest path between source and target using Dijkstra's algorithm.

        Args:
            graph: Network graph
            source: Source node ID
            target: Target node ID

        Returns:
            Shortest path as a list of node IDs
        """
        if source not in graph or target not in graph:
            return []

        if source == target:
            return [source]

        # Initialize
        distances = {node: float("inf") for node in graph}
        distances[source] = 0
        previous = {node: None for node in graph}
        unvisited = list(graph.keys())

        while unvisited:
            # Find the unvisited node with the smallest distance
            current = min(unvisited, key=lambda node: distances[node])

            # If we've reached the target or there's no path
            if current == target or distances[current] == float("inf"):
                break

            # Remove current from unvisited
            unvisited.remove(current)

            # Check neighbors
            for neighbor, weight in graph[current].items():
                if neighbor in unvisited:
                    distance = distances[current] + weight

                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current

        # Build the path
        if previous[target] is None:
            return []

        path = [target]
        while path[0] != source:
            path.insert(0, previous[path[0]])

        return path

    def _path_cost(self, graph: Dict[str, Dict[str, float]], path: List[str]) -> float:
        """
        Calculate the cost of a path.

        Args:
            graph: Network graph
            path: Path as a list of node IDs

        Returns:
            Path cost
        """
        cost = 0.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]

            if u in graph and v in graph[u]:
                cost += graph[u][v]
            else:
                return float("inf")

        return cost

    def get_route(self, source_id: str, target_id: str) -> List[str]:
        """
        Get the best route between source and target shards.

        Args:
            source_id: Source shard ID
            target_id: Target shard ID

        Returns:
            Best route as a list of shard IDs
        """
        # Check cache first
        cached_route = self.route_cache.get((source_id, target_id))
        if cached_route:
            return cached_route

        with self.lock:
            if source_id not in self.shards or target_id not in self.shards:
                return [source_id, target_id]

            if source_id == target_id:
                return [source_id]

            # Get paths from routing table
            paths = self.routing_table.get((source_id, target_id), [])

            if not paths:
                # Calculate paths if not in routing table
                self._update_route(source_id, target_id)
                paths = self.routing_table.get((source_id, target_id), [])

            if not paths:
                # Fallback to direct path
                return [source_id, target_id]

            # Select best path based on algorithm
            if self.path_selection_algorithm == "shortest":
                # Select shortest path
                best_path = min(paths, key=len)
            elif self.path_selection_algorithm == "random":
                # Select random path
                best_path = random.choice(paths)
            else:  # 'weighted' (default)
                # Select path with lowest weight
                best_path = min(
                    paths, key=lambda path: self._calculate_path_weight(path)
                )

            # Cache the result
            self.route_cache.set((source_id, target_id), best_path)

            return best_path

    def _calculate_path_weight(self, path: List[str]) -> float:
        """
        Calculate the weight of a path.

        Args:
            path: Path as a list of shard IDs

        Returns:
            Path weight
        """
        weight = 0.0

        for i in range(len(path) - 1):
            shard1, shard2 = path[i], path[i + 1]

            # Get link metrics
            metrics = self.link_metrics.get((shard1, shard2), {})

            # Calculate link weight
            latency = metrics.get("latency", 100.0)
            congestion = metrics.get("congestion", 0.0)
            reliability = metrics.get("reliability", 1.0)

            # Weight formula: latency * (1 + congestion) / reliability
            link_weight = latency * (1.0 + congestion) / max(0.1, reliability)

            weight += link_weight

        return weight

    def record_transaction(self, transaction: Dict[str, Any]):
        """
        Record a cross-shard transaction.

        Args:
            transaction: Transaction information
        """
        with self.lock:
            # Add timestamp if not present
            if "timestamp" not in transaction:
                transaction["timestamp"] = time.time()

            # Add to recent transactions
            self.recent_transactions.append(transaction)

            # Trim if too many
            if len(self.recent_transactions) > 10000:
                self.recent_transactions = self.recent_transactions[-10000:]

            # Update shard cluster
            source_shard = transaction.get("source_shard")
            dest_shard = transaction.get("dest_shard")

            if source_shard and dest_shard and source_shard != dest_shard:
                self.shard_cluster.record_transaction(source_shard, dest_shard, 1.0)

    def _invalidate_routes_for_shard(self, shard_id: str):
        """
        Invalidate routes involving a shard.

        Args:
            shard_id: Shard ID
        """
        keys_to_remove = []

        for key in self.routing_table:
            source_id, target_id = key
            if source_id == shard_id or target_id == shard_id:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            if key in self.routing_table:
                del self.routing_table[key]

            self.route_cache.remove(key)

    def _invalidate_routes_for_link(self, shard_id1: str, shard_id2: str):
        """
        Invalidate routes involving a link.

        Args:
            shard_id1: First shard ID
            shard_id2: Second shard ID
        """
        keys_to_remove = []

        for key, paths in self.routing_table.items():
            for path in paths:
                for i in range(len(path) - 1):
                    if (path[i] == shard_id1 and path[i + 1] == shard_id2) or (
                        path[i] == shard_id2 and path[i + 1] == shard_id1
                    ):
                        keys_to_remove.append(key)
                        break

        for key in keys_to_remove:
            if key in self.routing_table:
                del self.routing_table[key]

            self.route_cache.remove(key)
