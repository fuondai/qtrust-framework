#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Path Cache Optimizer
This module implements path caching for frequently used routes to reduce routing overhead.
"""

import time
import threading
import random
import heapq
from typing import Dict, List, Tuple, Set, Optional, Any, Callable

from ..common.async_utils import AsyncCache


class PathCacheOptimizer:
    """
    Implements path caching for frequently used routes to reduce routing overhead.
    Optimizes routing performance by caching and prefetching common paths.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the path cache optimizer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Default configuration
        self.cache_size = self.config.get("cache_size", 10000)
        self.cache_ttl = self.config.get("cache_ttl", 300.0)  # 5 minutes
        self.prefetch_threshold = self.config.get(
            "prefetch_threshold", 5
        )  # Frequency threshold for prefetching
        self.prefetch_limit = self.config.get(
            "prefetch_limit", 100
        )  # Max paths to prefetch
        self.update_interval = self.config.get("update_interval", 60.0)  # seconds
        self.stats_window = self.config.get("stats_window", 3600.0)  # 1 hour

        # Path cache
        self.path_cache = AsyncCache(max_size=self.cache_size, ttl=self.cache_ttl)

        # Usage statistics
        self.path_usage = {}  # (source_id, target_id) -> usage_count
        self.path_timestamps = {}  # (source_id, target_id) -> [timestamps]
        self.path_latencies = {}  # (source_id, target_id) -> [latencies]

        # Prefetch queue
        self.prefetch_queue = []  # List of (source_id, target_id) pairs to prefetch

        # Router reference (set by the router)
        self.router = None

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.update_thread = None

    def start(self):
        """
        Start the path cache optimizer.
        """
        if self.running:
            return

        self.running = True

        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

    def stop(self):
        """
        Stop the path cache optimizer.
        """
        self.running = False

        if self.update_thread:
            self.update_thread.join(timeout=5.0)
            self.update_thread = None

    def _update_loop(self):
        """
        Background thread for periodic cache updates.
        """
        while self.running:
            try:
                self._update_cache()
                self._prefetch_paths()
                self._clean_statistics()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in path cache update: {e}")
                time.sleep(5.0)  # Shorter interval on error

    def set_router(self, router):
        """
        Set the router reference.

        Args:
            router: Router object
        """
        self.router = router

    def get_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """
        Get a cached path.

        Args:
            source_id: Source shard ID
            target_id: Target shard ID

        Returns:
            Cached path or None if not found
        """
        with self.lock:
            # Record usage
            self._record_usage(source_id, target_id)

            # Check cache
            return self.path_cache.get((source_id, target_id))

    def put_path(
        self,
        source_id: str,
        target_id: str,
        path: List[str],
        latency: Optional[float] = None,
    ):
        """
        Put a path in the cache.

        Args:
            source_id: Source shard ID
            target_id: Target shard ID
            path: Path as a list of shard IDs
            latency: Optional measured latency
        """
        with self.lock:
            # Store in cache
            self.path_cache.put((source_id, target_id), path)

            # Record usage and latency
            self._record_usage(source_id, target_id)

            if latency is not None:
                self._record_latency(source_id, target_id, latency)

    def invalidate_path(self, source_id: str, target_id: str):
        """
        Invalidate a cached path.

        Args:
            source_id: Source shard ID
            target_id: Target shard ID
        """
        with self.lock:
            self.path_cache.remove((source_id, target_id))

    def invalidate_shard(self, shard_id: str):
        """
        Invalidate all paths involving a shard.

        Args:
            shard_id: Shard ID
        """
        with self.lock:
            # Find all keys involving this shard
            keys_to_remove = []

            for key in list(self.path_cache.cache.keys()):
                if shard_id in key:
                    keys_to_remove.append(key)

            # Remove from cache
            for key in keys_to_remove:
                self.path_cache.remove(key)

    def _record_usage(self, source_id: str, target_id: str):
        """
        Record usage of a path.

        Args:
            source_id: Source shard ID
            target_id: Target shard ID
        """
        key = (source_id, target_id)

        # Increment usage count
        self.path_usage[key] = self.path_usage.get(key, 0) + 1

        # Record timestamp
        if key not in self.path_timestamps:
            self.path_timestamps[key] = []

        self.path_timestamps[key].append(time.time())

    def _record_latency(self, source_id: str, target_id: str, latency: float):
        """
        Record latency for a path.

        Args:
            source_id: Source shard ID
            target_id: Target shard ID
            latency: Measured latency
        """
        key = (source_id, target_id)

        if key not in self.path_latencies:
            self.path_latencies[key] = []

        self.path_latencies[key].append(latency)

        # Keep only the last 100 measurements
        if len(self.path_latencies[key]) > 100:
            self.path_latencies[key] = self.path_latencies[key][-100:]

    def _update_cache(self):
        """
        Update the cache based on usage statistics.
        """
        with self.lock:
            # Calculate frequency scores
            frequency_scores = []

            for key, count in self.path_usage.items():
                # Calculate recency score based on timestamps
                timestamps = self.path_timestamps.get(key, [])
                if not timestamps:
                    continue

                # Calculate frequency (usage per hour)
                current_time = time.time()
                recent_time = current_time - self.stats_window
                recent_count = sum(1 for ts in timestamps if ts >= recent_time)

                frequency = recent_count / (self.stats_window / 3600.0)

                # Add to scores
                frequency_scores.append((frequency, key))

            # Sort by frequency (highest first)
            frequency_scores.sort(reverse=True)

            # Add high-frequency paths to prefetch queue
            self.prefetch_queue = []

            for frequency, key in frequency_scores:
                if frequency >= self.prefetch_threshold:
                    self.prefetch_queue.append(key)

                if len(self.prefetch_queue) >= self.prefetch_limit:
                    break

    def _prefetch_paths(self):
        """
        Prefetch paths for high-frequency routes.
        """
        if not self.router:
            return

        with self.lock:
            # Shuffle to avoid always prefetching in the same order
            random.shuffle(self.prefetch_queue)

            # Prefetch paths
            for source_id, target_id in self.prefetch_queue[: self.prefetch_limit]:
                # Skip if already in cache
                if self.path_cache.get((source_id, target_id)) is not None:
                    continue

                # Request path from router
                try:
                    # In a real implementation, this would be an async call
                    # For simulation, we'll just call the router directly
                    tx = {
                        "source_shard": source_id,
                        "target_shard": target_id,
                        "prefetch": True,
                    }

                    path = self.router.route_transaction(tx)

                    if path:
                        self.put_path(source_id, target_id, path)
                except Exception as e:
                    print(f"Error prefetching path {source_id} -> {target_id}: {e}")

    def _clean_statistics(self):
        """
        Clean up old statistics.
        """
        with self.lock:
            current_time = time.time()
            cutoff_time = current_time - self.stats_window

            # Clean timestamps
            for key, timestamps in list(self.path_timestamps.items()):
                # Remove old timestamps
                self.path_timestamps[key] = [
                    ts for ts in timestamps if ts >= cutoff_time
                ]

                # Remove entry if empty
                if not self.path_timestamps[key]:
                    del self.path_timestamps[key]

            # Reset usage counts for paths with no recent usage
            for key in list(self.path_usage.keys()):
                if key not in self.path_timestamps:
                    del self.path_usage[key]

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the path cache.

        Returns:
            Dictionary with cache statistics
        """
        with self.lock:
            stats = {
                "cache_size": self.path_cache.size(),
                "cache_capacity": self.cache_size,
                "prefetch_queue_size": len(self.prefetch_queue),
                "avg_latency": 0.0,
                "hit_ratio": 0.0,
                "popular_paths": [],
            }

            # Calculate average latency
            all_latencies = []
            for latencies in self.path_latencies.values():
                all_latencies.extend(latencies)

            if all_latencies:
                stats["avg_latency"] = sum(all_latencies) / len(all_latencies)

            # Calculate hit ratio
            cache_stats = getattr(
                self.path_cache, "cache_stats", {"hits": 0, "misses": 0}
            )
            hits = cache_stats.get("hits", 0)
            misses = cache_stats.get("misses", 0)
            total = hits + misses

            if total > 0:
                stats["hit_ratio"] = hits / total

            # Get popular paths
            popular_paths = []

            for key, count in sorted(
                self.path_usage.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                source_id, target_id = key
                latencies = self.path_latencies.get(key, [])
                avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

                popular_paths.append(
                    {
                        "source": source_id,
                        "target": target_id,
                        "usage_count": count,
                        "avg_latency": avg_latency,
                    }
                )

            stats["popular_paths"] = popular_paths

            return stats
