#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Load Monitoring
This module implements resource monitoring for shard load calculation.
"""

import time
import threading
import psutil
import os
from typing import Dict, List, Any, Optional, Callable


class ResourceMonitor:
    """
    Monitors system resources including CPU, memory, and network utilization.
    Provides metrics for shard load calculation.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the resource monitor.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Default configuration
        self.update_interval = self.config.get("update_interval", 1.0)  # seconds
        self.history_size = self.config.get(
            "history_size", 60
        )  # Keep 1 minute of history by default

        # Resource metrics
        self.cpu_usage = []
        self.memory_usage = []
        self.disk_io = []
        self.network_io = []

        # Transaction metrics
        self.pending_tx_count = 0
        self.pending_tx_history = []

        # Cross-shard metrics
        self.cross_shard_traffic = {}  # shard_id -> bytes
        self.cross_shard_traffic_history = []

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.monitor_thread = None

    def start(self):
        """
        Start the resource monitor.
        """
        if self.running:
            return

        self.running = True

        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self):
        """
        Stop the resource monitor.
        """
        self.running = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None

    def _monitor_loop(self):
        """
        Background thread for periodic resource monitoring.
        """
        # Initialize network IO counters
        last_net_io = psutil.net_io_counters()
        last_time = time.time()

        while self.running:
            try:
                current_time = time.time()
                elapsed = current_time - last_time

                # Collect CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)

                # Collect memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent

                # Collect disk IO
                disk_io_counters = psutil.disk_io_counters()

                # Collect network IO
                net_io = psutil.net_io_counters()

                # Calculate network IO rates
                net_sent_rate = (net_io.bytes_sent - last_net_io.bytes_sent) / elapsed
                net_recv_rate = (net_io.bytes_recv - last_net_io.bytes_recv) / elapsed

                # Update last values
                last_net_io = net_io
                last_time = current_time

                # Update metrics with lock
                with self.lock:
                    # Add current metrics
                    self.cpu_usage.append(cpu_percent)
                    self.memory_usage.append(memory_percent)

                    self.disk_io.append(
                        {
                            "read_bytes": disk_io_counters.read_bytes,
                            "write_bytes": disk_io_counters.write_bytes,
                            "timestamp": current_time,
                        }
                    )

                    self.network_io.append(
                        {
                            "sent_rate": net_sent_rate,
                            "recv_rate": net_recv_rate,
                            "timestamp": current_time,
                        }
                    )

                    self.pending_tx_history.append(self.pending_tx_count)

                    # Add cross-shard traffic snapshot
                    self.cross_shard_traffic_history.append(
                        {
                            "traffic": self.cross_shard_traffic.copy(),
                            "timestamp": current_time,
                        }
                    )

                    # Trim history to configured size
                    if len(self.cpu_usage) > self.history_size:
                        self.cpu_usage = self.cpu_usage[-self.history_size :]

                    if len(self.memory_usage) > self.history_size:
                        self.memory_usage = self.memory_usage[-self.history_size :]

                    if len(self.disk_io) > self.history_size:
                        self.disk_io = self.disk_io[-self.history_size :]

                    if len(self.network_io) > self.history_size:
                        self.network_io = self.network_io[-self.history_size :]

                    if len(self.pending_tx_history) > self.history_size:
                        self.pending_tx_history = self.pending_tx_history[
                            -self.history_size :
                        ]

                    if len(self.cross_shard_traffic_history) > self.history_size:
                        self.cross_shard_traffic_history = (
                            self.cross_shard_traffic_history[-self.history_size :]
                        )

                # Sleep until next update
                time.sleep(self.update_interval)

            except Exception as e:
                print(f"Error in resource monitor: {e}")
                time.sleep(1.0)  # Shorter interval on error

    def update_pending_tx_count(self, count: int):
        """
        Update the pending transaction count.

        Args:
            count: Current pending transaction count
        """
        with self.lock:
            self.pending_tx_count = count

    def record_cross_shard_traffic(self, shard_id: str, bytes_sent: int):
        """
        Record cross-shard traffic.

        Args:
            shard_id: Target shard ID
            bytes_sent: Number of bytes sent
        """
        with self.lock:
            if shard_id in self.cross_shard_traffic:
                self.cross_shard_traffic[shard_id] += bytes_sent
            else:
                self.cross_shard_traffic[shard_id] = bytes_sent

    def get_cpu_usage(self, window: int = None) -> float:
        """
        Get average CPU usage over a time window.

        Args:
            window: Number of samples to average (None for all available)

        Returns:
            Average CPU usage as a percentage
        """
        with self.lock:
            if not self.cpu_usage:
                return 0.0

            if window is None or window >= len(self.cpu_usage):
                return sum(self.cpu_usage) / len(self.cpu_usage)

            return sum(self.cpu_usage[-window:]) / window

    def get_memory_usage(self, window: int = None) -> float:
        """
        Get average memory usage over a time window.

        Args:
            window: Number of samples to average (None for all available)

        Returns:
            Average memory usage as a percentage
        """
        with self.lock:
            if not self.memory_usage:
                return 0.0

            if window is None or window >= len(self.memory_usage):
                return sum(self.memory_usage) / len(self.memory_usage)

            return sum(self.memory_usage[-window:]) / window

    def get_network_io_rate(self, window: int = None) -> Dict[str, float]:
        """
        Get average network IO rates over a time window.

        Args:
            window: Number of samples to average (None for all available)

        Returns:
            Dictionary with average sent and received rates in bytes/second
        """
        with self.lock:
            if not self.network_io:
                return {"sent_rate": 0.0, "recv_rate": 0.0}

            if window is None or window >= len(self.network_io):
                samples = self.network_io
            else:
                samples = self.network_io[-window:]

            sent_rate = sum(sample["sent_rate"] for sample in samples) / len(samples)
            recv_rate = sum(sample["recv_rate"] for sample in samples) / len(samples)

            return {"sent_rate": sent_rate, "recv_rate": recv_rate}

    def get_pending_tx_count(self, window: int = None) -> float:
        """
        Get average pending transaction count over a time window.

        Args:
            window: Number of samples to average (None for all available)

        Returns:
            Average pending transaction count
        """
        with self.lock:
            if not self.pending_tx_history:
                return 0.0

            if window is None or window >= len(self.pending_tx_history):
                return sum(self.pending_tx_history) / len(self.pending_tx_history)

            return sum(self.pending_tx_history[-window:]) / window

    def get_cross_shard_traffic_rate(self, window: int = None) -> Dict[str, float]:
        """
        Get average cross-shard traffic rates over a time window.

        Args:
            window: Number of samples to average (None for all available)

        Returns:
            Dictionary mapping shard IDs to average traffic rates in bytes/second
        """
        with self.lock:
            if not self.cross_shard_traffic_history:
                return {}

            if window is None or window >= len(self.cross_shard_traffic_history):
                samples = self.cross_shard_traffic_history
            else:
                samples = self.cross_shard_traffic_history[-window:]

            # Collect all shard IDs
            all_shards = set()
            for sample in samples:
                all_shards.update(sample["traffic"].keys())

            # Calculate average rates
            result = {}
            for shard_id in all_shards:
                # Get traffic values for this shard
                values = [sample["traffic"].get(shard_id, 0) for sample in samples]

                # Calculate average
                if values:
                    result[shard_id] = sum(values) / len(values)
                else:
                    result[shard_id] = 0.0

            return result

    def get_load_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive load metrics.

        Returns:
            Dictionary with all load metrics
        """
        with self.lock:
            return {
                "cpu_usage": self.get_cpu_usage(),
                "memory_usage": self.get_memory_usage(),
                "network_io": self.get_network_io_rate(),
                "pending_tx_count": self.get_pending_tx_count(),
                "cross_shard_traffic": self.get_cross_shard_traffic_rate(),
                "timestamp": time.time(),
            }
