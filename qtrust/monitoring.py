#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Monitoring System
This module implements comprehensive monitoring capabilities for the QTrust system.
"""

import os
import sys
import time
import json
import csv
import threading
import logging
import datetime
import psutil
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("qtrust_monitor.log"), logging.StreamHandler()],
)


class MetricsCollector:
    """
    Collects and stores system-wide metrics.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the metrics collector.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Default configuration
        self.collection_interval = self.config.get(
            "collection_interval", 1.0
        )  # seconds
        self.history_size = self.config.get(
            "history_size", 3600
        )  # Keep 1 hour of history by default
        self.export_interval = self.config.get(
            "export_interval", 60.0
        )  # Export every minute
        self.export_dir = self.config.get("export_dir", "metrics")

        # Create export directory if it doesn't exist
        os.makedirs(self.export_dir, exist_ok=True)

        # Metrics storage
        self.metrics = {
            "timestamp": [],
            "tps": [],
            "latency": [],
            "cpu_usage": [],
            "memory_usage": [],
            "disk_io": [],
            "network_io": [],
            "cross_shard_traffic": [],
            "trust_convergence": [],
            "consensus_rounds": [],
            "proposal_success_rate": [],
            "transaction_validation_rate": [],
        }

        # Custom metrics
        self.custom_metrics = {}

        # Metric callbacks
        self.metric_callbacks = {}

        # Register default callbacks
        self._register_default_callbacks()

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.collection_thread = None
        self.export_thread = None

    def _register_default_callbacks(self):
        """
        Register default metric callbacks.
        """
        self.register_metric_callback("cpu_usage", self._collect_cpu_usage)
        self.register_metric_callback("memory_usage", self._collect_memory_usage)
        self.register_metric_callback("disk_io", self._collect_disk_io)
        self.register_metric_callback("network_io", self._collect_network_io)

    def register_metric_callback(self, metric_name: str, callback: Callable[[], Any]):
        """
        Register a callback for collecting a metric.

        Args:
            metric_name: Metric name
            callback: Callback function that returns the metric value
        """
        self.metric_callbacks[metric_name] = callback

        # Initialize metric storage if not exists
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

    def register_custom_metric(self, metric_name: str, initial_value: Any = None):
        """
        Register a custom metric.

        Args:
            metric_name: Metric name
            initial_value: Initial value for the metric
        """
        self.custom_metrics[metric_name] = initial_value

        # Initialize metric storage if not exists
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []

    def update_custom_metric(self, metric_name: str, value: Any):
        """
        Update a custom metric.

        Args:
            metric_name: Metric name
            value: New value for the metric
        """
        if metric_name in self.custom_metrics:
            self.custom_metrics[metric_name] = value

    def update_tps(self, tps: float):
        """
        Update transactions per second metric.

        Args:
            tps: Transactions per second
        """
        with self.lock:
            self.metrics["tps"].append(tps)
            self._trim_metric("tps")

    def update_latency(self, latency: float):
        """
        Update latency metric.

        Args:
            latency: Latency in milliseconds
        """
        with self.lock:
            self.metrics["latency"].append(latency)
            self._trim_metric("latency")

    def update_cross_shard_traffic(self, traffic: float):
        """
        Update cross-shard traffic metric.

        Args:
            traffic: Cross-shard traffic in bytes
        """
        with self.lock:
            self.metrics["cross_shard_traffic"].append(traffic)
            self._trim_metric("cross_shard_traffic")

    def update_trust_convergence(self, convergence: float):
        """
        Update trust convergence metric.

        Args:
            convergence: Trust convergence value in range [0, 1]
        """
        with self.lock:
            self.metrics["trust_convergence"].append(convergence)
            self._trim_metric("trust_convergence")

    def update_consensus_rounds(self, rounds: int):
        """
        Update consensus rounds metric.

        Args:
            rounds: Number of consensus rounds
        """
        with self.lock:
            self.metrics["consensus_rounds"].append(rounds)
            self._trim_metric("consensus_rounds")

    def update_proposal_success_rate(self, rate: float):
        """
        Update proposal success rate metric.

        Args:
            rate: Proposal success rate in range [0, 1]
        """
        with self.lock:
            self.metrics["proposal_success_rate"].append(rate)
            self._trim_metric("proposal_success_rate")

    def update_transaction_validation_rate(self, rate: float):
        """
        Update transaction validation rate metric.

        Args:
            rate: Transaction validation rate in range [0, 1]
        """
        with self.lock:
            self.metrics["transaction_validation_rate"].append(rate)
            self._trim_metric("transaction_validation_rate")

    def start(self):
        """
        Start metrics collection.
        """
        if self.running:
            return

        self.running = True

        # Start collection thread
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()

        # Start export thread
        self.export_thread = threading.Thread(target=self._export_loop)
        self.export_thread.daemon = True
        self.export_thread.start()

        logging.info("Metrics collection started")

    def stop(self):
        """
        Stop metrics collection.
        """
        if not self.running:
            return

        self.running = False

        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
            self.collection_thread = None

        if self.export_thread:
            self.export_thread.join(timeout=5.0)
            self.export_thread = None

        logging.info("Metrics collection stopped")

    def get_metric(self, metric_name: str, window: int = None) -> List[Any]:
        """
        Get metric values.

        Args:
            metric_name: Metric name
            window: Number of most recent values to return (None for all)

        Returns:
            List of metric values
        """
        with self.lock:
            if metric_name not in self.metrics:
                return []

            values = self.metrics[metric_name]

            if window is None or window >= len(values):
                return values.copy()

            return values[-window:].copy()

    def get_metric_average(
        self, metric_name: str, window: int = None
    ) -> Optional[float]:
        """
        Get average metric value.

        Args:
            metric_name: Metric name
            window: Number of most recent values to average (None for all)

        Returns:
            Average metric value or None if no values
        """
        values = self.get_metric(metric_name, window)

        if not values:
            return None

        # Filter out non-numeric values
        numeric_values = [v for v in values if isinstance(v, (int, float))]

        if not numeric_values:
            return None

        return sum(numeric_values) / len(numeric_values)

    def get_metric_max(self, metric_name: str, window: int = None) -> Optional[float]:
        """
        Get maximum metric value.

        Args:
            metric_name: Metric name
            window: Number of most recent values to consider (None for all)

        Returns:
            Maximum metric value or None if no values
        """
        values = self.get_metric(metric_name, window)

        if not values:
            return None

        # Filter out non-numeric values
        numeric_values = [v for v in values if isinstance(v, (int, float))]

        if not numeric_values:
            return None

        return max(numeric_values)

    def get_metric_min(self, metric_name: str, window: int = None) -> Optional[float]:
        """
        Get minimum metric value.

        Args:
            metric_name: Metric name
            window: Number of most recent values to consider (None for all)

        Returns:
            Minimum metric value or None if no values
        """
        values = self.get_metric(metric_name, window)

        if not values:
            return None

        # Filter out non-numeric values
        numeric_values = [v for v in values if isinstance(v, (int, float))]

        if not numeric_values:
            return None

        return min(numeric_values)

    def get_all_metrics(self, window: int = None) -> Dict[str, List[Any]]:
        """
        Get all metrics.

        Args:
            window: Number of most recent values to return (None for all)

        Returns:
            Dictionary mapping metric names to values
        """
        result = {}

        for metric_name in self.metrics:
            result[metric_name] = self.get_metric(metric_name, window)

        return result

    def export_metrics_json(self, filename: str = None) -> str:
        """
        Export metrics to JSON file.

        Args:
            filename: Output filename (None for auto-generated)

        Returns:
            Path to the exported file
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        filepath = os.path.join(self.export_dir, filename)

        with self.lock:
            with open(filepath, "w") as f:
                json.dump(self.metrics, f, indent=2)

        logging.info(f"Metrics exported to {filepath}")

        return filepath

    def export_metrics_csv(self, filename: str = None) -> str:
        """
        Export metrics to CSV file.

        Args:
            filename: Output filename (None for auto-generated)

        Returns:
            Path to the exported file
        """
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.csv"

        filepath = os.path.join(self.export_dir, filename)

        with self.lock:
            # Prepare data
            data = []
            timestamps = self.metrics.get("timestamp", [])

            # Create header row
            header = ["timestamp"]
            for metric_name in sorted(self.metrics.keys()):
                if metric_name != "timestamp":
                    header.append(metric_name)

            # Create data rows
            for i in range(len(timestamps)):
                row = [timestamps[i]]

                for metric_name in sorted(self.metrics.keys()):
                    if metric_name != "timestamp":
                        values = self.metrics[metric_name]
                        row.append(values[i] if i < len(values) else "")

                data.append(row)

            # Write to CSV
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(data)

        logging.info(f"Metrics exported to {filepath}")

        return filepath

    def generate_plots(self, output_dir: str = None) -> List[str]:
        """
        Generate plots for metrics.

        Args:
            output_dir: Output directory (None for default)

        Returns:
            List of paths to generated plot files
        """
        if output_dir is None:
            output_dir = os.path.join(self.export_dir, "plots")

        os.makedirs(output_dir, exist_ok=True)

        plot_files = []

        with self.lock:
            timestamps = self.metrics.get("timestamp", [])

            if not timestamps:
                return plot_files

            # Convert timestamps to datetime objects
            datetime_timestamps = [
                datetime.datetime.fromtimestamp(ts) for ts in timestamps
            ]

            # Generate plots for each metric
            for metric_name, values in self.metrics.items():
                if metric_name == "timestamp" or not values:
                    continue

                # Filter out non-numeric values
                numeric_indices = [
                    i for i, v in enumerate(values) if isinstance(v, (int, float))
                ]

                if not numeric_indices:
                    continue

                filtered_timestamps = [
                    datetime_timestamps[i]
                    for i in numeric_indices
                    if i < len(datetime_timestamps)
                ]
                filtered_values = [values[i] for i in numeric_indices]

                if not filtered_timestamps or not filtered_values:
                    continue

                # Create plot
                plt.figure(figsize=(10, 6))
                plt.plot(filtered_timestamps, filtered_values)
                plt.title(f"{metric_name.replace('_', ' ').title()} over Time")
                plt.xlabel("Time")
                plt.ylabel(metric_name.replace("_", " ").title())
                plt.grid(True)

                # Format x-axis
                plt.gcf().autofmt_xdate()

                # Save plot
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{metric_name}_{timestamp}.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath)
                plt.close()

                plot_files.append(filepath)

        logging.info(f"Generated {len(plot_files)} plots in {output_dir}")

        return plot_files

    def _collection_loop(self):
        """
        Main collection loop.
        """
        while self.running:
            try:
                self._collect_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                logging.error(f"Error in metrics collection: {e}")
                time.sleep(1.0)

    def _export_loop(self):
        """
        Main export loop.
        """
        while self.running:
            try:
                time.sleep(self.export_interval)
                self.export_metrics_json()
                self.export_metrics_csv()
            except Exception as e:
                logging.error(f"Error in metrics export: {e}")
                time.sleep(1.0)

    def _collect_metrics(self):
        """
        Collect all metrics.
        """
        with self.lock:
            # Add timestamp
            current_time = time.time()
            self.metrics["timestamp"].append(current_time)
            self._trim_metric("timestamp")

            # Collect metrics using callbacks
            for metric_name, callback in self.metric_callbacks.items():
                try:
                    value = callback()
                    self.metrics[metric_name].append(value)
                    self._trim_metric(metric_name)
                except Exception as e:
                    logging.error(f"Error collecting metric {metric_name}: {e}")

            # Add custom metrics
            for metric_name, value in self.custom_metrics.items():
                self.metrics[metric_name].append(value)
                self._trim_metric(metric_name)

    def _trim_metric(self, metric_name: str):
        """
        Trim a metric to the configured history size.

        Args:
            metric_name: Metric name
        """
        if (
            metric_name in self.metrics
            and len(self.metrics[metric_name]) > self.history_size
        ):
            self.metrics[metric_name] = self.metrics[metric_name][-self.history_size :]

    def _collect_cpu_usage(self) -> float:
        """
        Collect CPU usage.

        Returns:
            CPU usage as a percentage
        """
        return psutil.cpu_percent(interval=None)

    def _collect_memory_usage(self) -> float:
        """
        Collect memory usage.

        Returns:
            Memory usage as a percentage
        """
        return psutil.virtual_memory().percent

    def _collect_disk_io(self) -> Dict[str, int]:
        """
        Collect disk I/O.

        Returns:
            Dictionary with disk I/O statistics
        """
        io_counters = psutil.disk_io_counters()
        return {
            "read_bytes": io_counters.read_bytes,
            "write_bytes": io_counters.write_bytes,
        }

    def _collect_network_io(self) -> Dict[str, int]:
        """
        Collect network I/O.

        Returns:
            Dictionary with network I/O statistics
        """
        io_counters = psutil.net_io_counters()
        return {
            "bytes_sent": io_counters.bytes_sent,
            "bytes_recv": io_counters.bytes_recv,
        }


class TrustConvergenceMonitor:
    """
    Monitors trust convergence across the network.
    """

    def __init__(
        self, metrics_collector: MetricsCollector, config: Dict[str, Any] = None
    ):
        """
        Initialize the trust convergence monitor.

        Args:
            metrics_collector: Metrics collector
            config: Configuration dictionary
        """
        self.metrics_collector = metrics_collector
        self.config = config or {}

        # Default configuration
        self.update_interval = self.config.get("update_interval", 5.0)  # seconds

        # Trust data
        self.node_trust = {}  # node_id -> trust_value
        self.trust_history = []  # List of (timestamp, convergence) tuples

        # Register metrics
        self.metrics_collector.register_metric_callback(
            "trust_convergence", self._calculate_convergence
        )

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.monitor_thread = None

    def start(self):
        """
        Start trust convergence monitoring.
        """
        if self.running:
            return

        self.running = True

        # Start monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logging.info("Trust convergence monitoring started")

    def stop(self):
        """
        Stop trust convergence monitoring.
        """
        if not self.running:
            return

        self.running = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            self.monitor_thread = None

        logging.info("Trust convergence monitoring stopped")

    def update_node_trust(self, node_id: str, trust_value: float):
        """
        Update trust value for a node.

        Args:
            node_id: Node identifier
            trust_value: Trust value in range [0, 1]
        """
        with self.lock:
            self.node_trust[node_id] = trust_value

    def get_node_trust(self, node_id: str) -> Optional[float]:
        """
        Get trust value for a node.

        Args:
            node_id: Node identifier

        Returns:
            Trust value or None if node not found
        """
        with self.lock:
            return self.node_trust.get(node_id)

    def get_all_node_trust(self) -> Dict[str, float]:
        """
        Get trust values for all nodes.

        Returns:
            Dictionary mapping node IDs to trust values
        """
        with self.lock:
            return self.node_trust.copy()

    def get_trust_convergence(self) -> float:
        """
        Get current trust convergence value.

        Returns:
            Trust convergence value in range [0, 1]
        """
        return self._calculate_convergence()

    def get_trust_history(self) -> List[Tuple[float, float]]:
        """
        Get trust convergence history.

        Returns:
            List of (timestamp, convergence) tuples
        """
        with self.lock:
            return self.trust_history.copy()

    def _monitor_loop(self):
        """
        Main monitoring loop.
        """
        while self.running:
            try:
                # Calculate convergence
                convergence = self._calculate_convergence()

                # Update history
                with self.lock:
                    self.trust_history.append((time.time(), convergence))

                    # Trim history (keep last 1000 entries)
                    if len(self.trust_history) > 1000:
                        self.trust_history = self.trust_history[-1000:]

                # Sleep until next update
                time.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Error in trust convergence monitoring: {e}")
                time.sleep(1.0)

    def _calculate_convergence(self) -> float:
        """
        Calculate trust convergence.

        Returns:
            Trust convergence value in range [0, 1]
        """
        with self.lock:
            if not self.node_trust:
                return 0.0

            # Get trust values
            trust_values = list(self.node_trust.values())

            # Calculate standard deviation
            if len(trust_values) > 1:
                mean = sum(trust_values) / len(trust_values)
                variance = sum((x - mean) ** 2 for x in trust_values) / len(
                    trust_values
                )
                std_dev = variance**0.5

                # Calculate convergence (1 - normalized std_dev)
                # Lower std_dev means higher convergence
                if mean > 0:
                    normalized_std_dev = std_dev / mean
                    convergence = max(0.0, min(1.0, 1.0 - normalized_std_dev))
                else:
                    convergence = 0.0
            else:
                # Only one node, perfect convergence
                convergence = 1.0

            return convergence


class CrossShardMonitor:
    """
    Monitors cross-shard communication and routing delays.
    """

    def __init__(
        self, metrics_collector: MetricsCollector, config: Dict[str, Any] = None
    ):
        """
        Initialize the cross-shard monitor.

        Args:
            metrics_collector: Metrics collector
            config: Configuration dictionary
        """
        self.metrics_collector = metrics_collector
        self.config = config or {}

        # Default configuration
        self.update_interval = self.config.get("update_interval", 5.0)  # seconds

        # Cross-shard data
        self.shard_latencies = {}  # (source_shard, target_shard) -> latency
        self.shard_traffic = {}  # (source_shard, target_shard) -> traffic
        self.routing_delays = {}  # (source_shard, target_shard) -> delay

        # Register metrics
        self.metrics_collector.register_metric_callback(
            "cross_shard_latency", self._calculate_avg_latency
        )
        self.metrics_collector.register_metric_callback(
            "cross_shard_traffic", self._calculate_total_traffic
        )
        self.metrics_collector.register_metric_callback(
            "routing_delay", self._calculate_avg_routing_delay
        )

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.monitor_thread = None

    def start(self):
        """
        Start cross-shard monitoring.
        """
        if self.running:
            return

        self.running = True

        # Start monitor thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

        logging.info("Cross-shard monitoring started")

    def stop(self):
        """
        Stop cross-shard monitoring.
        """
        if not self.running:
            return

        self.running = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
            self.monitor_thread = None

        logging.info("Cross-shard monitoring stopped")

    def update_shard_latency(
        self, source_shard: str, target_shard: str, latency: float
    ):
        """
        Update latency between two shards.

        Args:
            source_shard: Source shard identifier
            target_shard: Target shard identifier
            latency: Latency in milliseconds
        """
        with self.lock:
            self.shard_latencies[(source_shard, target_shard)] = latency

    def update_shard_traffic(
        self, source_shard: str, target_shard: str, traffic: float
    ):
        """
        Update traffic between two shards.

        Args:
            source_shard: Source shard identifier
            target_shard: Target shard identifier
            traffic: Traffic in bytes
        """
        with self.lock:
            self.shard_traffic[(source_shard, target_shard)] = traffic

    def update_routing_delay(self, source_shard: str, target_shard: str, delay: float):
        """
        Update routing delay between two shards.

        Args:
            source_shard: Source shard identifier
            target_shard: Target shard identifier
            delay: Routing delay in milliseconds
        """
        with self.lock:
            self.routing_delays[(source_shard, target_shard)] = delay

    def get_shard_latency(
        self, source_shard: str, target_shard: str
    ) -> Optional[float]:
        """
        Get latency between two shards.

        Args:
            source_shard: Source shard identifier
            target_shard: Target shard identifier

        Returns:
            Latency in milliseconds or None if not found
        """
        with self.lock:
            return self.shard_latencies.get((source_shard, target_shard))

    def get_shard_traffic(
        self, source_shard: str, target_shard: str
    ) -> Optional[float]:
        """
        Get traffic between two shards.

        Args:
            source_shard: Source shard identifier
            target_shard: Target shard identifier

        Returns:
            Traffic in bytes or None if not found
        """
        with self.lock:
            return self.shard_traffic.get((source_shard, target_shard))

    def get_routing_delay(
        self, source_shard: str, target_shard: str
    ) -> Optional[float]:
        """
        Get routing delay between two shards.

        Args:
            source_shard: Source shard identifier
            target_shard: Target shard identifier

        Returns:
            Routing delay in milliseconds or None if not found
        """
        with self.lock:
            return self.routing_delays.get((source_shard, target_shard))

    def get_all_shard_latencies(self) -> Dict[Tuple[str, str], float]:
        """
        Get all shard latencies.

        Returns:
            Dictionary mapping (source_shard, target_shard) to latency
        """
        with self.lock:
            return self.shard_latencies.copy()

    def get_all_shard_traffic(self) -> Dict[Tuple[str, str], float]:
        """
        Get all shard traffic.

        Returns:
            Dictionary mapping (source_shard, target_shard) to traffic
        """
        with self.lock:
            return self.shard_traffic.copy()

    def get_all_routing_delays(self) -> Dict[Tuple[str, str], float]:
        """
        Get all routing delays.

        Returns:
            Dictionary mapping (source_shard, target_shard) to routing delay
        """
        with self.lock:
            return self.routing_delays.copy()

    def _monitor_loop(self):
        """
        Main monitoring loop.
        """
        while self.running:
            try:
                # Calculate metrics
                avg_latency = self._calculate_avg_latency()
                total_traffic = self._calculate_total_traffic()
                avg_routing_delay = self._calculate_avg_routing_delay()

                # Log metrics
                logging.info(
                    f"Cross-shard metrics: avg_latency={avg_latency:.2f}ms, "
                    f"total_traffic={total_traffic:.2f}B, "
                    f"avg_routing_delay={avg_routing_delay:.2f}ms"
                )

                # Sleep until next update
                time.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Error in cross-shard monitoring: {e}")
                time.sleep(1.0)

    def _calculate_avg_latency(self) -> float:
        """
        Calculate average shard latency.

        Returns:
            Average latency in milliseconds
        """
        with self.lock:
            if not self.shard_latencies:
                return 0.0

            return sum(self.shard_latencies.values()) / len(self.shard_latencies)

    def _calculate_total_traffic(self) -> float:
        """
        Calculate total shard traffic.

        Returns:
            Total traffic in bytes
        """
        with self.lock:
            return sum(self.shard_traffic.values())

    def _calculate_avg_routing_delay(self) -> float:
        """
        Calculate average routing delay.

        Returns:
            Average routing delay in milliseconds
        """
        with self.lock:
            if not self.routing_delays:
                return 0.0

            return sum(self.routing_delays.values()) / len(self.routing_delays)


class QTrustMonitor:
    """
    Main monitoring system for the QTrust blockchain sharding framework.
    """

    def __init__(self, config_file: str = None):
        """
        Initialize the QTrust monitor.

        Args:
            config_file: Path to configuration file
        """
        self.config = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                self.config = json.load(f)

        # Create metrics collector
        self.metrics_collector = MetricsCollector(
            self.config.get("metrics_collector", {})
        )

        # Create trust convergence monitor
        self.trust_monitor = TrustConvergenceMonitor(
            self.metrics_collector, self.config.get("trust_monitor", {})
        )

        # Create cross-shard monitor
        self.cross_shard_monitor = CrossShardMonitor(
            self.metrics_collector, self.config.get("cross_shard_monitor", {})
        )

        # Running flag
        self.running = False

    def start(self):
        """
        Start the QTrust monitor.
        """
        if self.running:
            return

        self.running = True

        # Start components
        self.metrics_collector.start()
        self.trust_monitor.start()
        self.cross_shard_monitor.start()

        logging.info("QTrust monitor started")

    def stop(self):
        """
        Stop the QTrust monitor.
        """
        if not self.running:
            return

        self.running = False

        # Stop components
        self.metrics_collector.stop()
        self.trust_monitor.stop()
        self.cross_shard_monitor.stop()

        logging.info("QTrust monitor stopped")

    def update_tps(self, tps: float):
        """
        Update transactions per second metric.

        Args:
            tps: Transactions per second
        """
        self.metrics_collector.update_tps(tps)

    def update_latency(self, latency: float):
        """
        Update latency metric.

        Args:
            latency: Latency in milliseconds
        """
        self.metrics_collector.update_latency(latency)

    def update_node_trust(self, node_id: str, trust_value: float):
        """
        Update trust value for a node.

        Args:
            node_id: Node identifier
            trust_value: Trust value in range [0, 1]
        """
        self.trust_monitor.update_node_trust(node_id, trust_value)

    def update_shard_metrics(
        self,
        source_shard: str,
        target_shard: str,
        latency: float,
        traffic: float,
        routing_delay: float,
    ):
        """
        Update metrics between two shards.

        Args:
            source_shard: Source shard identifier
            target_shard: Target shard identifier
            latency: Latency in milliseconds
            traffic: Traffic in bytes
            routing_delay: Routing delay in milliseconds
        """
        self.cross_shard_monitor.update_shard_latency(
            source_shard, target_shard, latency
        )
        self.cross_shard_monitor.update_shard_traffic(
            source_shard, target_shard, traffic
        )
        self.cross_shard_monitor.update_routing_delay(
            source_shard, target_shard, routing_delay
        )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all metrics.

        Returns:
            Dictionary with metrics summary
        """
        return {
            "tps": self.metrics_collector.get_metric_average("tps", 10),
            "latency": self.metrics_collector.get_metric_average("latency", 10),
            "cpu_usage": self.metrics_collector.get_metric_average("cpu_usage", 10),
            "memory_usage": self.metrics_collector.get_metric_average(
                "memory_usage", 10
            ),
            "cross_shard_latency": self.cross_shard_monitor._calculate_avg_latency(),
            "cross_shard_traffic": self.cross_shard_monitor._calculate_total_traffic(),
            "routing_delay": self.cross_shard_monitor._calculate_avg_routing_delay(),
            "trust_convergence": self.trust_monitor.get_trust_convergence(),
            "timestamp": time.time(),
        }

    def export_all_metrics(self) -> Dict[str, str]:
        """
        Export all metrics to files.

        Returns:
            Dictionary mapping export type to file path
        """
        exports = {}

        # Export metrics to JSON
        exports["json"] = self.metrics_collector.export_metrics_json()

        # Export metrics to CSV
        exports["csv"] = self.metrics_collector.export_metrics_csv()

        # Generate plots
        plot_files = self.metrics_collector.generate_plots()
        if plot_files:
            exports["plots"] = plot_files[0]  # Return first plot file

        return exports


if __name__ == "__main__":
    # Example usage
    monitor = QTrustMonitor()

    # Start monitoring
    monitor.start()

    # Simulate some metrics
    for i in range(100):
        # Update TPS and latency
        monitor.update_tps(random.uniform(100, 1000))
        monitor.update_latency(random.uniform(10, 500))

        # Update node trust
        for node_id in range(10):
            monitor.update_node_trust(f"node_{node_id}", random.uniform(0.5, 1.0))

        # Update shard metrics
        for source in range(5):
            for target in range(5):
                if source != target:
                    monitor.update_shard_metrics(
                        f"shard_{source}",
                        f"shard_{target}",
                        random.uniform(10, 200),  # latency
                        random.uniform(1000, 10000),  # traffic
                        random.uniform(5, 50),  # routing delay
                    )

        # Sleep for a bit
        time.sleep(0.1)

    # Get metrics summary
    summary = monitor.get_metrics_summary()
    print("Metrics summary:", json.dumps(summary, indent=2))

    # Export metrics
    exports = monitor.export_all_metrics()
    print("Exported metrics:", exports)

    # Stop monitoring
    monitor.stop()
