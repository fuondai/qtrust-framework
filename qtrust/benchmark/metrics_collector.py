#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Benchmark Metrics Collector
This module implements comprehensive metrics collection for benchmarking.
"""

import os
import sys
import time
import json
import csv
import logging
import threading
import datetime
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("metrics_collector.log"), logging.StreamHandler()],
)
logger = logging.getLogger("MetricsCollector")


class MetricsCollector:
    """
    Collects and analyzes benchmark metrics.
    """

    def __init__(self, config_file: str = None):
        """
        Initialize the metrics collector.

        Args:
            config_file: Path to configuration file
        """
        self.config = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                self.config = json.load(f)

        # Default configuration
        self.collection_interval = self.config.get(
            "collection_interval", 1.0
        )  # seconds
        self.output_dir = self.config.get("output_dir", "benchmark_results")
        self.metrics_file = self.config.get("metrics_file", "metrics.json")
        self.csv_export = self.config.get("csv_export", True)
        self.plot_graphs = self.config.get("plot_graphs", True)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Metrics storage
        self.metrics = {
            "timestamp": [],
            "system": {
                "cpu_usage": [],
                "memory_usage": [],
                "network_in": [],
                "network_out": [],
            },
            "transactions": {
                "generated": [],
                "pending": [],
                "completed": [],
                "failed": [],
                "tps": [],
                "latency_avg": [],
                "latency_p95": [],
                "latency_p99": [],
            },
            "shards": {"count": [], "load_balance": [], "cross_shard_traffic": []},
            "consensus": {
                "rounds": [],
                "time_to_finality": [],
                "byzantine_detected": [],
            },
            "trust": {
                "average_score": [],
                "min_score": [],
                "max_score": [],
                "convergence_time": [],
            },
        }

        # Time series data for each node and shard
        self.node_metrics = defaultdict(
            lambda: defaultdict(list)
        )  # node_id -> metric -> [values]
        self.shard_metrics = defaultdict(
            lambda: defaultdict(list)
        )  # shard_id -> metric -> [values]

        # Event log
        self.events = []

        # Running flag
        self.running = False
        self.collection_thread = None
        self.start_time = None
        self.simulation = None
        self.tx_generator = None

    def register_event(self, event_type: str, details: Dict[str, Any]):
        """
        Register a benchmark event.

        Args:
            event_type: Event type
            details: Event details
        """
        self.events.append(
            {"timestamp": time.time(), "type": event_type, "details": details}
        )

    def start_collection(self, simulation, tx_generator):
        self.simulation = simulation
        self.tx_generator = tx_generator
        self.running = True
        self.start_time = time.time()
        print("Metrics collection started")
        
    def stop_collection(self):
        self.running = False
        print("Metrics collection stopped")
        
    def get_summary_report(self):
        """Generate a simple summary report for the demo"""
        total_tx = self.tx_generator.total_generated if self.tx_generator else 0
        tps = self.tx_generator.get_tps() if self.tx_generator else 0
        
        report = (
            f"Total transactions: {total_tx}\n"
            f"Average TPS: {tps:.2f}\n"
            f"Network configuration: {self.simulation.num_shards} shards, "
            f"{self.simulation.nodes_per_shard} nodes per shard\n"
        )
        
        # Create a simple chart file for demonstration
        os.makedirs(self.config.get("output_dir", "demo_results"), exist_ok=True)
        chart_file = os.path.join(self.config.get("output_dir", "demo_results"), "demo_tps.txt")
        with open(chart_file, "w") as f:
            f.write(f"Demo completed with {tps:.2f} TPS\n")
            
        return report

    def _collection_loop(self, simulation=None, tx_generator=None):
        """
        Main collection loop.

        Args:
            simulation: Optional BenchmarkSimulation instance
            tx_generator: Optional TransactionGenerator instance
        """
        while self.running:
            try:
                # Record timestamp
                current_time = time.time()
                self.metrics["timestamp"].append(current_time)

                # Collect system metrics
                self._collect_system_metrics()

                # Collect transaction metrics
                if tx_generator:
                    self._collect_transaction_metrics(tx_generator)

                # Collect simulation metrics
                if simulation:
                    self._collect_simulation_metrics(simulation)

                # Sleep until next collection
                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                time.sleep(1.0)

    def _collect_system_metrics(self):
        """
        Collect system-wide metrics.
        """
        try:
            # In a real implementation, this would use psutil to get actual system metrics
            # For the mock implementation, we'll generate random values

            # CPU usage (percentage)
            cpu_usage = 20.0 + 10.0 * np.random.random()
            self.metrics["system"]["cpu_usage"].append(cpu_usage)

            # Memory usage (percentage)
            memory_usage = 30.0 + 15.0 * np.random.random()
            self.metrics["system"]["memory_usage"].append(memory_usage)

            # Network I/O (bytes)
            network_in = int(1000 * np.random.random())
            network_out = int(800 * np.random.random())
            self.metrics["system"]["network_in"].append(network_in)
            self.metrics["system"]["network_out"].append(network_out)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")

    def _collect_transaction_metrics(self, tx_generator):
        """
        Collect transaction-related metrics.

        Args:
            tx_generator: TransactionGenerator instance
        """
        try:
            # Transaction counts
            self.metrics["transactions"]["generated"].append(
                tx_generator.total_generated
            )
            self.metrics["transactions"]["pending"].append(
                tx_generator.get_pending_count()
            )
            self.metrics["transactions"]["completed"].append(
                tx_generator.total_completed
            )
            self.metrics["transactions"]["failed"].append(tx_generator.total_failed)

            # Transactions per second
            self.metrics["transactions"]["tps"].append(tx_generator.get_tps())

            # Latency metrics
            self.metrics["transactions"]["latency_avg"].append(
                tx_generator.get_average_latency()
            )
            self.metrics["transactions"]["latency_p95"].append(
                tx_generator.get_percentile_latency(0.95)
            )
            self.metrics["transactions"]["latency_p99"].append(
                tx_generator.get_percentile_latency(0.99)
            )

        except Exception as e:
            logger.error(f"Error collecting transaction metrics: {e}")

    def _collect_simulation_metrics(self, simulation):
        """
        Collect simulation-related metrics.

        Args:
            simulation: BenchmarkSimulation instance
        """
        try:
            # Get simulation state
            state = simulation.get_state()

            # Shard metrics
            self.metrics["shards"]["count"].append(len(state["shards"]))

            # Calculate load balance (standard deviation of transactions per shard)
            shard_loads = [
                shard["transactions_pending"] for shard in state["shards"].values()
            ]
            load_balance = np.std(shard_loads) if shard_loads else 0
            self.metrics["shards"]["load_balance"].append(load_balance)

            # Calculate cross-shard traffic
            cross_shard_traffic = sum(
                sum(messages.values())
                for shard in state["shards"].values()
                for messages in [shard["cross_shard_messages"]]
            )
            self.metrics["shards"]["cross_shard_traffic"].append(cross_shard_traffic)

            # Consensus metrics
            consensus_rounds = sum(
                shard["consensus_rounds"] for shard in state["shards"].values()
            )
            self.metrics["consensus"]["rounds"].append(consensus_rounds)

            # Time to finality (mock value for now)
            time_to_finality = 500 + 100 * np.random.random()  # milliseconds
            self.metrics["consensus"]["time_to_finality"].append(time_to_finality)

            # Byzantine nodes detected
            byzantine_detected = sum(
                1 for node in state["nodes"].values() if node["is_byzantine"]
            )
            self.metrics["consensus"]["byzantine_detected"].append(byzantine_detected)

            # Trust metrics
            trust_scores = [node["trust_score"] for node in state["nodes"].values()]
            avg_trust = np.mean(trust_scores) if trust_scores else 0
            min_trust = np.min(trust_scores) if trust_scores else 0
            max_trust = np.max(trust_scores) if trust_scores else 0

            self.metrics["trust"]["average_score"].append(avg_trust)
            self.metrics["trust"]["min_score"].append(min_trust)
            self.metrics["trust"]["max_score"].append(max_trust)

            # Trust convergence time (mock value for now)
            convergence_time = 2000 + 500 * np.random.random()  # milliseconds
            self.metrics["trust"]["convergence_time"].append(convergence_time)

            # Collect per-node metrics
            for node_id, node in state["nodes"].items():
                self.node_metrics[node_id]["cpu_usage"].append(node["cpu_usage"])
                self.node_metrics[node_id]["memory_usage"].append(node["memory_usage"])
                self.node_metrics[node_id]["network_in"].append(node["network_in"])
                self.node_metrics[node_id]["network_out"].append(node["network_out"])
                self.node_metrics[node_id]["transactions_processed"].append(
                    node["transactions_processed"]
                )
                self.node_metrics[node_id]["trust_score"].append(node["trust_score"])

            # Collect per-shard metrics
            for shard_id, shard in state["shards"].items():
                self.shard_metrics[shard_id]["transactions_pending"].append(
                    shard["transactions_pending"]
                )
                self.shard_metrics[shard_id]["transactions_processed"].append(
                    shard["transactions_processed"]
                )
                self.shard_metrics[shard_id]["consensus_rounds"].append(
                    shard["consensus_rounds"]
                )

                # Calculate cross-shard traffic for this shard
                cross_traffic = sum(shard["cross_shard_messages"].values())
                self.shard_metrics[shard_id]["cross_shard_traffic"].append(
                    cross_traffic
                )

        except Exception as e:
            logger.error(f"Error collecting simulation metrics: {e}")

    def save_metrics(self):
        """
        Save collected metrics to file.

        Returns:
            Path to saved metrics file
        """
        # Create metrics object
        metrics_obj = {
            "timestamp": datetime.datetime.now().isoformat(),
            "duration": time.time() - self.start_time if self.start_time else 0,
            "metrics": self.metrics,
            "events": self.events,
            "summary": self.calculate_summary(),
        }

        # Save to JSON file
        metrics_path = os.path.join(self.output_dir, self.metrics_file)
        with open(metrics_path, "w") as f:
            json.dump(metrics_obj, f, indent=2)

        logger.info(f"Saved metrics to {metrics_path}")

        # Export to CSV if enabled
        if self.csv_export:
            self._export_to_csv()

        # Generate plots if enabled
        if self.plot_graphs:
            self._generate_plots()

        return metrics_path

    def _export_to_csv(self):
        """
        Export metrics to CSV files.
        """
        try:
            # System metrics
            self._export_metric_to_csv("system", self.metrics["system"])

            # Transaction metrics
            self._export_metric_to_csv("transactions", self.metrics["transactions"])

            # Shard metrics
            self._export_metric_to_csv("shards", self.metrics["shards"])

            # Consensus metrics
            self._export_metric_to_csv("consensus", self.metrics["consensus"])

            # Trust metrics
            self._export_metric_to_csv("trust", self.metrics["trust"])

            # Export events
            events_path = os.path.join(self.output_dir, "events.csv")
            with open(events_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "type", "details"])

                for event in self.events:
                    writer.writerow(
                        [
                            event["timestamp"],
                            event["type"],
                            json.dumps(event["details"]),
                        ]
                    )

            logger.info(f"Exported events to {events_path}")

        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")

    def _export_metric_to_csv(self, category: str, metrics: Dict[str, List[Any]]):
        """
        Export a category of metrics to CSV.

        Args:
            category: Metrics category
            metrics: Metrics dictionary
        """
        csv_path = os.path.join(self.output_dir, f"{category}_metrics.csv")

        with open(csv_path, "w", newline="") as f:
            # Get all metric names
            metric_names = list(metrics.keys())

            # Create writer and write header
            writer = csv.writer(f)
            writer.writerow(["timestamp"] + metric_names)

            # Write data rows
            for i in range(len(self.metrics["timestamp"])):
                row = [self.metrics["timestamp"][i]]

                for name in metric_names:
                    if i < len(metrics[name]):
                        row.append(metrics[name][i])
                    else:
                        row.append(None)

                writer.writerow(row)

        logger.info(f"Exported {category} metrics to {csv_path}")

    def _generate_plots(self):
        """
        Generate plots from collected metrics.
        """
        try:
            # Create plots directory
            plots_dir = os.path.join(self.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # Generate system metrics plots
            self._plot_metrics(
                "System Metrics",
                ["cpu_usage", "memory_usage"],
                self.metrics["system"],
                os.path.join(plots_dir, "system_usage.png"),
            )

            self._plot_metrics(
                "Network I/O",
                ["network_in", "network_out"],
                self.metrics["system"],
                os.path.join(plots_dir, "network_io.png"),
            )

            # Generate transaction metrics plots
            self._plot_metrics(
                "Transaction Counts",
                ["generated", "pending", "completed", "failed"],
                self.metrics["transactions"],
                os.path.join(plots_dir, "transaction_counts.png"),
            )

            self._plot_metrics(
                "Transaction Performance",
                ["tps"],
                self.metrics["transactions"],
                os.path.join(plots_dir, "transaction_tps.png"),
            )

            self._plot_metrics(
                "Transaction Latency",
                ["latency_avg", "latency_p95", "latency_p99"],
                self.metrics["transactions"],
                os.path.join(plots_dir, "transaction_latency.png"),
            )

            # Generate shard metrics plots
            self._plot_metrics(
                "Shard Metrics",
                ["count", "load_balance", "cross_shard_traffic"],
                self.metrics["shards"],
                os.path.join(plots_dir, "shard_metrics.png"),
            )

            # Generate consensus metrics plots
            self._plot_metrics(
                "Consensus Metrics",
                ["rounds", "time_to_finality", "byzantine_detected"],
                self.metrics["consensus"],
                os.path.join(plots_dir, "consensus_metrics.png"),
            )

            # Generate trust metrics plots
            self._plot_metrics(
                "Trust Scores",
                ["average_score", "min_score", "max_score"],
                self.metrics["trust"],
                os.path.join(plots_dir, "trust_scores.png"),
            )

            self._plot_metrics(
                "Trust Convergence",
                ["convergence_time"],
                self.metrics["trust"],
                os.path.join(plots_dir, "trust_convergence.png"),
            )

            logger.info(f"Generated plots in {plots_dir}")

        except Exception as e:
            logger.error(f"Error generating plots: {e}")

    def _plot_metrics(
        self,
        title: str,
        metric_names: List[str],
        metrics: Dict[str, List[Any]],
        output_path: str,
    ):
        """
        Plot a set of metrics.

        Args:
            title: Plot title
            metric_names: Names of metrics to plot
            metrics: Metrics dictionary
            output_path: Output file path
        """
        plt.figure(figsize=(10, 6))

        for name in metric_names:
            if name in metrics and metrics[name]:
                plt.plot(self.metrics["timestamp"], metrics[name], label=name)

        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)

        # Convert timestamps to relative time
        if self.metrics["timestamp"]:
            start_time = self.metrics["timestamp"][0]
            time_labels = [f"{t - start_time:.1f}" for t in self.metrics["timestamp"]]

            # Set x-ticks at regular intervals
            num_ticks = min(10, len(time_labels))
            if num_ticks > 1:
                tick_indices = np.linspace(
                    0, len(time_labels) - 1, num_ticks, dtype=int
                )
                plt.xticks(
                    np.array(self.metrics["timestamp"])[tick_indices],
                    np.array(time_labels)[tick_indices],
                )

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def calculate_summary(self) -> Dict[str, Any]:
        """
        Calculate summary statistics from collected metrics.

        Returns:
            Dictionary with summary statistics
        """
        summary = {}

        # System metrics
        if self.metrics["system"]["cpu_usage"]:
            summary["avg_cpu_usage"] = np.mean(self.metrics["system"]["cpu_usage"])
            summary["max_cpu_usage"] = np.max(self.metrics["system"]["cpu_usage"])

        if self.metrics["system"]["memory_usage"]:
            summary["avg_memory_usage"] = np.mean(
                self.metrics["system"]["memory_usage"]
            )
            summary["max_memory_usage"] = np.max(self.metrics["system"]["memory_usage"])

        # Transaction metrics
        if self.metrics["transactions"]["tps"]:
            summary["avg_tps"] = np.mean(self.metrics["transactions"]["tps"])
            summary["max_tps"] = np.max(self.metrics["transactions"]["tps"])

        if self.metrics["transactions"]["latency_avg"]:
            summary["avg_latency"] = np.mean(
                self.metrics["transactions"]["latency_avg"]
            )

        if self.metrics["transactions"]["latency_p95"]:
            summary["avg_p95_latency"] = np.mean(
                self.metrics["transactions"]["latency_p95"]
            )

        if self.metrics["transactions"]["latency_p99"]:
            summary["avg_p99_latency"] = np.mean(
                self.metrics["transactions"]["latency_p99"]
            )

        # Transaction counts
        if self.metrics["transactions"]["generated"]:
            summary["total_transactions"] = self.metrics["transactions"]["generated"][
                -1
            ]

        if (
            self.metrics["transactions"]["completed"]
            and self.metrics["transactions"]["generated"]
        ):
            last_completed = self.metrics["transactions"]["completed"][-1]
            last_generated = self.metrics["transactions"]["generated"][-1]

            if last_generated > 0:
                summary["completion_rate"] = last_completed / last_generated

        # Consensus metrics
        if self.metrics["consensus"]["time_to_finality"]:
            summary["avg_time_to_finality"] = np.mean(
                self.metrics["consensus"]["time_to_finality"]
            )

        # Trust metrics
        if self.metrics["trust"]["average_score"]:
            summary["final_avg_trust"] = self.metrics["trust"]["average_score"][-1]

        if self.metrics["trust"]["convergence_time"]:
            summary["avg_trust_convergence"] = np.mean(
                self.metrics["trust"]["convergence_time"]
            )

        return summary


if __name__ == "__main__":
    # Example usage
    collector = MetricsCollector(
        {
            "collection_interval": 0.5,
            "output_dir": "benchmark_results",
            "plot_graphs": True,
        }
    )

    # Start collection
    collector.start_collection()

    # Simulate some activity
    for i in range(20):
        # Register some events
        if i % 5 == 0:
            collector.register_event("test_event", {"iteration": i, "value": i * 10})

        # Sleep a bit
        time.sleep(0.5)

    # Stop collection
    collector.stop_collection()

    # Save metrics
    collector.save_metrics()

    # Print summary report
    print(collector.get_summary_report())
