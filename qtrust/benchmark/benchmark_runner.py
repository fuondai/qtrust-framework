#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Benchmark Runner
This module implements a complete benchmark pipeline for the QTrust framework.
"""

import os
import sys
import time
import json
import argparse
import logging
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import benchmark components
from qtrust.benchmark.transaction_generator import TransactionGenerator
from qtrust.benchmark.metrics_collector import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark_runner.log"), logging.StreamHandler()],
)
logger = logging.getLogger("BenchmarkRunner")


class BenchmarkRunner:
    """
    Runs a complete benchmark pipeline for the QTrust framework.
    """

    def __init__(self, config_file: str = None):
        """
        Initialize the benchmark runner.

        Args:
            config_file: Path to configuration file
        """
        self.config = {}
        if config_file and os.path.exists(config_file):
            with open(config_file, "r") as f:
                self.config = json.load(f)

        # Default configuration
        self.output_dir = self.config.get("output_dir", "benchmark_results")
        self.warmup_duration = self.config.get("warmup_duration", 30)  # seconds
        self.benchmark_duration = self.config.get("benchmark_duration", 300)  # seconds
        self.cooldown_duration = self.config.get("cooldown_duration", 30)  # seconds

        # Simulation configuration
        self.simulation_config = self.config.get(
            "simulation",
            {
                "num_regions": 5,
                "num_shards": 64,
                "nodes_per_shard": 3,
                "byzantine_ratio": 0.2,
                "sybil_ratio": 0.1,
            },
        )

        # Transaction generator configuration
        self.transaction_config = self.config.get(
            "transactions",
            {
                "tx_rate": 100,  # transactions per second
                "tx_distribution": {
                    "transfer": 0.7,
                    "cross_shard": 0.2,
                    "contract": 0.1,
                },
            },
        )

        # Metrics collector configuration
        self.metrics_config = self.config.get(
            "metrics",
            {
                "collection_interval": 1.0,
                "output_dir": self.output_dir,
                "plot_graphs": True,
            },
        )

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Components
        self.simulation = None
        self.tx_generator = None
        self.metrics_collector = None

    def setup(self):
        """
        Set up the benchmark environment.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Setting up benchmark environment...")

            # Create simulation environment
            from qtrust.simulation.network_simulation import NetworkSimulation
            
            # Create simulation with direct config
            self.simulation = NetworkSimulation(**self.simulation_config)
            
            # Set up simulation
            if not self.simulation.initialize():
                logger.error("Failed to set up simulation")
                return False

            # Create transaction generator with direct config
            self.tx_generator = TransactionGenerator()
            self.tx_generator.config = self.transaction_config

            # Create metrics collector with direct config
            self.metrics_collector = MetricsCollector()
            self.metrics_collector.config = self.metrics_config

            logger.info("Benchmark environment set up successfully")
            return True

        except Exception as e:
            logger.error(f"Error setting up benchmark environment: {e}")
            return False

    def run(self):
        """
        Run the benchmark.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting benchmark...")

            # Start simulation
            if not self.simulation.start_simulation():
                logger.error("Failed to start simulation")
                return False

            # Start metrics collection
            self.metrics_collector.start_collection(self.simulation, self.tx_generator)

            # Warmup phase
            logger.info(f"Warmup phase ({self.warmup_duration}s)...")
            time.sleep(self.warmup_duration)

            # Start transaction generation
            logger.info(f"Benchmark phase ({self.benchmark_duration}s)...")
            self.tx_generator.start_generation(self.benchmark_duration)

            # Wait for transaction generation to complete
            while self.tx_generator.running:
                time.sleep(1.0)

                # Log progress
                if self.tx_generator.total_generated > 0:
                    completion = min(
                        100.0,
                        100.0
                        * (time.time() - self.tx_generator.start_time)
                        / self.benchmark_duration,
                    )
                    logger.info(
                        f"Progress: {completion:.1f}%, Generated: {self.tx_generator.total_generated}, TPS: {self.tx_generator.get_tps():.2f}"
                    )

            # Cooldown phase
            logger.info(f"Cooldown phase ({self.cooldown_duration}s)...")
            time.sleep(self.cooldown_duration)

            # Stop metrics collection
            self.metrics_collector.stop_collection()

            # Stop simulation
            self.simulation.stop_simulation()

            # Save metrics
            metrics_path = self.metrics_collector.save_metrics()

            # Export transactions
            tx_path = self.tx_generator.export_transactions(
                os.path.join(self.output_dir, "transactions.csv")
            )

            logger.info(f"Benchmark completed successfully")
            logger.info(f"Metrics saved to: {metrics_path}")
            logger.info(f"Transactions saved to: {tx_path}")

            return True

        except Exception as e:
            logger.error(f"Error running benchmark: {e}")

            # Try to stop components
            if self.metrics_collector:
                self.metrics_collector.stop_collection()

            if self.simulation:
                self.simulation.stop()

            return False

    def generate_report(self):
        """
        Generate a comprehensive benchmark report.

        Returns:
            Path to the report file
        """
        try:
            logger.info("Generating benchmark report...")

            # Get summary report from metrics collector
            summary_report = self.metrics_collector.get_summary_report()

            # Create report file
            report_path = os.path.join(self.output_dir, "benchmark_report.md")

            with open(report_path, "w") as f:
                f.write("# QTrust Blockchain Sharding Framework - Benchmark Report\n\n")

                # Add timestamp
                f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Add configuration summary
                f.write("## Benchmark Configuration\n\n")
                f.write("### Simulation Parameters\n\n")
                f.write("| Parameter | Value |\n")
                f.write("|-----------|-------|\n")

                for key, value in self.simulation_config.items():
                    f.write(f"| {key} | {value} |\n")

                f.write("\n### Transaction Parameters\n\n")
                f.write("| Parameter | Value |\n")
                f.write("|-----------|-------|\n")

                for key, value in self.transaction_config.items():
                    if isinstance(value, dict):
                        f.write(f"| {key} | {json.dumps(value)} |\n")
                    else:
                        f.write(f"| {key} | {value} |\n")

                f.write("\n### Benchmark Duration\n\n")
                f.write(f"- Warmup: {self.warmup_duration} seconds\n")
                f.write(f"- Benchmark: {self.benchmark_duration} seconds\n")
                f.write(f"- Cooldown: {self.cooldown_duration} seconds\n")

                # Add summary report
                f.write("\n## Performance Summary\n\n")
                f.write("```\n")
                f.write(summary_report)
                f.write("\n```\n\n")

                # Add plots
                f.write("## Performance Graphs\n\n")

                plots_dir = os.path.join(self.output_dir, "plots")
                if os.path.exists(plots_dir):
                    plot_files = [
                        f for f in os.listdir(plots_dir) if f.endswith(".png")
                    ]

                    for plot_file in sorted(plot_files):
                        plot_path = os.path.join("plots", plot_file)
                        plot_name = (
                            plot_file.replace(".png", "").replace("_", " ").title()
                        )

                        f.write(f"### {plot_name}\n\n")
                        f.write(f"![{plot_name}]({plot_path})\n\n")

                # Add conclusion
                f.write("## Conclusion\n\n")
                f.write(
                    "This benchmark demonstrates the performance characteristics of the QTrust Blockchain Sharding Framework "
                )
                f.write(
                    "under various conditions. The results show the system's throughput, latency, and resource utilization "
                )
                f.write(
                    "with the configured number of shards, nodes, and Byzantine actors.\n\n"
                )

                f.write(
                    "For detailed analysis, refer to the raw metrics and transaction data in the benchmark results directory.\n"
                )

            logger.info(f"Benchmark report generated: {report_path}")
            return report_path

        except Exception as e:
            logger.error(f"Error generating benchmark report: {e}")
            return None


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="QTrust Blockchain Sharding Framework Benchmark Runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="benchmark_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory for benchmark results",
    )
    args = parser.parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)

    # Override output directory if specified
    if args.output_dir:
        config["output_dir"] = args.output_dir

    # Create benchmark runner
    runner = BenchmarkRunner()
    runner.config = config

    # Extract configuration sections
    runner.output_dir = config.get("output_dir", "benchmark_results")
    runner.warmup_duration = config.get("warmup_duration", 30)
    runner.benchmark_duration = config.get("benchmark_duration", 300)
    runner.cooldown_duration = config.get("cooldown_duration", 30)
    runner.simulation_config = config.get("simulation", {})
    runner.transaction_config = config.get("transactions", {})
    runner.metrics_config = config.get("metrics", {})

    # Set up benchmark
    if not runner.setup():
        logger.error("Failed to set up benchmark")
        return 1

    # Run benchmark
    if not runner.run():
        logger.error("Failed to run benchmark")
        return 1

    # Generate report
    report_path = runner.generate_report()
    if not report_path:
        logger.error("Failed to generate benchmark report")
        return 1

    logger.info(f"Benchmark completed successfully. Report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
