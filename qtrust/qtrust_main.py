#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Main Module
Version: 3.0.0
"""

import os
import sys
import argparse
import logging
import json
import time
from typing import Dict, List, Any

# Import the unified framework
from qtrust.qtrust_framework import QTrustFramework


def parse_arguments():
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="QTrust Blockchain Sharding Framework v3.0.0"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["all", "throughput", "latency", "cross_shard", "byzantine"],
        help="Run benchmark",
    )
    parser.add_argument(
        "--output", type=str, help="Path to output file for benchmark results"
    )
    parser.add_argument("--shards", type=int, help="Number of shards")
    parser.add_argument("--validators", type=int, help="Number of validators")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level",
    )

    return parser.parse_args()


def run_event_loop(framework):
    """
    Run the main event loop.

    Args:
        framework: QTrust framework instance
    """
    try:
        # Keep running until interrupted
        while True:
            # Process any pending tasks
            framework.process_pending_tasks()

            # Sleep to avoid high CPU usage
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down QTrust framework...")
    finally:
        # Stop the framework
        framework.stop()


def main():
    """
    Main entry point for the QTrust blockchain sharding framework.
    """
    # Parse command line arguments
    args = parse_arguments()

    # Initialize the unified QTrust framework with config file
    config = None
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            print("Using default configuration")
    
    # Create the framework with the loaded config or None for defaults
    framework = QTrustFramework(config)

    # Apply command line overrides
    if args.shards:
        framework.set_num_shards(args.shards)

    if args.validators:
        framework.set_num_validators(args.validators)

    if args.log_level:
        logging_level = getattr(logging, args.log_level)
        framework.logger.setLevel(logging_level)
        logging.basicConfig(level=logging_level)

    # Run benchmark if specified
    if args.benchmark:
        print(f"Running {args.benchmark} benchmark...")
        results = framework.run_benchmark(args.benchmark, args.output)
        print(json.dumps(results, indent=2))
        return

    # Start the framework
    framework.start()
    print(f"QTrust framework v3.0.0 started successfully")

    # Print status information
    status = framework.get_status()
    print(
        f"Running with {status['shard_cluster']['num_shards']} shards and "
        f"{status['shard_cluster']['num_validators']} validators"
    )

    # Run the main event loop
    run_event_loop(framework)


if __name__ == "__main__":
    main()
