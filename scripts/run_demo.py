#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Demo Script
This script runs a simple demonstration of the QTrust Blockchain Sharding Framework
with a small network of 4 nodes in a single region.
"""

import os
import sys
import time
import json
import logging
import argparse

# Add the QTrust source directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrust.simulation.fixed_network_simulation import NetworkSimulation
from qtrust.benchmark.transaction_generator import TransactionGenerator
from qtrust.benchmark.metrics_collector import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("demo_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QTrustDemo")

def run_demo(verify_mode=False):
    """
    Run a simple demonstration of QTrust with 4 nodes in a single region.
    
    Args:
        verify_mode (bool): If True, run in verification mode (shorter duration)
    """
    logger.info("Starting QTrust demonstration with 4 nodes in a single region")
    
    # Demo configuration
    config = {
        "simulation": {
            "num_regions": 1,
            "num_shards": 1,
            "nodes_per_shard": 4,
            "byzantine_ratio": 0.0,
            "network_latency": {
                "intra_region_ms": 10
            }
        },
        "transactions": {
            "tx_rate": 100,
            "tx_distribution": {
                "transfer": 0.8,
                "cross_shard": 0.0,
                "contract": 0.2
            }
        },
        "benchmark": {
            "duration": 10 if verify_mode else 60,  # Shorter duration in verify mode
            "warmup": 2 if verify_mode else 10,     # Shorter warmup in verify mode
            "cooldown": 2 if verify_mode else 10    # Shorter cooldown in verify mode
        }
    }
    
    # Create simulation
    simulation = NetworkSimulation(
        num_regions=config["simulation"]["num_regions"],
        num_shards=config["simulation"]["num_shards"], 
        nodes_per_shard=config["simulation"]["nodes_per_shard"],
        byzantine_ratio=config["simulation"]["byzantine_ratio"]
    )
    
    # Create transaction generator
    tx_generator = TransactionGenerator()
    tx_generator.config = config["transactions"]
    
    # Create metrics collector
    metrics_collector = MetricsCollector()
    metrics_collector.config = {
        "collection_interval": 1.0,
        "output_dir": "demo_results",
        "plot_graphs": True
    }
    
    try:
        # Initialize simulation
        logger.info("Initializing network simulation...")
        if not simulation.initialize():
            logger.error("Failed to initialize simulation")
            return False
        
        # Start simulation
        logger.info("Starting network simulation...")
        if not simulation.start_simulation():
            logger.error("Failed to start simulation")
            return False
        
        # Start metrics collection
        metrics_collector.start_collection(simulation, tx_generator)
        
        # Warmup phase
        logger.info(f"Warmup phase ({config['benchmark']['warmup']}s)...")
        time.sleep(config['benchmark']['warmup'])
        
        # Transaction generation phase
        logger.info(f"Transaction generation phase ({config['benchmark']['duration']}s)...")
        tx_generator.start_generation(config['benchmark']['duration'])
        
        # Wait for transaction generation to complete
        start_time = time.time()
        while tx_generator.running:
            elapsed = time.time() - start_time
            if elapsed > config['benchmark']['duration'] + 5:
                logger.warning("Transaction generation taking longer than expected, stopping...")
                tx_generator.stop_generation()
                break
                
            time.sleep(1.0)
            completion = min(100.0, 100.0 * elapsed / config['benchmark']['duration'])
            logger.info(f"Progress: {completion:.1f}%, Generated: {tx_generator.total_generated}, TPS: {tx_generator.get_tps():.2f}")
        
        # Cooldown phase
        logger.info(f"Cooldown phase ({config['benchmark']['cooldown']}s)...")
        time.sleep(config['benchmark']['cooldown'])
        
        # Stop metrics collection
        metrics_collector.stop_collection()
        
        # Stop simulation
        simulation.stop_simulation()
        
        # Generate report
        logger.info("Generating demo report...")
        summary = metrics_collector.get_summary_report()
        
        # Print summary
        logger.info("Demo completed successfully!")
        logger.info("\nPerformance Summary:")
        logger.info(summary)
        
        return True
        
    except Exception as e:
        logger.error(f"Error running demo: {e}")
        
        # Try to stop components
        if 'metrics_collector' in locals() and metrics_collector:
            metrics_collector.stop_collection()
        
        if 'simulation' in locals() and simulation:
            simulation.stop_simulation()
        
        return False

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="QTrust Demo Script")
    parser.add_argument("--verify", action="store_true", 
                        help="Run in verification mode with shorter duration")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs("demo_results", exist_ok=True)
    
    # Run the demo with appropriate mode
    success = run_demo(verify_mode=args.verify)
    
    # Exit with appropriate status
    sys.exit(0 if success else 1)
