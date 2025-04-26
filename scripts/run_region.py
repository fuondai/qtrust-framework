#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Region Simulator
This script simulates a region with multiple nodes and shards
Used for distributed running of the QTrust framework across multiple containers
"""

import os
import sys
import time
import json
import logging
import argparse
import threading
import requests
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Add the QTrust source directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qtrust.qtrust_framework import QTrustFramework

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"region_simulation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("RegionSimulator")

class RegionNode:
    """
    Simulates a single node in a region.
    Each node is a validator in a specific shard.
    """
    
    def __init__(self, node_id: str, region: str, shard_id: int, config: Dict[str, Any]):
        """
        Initialize a region node.
        
        Args:
            node_id: Unique identifier for this node
            region: Region identifier
            shard_id: Shard this node belongs to
            config: Configuration parameters
        """
        self.node_id = node_id
        self.region = region
        self.shard_id = shard_id
        self.config = config
        self.running = False
        self.framework = None
        self.logger = logging.getLogger(f"Node-{node_id}")
        
    def start(self) -> bool:
        """
        Start the node.
        
        Returns:
            Success status
        """
        try:
            # Create a QTrust framework instance with custom configuration
            node_config = self.config.copy()
            node_config["node_id"] = self.node_id
            node_config["region"] = self.region
            node_config["shard_id"] = self.shard_id
            
            # Initialize the framework
            self.framework = QTrustFramework(node_config)
            
            # Start the framework
            self.framework.start()
            self.running = True
            
            self.logger.info(f"Node {self.node_id} in region {self.region}, shard {self.shard_id} started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start node {self.node_id}: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the node.
        
        Returns:
            Success status
        """
        if not self.running:
            return True
            
        try:
            # Stop the framework
            if self.framework:
                self.framework.stop()
            
            self.running = False
            self.logger.info(f"Node {self.node_id} stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop node {self.node_id}: {e}")
            return False
    
    def process_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of transactions.
        
        Args:
            transactions: List of transactions to process
            
        Returns:
            List of transaction results
        """
        if not self.running or not self.framework:
            return [{"success": False, "error": "Node not running"}] * len(transactions)
            
        results = []
        for tx in transactions:
            result = self.framework.process_transaction(tx)
            results.append(result)
            
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the node.
        
        Returns:
            Status dictionary
        """
        if not self.framework:
            return {
                "node_id": self.node_id,
                "region": self.region,
                "shard_id": self.shard_id,
                "running": self.running,
                "status": "not_initialized"
            }
            
        # Get framework status
        framework_status = self.framework.get_status()
        
        return {
            "node_id": self.node_id,
            "region": self.region,
            "shard_id": self.shard_id,
            "running": self.running,
            "status": "running" if self.running else "stopped",
            "framework": framework_status
        }


class RegionSimulator:
    """
    Simulates a region with multiple nodes distributed across shards.
    """
    
    def __init__(self, region: str, node_count: int, shard_count: int, config_file: str):
        """
        Initialize the region simulator.
        
        Args:
            region: Region identifier
            node_count: Number of nodes in this region
            shard_count: Number of shards assigned to this region
            config_file: Path to configuration file
        """
        self.region = region
        self.node_count = node_count
        self.shard_count = shard_count
        self.config_file = config_file
        self.nodes = {}
        self.running = False
        self.control_node_url = os.environ.get("CONTROL_NODE", "qtrust-control:8000")
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = json.load(f)
            
        # Set up logger
        self.logger = logging.getLogger(f"Region-{region}")
        
    def initialize(self) -> bool:
        """
        Initialize the region by creating nodes and distributing them across shards.
        
        Returns:
            Success status
        """
        try:
            # Calculate nodes per shard (not all shards will have the same number)
            base_nodes_per_shard = self.node_count // self.shard_count
            extra_nodes = self.node_count % self.shard_count
            
            # Create nodes
            node_index = 0
            for shard_id in range(self.shard_count):
                # Calculate how many nodes for this shard
                shard_nodes = base_nodes_per_shard + (1 if shard_id < extra_nodes else 0)
                
                for i in range(shard_nodes):
                    node_id = f"{self.region}-s{shard_id}-n{i}"
                    self.nodes[node_id] = RegionNode(node_id, self.region, shard_id, self.config)
                    node_index += 1
            
            self.logger.info(f"Region {self.region} initialized with {len(self.nodes)} nodes across {self.shard_count} shards")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize region {self.region}: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start all nodes in the region.
        
        Returns:
            Success status
        """
        if self.running:
            self.logger.warning(f"Region {self.region} is already running")
            return True
            
        try:
            # Start nodes in parallel using a thread pool
            with ThreadPoolExecutor(max_workers=min(32, len(self.nodes))) as executor:
                futures = {executor.submit(node.start): node_id for node_id, node in self.nodes.items()}
                
                # Wait for all nodes to start
                success_count = 0
                for future in futures:
                    if future.result():
                        success_count += 1
            
            self.running = True
            self.logger.info(f"Region {self.region} started with {success_count}/{len(self.nodes)} nodes")
            
            # Register with control node
            self._register_with_control()
            
            return success_count == len(self.nodes)
            
        except Exception as e:
            self.logger.error(f"Failed to start region {self.region}: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop all nodes in the region.
        
        Returns:
            Success status
        """
        if not self.running:
            return True
            
        try:
            # Stop nodes in parallel using a thread pool
            with ThreadPoolExecutor(max_workers=min(32, len(self.nodes))) as executor:
                futures = {executor.submit(node.stop): node_id for node_id, node in self.nodes.items()}
                
                # Wait for all nodes to stop
                success_count = 0
                for future in futures:
                    if future.result():
                        success_count += 1
            
            self.running = False
            self.logger.info(f"Region {self.region} stopped with {success_count}/{len(self.nodes)} nodes")
            return success_count == len(self.nodes)
            
        except Exception as e:
            self.logger.error(f"Failed to stop region {self.region}: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the region.
        
        Returns:
            Status dictionary
        """
        node_statuses = {}
        for node_id, node in self.nodes.items():
            node_statuses[node_id] = {
                "running": node.running,
                "shard_id": node.shard_id
            }
            
        return {
            "region": self.region,
            "node_count": self.node_count,
            "shard_count": self.shard_count,
            "running": self.running,
            "nodes": node_statuses
        }
        
    def _register_with_control(self) -> bool:
        """
        Register this region with the control node.
        
        Returns:
            Success status
        """
        try:
            url = f"http://{self.control_node_url}/api/register_region"
            data = {
                "region": self.region,
                "node_count": self.node_count,
                "shard_count": self.shard_count,
                "status": "running"
            }
            
            response = requests.post(url, json=data, timeout=5)
            if response.status_code == 200:
                self.logger.info(f"Region {self.region} registered with control node")
                return True
            else:
                self.logger.warning(f"Failed to register region {self.region} with control node: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Failed to register region {self.region} with control node: {e}")
            return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run a QTrust region simulation')
    parser.add_argument('--region', type=str, required=True, help='Region identifier')
    parser.add_argument('--nodes', type=int, required=True, help='Number of nodes in this region')
    parser.add_argument('--shards', type=int, required=True, help='Number of shards assigned to this region')
    parser.add_argument('--config', type=str, default='/app/config/paper_large_scale.json', help='Path to configuration file')
    
    return parser.parse_args()


def main():
    """Main entry point for the region simulator."""
    # Parse arguments
    args = parse_arguments()
    
    # Create simulator
    simulator = RegionSimulator(args.region, args.nodes, args.shards, args.config)
    
    # Initialize simulator
    if not simulator.initialize():
        logger.error(f"Failed to initialize region {args.region}")
        sys.exit(1)
    
    # Start simulator
    if not simulator.start():
        logger.error(f"Failed to start region {args.region}")
        sys.exit(1)
    
    # Keep running until interrupted
    try:
        while True:
            time.sleep(60)
            logger.info(f"Region {args.region} running with {args.nodes} nodes across {args.shards} shards")
    except KeyboardInterrupt:
        logger.info(f"Stopping region {args.region}")
        simulator.stop()
    
    
if __name__ == "__main__":
    main() 