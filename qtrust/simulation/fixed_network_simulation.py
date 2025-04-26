#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Simplified Network Simulation
This module provides a simplified network simulation for demonstration purposes.
"""

import random
import time
import os
import sys
from typing import Dict, List, Any, Optional, Tuple

class SimpleNetworkSimulation:
    """
    Simplified network simulation for demonstration and testing.
    """

    def __init__(self, num_regions=5, num_shards=64, nodes_per_shard=3, 
                 byzantine_ratio=0.2, sybil_ratio=0.1, **kwargs):
        """
        Initialize the network simulation.

        Args:
            num_regions: Number of network regions
            num_shards: Number of shards
            nodes_per_shard: Number of nodes per shard
            byzantine_ratio: Ratio of Byzantine nodes
            sybil_ratio: Ratio of Sybil nodes
            **kwargs: Additional parameters
        """
        self.num_regions = num_regions
        self.num_shards = num_shards
        self.nodes_per_shard = nodes_per_shard
        self.byzantine_ratio = byzantine_ratio
        self.sybil_ratio = sybil_ratio
        
        # Additional parameters
        self.config = kwargs
        
        # Node and shard mapping
        self.nodes = []
        self.shards = {}
        self.node_to_shard = {}
        
        # Node roles
        self.byzantine_nodes = []
        self.sybil_nodes = []
        
        # Simulation state
        self.is_running = False
        
    def initialize(self):
        """
        Initialize the simulation environment.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Generate regions
            regions = ["us-east", "us-west", "eu-central", "asia-east", "asia-south"][:self.num_regions]
            
            # Generate nodes and assign to shards
            node_id = 0
            for shard_id in range(self.num_shards):
                shard_nodes = []
                for _ in range(self.nodes_per_shard):
                    node_name = f"node_{node_id}"
                    region = random.choice(regions)
                    
                    # Track node
                    self.nodes.append(node_name)
                    shard_nodes.append(node_name)
                    self.node_to_shard[node_name] = shard_id
                    
                    node_id += 1
                
                self.shards[shard_id] = shard_nodes
            
            # Designate Byzantine nodes
            num_byzantine = int(len(self.nodes) * self.byzantine_ratio)
            self.byzantine_nodes = random.sample(self.nodes, num_byzantine)
            
            # Designate Sybil nodes
            remaining_nodes = [n for n in self.nodes if n not in self.byzantine_nodes]
            num_sybil = int(len(self.nodes) * self.sybil_ratio)
            self.sybil_nodes = random.sample(remaining_nodes, min(num_sybil, len(remaining_nodes)))
            
            print(f"Initialized network with {len(self.nodes)} nodes in {self.num_shards} shards")
            print(f"Byzantine nodes: {len(self.byzantine_nodes)}")
            print(f"Sybil nodes: {len(self.sybil_nodes)}")
            
            return True
        
        except Exception as e:
            print(f"Error initializing simulation: {e}")
            return False
    
    def start_simulation(self):
        """
        Start the simulation.
        
        Returns:
            True if successful, False otherwise
        """
        if self.is_running:
            return True
            
        try:
            print("Starting network simulation...")
            self.is_running = True
            return True
        
        except Exception as e:
            print(f"Error starting simulation: {e}")
            return False
    
    def stop_simulation(self):
        """
        Stop the simulation.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_running:
            return True
            
        try:
            print("Stopping network simulation...")
            self.is_running = False
            return True
        
        except Exception as e:
            print(f"Error stopping simulation: {e}")
            return False
    
    def create_network_partition(self):
        """
        Create a network partition.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("Creating network partition...")
            return True
        
        except Exception as e:
            print(f"Error creating network partition: {e}")
            return False
    
    def heal_network_partition(self):
        """
        Heal all network partitions.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            print("Healing network partition...")
            return True
        
        except Exception as e:
            print(f"Error healing network partition: {e}")
            return False
            
    def get_state(self):
        """
        Get the current state of the simulation.
        Added for compatibility with full MetricsCollector implementation.
        
        Returns:
            Dictionary containing the current simulation state
        """
        # Create a simplified state representation
        state = {
            "nodes": {},
            "shards": {}
        }
        
        # Add basic info for each node
        for node_id in self.nodes:
            state["nodes"][node_id] = {
                "cpu_usage": random.uniform(10, 50),
                "memory_usage": random.uniform(20, 60),
                "network_in": random.randint(1000, 5000),
                "network_out": random.randint(800, 4000),
                "transactions_processed": random.randint(10, 100),
                "trust_score": random.uniform(0.5, 1.0),
                "is_byzantine": node_id in self.byzantine_nodes
            }
            
        # Add basic info for each shard
        for shard_id, nodes in self.shards.items():
            state["shards"][shard_id] = {
                "transactions_pending": random.randint(5, 50),
                "transactions_processed": random.randint(50, 200),
                "consensus_rounds": random.randint(1, 10),
                "cross_shard_messages": {
                    f"shard_{i}": random.randint(0, 20) 
                    for i in range(self.num_shards) 
                    if i != shard_id
                }
            }
            
        return state

# For backward compatibility
NetworkSimulation = SimpleNetworkSimulation 