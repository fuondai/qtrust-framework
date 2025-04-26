"""
Integration tests for QTrust framework.

This module contains integration tests that verify the interaction
between different components of the QTrust framework.
"""

import os
import sys
import unittest
import json
from typing import Dict, List, Any

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.qtrust_framework import QTrustFramework
from qtrust.agents.rainbow_agent import RainbowDQNAgent
from qtrust.trust.htdcm import HTDCM
from qtrust.routing.mad_rapid import MADRAPIDRouter
from qtrust.consensus.adaptive_consensus import AdaptiveConsensusSelector


class TestComponentIntegration(unittest.TestCase):
    """Integration tests for QTrust framework components."""

    def setUp(self):
        """Set up test environment."""
        self.framework = QTrustFramework()
        self.rainbow_agent = RainbowDQNAgent()
        self.htdcm = HTDCM()
        self.mad_rapid = MADRAPIDRouter()
        self.adaptive_consensus = AdaptiveConsensusSelector()
        
        # Create results directory if it doesn't exist
        os.makedirs('integration_results', exist_ok=True)

    def test_rainbow_htdcm_integration(self):
        """Test integration between Rainbow DQN and HTDCM."""
        # Configure test network
        config = {
            'shards': 8,
            'nodes_per_shard': 12,
            'simulation_duration': 300  # seconds
        }
        
        # Initialize components
        self.rainbow_agent.initialize(config)
        self.htdcm.initialize(config)
        
        # Connect components
        self.rainbow_agent.register_trust_provider(self.htdcm)
        self.htdcm.register_shard_manager(self.rainbow_agent)
        
        # Run simulation
        for i in range(10):
            # Simulate network changes
            network_state = self.generate_network_state(config, i)
            
            # Update Rainbow agent with network state
            self.rainbow_agent.update_state(network_state)
            
            # Update HTDCM with trust information
            trust_updates = self.generate_trust_updates(config, i)
            self.htdcm.update_trust_metrics(trust_updates)
            
            # Get shard allocation from Rainbow agent
            shard_allocation = self.rainbow_agent.get_shard_allocation()
            
            # Verify that shard allocation considers trust metrics
            for node_id, shard_id in shard_allocation.items():
                node_trust = self.htdcm.get_node_trust(node_id)
                self.assertIsNotNone(node_trust)
                
                # Nodes with very low trust should not be assigned to shards
                if node_trust < 0.2:
                    self.assertEqual(shard_id, -1)  # -1 indicates no shard assignment
        
        # Save results
        results = {
            'rainbow_metrics': self.rainbow_agent.get_metrics(),
            'htdcm_metrics': self.htdcm.get_metrics()
        }
        
        with open('integration_results/rainbow_htdcm_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    def test_htdcm_consensus_integration(self):
        """Test integration between HTDCM and Adaptive Consensus."""
        # Configure test network
        config = {
            'shards': 8,
            'nodes_per_shard': 12,
            'simulation_duration': 300  # seconds
        }
        
        # Initialize components
        self.htdcm.initialize(config)
        self.adaptive_consensus.initialize(config)
        
        # Connect components
        self.adaptive_consensus.register_trust_provider(self.htdcm)
        
        # Run simulation
        for i in range(10):
            # Update HTDCM with trust information
            trust_updates = self.generate_trust_updates(config, i)
            self.htdcm.update_trust_metrics(trust_updates)
            
            # Get consensus protocol selection for each shard
            for shard_id in range(config['shards']):
                # Get nodes in shard
                shard_nodes = self.get_nodes_in_shard(shard_id, config)
                
                # Get trust metrics for shard nodes
                trust_metrics = {node_id: self.htdcm.get_node_trust(node_id) for node_id in shard_nodes}
                
                # Select consensus protocol based on trust metrics
                protocol = self.adaptive_consensus.select_protocol(shard_id, trust_metrics)
                
                # Verify protocol selection logic
                min_trust = min(trust_metrics.values())
                if min_trust < 0.3:
                    # Should select more robust protocol for low trust
                    self.assertEqual(protocol, 'pbft')
                elif min_trust > 0.8:
                    # Should select more efficient protocol for high trust
                    self.assertEqual(protocol, 'tendermint')
        
        # Save results
        results = {
            'htdcm_metrics': self.htdcm.get_metrics(),
            'consensus_metrics': self.adaptive_consensus.get_metrics()
        }
        
        with open('integration_results/htdcm_consensus_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    def test_mad_rapid_consensus_integration(self):
        """Test integration between MAD-RAPID and Adaptive Consensus."""
        # Configure test network
        config = {
            'shards': 8,
            'nodes_per_shard': 12,
            'simulation_duration': 300  # seconds
        }
        
        # Initialize components
        self.mad_rapid.initialize(config)
        self.adaptive_consensus.initialize(config)
        
        # Connect components
        self.mad_rapid.register_consensus_provider(self.adaptive_consensus)
        
        # Run simulation
        for i in range(10):
            # Generate cross-shard transactions
            transactions = self.generate_transactions(config, i)
            
            # Process transactions through MAD-RAPID
            for tx in transactions:
                route = self.mad_rapid.find_optimal_route(tx)
                
                # Verify route considers consensus protocols
                for shard_id in route:
                    protocol = self.adaptive_consensus.get_protocol(shard_id)
                    self.assertIsNotNone(protocol)
                    
                    # Verify route optimization based on consensus
                    if protocol == 'pbft':
                        # PBFT has higher latency, should minimize hops
                        self.assertLessEqual(len(route), 3)
        
        # Save results
        results = {
            'mad_rapid_metrics': self.mad_rapid.get_metrics(),
            'consensus_metrics': self.adaptive_consensus.get_metrics()
        }
        
        with open('integration_results/mad_rapid_consensus_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    def test_full_framework_integration(self):
        """Test integration of all components in the QTrust framework."""
        # Configure test network
        config = {
            'shards': 8,
            'nodes_per_shard': 12,
            'simulation_duration': 300  # seconds
        }
        
        # Initialize framework
        self.framework.initialize(config)
        
        # Run simulation
        results = self.framework.run_simulation(config)
        
        # Verify framework metrics
        self.assertGreaterEqual(results['throughput'], 10000)  # At least 10,000 TPS
        self.assertLessEqual(results['latency'], 1.5)  # At most 1.5 seconds latency
        self.assertGreaterEqual(results['efficiency'], 85)  # At least 85% efficiency
        
        # Save results
        with open('integration_results/framework_integration_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Helper methods
    def generate_network_state(self, config, iteration):
        """Generate simulated network state for testing."""
        return {
            'iteration': iteration,
            'nodes': {
                f'node_{i}': {
                    'cpu_usage': 0.5 + (i % 10) * 0.05,
                    'memory_usage': 0.4 + (i % 8) * 0.05,
                    'bandwidth': 100 - (i % 5) * 10,
                    'location': f'region_{i % 5}'
                }
                for i in range(config['shards'] * config['nodes_per_shard'])
            },
            'shards': {
                f'shard_{i}': {
                    'transaction_count': 1000 + (i % 5) * 200,
                    'cross_shard_ratio': 0.2 + (i % 5) * 0.05
                }
                for i in range(config['shards'])
            }
        }

    def generate_trust_updates(self, config, iteration):
        """Generate simulated trust updates for testing."""
        return {
            f'node_{i}': {
                'transaction_success_rate': 0.95 - (i % 20) * 0.05,
                'response_time': 0.1 + (i % 10) * 0.02,
                'uptime': 0.98 - (i % 15) * 0.02,
                'validation_accuracy': 0.99 - (i % 25) * 0.04
            }
            for i in range(config['shards'] * config['nodes_per_shard'])
        }

    def get_nodes_in_shard(self, shard_id, config):
        """Get list of nodes in a specific shard."""
        return [
            f'node_{shard_id * config["nodes_per_shard"] + i}'
            for i in range(config['nodes_per_shard'])
        ]

    def generate_transactions(self, config, iteration):
        """Generate simulated transactions for testing."""
        return [
            {
                'id': f'tx_{iteration}_{i}',
                'from_shard': i % config['shards'],
                'to_shard': (i + 1 + (i % 3)) % config['shards'],
                'size': 1000 + (i % 5) * 200,
                'priority': 1 + (i % 3)
            }
            for i in range(100)
        ]


if __name__ == '__main__':
    unittest.main()
