#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration tests for the QTrust framework.
Tests the unified framework with all components integrated.
"""

import os
import sys
import unittest
import json
import tempfile
from unittest.mock import MagicMock, patch

# Add parent directory to path to import QTrust modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.qtrust_framework import QTrustFramework
from qtrust.agents.rainbow_agent import RainbowDQNAgent
from qtrust.trust.htdcm import HTDCM
from qtrust.routing.mad_rapid import MADRAPIDRouter
from qtrust.consensus.adaptive_consensus import AdaptiveConsensusSelector
from qtrust.federated.privacy_preserving_fl import HierarchicalFederatedLearning


class TestFrameworkIntegration(unittest.TestCase):
    """Test cases for the integrated QTrust framework."""

    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary config file
        self.config_file = tempfile.NamedTemporaryFile(delete=False, mode='w+')
        config = {
            'log_level': 'DEBUG',
            'num_shards': 4,
            'num_validators': 16,
            'htdcm': {
                'trust_threshold': 0.7,
                'update_frequency': 5
            },
            'rainbow_dqn': {
                'learning_rate': 0.001,
                'gamma': 0.99,
                'batch_size': 32
            },
            'consensus': {
                'default_protocol': 'pbft',
                'transition_blocks': 10
            },
            'mad_rapid': {
                'congestion_threshold': 0.8,
                'path_cache_size': 100
            },
            'federated_learning': {
                'local_epochs': 2,
                'privacy_budget': 1.0
            }
        }
        json.dump(config, self.config_file)
        self.config_file.flush()
        
        # Initialize framework with test config
        self.framework = QTrustFramework(self.config_file.name)
        
        # Mock some component methods to avoid actual network/disk operations
        self.framework.htdcm.start = MagicMock()
        self.framework.rainbow_dqn.start = MagicMock()
        self.framework.adaptive_consensus.start = MagicMock()
        self.framework.mad_rapid.start = MagicMock()
        self.framework.federated_learning.start = MagicMock()
        self.framework.metrics_collector.start = MagicMock()

    def tearDown(self):
        """Clean up after each test."""
        # Close and remove the temporary config file
        self.config_file.close()
        os.unlink(self.config_file.name)

    def test_framework_initialization(self):
        """Test that the framework initializes all components correctly."""
        # Check that all main components are initialized
        self.assertIsInstance(self.framework.htdcm, HTDCM)
        self.assertIsInstance(self.framework.rainbow_dqn, RainbowDQNAgent)
        self.assertIsInstance(self.framework.adaptive_consensus, AdaptiveConsensusSelector)
        self.assertIsInstance(self.framework.mad_rapid, MADRAPIDRouter)
        self.assertIsInstance(self.framework.federated_learning, HierarchicalFederatedLearning)
        
        # Check that configuration was loaded correctly
        self.assertEqual(self.framework.config['num_shards'], 4)
        self.assertEqual(self.framework.config['num_validators'], 16)
        self.assertEqual(self.framework.config['htdcm']['trust_threshold'], 0.7)

    def test_component_interactions(self):
        """Test that component interactions are established correctly."""
        # Test that Rainbow DQN is connected to HTDCM
        self.framework.rainbow_dqn.register_trust_provider = MagicMock()
        self.framework._establish_component_interactions()
        self.framework.rainbow_dqn.register_trust_provider.assert_called_with(self.framework.htdcm)
        
        # Test that HTDCM is connected to Adaptive Consensus
        self.framework.adaptive_consensus.register_trust_provider = MagicMock()
        self.framework._establish_component_interactions()
        self.framework.adaptive_consensus.register_trust_provider.assert_called_with(self.framework.htdcm)
        
        # Test that components are connected to MAD-RAPID
        self.framework.mad_rapid.register_trust_provider = MagicMock()
        self.framework.mad_rapid.register_network_state_provider = MagicMock()
        self.framework.mad_rapid.register_consensus_provider = MagicMock()
        self.framework._establish_component_interactions()
        self.framework.mad_rapid.register_trust_provider.assert_called_with(self.framework.htdcm)
        self.framework.mad_rapid.register_network_state_provider.assert_called_with(self.framework.rainbow_dqn)
        self.framework.mad_rapid.register_consensus_provider.assert_called_with(self.framework.adaptive_consensus)

    def test_framework_start_stop(self):
        """Test that the framework starts and stops all components in the correct order."""
        # Test start
        self.framework.start()
        
        # Verify that all components were started
        self.framework.htdcm.start.assert_called_once()
        self.framework.rainbow_dqn.start.assert_called_once()
        self.framework.adaptive_consensus.start.assert_called_once()
        self.framework.mad_rapid.start.assert_called_once()
        self.framework.federated_learning.start.assert_called_once()
        self.framework.metrics_collector.start.assert_called_once()
        
        # Mock stop methods
        self.framework.htdcm.stop = MagicMock()
        self.framework.rainbow_dqn.stop = MagicMock()
        self.framework.adaptive_consensus.stop = MagicMock()
        self.framework.mad_rapid.stop = MagicMock()
        self.framework.federated_learning.stop = MagicMock()
        self.framework.metrics_collector.stop = MagicMock()
        
        # Test stop
        self.framework.stop()
        
        # Verify that all components were stopped
        self.framework.htdcm.stop.assert_called_once()
        self.framework.rainbow_dqn.stop.assert_called_once()
        self.framework.adaptive_consensus.stop.assert_called_once()
        self.framework.mad_rapid.stop.assert_called_once()
        self.framework.federated_learning.stop.assert_called_once()
        self.framework.metrics_collector.stop.assert_called_once()

    def test_transaction_processing(self):
        """Test that transactions are processed correctly through the framework."""
        # Create a mock transaction
        mock_transaction = MagicMock()
        mock_transaction.is_cross_shard.return_value = True
        mock_transaction.id = "test_tx_123"
        
        # Mock the required methods
        self.framework.rainbow_dqn.get_shard_congestion_predictions = MagicMock(return_value={})
        self.framework.mad_rapid.optimize_transaction_path = MagicMock(return_value=["shard1", "shard2"])
        self.framework.mad_rapid.execute_cross_shard_transaction = MagicMock(return_value={"success": True})
        self.framework.metrics_collector.record_transaction_result = MagicMock()
        
        # Process the transaction
        result = self.framework.process_transaction(mock_transaction)
        
        # Verify the transaction was processed correctly
        self.framework.rainbow_dqn.get_shard_congestion_predictions.assert_called_once()
        self.framework.mad_rapid.optimize_transaction_path.assert_called_once_with(
            mock_transaction, self.framework.rainbow_dqn.get_shard_congestion_predictions())
        self.framework.mad_rapid.execute_cross_shard_transaction.assert_called_once_with(
            mock_transaction, ["shard1", "shard2"])
        self.framework.metrics_collector.record_transaction_result.assert_called_once()
        self.assertEqual(result, {"success": True})
        
        # Test single-shard transaction
        mock_transaction.is_cross_shard.return_value = False
        mock_transaction.shard_id = "shard1"
        
        mock_consensus = MagicMock()
        mock_consensus.process_transaction.return_value = {"success": True}
        self.framework.adaptive_consensus.get_current_protocol = MagicMock(return_value=mock_consensus)
        
        # Process the transaction
        result = self.framework.process_transaction(mock_transaction)
        
        # Verify the transaction was processed correctly
        self.framework.adaptive_consensus.get_current_protocol.assert_called_once_with("shard1")
        mock_consensus.process_transaction.assert_called_once_with(mock_transaction)
        self.assertEqual(result, {"success": True})

    def test_benchmark_functionality(self):
        """Test that benchmarks can be run through the framework."""
        # Mock the metrics collector
        self.framework.metrics_collector.run_benchmark = MagicMock(return_value={
            "throughput": 10000,
            "latency": 500,
            "success_rate": 0.99
        })
        
        # Run a benchmark
        results = self.framework.run_benchmark("throughput")
        
        # Verify the benchmark was run correctly
        self.framework.metrics_collector.run_benchmark.assert_called_once()
        self.assertEqual(results["throughput"], 10000)
        self.assertEqual(results["latency"], 500)
        self.assertEqual(results["success_rate"], 0.99)


if __name__ == '__main__':
    unittest.main()
