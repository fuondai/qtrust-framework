import os
import sys
import unittest
import numpy as np
from typing import Dict, List, Tuple, Any

# Import the QTrust framework from mocks directory
from qtrust.mocks.mock_qtrust_framework import QTrustFramework

class TestMockQTrustFramework(unittest.TestCase):
    """Test cases for the mock QTrust framework."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'num_shards': 4,
            'num_nodes_per_shard': 8,
            'state_dim': 32,
            'action_dim': 4,
            'trust_threshold': 0.7,
            'consensus_update_frequency': 50,
            'routing_optimization_frequency': 25,
            'federated_learning_frequency': 100,
            'use_pytorch': False
        }
        self.framework = QTrustFramework(self.config)
        
    def test_initialization(self):
        """Test framework initialization."""
        self.assertEqual(self.framework.config['num_shards'], 4)
        self.assertEqual(self.framework.config['num_nodes_per_shard'], 8)
        self.assertEqual(self.framework.config['state_dim'], 32)
        self.assertEqual(self.framework.config['action_dim'], 4)
        
    def test_update(self):
        """Test framework update."""
        state = np.random.rand(self.config['state_dim'])
        action = self.framework.update(state)
        self.assertIsInstance(action, int)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, self.config['action_dim'])
        
    def test_process_transaction(self):
        """Test transaction processing."""
        # Test same-shard transaction
        transaction = {
            'source_shard': 'shard_0',
            'dest_shard': 'shard_0',
            'amount': 100
        }
        result = self.framework.process_transaction(transaction)
        self.assertTrue(result)
        
        # Test cross-shard transaction
        transaction = {
            'source_shard': 'shard_0',
            'dest_shard': 'shard_1',
            'amount': 100
        }
        result = self.framework.process_transaction(transaction)
        self.assertTrue(result)
        
    def test_evaluate_trust(self):
        """Test trust evaluation."""
        node_id = 'node_0_0'
        trust = self.framework.evaluate_trust(node_id)
        self.assertGreaterEqual(trust, 0.0)
        self.assertLessEqual(trust, 1.0)
        
    def test_save_load(self):
        """Test save and load functionality."""
        # Test save
        save_path = '/tmp/qtrust_test'
        self.framework.save(save_path)
        
        # Test load
        self.framework.load(save_path)
        
        # No assertions needed as we're just testing that the methods don't raise exceptions

if __name__ == '__main__':
    unittest.main()
