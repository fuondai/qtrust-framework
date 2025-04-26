import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from qtrust.trust.trust_vector import TrustVector
from qtrust.routing.mad_rapid import MADRAPIDRouter
from qtrust.consensus.dynamic_consensus import DynamicConsensus

class TestCrossShard(unittest.TestCase):
    def setUp(self):
        self.num_shards = 16
        self.num_nodes = 64
        self.router = MADRAPIDRouter(num_shards=self.num_shards, num_nodes=self.num_nodes)
        self.consensus = DynamicConsensus(num_nodes=self.num_nodes, byzantine_threshold=0.2)
        
        # Create trust vectors for nodes
        self.trust_vectors = {}
        for i in range(self.num_nodes):
            node_id = f'node_{i}'
            self.trust_vectors[node_id] = TrustVector()
            # Assign random trust values
            self.trust_vectors[node_id].update_dimension('transaction_validation', 0.5 + (i % 5) * 0.1)
            self.trust_vectors[node_id].update_dimension('block_proposal', 0.5 + (i % 4) * 0.1)
            self.trust_vectors[node_id].update_dimension('response_time', 0.5 + (i % 3) * 0.1)
        
        # Update router with trust scores
        trust_scores = {node_id: tv.get_aggregate_trust() for node_id, tv in self.trust_vectors.items()}
        self.router.update_routing_table(trust_scores)
        
    def test_cross_shard_transaction_routing(self):
        """Test routing of cross-shard transactions."""
        # Create a cross-shard transaction
        transaction = {
            'sender': 'node_5',
            'receiver': 'node_42',
            'amount': 100,
            'timestamp': 1619123456,
            'cross_shard': True
        }
        
        # Route the transaction
        source_shard, dest_shard = self.router.route_cross_shard_transaction(transaction)
        
        # Verify routing
        self.assertTrue(0 <= source_shard < self.num_shards)
        self.assertTrue(0 <= dest_shard < self.num_shards)
        self.assertNotEqual(source_shard, dest_shard)
        
    def test_cross_shard_transaction_validation(self):
        """Test validation of cross-shard transactions."""
        # Create a cross-shard transaction
        transaction = {
            'sender': 'node_5',
            'receiver': 'node_42',
            'amount': 100,
            'timestamp': 1619123456,
            'cross_shard': True
        }
        
        # Route the transaction
        source_shard, dest_shard = self.router.route_cross_shard_transaction(transaction)
        
        # Create blocks for both shards
        source_block = {
            'transactions': [transaction],
            'timestamp': 1619123456,
            'proposer': 'node_5',
            'shard_id': source_shard
        }
        
        dest_block = {
            'transactions': [transaction],
            'timestamp': 1619123456,
            'proposer': 'node_42',
            'shard_id': dest_shard
        }
        
        # Validate blocks
        source_valid = self.consensus.validate_block(source_block)
        dest_valid = self.consensus.validate_block(dest_block)
        
        # Verify validation
        self.assertTrue(source_valid)
        self.assertTrue(dest_valid)
        
    def test_cross_shard_transaction_finalization(self):
        """Test finalization of cross-shard transactions."""
        # Create a cross-shard transaction
        transaction = {
            'sender': 'node_5',
            'receiver': 'node_42',
            'amount': 100,
            'timestamp': 1619123456,
            'cross_shard': True
        }
        
        # Route the transaction
        source_shard, dest_shard = self.router.route_cross_shard_transaction(transaction)
        
        # Create blocks for both shards
        source_block = {
            'transactions': [transaction],
            'timestamp': 1619123456,
            'proposer': 'node_5',
            'shard_id': source_shard,
            'hash': '0x1234567890abcdef'
        }
        
        dest_block = {
            'transactions': [transaction],
            'timestamp': 1619123456,
            'proposer': 'node_42',
            'shard_id': dest_shard,
            'hash': '0xfedcba0987654321'
        }
        
        # Create votes for both blocks
        source_votes = {f'node_{i}': True for i in range(10)}
        dest_votes = {f'node_{i+30}': True for i in range(10)}
        
        # Finalize blocks
        source_finalized = self.consensus.finalize_block(source_block, source_votes)
        dest_finalized = self.consensus.finalize_block(dest_block, dest_votes)
        
        # Verify finalization
        self.assertTrue(source_finalized)
        self.assertTrue(dest_finalized)
        
    def test_cross_shard_transaction_atomicity(self):
        """Test atomicity of cross-shard transactions."""
        # Create a cross-shard transaction
        transaction = {
            'sender': 'node_5',
            'receiver': 'node_42',
            'amount': 100,
            'timestamp': 1619123456,
            'cross_shard': True
        }
        
        # Route the transaction
        source_shard, dest_shard = self.router.route_cross_shard_transaction(transaction)
        
        # Create blocks for both shards
        source_block = {
            'transactions': [transaction],
            'timestamp': 1619123456,
            'proposer': 'node_5',
            'shard_id': source_shard,
            'hash': '0x1234567890abcdef'
        }
        
        dest_block = {
            'transactions': [transaction],
            'timestamp': 1619123456,
            'proposer': 'node_42',
            'shard_id': dest_shard,
            'hash': '0xfedcba0987654321'
        }
        
        # Create votes for both blocks
        source_votes = {f'node_{i}': True for i in range(10)}
        dest_votes = {f'node_{i+30}': True for i in range(10)}
        
        # Finalize source block but not dest block
        source_finalized = self.consensus.finalize_block(source_block, source_votes)
        
        # Simulate timeout on dest block
        dest_timeout = self.consensus.handle_timeout(1)
        
        # Verify atomicity
        self.assertTrue(source_finalized)
        self.assertEqual(dest_timeout, 2)  # Should move to next round
        
        # Check if transaction is marked for rollback in source shard
        rollback_status = self.consensus.check_cross_shard_status(transaction)
        self.assertEqual(rollback_status, 'pending')  # Should be pending until dest shard confirms

if __name__ == '__main__':
    unittest.main()
