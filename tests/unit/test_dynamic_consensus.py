import unittest
from unittest.mock import MagicMock, patch
from qtrust.consensus.dynamic_consensus import DynamicConsensus

class TestDynamicConsensus(unittest.TestCase):
    def setUp(self):
        self.consensus = DynamicConsensus(num_nodes=64, byzantine_threshold=0.2)
        
    def test_initialization(self):
        """Test that consensus initializes with correct parameters."""
        self.assertEqual(self.consensus.num_nodes, 64)
        self.assertEqual(self.consensus.byzantine_threshold, 0.2)
        self.assertIsNotNone(self.consensus.validators)
        
    def test_propose_block(self):
        """Test block proposal."""
        # Mock block data
        block_data = {
            'transactions': ['tx1', 'tx2', 'tx3'],
            'timestamp': 1619123456,
            'proposer': 'node_1'
        }
        
        # Propose block
        block = self.consensus.propose_block(block_data)
        
        # Verify block proposal
        self.assertIsNotNone(block)
        self.assertEqual(block['transactions'], block_data['transactions'])
        self.assertEqual(block['proposer'], block_data['proposer'])
        self.assertIn('hash', block)
        
    def test_validate_block(self):
        """Test block validation."""
        # Mock block
        block = {
            'transactions': ['tx1', 'tx2', 'tx3'],
            'timestamp': 1619123456,
            'proposer': 'node_1',
            'hash': '0x1234567890abcdef'
        }
        
        # Validate block
        is_valid = self.consensus.validate_block(block)
        
        # Verify validation result
        self.assertTrue(is_valid)
        
    def test_finalize_block(self):
        """Test block finalization."""
        # Mock block and votes
        block = {
            'transactions': ['tx1', 'tx2', 'tx3'],
            'timestamp': 1619123456,
            'proposer': 'node_1',
            'hash': '0x1234567890abcdef'
        }
        
        votes = {
            'node_1': True,
            'node_2': True,
            'node_3': True,
            'node_4': False,
            'node_5': True
        }
        
        # Finalize block
        is_finalized = self.consensus.finalize_block(block, votes)
        
        # Verify finalization result
        self.assertTrue(is_finalized)
        
    def test_byzantine_detection(self):
        """Test detection of Byzantine behavior."""
        # Mock Byzantine behavior
        byzantine_nodes = ['node_3', 'node_7']
        
        # Detect Byzantine nodes
        detected = self.consensus.detect_byzantine_behavior(byzantine_nodes)
        
        # Verify detection result
        self.assertEqual(set(detected), set(byzantine_nodes))
        
    def test_adjust_consensus_parameters(self):
        """Test adjustment of consensus parameters based on network conditions."""
        # Mock network conditions
        network_conditions = {
            'average_latency': 200,  # ms
            'node_count': 64,
            'byzantine_ratio': 0.15
        }
        
        # Adjust parameters
        self.consensus.adjust_parameters(network_conditions)
        
        # Verify parameter adjustment
        self.assertNotEqual(self.consensus.timeout, self.consensus.default_timeout)
        
    def test_handle_timeout(self):
        """Test handling of consensus timeout."""
        # Mock timeout scenario
        round_number = 3
        
        # Handle timeout
        new_round = self.consensus.handle_timeout(round_number)
        
        # Verify timeout handling
        self.assertEqual(new_round, round_number + 1)
        
    def test_view_change(self):
        """Test view change procedure."""
        # Mock view change
        old_leader = 'node_1'
        
        # Change view
        new_leader = self.consensus.change_view(old_leader)
        
        # Verify view change
        self.assertNotEqual(new_leader, old_leader)
        
    def test_trust_based_validation(self):
        """Test trust-based validation weight."""
        # Mock trust scores
        trust_scores = {
            'node_1': 0.9,
            'node_2': 0.8,
            'node_3': 0.7,
            'node_4': 0.6
        }
        
        # Get validation weights
        weights = self.consensus.get_validation_weights(trust_scores)
        
        # Verify weights
        for node, score in trust_scores.items():
            self.assertIn(node, weights)
            self.assertAlmostEqual(weights[node], score / sum(trust_scores.values()))

if __name__ == '__main__':
    unittest.main()
