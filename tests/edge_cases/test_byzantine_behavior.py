import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from qtrust.trust.trust_vector import TrustVector
from qtrust.routing.mad_rapid import MADRAPIDRouter
from qtrust.consensus.dynamic_consensus import DynamicConsensus

class TestByzantineBehavior(unittest.TestCase):
    def setUp(self):
        self.num_shards = 16
        self.num_nodes = 64
        self.byzantine_threshold = 0.2
        self.router = MADRAPIDRouter(num_shards=self.num_shards, num_nodes=self.num_nodes)
        self.consensus = DynamicConsensus(num_nodes=self.num_nodes, byzantine_threshold=self.byzantine_threshold)
        
        # Create trust vectors for nodes
        self.trust_vectors = {}
        for i in range(self.num_nodes):
            node_id = f'node_{i}'
            self.trust_vectors[node_id] = TrustVector()
            # Assign random trust values
            self.trust_vectors[node_id].update_dimension('transaction_validation', 0.5 + (i % 5) * 0.1)
            self.trust_vectors[node_id].update_dimension('block_proposal', 0.5 + (i % 4) * 0.1)
            self.trust_vectors[node_id].update_dimension('response_time', 0.5 + (i % 3) * 0.1)
        
        # Designate some nodes as Byzantine
        self.byzantine_nodes = [f'node_{i}' for i in range(10)]
        
    def test_double_voting_detection(self):
        """Test detection of double voting Byzantine behavior."""
        # Create a block
        block = {
            'transactions': ['tx1', 'tx2', 'tx3'],
            'timestamp': 1619123456,
            'proposer': 'node_20',
            'hash': '0x1234567890abcdef'
        }
        
        # Create conflicting block
        conflicting_block = {
            'transactions': ['tx4', 'tx5', 'tx6'],
            'timestamp': 1619123456,
            'proposer': 'node_20',
            'hash': '0xfedcba0987654321'
        }
        
        # Byzantine node votes for both blocks
        byzantine_node = self.byzantine_nodes[0]
        
        # Record votes
        votes_block1 = {byzantine_node: True}
        votes_block2 = {byzantine_node: True}
        
        # Detect double voting
        double_voters = self.consensus.detect_double_voting([
            (block, votes_block1),
            (conflicting_block, votes_block2)
        ])
        
        # Verify detection
        self.assertIn(byzantine_node, double_voters)
        
    def test_equivocation_detection(self):
        """Test detection of equivocation (proposing conflicting blocks)."""
        # Byzantine node proposes two conflicting blocks
        byzantine_node = self.byzantine_nodes[1]
        
        # Create two conflicting blocks proposed by the same node
        block1 = {
            'transactions': ['tx1', 'tx2', 'tx3'],
            'timestamp': 1619123456,
            'proposer': byzantine_node,
            'hash': '0x1234567890abcdef'
        }
        
        block2 = {
            'transactions': ['tx4', 'tx5', 'tx6'],
            'timestamp': 1619123456,
            'proposer': byzantine_node,
            'hash': '0xfedcba0987654321'
        }
        
        # Detect equivocation
        equivocators = self.consensus.detect_equivocation([block1, block2])
        
        # Verify detection
        self.assertIn(byzantine_node, equivocators)
        
    def test_selective_non_responsiveness(self):
        """Test detection of selective non-responsiveness."""
        # Byzantine node responds to some nodes but not others
        byzantine_node = self.byzantine_nodes[2]
        
        # Create response records
        response_records = {
            byzantine_node: {
                'node_20': True,  # Responded to node_20
                'node_21': True,  # Responded to node_21
                'node_22': False, # Did not respond to node_22
                'node_23': False  # Did not respond to node_23
            }
        }
        
        # Detect selective non-responsiveness
        selective_nodes = self.consensus.detect_selective_responsiveness(response_records)
        
        # Verify detection
        self.assertIn(byzantine_node, selective_nodes)
        
    def test_invalid_transaction_proposal(self):
        """Test detection of invalid transaction proposals."""
        # Byzantine node proposes invalid transactions
        byzantine_node = self.byzantine_nodes[3]
        
        # Create invalid transaction
        invalid_transaction = {
            'sender': byzantine_node,
            'receiver': 'node_30',
            'amount': -100,  # Negative amount, invalid
            'timestamp': 1619123456
        }
        
        # Create block with invalid transaction
        block = {
            'transactions': [invalid_transaction],
            'timestamp': 1619123456,
            'proposer': byzantine_node,
            'hash': '0x1234567890abcdef'
        }
        
        # Validate block
        is_valid = self.consensus.validate_block(block)
        
        # Verify validation result
        self.assertFalse(is_valid)
        
        # Record invalid proposal
        self.consensus.record_invalid_proposal(byzantine_node)
        self.consensus.record_invalid_proposal(byzantine_node)
        
        # Get nodes with multiple invalid proposals
        invalid_proposers = self.consensus.get_invalid_proposers(threshold=2)
        
        # Verify detection
        self.assertIn(byzantine_node, invalid_proposers)
        
    def test_trust_score_degradation(self):
        """Test trust score degradation for Byzantine nodes."""
        # Byzantine node with initial trust
        byzantine_node = self.byzantine_nodes[4]
        initial_trust = self.trust_vectors[byzantine_node].get_aggregate_trust()
        
        # Record Byzantine behavior
        for _ in range(5):
            self.trust_vectors[byzantine_node].update_dimension('transaction_validation', 
                                                              self.trust_vectors[byzantine_node].get_dimension('transaction_validation') * 0.8)
        
        # Get updated trust
        updated_trust = self.trust_vectors[byzantine_node].get_aggregate_trust()
        
        # Verify trust degradation
        self.assertLess(updated_trust, initial_trust)
        
    def test_byzantine_node_isolation(self):
        """Test isolation of Byzantine nodes from consensus."""
        # Identify Byzantine nodes
        byzantine_nodes = self.byzantine_nodes[:5]
        
        # Create block
        block = {
            'transactions': ['tx1', 'tx2', 'tx3'],
            'timestamp': 1619123456,
            'proposer': 'node_20',
            'hash': '0x1234567890abcdef'
        }
        
        # Create votes including Byzantine nodes
        votes = {node: True for node in byzantine_nodes}
        # Add honest nodes' votes
        for i in range(20, 30):
            votes[f'node_{i}'] = True
        
        # Isolate Byzantine nodes
        self.consensus.isolate_byzantine_nodes(byzantine_nodes)
        
        # Finalize block with votes
        is_finalized = self.consensus.finalize_block(block, votes)
        
        # Verify Byzantine votes were ignored
        self.assertTrue(is_finalized)
        for node in byzantine_nodes:
            self.assertNotIn(node, self.consensus.active_validators)

if __name__ == '__main__':
    unittest.main()
