import unittest
from unittest.mock import MagicMock, patch
from qtrust.routing.mad_rapid import MADRAPIDRouter

class TestMADRAPIDRouter(unittest.TestCase):
    def setUp(self):
        self.router = MADRAPIDRouter(num_shards=64, num_nodes=200)
        
    def test_initialization(self):
        """Test that router initializes with correct parameters."""
        self.assertEqual(self.router.num_shards, 64)
        self.assertEqual(self.router.num_nodes, 200)
        self.assertIsNotNone(self.router.routing_table)
        
    def test_route_transaction(self):
        """Test transaction routing to correct shard."""
        # Mock transaction
        transaction = {
            'sender': 'node_123',
            'receiver': 'node_456',
            'amount': 100,
            'timestamp': 1619123456
        }
        
        # Route transaction
        target_shard = self.router.route_transaction(transaction)
        
        # Verify routing result
        self.assertIsNotNone(target_shard)
        self.assertTrue(0 <= target_shard < self.router.num_shards)
        
    def test_cross_shard_routing(self):
        """Test cross-shard transaction routing."""
        # Mock cross-shard transaction
        transaction = {
            'sender': 'node_123',
            'receiver': 'node_456',
            'amount': 100,
            'timestamp': 1619123456,
            'cross_shard': True,
            'source_shard': 10,
            'destination_shard': 20
        }
        
        # Route transaction
        source_shard, dest_shard = self.router.route_cross_shard_transaction(transaction)
        
        # Verify routing result
        self.assertEqual(source_shard, 10)
        self.assertEqual(dest_shard, 20)
        
    def test_update_routing_table(self):
        """Test routing table updates based on trust scores."""
        # Mock trust scores
        trust_scores = {
            'node_1': 0.9,
            'node_2': 0.8,
            'node_3': 0.7,
            'node_4': 0.6
        }
        
        # Update routing table
        self.router.update_routing_table(trust_scores)
        
        # Verify routing table was updated
        for node, score in trust_scores.items():
            self.assertIn(node, self.router.node_trust_scores)
            self.assertEqual(self.router.node_trust_scores[node], score)
        
    def test_optimize_routes(self):
        """Test route optimization based on network conditions."""
        # Mock network conditions
        network_conditions = {
            'shard_1_to_shard_2': 0.9,  # Good connection
            'shard_2_to_shard_3': 0.5,  # Average connection
            'shard_3_to_shard_4': 0.2   # Poor connection
        }
        
        # Optimize routes
        self.router.optimize_routes(network_conditions)
        
        # Verify route optimization
        # This would typically check internal state changes
        self.assertTrue(hasattr(self.router, 'route_quality'))
        
    def test_handle_network_partition(self):
        """Test handling of network partitions."""
        # Mock network partition
        partitioned_shards = [5, 10, 15]
        
        # Handle partition
        self.router.handle_network_partition(partitioned_shards)
        
        # Verify partition handling
        for shard in partitioned_shards:
            self.assertIn(shard, self.router.partitioned_shards)
        
    def test_route_recovery(self):
        """Test recovery from network partitions."""
        # Mock network partition and recovery
        partitioned_shards = [5, 10, 15]
        self.router.handle_network_partition(partitioned_shards)
        
        # Recover routes
        self.router.recover_routes([5, 15])  # Recover shards 5 and 15
        
        # Verify recovery
        self.assertIn(10, self.router.partitioned_shards)  # Shard 10 still partitioned
        self.assertNotIn(5, self.router.partitioned_shards)  # Shard 5 recovered
        self.assertNotIn(15, self.router.partitioned_shards)  # Shard 15 recovered
        
    def test_load_balancing(self):
        """Test load balancing across shards."""
        # Mock shard loads
        shard_loads = {
            0: 0.9,  # High load
            1: 0.5,  # Medium load
            2: 0.2   # Low load
        }
        
        # Balance load
        target_shard = self.router.balance_load(shard_loads)
        
        # Verify load balancing
        self.assertEqual(target_shard, 2)  # Should choose lowest load
        
    def test_route_caching(self):
        """Test route caching for performance."""
        # Mock transaction
        transaction = {
            'sender': 'node_123',
            'receiver': 'node_456',
            'amount': 100,
            'timestamp': 1619123456
        }
        
        # Route transaction twice
        first_route = self.router.route_transaction(transaction)
        
        # Mock cache hit
        self.router.get_cached_route = MagicMock(return_value=first_route)
        
        second_route = self.router.route_transaction(transaction)
        
        # Verify cache was used
        self.router.get_cached_route.assert_called_once()
        self.assertEqual(first_route, second_route)

if __name__ == '__main__':
    unittest.main()
