"""
Unit tests for shard clustering module.

This module contains unit tests for the shard clustering functionality
in the QTrust blockchain sharding framework.
"""

import os
import sys
import unittest
import time
import threading
from typing import Dict, List, Any

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.shard_clustering import ShardCluster, ShardManager


class TestShardCluster(unittest.TestCase):
    """Unit tests for the ShardCluster class."""

    def setUp(self):
        """Set up test environment."""
        self.cluster = ShardCluster("test_cluster", capacity=5)

    def test_cluster_initialization(self):
        """Test initialization of shard cluster."""
        self.assertEqual(self.cluster.cluster_id, "test_cluster")
        self.assertEqual(self.cluster.capacity, 5)
        self.assertEqual(len(self.cluster.nodes), 0)
        self.assertFalse(self.cluster.active)
        self.assertIsNotNone(self.cluster.creation_time)
        self.assertEqual(self.cluster.creation_time, self.cluster.last_rebalance_time)

    def test_add_remove_node(self):
        """Test adding and removing nodes from cluster."""
        # Add nodes
        self.assertTrue(self.cluster.add_node("node1"))
        self.assertTrue(self.cluster.add_node("node2"))
        self.assertTrue(self.cluster.add_node("node3"))
        
        # Check node count
        self.assertEqual(self.cluster.get_node_count(), 3)
        
        # Try adding duplicate node (should still return True)
        self.assertTrue(self.cluster.add_node("node1"))
        self.assertEqual(self.cluster.get_node_count(), 3)  # Count shouldn't change
        
        # Add more nodes up to capacity
        self.assertTrue(self.cluster.add_node("node4"))
        self.assertTrue(self.cluster.add_node("node5"))
        
        # Try adding beyond capacity
        self.assertFalse(self.cluster.add_node("node6"))
        
        # Remove nodes
        self.assertTrue(self.cluster.remove_node("node1"))
        self.assertEqual(self.cluster.get_node_count(), 4)
        
        # Try removing non-existent node
        self.assertFalse(self.cluster.remove_node("node99"))
        
        # Remove remaining nodes
        self.assertTrue(self.cluster.remove_node("node2"))
        self.assertTrue(self.cluster.remove_node("node3"))
        self.assertTrue(self.cluster.remove_node("node4"))
        self.assertTrue(self.cluster.remove_node("node5"))
        
        # Check empty cluster
        self.assertEqual(self.cluster.get_node_count(), 0)

    def test_activation(self):
        """Test cluster activation and deactivation."""
        self.assertFalse(self.cluster.is_active())
        
        # Activate
        self.cluster.activate()
        self.assertTrue(self.cluster.is_active())
        
        # Deactivate
        self.cluster.deactivate()
        self.assertFalse(self.cluster.is_active())

    def test_metrics(self):
        """Test cluster metrics update and retrieval."""
        # Initial metrics
        initial_metrics = self.cluster.get_metrics()
        self.assertEqual(initial_metrics['transaction_count'], 0)
        self.assertEqual(initial_metrics['cross_shard_ratio'], 0.0)
        
        # Update metrics
        new_metrics = {
            'transaction_count': 1000,
            'cross_shard_ratio': 0.25,
            'avg_latency': 1.5,
            'throughput': 5000.0
        }
        self.cluster.update_metrics(new_metrics)
        
        # Check updated metrics
        updated_metrics = self.cluster.get_metrics()
        self.assertEqual(updated_metrics['transaction_count'], 1000)
        self.assertEqual(updated_metrics['cross_shard_ratio'], 0.25)
        self.assertEqual(updated_metrics['avg_latency'], 1.5)
        self.assertEqual(updated_metrics['throughput'], 5000.0)

    def test_load_factor(self):
        """Test load factor calculation."""
        # Set metrics that affect load factor
        metrics = {
            'throughput': 5000.0,  # 50% of max (10000)
            'cross_shard_ratio': 0.4,
            'avg_latency': 2.0  # 40% of max (5.0)
        }
        self.cluster.update_metrics(metrics)
        
        # Calculate expected load factor
        # 0.5 * 0.5 (throughput) + 0.3 * 0.4 (cross_shard) + 0.2 * 0.4 (latency) = 0.45
        expected_load = 0.5 * 0.5 + 0.3 * 0.4 + 0.2 * 0.4
        
        # Check load factor
        self.assertAlmostEqual(self.cluster.get_load_factor(), expected_load, places=6)
        
        # Test with different metrics
        metrics = {
            'throughput': 10000.0,  # 100% of max
            'cross_shard_ratio': 0.8,
            'avg_latency': 5.0  # 100% of max
        }
        self.cluster.update_metrics(metrics)
        
        # Calculate expected load factor
        # 0.5 * 1.0 (throughput) + 0.3 * 0.8 (cross_shard) + 0.2 * 1.0 (latency) = 0.94
        expected_load = 0.5 * 1.0 + 0.3 * 0.8 + 0.2 * 1.0
        
        # Check load factor
        self.assertAlmostEqual(self.cluster.get_load_factor(), expected_load, places=6)


class TestShardManager(unittest.TestCase):
    """Unit tests for the ShardManager class."""

    def setUp(self):
        """Set up test environment."""
        self.manager = ShardManager(initial_shards=3, nodes_per_shard=4)
        self.manager.initialize()

    def tearDown(self):
        """Clean up after tests."""
        self.manager.shutdown()

    def test_initialization(self):
        """Test initialization of shard manager."""
        # Check initial clusters
        clusters = self.manager.get_all_clusters()
        self.assertEqual(len(clusters), 3)
        
        # Check cluster IDs
        cluster_ids = [cluster.cluster_id for cluster in clusters]
        self.assertIn("shard_0", cluster_ids)
        self.assertIn("shard_1", cluster_ids)
        self.assertIn("shard_2", cluster_ids)
        
        # Check all clusters are active
        for cluster in clusters:
            self.assertTrue(cluster.is_active())
            self.assertEqual(cluster.capacity, 4)

    def test_node_assignment(self):
        """Test node assignment to clusters."""
        # Add nodes
        cluster_id1 = self.manager.add_node("node1")
        cluster_id2 = self.manager.add_node("node2")
        cluster_id3 = self.manager.add_node("node3")
        
        # Check assignments
        self.assertIsNotNone(cluster_id1)
        self.assertIsNotNone(cluster_id2)
        self.assertIsNotNone(cluster_id3)
        
        # Check node assignments
        self.assertEqual(self.manager.get_node_assignment("node1"), cluster_id1)
        self.assertEqual(self.manager.get_node_assignment("node2"), cluster_id2)
        self.assertEqual(self.manager.get_node_assignment("node3"), cluster_id3)
        
        # Check non-existent node
        self.assertIsNone(self.manager.get_node_assignment("node99"))
        
        # Add more nodes to fill clusters
        for i in range(4, 13):
            node_id = f"node{i}"
            cluster_id = self.manager.add_node(node_id)
            self.assertIsNotNone(cluster_id)
            
        # Check cluster node counts
        clusters = self.manager.get_all_clusters()
        total_nodes = sum(cluster.get_node_count() for cluster in clusters)
        self.assertEqual(total_nodes, 12)
        
        # Add one more node (should create a new cluster)
        cluster_id13 = self.manager.add_node("node13")
        self.assertIsNotNone(cluster_id13)
        
        # Check new cluster was created
        clusters = self.manager.get_all_clusters()
        self.assertEqual(len(clusters), 4)

    def test_node_removal(self):
        """Test node removal from clusters."""
        # Add nodes
        cluster_id1 = self.manager.add_node("node1")
        cluster_id2 = self.manager.add_node("node2")
        
        # Remove node
        self.assertTrue(self.manager.remove_node("node1"))
        
        # Check node assignment is removed
        self.assertIsNone(self.manager.get_node_assignment("node1"))
        
        # Try removing non-existent node
        self.assertFalse(self.manager.remove_node("node99"))
        
        # Check other node is still assigned
        self.assertEqual(self.manager.get_node_assignment("node2"), cluster_id2)

    def test_cluster_metrics(self):
        """Test updating and retrieving cluster metrics."""
        # Add a node to first cluster
        cluster_id = self.manager.add_node("test_node")
        self.assertIsNotNone(cluster_id)
        
        # Update metrics
        metrics = {
            'transaction_count': 1000,
            'cross_shard_ratio': 0.25,
            'avg_latency': 1.5,
            'throughput': 5000.0
        }
        self.assertTrue(self.manager.update_cluster_metrics(cluster_id, metrics))
        
        # Get cluster and check metrics
        cluster = self.manager.get_cluster(cluster_id)
        self.assertIsNotNone(cluster)
        
        updated_metrics = cluster.get_metrics()
        self.assertEqual(updated_metrics['transaction_count'], 1000)
        self.assertEqual(updated_metrics['cross_shard_ratio'], 0.25)
        
        # Try updating non-existent cluster
        self.assertFalse(self.manager.update_cluster_metrics("non_existent", metrics))

    def test_network_state(self):
        """Test retrieving network state."""
        # Add nodes
        for i in range(1, 7):
            node_id = f"node{i}"
            self.manager.add_node(node_id)
            
        # Get network state
        state = self.manager.get_network_state()
        
        # Check state properties
        self.assertEqual(state["node_count"], 6)
        self.assertEqual(state["cluster_count"], 3)
        self.assertEqual(len(state["clusters"]), 3)
        self.assertEqual(len(state["node_assignments"]), 6)

    def test_rebalance_clusters(self):
        """Test cluster rebalancing."""
        # Add nodes to fill first cluster
        for i in range(1, 5):
            self.manager.add_node(f"node{i}")
            
        # Get first cluster
        clusters = self.manager.get_all_clusters()
        first_cluster = clusters[0]
        
        # Update metrics to make first cluster overloaded
        overload_metrics = {
            'throughput': 9000.0,
            'cross_shard_ratio': 0.7,
            'avg_latency': 4.0
        }
        self.manager.update_cluster_metrics(first_cluster.cluster_id, overload_metrics)
        
        # Update metrics to make second cluster underloaded
        second_cluster = clusters[1]
        underload_metrics = {
            'throughput': 1000.0,
            'cross_shard_ratio': 0.1,
            'avg_latency': 0.5
        }
        self.manager.update_cluster_metrics(second_cluster.cluster_id, underload_metrics)
        
        # Trigger rebalance
        result = self.manager.rebalance_clusters()
        self.assertTrue(result.wait(timeout=2.0))
        
        # Check rebalance result
        self.assertTrue(result.success)
        rebalance_data = result.result
        
        # Should have rebalanced
        self.assertTrue(rebalance_data["rebalanced"])
        self.assertGreater(len(rebalance_data["moves"]), 0)


if __name__ == '__main__':
    unittest.main()
