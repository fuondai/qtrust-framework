#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the HTDCM (Hierarchical Trust-based Data Center Mechanism).
Tests the multi-dimensional trust metrics, hierarchical trust architecture,
and modified PageRank algorithm for trust aggregation.
"""

import os
import sys
import unittest
import time
import networkx as nx
from unittest.mock import MagicMock, patch

# Add parent directory to path to import QTrust modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.trust.htdcm import HTDCM, TrustScore, TrustDimension, TrustEntity, TrustHierarchy


class TestTrustScore(unittest.TestCase):
    """Test cases for the TrustScore class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.trust_score = TrustScore(0.5, 0.5)

    def test_initialization(self):
        """Test trust score initialization."""
        self.assertEqual(self.trust_score.score, 0.5)
        self.assertEqual(self.trust_score.confidence, 0.5)
        self.assertTrue(hasattr(self.trust_score, 'last_update'))
        self.assertEqual(len(self.trust_score.history), 1)

    def test_update(self):
        """Test trust score update."""
        old_score = self.trust_score.score
        old_confidence = self.trust_score.confidence
        old_update_time = self.trust_score.last_update
        
        # Wait a moment to ensure time difference
        time.sleep(0.01)
        
        # Update with new score
        self.trust_score.update(0.8, 0.6)
        
        # Check that score was updated
        self.assertNotEqual(self.trust_score.score, old_score)
        self.assertNotEqual(self.trust_score.confidence, old_confidence)
        self.assertGreater(self.trust_score.last_update, old_update_time)
        self.assertEqual(len(self.trust_score.history), 2)
        
        # Check that history contains the update
        self.assertEqual(self.trust_score.history[-1][0], self.trust_score.score)
        self.assertEqual(self.trust_score.history[-1][1], self.trust_score.confidence)

    def test_decay(self):
        """Test trust score decay."""
        self.trust_score.score = 0.8
        self.trust_score.confidence = 0.8
        old_update_time = self.trust_score.last_update
        
        # Wait a moment to ensure time difference
        time.sleep(0.01)
        
        # Apply decay
        self.trust_score.decay(0.9)
        
        # Check that score was decayed
        self.assertEqual(self.trust_score.score, 0.8 * 0.9)
        self.assertEqual(self.trust_score.confidence, 0.8 * 0.9)
        self.assertGreater(self.trust_score.last_update, old_update_time)
        self.assertEqual(len(self.trust_score.history), 2)

    def test_get_trend(self):
        """Test trust score trend calculation."""
        # Add some history
        self.trust_score.update(0.6, 0.5)
        self.trust_score.update(0.7, 0.5)
        self.trust_score.update(0.8, 0.5)
        
        # Calculate trend
        trend = self.trust_score.get_trend()
        
        # Trend should be positive
        self.assertGreater(trend, 0)
        
        # Add decreasing scores
        self.trust_score.update(0.7, 0.5)
        self.trust_score.update(0.6, 0.5)
        self.trust_score.update(0.5, 0.5)
        
        # Calculate trend
        trend = self.trust_score.get_trend()
        
        # Trend should be negative
        self.assertLess(trend, 0)


class TestTrustDimension(unittest.TestCase):
    """Test cases for the TrustDimension class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.dimension = TrustDimension("test_dimension", 0.8, 0.95)

    def test_initialization(self):
        """Test trust dimension initialization."""
        self.assertEqual(self.dimension.name, "test_dimension")
        self.assertEqual(self.dimension.weight, 0.8)
        self.assertEqual(self.dimension.decay_rate, 0.95)

    def test_getters_setters(self):
        """Test trust dimension getters and setters."""
        self.assertEqual(self.dimension.get_name(), "test_dimension")
        self.assertEqual(self.dimension.get_weight(), 0.8)
        self.assertEqual(self.dimension.get_decay_rate(), 0.95)
        
        # Test setters
        self.dimension.set_weight(0.9)
        self.dimension.set_decay_rate(0.98)
        
        self.assertEqual(self.dimension.get_weight(), 0.9)
        self.assertEqual(self.dimension.get_decay_rate(), 0.98)


class TestTrustEntity(unittest.TestCase):
    """Test cases for the TrustEntity class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.entity = TrustEntity("test_entity", "node")

    def test_initialization(self):
        """Test trust entity initialization."""
        self.assertEqual(self.entity.entity_id, "test_entity")
        self.assertEqual(self.entity.entity_type, "node")
        self.assertEqual(len(self.entity.trust_scores), 0)
        self.assertFalse(self.entity.verification_status)

    def test_trust_score_management(self):
        """Test trust score management."""
        # Set trust score
        self.entity.set_trust_score("test_dimension", 0.7, 0.6)
        
        # Check that score was set
        self.assertIn("test_dimension", self.entity.trust_scores)
        self.assertEqual(self.entity.get_trust_value("test_dimension"), 0.7)
        
        # Update trust score
        self.entity.update_trust_score("test_dimension", 0.8, 0.7)
        
        # Check that score was updated
        self.assertNotEqual(self.entity.get_trust_value("test_dimension"), 0.7)
        
        # Get trust score object
        trust_score = self.entity.get_trust_score("test_dimension")
        self.assertIsInstance(trust_score, TrustScore)
        
        # Get dimensions
        dimensions = self.entity.get_dimensions()
        self.assertEqual(len(dimensions), 1)
        self.assertEqual(dimensions[0], "test_dimension")

    def test_aggregate_trust(self):
        """Test aggregate trust calculation."""
        # Set multiple trust scores
        self.entity.set_trust_score("dimension1", 0.7, 0.6)
        self.entity.set_trust_score("dimension2", 0.8, 0.7)
        self.entity.set_trust_score("dimension3", 0.9, 0.8)
        
        # Calculate aggregate trust with equal weights
        aggregate = self.entity.calculate_aggregate_trust()
        
        # Should be average of scores
        expected = (0.7 + 0.8 + 0.9) / 3
        self.assertAlmostEqual(aggregate, expected, places=6)
        
        # Calculate with custom weights
        weights = {"dimension1": 0.5, "dimension2": 1.0, "dimension3": 2.0}
        aggregate = self.entity.calculate_aggregate_trust(weights)
        
        # Should be weighted average
        expected = (0.7 * 0.5 + 0.8 * 1.0 + 0.9 * 2.0) / (0.5 + 1.0 + 2.0)
        self.assertAlmostEqual(aggregate, expected, places=6)
        
        # Get cached aggregate
        cached = self.entity.get_aggregate_trust()
        self.assertEqual(cached, aggregate)

    def test_verification(self):
        """Test verification status."""
        self.assertFalse(self.entity.is_verified())
        
        # Set verified
        self.entity.set_verified(True)
        self.assertTrue(self.entity.is_verified())
        
        # Check verification age
        age = self.entity.get_verification_age()
        self.assertGreaterEqual(age, 0)
        
        # Set not verified
        self.entity.set_verified(False)
        self.assertFalse(self.entity.is_verified())
        
        # Check verification age
        age = self.entity.get_verification_age()
        self.assertEqual(age, -1)

    def test_metadata(self):
        """Test metadata management."""
        # Set metadata
        self.entity.set_metadata("key1", "value1")
        self.entity.set_metadata("key2", 123)
        
        # Get metadata
        self.assertEqual(self.entity.get_metadata("key1"), "value1")
        self.assertEqual(self.entity.get_metadata("key2"), 123)
        self.assertIsNone(self.entity.get_metadata("nonexistent"))
        
        # Get all metadata
        all_metadata = self.entity.get_all_metadata()
        self.assertEqual(len(all_metadata), 2)
        self.assertEqual(all_metadata["key1"], "value1")
        self.assertEqual(all_metadata["key2"], 123)


class TestTrustHierarchy(unittest.TestCase):
    """Test cases for the TrustHierarchy class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.hierarchy = TrustHierarchy()

    def test_initialization(self):
        """Test trust hierarchy initialization."""
        # Check default dimensions
        dimensions = self.hierarchy.get_dimensions()
        self.assertEqual(len(dimensions), 4)
        self.assertIn("transaction_success", dimensions)
        self.assertIn("response_time", dimensions)
        self.assertIn("peer_rating", dimensions)
        self.assertIn("historical_trust", dimensions)

    def test_node_management(self):
        """Test node management."""
        # Add node
        node = self.hierarchy.add_node("node1")
        self.assertIsInstance(node, TrustEntity)
        self.assertEqual(node.get_id(), "node1")
        self.assertEqual(node.get_type(), "node")
        
        # Get node
        node = self.hierarchy.get_node("node1")
        self.assertIsNotNone(node)
        self.assertEqual(node.get_id(), "node1")
        
        # Get nonexistent node
        node = self.hierarchy.get_node("nonexistent")
        self.assertIsNone(node)
        
        # Get all nodes
        nodes = self.hierarchy.get_all_nodes()
        self.assertEqual(len(nodes), 1)
        self.assertIn("node1", nodes)

    def test_shard_management(self):
        """Test shard management."""
        # Add shard
        shard = self.hierarchy.add_shard("shard1")
        self.assertIsInstance(shard, TrustEntity)
        self.assertEqual(shard.get_id(), "shard1")
        self.assertEqual(shard.get_type(), "shard")
        
        # Get shard
        shard = self.hierarchy.get_shard("shard1")
        self.assertIsNotNone(shard)
        self.assertEqual(shard.get_id(), "shard1")
        
        # Get nonexistent shard
        shard = self.hierarchy.get_shard("nonexistent")
        self.assertIsNone(shard)
        
        # Get all shards
        shards = self.hierarchy.get_all_shards()
        self.assertEqual(len(shards), 1)
        self.assertIn("shard1", shards)

    def test_node_shard_assignment(self):
        """Test node to shard assignment."""
        # Add nodes and shards
        self.hierarchy.add_node("node1")
        self.hierarchy.add_node("node2")
        self.hierarchy.add_shard("shard1")
        self.hierarchy.add_shard("shard2")
        
        # Assign nodes to shards
        self.hierarchy.assign_node_to_shard("node1", "shard1")
        self.hierarchy.assign_node_to_shard("node2", "shard2")
        
        # Check assignments
        self.assertEqual(self.hierarchy.get_node_shard("node1"), "shard1")
        self.assertEqual(self.hierarchy.get_node_shard("node2"), "shard2")
        
        # Get shard nodes
        shard1_nodes = self.hierarchy.get_shard_nodes("shard1")
        shard2_nodes = self.hierarchy.get_shard_nodes("shard2")
        
        self.assertEqual(len(shard1_nodes), 1)
        self.assertEqual(len(shard2_nodes), 1)
        self.assertIn("node1", shard1_nodes)
        self.assertIn("node2", shard2_nodes)
        
        # Reassign node
        self.hierarchy.assign_node_to_shard("node1", "shard2")
        
        # Check new assignments
        self.assertEqual(self.hierarchy.get_node_shard("node1"), "shard2")
        
        # Get shard nodes
        shard1_nodes = self.hierarchy.get_shard_nodes("shard1")
        shard2_nodes = self.hierarchy.get_shard_nodes("shard2")
        
        self.assertEqual(len(shard1_nodes), 0)
        self.assertEqual(len(shard2_nodes), 2)
        self.assertIn("node1", shard2_nodes)
        self.assertIn("node2", shard2_nodes)

    def test_trust_propagation(self):
        """Test trust propagation from nodes to shards to network."""
        # Add nodes and shards
        self.hierarchy.add_node("node1")
        self.hierarchy.add_node("node2")
        self.hierarchy.add_shard("shard1")
        
        # Assign nodes to shards
        self.hierarchy.assign_node_to_shard("node1", "shard1")
        self.hierarchy.assign_node_to_shard("node2", "shard1")
        
        # Update node trust
        self.hierarchy.update_node_trust("node1", "transaction_success", 0.8, 0.7)
        self.hierarchy.update_node_trust("node2", "transaction_success", 0.9, 0.8)
        
        # Check that shard trust was updated
        shard = self.hierarchy.get_shard("shard1")
        shard_trust = shard.get_trust_value("transaction_success")
        self.assertGreater(shard_trust, 0.5)  # Default is 0.5
        
        # Check that network trust was updated
        network = self.hierarchy.get_network()
        network_trust = network.get_trust_value("transaction_success")
        self.assertGreater(network_trust, 0.5)  # Default is 0.5

    def test_pagerank_trust(self):
        """Test PageRank-based trust calculation."""
        # Add nodes and shards
        self.hierarchy.add_node("node1")
        self.hierarchy.add_node("node2")
        self.hierarchy.add_node("node3")
        self.hierarchy.add_shard("shard1")
        self.hierarchy.add_shard("shard2")
        
        # Assign nodes to shards
        self.hierarchy.assign_node_to_shard("node1", "shard1")
        self.hierarchy.assign_node_to_shard("node2", "shard1")
        self.hierarchy.assign_node_to_shard("node3", "shard2")
        
        # Update node trust
        self.hierarchy.update_node_trust("node1", "transaction_success", 0.8, 0.7)
        self.hierarchy.update_node_trust("node2", "transaction_success", 0.9, 0.8)
        self.hierarchy.update_node_trust("node3", "transaction_success", 0.7, 0.6)
        
        # Add cross-shard transactions
        shard1 = self.hierarchy.get_shard("shard1")
        shard1.set_metadata("cross_shard_tx_shard2", 100)
        
        # Calculate PageRank
        pagerank = self.hierarchy.calculate_pagerank_trust()
        
        # Check that all entities have PageRank scores
        self.assertIn("node1", pagerank)
        self.assertIn("node2", pagerank)
        self.assertIn("node3", pagerank)
        self.assertIn("shard1", pagerank)
        self.assertIn("shard2", pagerank)
        
        # Update trust from PageRank
        self.hierarchy.update_trust_from_pagerank(pagerank)
        
        # Check that PageRank dimension was added
        node1 = self.hierarchy.get_node("node1")
        self.assertIsNotNone(node1.get_trust_score("pagerank"))

    def test_byzantine_detection(self):
        """Test Byzantine node detection."""
        # Add nodes
        self.hierarchy.add_node("good_node1")
        self.hierarchy.add_node("good_node2")
        self.hierarchy.add_node("bad_node")
        
        # Update trust scores
        self.hierarchy.update_node_trust("good_node1", "transaction_success", 0.8, 0.7)
        self.hierarchy.update_node_trust("good_node2", "transaction_success", 0.9, 0.8)
        self.hierarchy.update_node_trust("bad_node", "transaction_success", 0.2, 0.7)
        
        # Calculate aggregate trust
        for node_id in ["good_node1", "good_node2", "bad_node"]:
            node = self.hierarchy.get_node(node_id)
            node.calculate_aggregate_trust()
        
        # Detect Byzantine nodes
        byzantine_nodes = self.hierarchy.detect_byzantine_nodes(0.3)
        
        # Check detection
        self.assertEqual(len(byzantine_nodes), 1)
        self.assertIn("bad_node", byzantine_nodes)
        self.assertNotIn("good_node1", byzantine_nodes)
        self.assertNotIn("good_node2", byzantine_nodes)

    def test_trust_rankings(self):
        """Test trust rankings."""
        # Add nodes and shards
        self.hierarchy.add_node("node1")
        self.hierarchy.add_node("node2")
        self.hierarchy.add_node("node3")
        self.hierarchy.add_shard("shard1")
        self.hierarchy.add_shard("shard2")
        
        # Update trust scores
        self.hierarchy.update_node_trust("node1", "transaction_success", 0.8, 0.7)
        self.hierarchy.update_node_trust("node2", "transaction_success", 0.9, 0.8)
        self.hierarchy.update_node_trust("node3", "transaction_success", 0.7, 0.6)
        
        # Calculate aggregate trust
        for node_id in ["node1", "node2", "node3"]:
            node = self.hierarchy.get_node(node_id)
            node.calculate_aggregate_trust()
        
        # Get node rankings
        node_ranking = self.hierarchy.get_node_trust_ranking()
        
        # Check ranking
        self.assertEqual(len(node_ranking), 3)
        self.assertEqual(node_ranking[0][0], "node2")  # Highest trust
        self.assertEqual(node_ranking[2][0], "node3")  # Lowest trust


class TestHTDCM(unittest.TestCase):
    """Test cases for the HTDCM class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.config = {
            'trust_threshold': 0.7,
            'byzantine_threshold': 0.3,
            'decay_interval': 3600,
            'pagerank_interval': 600,
            'update_interval': 60
        }
        self.htdcm = HTDCM(self.config)

    def test_initialization(self):
        """Test HTDCM initialization."""
        self.assertEqual(self.htdcm.config['trust_threshold'], 0.7)
        self.assertEqual(self.htdcm.config['byzantine_threshold'], 0.3)
        self.assertIsInstance(self.htdcm.hierarchy, TrustHierarchy)
        self.assertIsInstance(self.htdcm.trust_graph, nx.DiGraph)
        self.assertEqual(self.htdcm.stats['updates'], 0)
        self.assertEqual(self.htdcm.stats['verifications'], 0)
        self.assertEqual(self.htdcm.stats['byzantine_detected'], 0)
        self.assertEqual(self.htdcm.stats['trust_propagations'], 0)

    def test_node_shard_management(self):
        """Test node and shard management."""
        # Add nodes and shards
        self.htdcm.add_node("node1", "shard1")
        self.htdcm.add_node("node2")
        self.htdcm.add_shard("shard2")
        
        # Assign node to shard
        self.htdcm.assign_node_to_shard("node2", "shard2")
        
        # Check that nodes and shards were added
        self.assertIn("node1", self.htdcm.hierarchy.nodes)
        self.assertIn("node2", self.htdcm.hierarchy.nodes)
        self.assertIn("shard1", self.htdcm.hierarchy.shards)
        self.assertIn("shard2", self.htdcm.hierarchy.shards)
        
        # Check assignments
        self.assertEqual(self.htdcm.hierarchy.get_node_shard("node1"), "shard1")
        self.assertEqual(self.htdcm.hierarchy.get_node_shard("node2"), "shard2")

    def test_trust_updates(self):
        """Test trust updates."""
        # Add nodes
        self.htdcm.add_node("node1", "shard1")
        
        # Update trust
        self.htdcm.update_trust("node1", "transaction_success", 0.8, 0.7)
        
        # Check that trust was updated
        trust = self.htdcm.get_trust_score("node1", "transaction_success")
        self.assertAlmostEqual(trust, 0.8, places=6)
        
        # Check stats
        self.assertEqual(self.htdcm.stats['updates'], 1)

    def test_transaction_trust(self):
        """Test transaction-based trust updates."""
        # Add nodes
        self.htdcm.add_node("node1", "shard1")
        
        # Update transaction trust
        self.htdcm.update_transaction_trust("node1", True, 200)
        
        # Check that trust was updated
        transaction_trust = self.htdcm.get_trust_score("node1", "transaction_success")
        response_trust = self.htdcm.get_trust_score("node1", "response_time")
        
        self.assertAlmostEqual(transaction_trust, 1.0, places=6)
        self.assertGreater(response_trust, 0.5)  # Should be good for 200ms
        
        # Update with failed transaction
        self.htdcm.update_transaction_trust("node1", False, 800)
        
        # Check that trust was updated
        transaction_trust = self.htdcm.get_trust_score("node1", "transaction_success")
        response_trust = self.htdcm.get_trust_score("node1", "response_time")
        
        self.assertLess(transaction_trust, 1.0)  # Should be reduced due to failure
        self.assertLess(response_trust, 0.5)  # Should be poor for 800ms

    def test_peer_rating(self):
        """Test peer rating trust updates."""
        # Add nodes
        self.htdcm.add_node("node1", "shard1")
        self.htdcm.add_node("node2", "shard1")
        
        # Set initial trust for rater
        self.htdcm.update_trust("node1", "transaction_success", 0.9, 0.8)
        
        # Update peer rating
        self.htdcm.update_peer_rating("node1", "node2", 0.8)
        
        # Check that trust was updated
        peer_trust = self.htdcm.get_trust_score("node2", "peer_rating")
        
        # Should be weighted by rater's trust
        self.assertGreater(peer_trust, 0.5)
        
        # Check stats
        self.assertEqual(self.htdcm.stats['trust_propagations'], 1)
        
        # Check trust graph
        self.assertTrue(self.htdcm.trust_graph.has_edge("node1", "node2"))
        self.assertGreater(self.htdcm.trust_graph["node1"]["node2"]["weight"], 0.5)

    def test_verification(self):
        """Test entity verification."""
        # Add node
        self.htdcm.add_node("node1", "shard1")
        
        # Verify node
        self.htdcm.verify_entity("node1", True)
        
        # Check verification status
        self.assertTrue(self.htdcm.is_verified("node1"))
        
        # Check stats
        self.assertEqual(self.htdcm.stats['verifications'], 1)
        
        # Unverify node
        self.htdcm.verify_entity("node1", False)
        
        # Check verification status
        self.assertFalse(self.htdcm.is_verified("node1"))

    def test_trusted_entities(self):
        """Test getting trusted entities."""
        # Add nodes
        self.htdcm.add_node("good_node", "shard1")
        self.htdcm.add_node("bad_node", "shard1")
        
        # Update trust
        self.htdcm.update_trust("good_node", "transaction_success", 0.9, 0.8)
        self.htdcm.update_trust("bad_node", "transaction_success", 0.2, 0.7)
        
        # Get trusted nodes
        trusted_nodes = self.htdcm.get_trusted_entities("node", 0.7)
        
        # Check trusted nodes
        self.assertEqual(len(trusted_nodes), 1)
        self.assertIn("good_node", trusted_nodes)
        self.assertNotIn("bad_node", trusted_nodes)

    def test_byzantine_detection(self):
        """Test Byzantine node detection."""
        # Add nodes
        self.htdcm.add_node("good_node", "shard1")
        self.htdcm.add_node("bad_node", "shard1")
        
        # Update trust
        self.htdcm.update_trust("good_node", "transaction_success", 0.9, 0.8)
        self.htdcm.update_trust("bad_node", "transaction_success", 0.2, 0.7)
        
        # Detect Byzantine nodes
        byzantine_nodes = self.htdcm.detect_byzantine_nodes()
        
        # Check detection
        self.assertEqual(len(byzantine_nodes), 1)
        self.assertIn("bad_node", byzantine_nodes)
        self.assertNotIn("good_node", byzantine_nodes)
        
        # Check stats
        self.assertEqual(self.htdcm.stats['byzantine_detected'], 1)

    def test_trusted_path(self):
        """Test finding trusted paths."""
        # Add nodes
        self.htdcm.add_node("node1", "shard1")
        self.htdcm.add_node("node2", "shard1")
        self.htdcm.add_node("node3", "shard2")
        
        # Add trust relationships
        self.htdcm.update_peer_rating("node1", "node2", 0.9)
        self.htdcm.update_peer_rating("node2", "node3", 0.8)
        
        # Find trusted path
        path = self.htdcm.get_trusted_path("node1", "node3")
        
        # Check path
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], "node1")
        self.assertEqual(path[1], "node2")
        self.assertEqual(path[2], "node3")
        
        # Find path with higher threshold
        path = self.htdcm.get_trusted_path("node1", "node3", 0.9)
        
        # Should not find path with higher threshold
        self.assertIsNone(path)

    def test_security_level(self):
        """Test security level calculation."""
        # Add nodes
        self.htdcm.add_node("good_node1", "shard1")
        self.htdcm.add_node("good_node2", "shard1")
        self.htdcm.add_node("bad_node", "shard1")
        
        # Update trust
        self.htdcm.update_trust("good_node1", "transaction_success", 0.9, 0.8)
        self.htdcm.update_trust("good_node2", "transaction_success", 0.8, 0.7)
        self.htdcm.update_trust("bad_node", "transaction_success", 0.2, 0.7)
        
        # Create mock state
        state = MagicMock()
        
        # Calculate security level
        security_level = self.htdcm.get_security_level(state)
        
        # Check security level
        self.assertGreaterEqual(security_level, 0.0)
        self.assertLessEqual(security_level, 1.0)

    def test_export_import(self):
        """Test exporting and importing trust network."""
        # Add nodes and shards
        self.htdcm.add_node("node1", "shard1")
        self.htdcm.add_node("node2", "shard1")
        self.htdcm.add_shard("shard2")
        
        # Update trust
        self.htdcm.update_trust("node1", "transaction_success", 0.9, 0.8)
        self.htdcm.update_trust("node2", "transaction_success", 0.8, 0.7)
        
        # Add trust relationships
        self.htdcm.update_peer_rating("node1", "node2", 0.9)
        
        # Export trust network
        network_data = self.htdcm.export_trust_network()
        
        # Check export
        self.assertIn("nodes", network_data)
        self.assertIn("edges", network_data)
        self.assertEqual(len(network_data["nodes"]), 4)  # 2 nodes + 1 shard + network
        self.assertEqual(len(network_data["edges"]), 4)  # 1 peer rating + 2 node-shard + 1 shard-network
        
        # Create new HTDCM
        new_htdcm = HTDCM(self.config)
        
        # Import trust network
        new_htdcm.import_trust_network(network_data)
        
        # Check import
        self.assertIn("node1", new_htdcm.hierarchy.nodes)
        self.assertIn("node2", new_htdcm.hierarchy.nodes)
        self.assertIn("shard1", new_htdcm.hierarchy.shards)
        self.assertTrue(new_htdcm.trust_graph.has_edge("node1", "node2"))


if __name__ == '__main__':
    unittest.main()
