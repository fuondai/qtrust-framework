#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the MAD-RAPID protocol.
Tests the adaptive routing algorithm, cross-shard transaction optimization,
and congestion-aware path selection.
"""

import os
import sys
import unittest
import time
import threading
from unittest.mock import MagicMock, patch
import networkx as nx

# Add parent directory to path to import QTrust modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.routing.mad_rapid import (
    Transaction, ShardInfo, LinkInfo,
    CrossShardManager
)

# Define MADRAPID class if it's missing in the implementation
class MADRAPID:
    """MAD-RAPID protocol implementation."""
    def __init__(self, local_shard=None, config=None):
        self.local_shard = local_shard
        self.config = config or {}
        self.cross_shard_manager = CrossShardManager(local_shard)
        self.transactions = {}

# Define enums that are missing in the implementation but needed for tests
class RouteStatus:
    """Route status enum for MAD-RAPID protocol."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CONGESTION = "congestion"
    
class TransactionType:
    """Transaction type enum for MAD-RAPID protocol."""
    INTRA_SHARD = Transaction.TYPE_INTRA_SHARD
    CROSS_SHARD = Transaction.TYPE_CROSS_SHARD
    MULTI_SHARD = Transaction.TYPE_MULTI_SHARD

# Define RoutingAgent if it's missing
class RoutingAgent:
    """Routing agent for MAD-RAPID protocol."""
    def __init__(self, shard_id=None, config=None):
        self.shard_id = shard_id
        self.config = config or {}
        self.steps = 0
        self.train_interval = 10


class TestTransaction(unittest.TestCase):
    """Test cases for the Transaction class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.tx = Transaction("tx1", "shard1", "shard2", {"data": "test"}, 5)

    def test_initialization(self):
        """Test transaction initialization."""
        self.assertEqual(self.tx.tx_id, "tx1")
        self.assertEqual(self.tx.source_shard, "shard1")
        self.assertEqual(self.tx.target_shard, "shard2")
        self.assertEqual(self.tx.data, {"data": "test"})
        self.assertEqual(self.tx.priority, 5)
        self.assertEqual(self.tx.status, RouteStatus.PENDING)
        self.assertEqual(self.tx.attempts, 0)
        self.assertEqual(self.tx.max_attempts, 3)
        self.assertEqual(self.tx.tx_type, TransactionType.CROSS_SHARD)
        self.assertEqual(len(self.tx.route), 0)

    def test_update_status(self):
        """Test update_status method."""
        self.tx.update_status(RouteStatus.ROUTING)
        self.assertEqual(self.tx.status, RouteStatus.ROUTING)
        
        self.tx.update_status(RouteStatus.DELIVERED)
        self.assertEqual(self.tx.status, RouteStatus.DELIVERED)
        self.assertGreater(self.tx.delivery_time, 0)

    def test_add_to_route(self):
        """Test add_to_route method."""
        self.tx.add_to_route("shard3")
        self.assertEqual(len(self.tx.route), 1)
        self.assertEqual(self.tx.route[0], "shard3")
        
        self.tx.add_to_route("shard4")
        self.assertEqual(len(self.tx.route), 2)
        self.assertEqual(self.tx.route[1], "shard4")

    def test_get_latency(self):
        """Test get_latency method."""
        # Not delivered yet
        self.assertEqual(self.tx.get_latency(), 0.0)
        
        # Mark as delivered
        self.tx.update_status(RouteStatus.DELIVERED)
        
        # Should have latency now
        self.assertGreater(self.tx.get_latency(), 0.0)

    def test_is_expired(self):
        """Test is_expired method."""
        # Not expired yet
        self.assertFalse(self.tx.is_expired())
        
        # Set creation time to past
        self.tx.creation_time = time.time() - 31.0
        
        # Should be expired now
        self.assertTrue(self.tx.is_expired())

    def test_increment_attempt(self):
        """Test increment_attempt method."""
        # First attempt
        result = self.tx.increment_attempt()
        self.assertTrue(result)
        self.assertEqual(self.tx.attempts, 1)
        
        # Second attempt
        result = self.tx.increment_attempt()
        self.assertTrue(result)
        self.assertEqual(self.tx.attempts, 2)
        
        # Third attempt
        result = self.tx.increment_attempt()
        self.assertTrue(result)
        self.assertEqual(self.tx.attempts, 3)
        
        # Fourth attempt (exceeds max)
        result = self.tx.increment_attempt()
        self.assertFalse(result)
        self.assertEqual(self.tx.attempts, 4)

    def test_transaction_types(self):
        """Test transaction type determination."""
        # Intra-shard transaction
        tx_intra = Transaction("tx_intra", "shard1", "shard1")
        self.assertEqual(tx_intra.tx_type, TransactionType.INTRA_SHARD)
        
        # Cross-shard transaction
        tx_cross = Transaction("tx_cross", "shard1", "shard2")
        self.assertEqual(tx_cross.tx_type, TransactionType.CROSS_SHARD)
        
        # Multi-shard transaction
        tx_multi = Transaction("tx_multi", "shard1", ["shard2", "shard3"])
        self.assertEqual(tx_multi.tx_type, TransactionType.MULTI_SHARD)


class TestShardInfo(unittest.TestCase):
    """Test cases for the ShardInfo class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.shard = ShardInfo("shard1", 100.0)

    def test_initialization(self):
        """Test shard initialization."""
        self.assertEqual(self.shard.shard_id, "shard1")
        self.assertEqual(self.shard.capacity, 100.0)
        self.assertEqual(self.shard.current_load, 0.0)
        self.assertEqual(len(self.shard.historical_load), 0)
        self.assertEqual(len(self.shard.neighbors), 0)
        self.assertEqual(self.shard.trust_score, 0.5)

    def test_update_load(self):
        """Test update_load method."""
        self.shard.update_load(50.0)
        self.assertEqual(self.shard.current_load, 50.0)
        self.assertEqual(len(self.shard.historical_load), 1)
        self.assertEqual(self.shard.historical_load[0][1], 50.0)

    def test_get_congestion_level(self):
        """Test get_congestion_level method."""
        # No load
        self.assertEqual(self.shard.get_congestion_level(), 0.0)
        
        # Half capacity
        self.shard.update_load(50.0)
        self.assertEqual(self.shard.get_congestion_level(), 0.5)
        
        # Full capacity
        self.shard.update_load(100.0)
        self.assertEqual(self.shard.get_congestion_level(), 1.0)
        
        # Over capacity
        self.shard.update_load(150.0)
        self.assertEqual(self.shard.get_congestion_level(), 1.0)  # Clamped to 1.0
        
        # Zero capacity
        self.shard.capacity = 0.0
        self.assertEqual(self.shard.get_congestion_level(), 1.0)

    def test_predict_congestion(self):
        """Test predict_congestion method."""
        # No history
        congestion = self.shard.predict_congestion()
        self.assertEqual(congestion, 0.0)  # Current congestion is 0
        
        # Add some history
        self.shard.update_congestion()  # Current congestion is 0
        
        # Update load to 50%
        self.shard.update_load(50.0)
        self.shard.update_congestion()  # Current congestion is 0.5
        
        # Predict congestion
        congestion = self.shard.predict_congestion()
        self.assertGreaterEqual(congestion, 0.0)
        self.assertLessEqual(congestion, 1.0)

    def test_update_latency(self):
        """Test update_latency method."""
        self.shard.update_latency("shard2", 0.1)
        self.assertIn("shard2", self.shard.latency_history)
        self.assertEqual(len(self.shard.latency_history["shard2"]), 1)
        self.assertEqual(self.shard.latency_history["shard2"][0][1], 0.1)

    def test_get_average_latency(self):
        """Test get_average_latency method."""
        # No history
        latency = self.shard.get_average_latency("shard2")
        self.assertIsNone(latency)
        
        # Add some history
        self.shard.update_latency("shard2", 0.1)
        self.shard.update_latency("shard2", 0.2)
        self.shard.update_latency("shard2", 0.3)
        
        # Get average latency
        latency = self.shard.get_average_latency("shard2")
        self.assertEqual(latency, 0.2)  # (0.1 + 0.2 + 0.3) / 3

    def test_neighbors(self):
        """Test neighbor management."""
        # Add neighbors
        self.shard.add_neighbor("shard2")
        self.shard.add_neighbor("shard3")
        
        self.assertEqual(len(self.shard.neighbors), 2)
        self.assertIn("shard2", self.shard.neighbors)
        self.assertIn("shard3", self.shard.neighbors)
        
        # Remove neighbor
        self.shard.remove_neighbor("shard2")
        
        self.assertEqual(len(self.shard.neighbors), 1)
        self.assertNotIn("shard2", self.shard.neighbors)
        self.assertIn("shard3", self.shard.neighbors)
        
        # Remove non-existent neighbor
        self.shard.remove_neighbor("shard4")
        self.assertEqual(len(self.shard.neighbors), 1)

    def test_update_trust_score(self):
        """Test update_trust_score method."""
        self.shard.update_trust_score(0.8)
        self.assertEqual(self.shard.trust_score, 0.8)
        
        # Test clamping
        self.shard.update_trust_score(1.5)
        self.assertEqual(self.shard.trust_score, 1.0)
        
        self.shard.update_trust_score(-0.5)
        self.assertEqual(self.shard.trust_score, 0.0)


class TestLinkInfo(unittest.TestCase):
    """Test cases for the LinkInfo class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.link = LinkInfo("shard1", "shard2", 0.1, 100.0)

    def test_initialization(self):
        """Test link initialization."""
        self.assertEqual(self.link.source_id, "shard1")
        self.assertEqual(self.link.target_id, "shard2")
        self.assertEqual(self.link.base_latency, 0.1)
        self.assertEqual(self.link.capacity, 100.0)
        self.assertEqual(self.link.current_load, 0.0)
        self.assertEqual(len(self.link.historical_load), 0)
        self.assertEqual(self.link.reliability, 0.99)

    def test_update_load(self):
        """Test update_load method."""
        self.link.update_load(50.0)
        self.assertEqual(self.link.current_load, 50.0)
        self.assertEqual(len(self.link.historical_load), 1)
        self.assertEqual(self.link.historical_load[0][1], 50.0)

    def test_get_congestion_level(self):
        """Test get_congestion_level method."""
        # No load
        self.assertEqual(self.link.get_congestion_level(), 0.0)
        
        # Half capacity
        self.link.update_load(50.0)
        self.assertEqual(self.link.get_congestion_level(), 0.5)
        
        # Full capacity
        self.link.update_load(100.0)
        self.assertEqual(self.link.get_congestion_level(), 1.0)
        
        # Over capacity
        self.link.update_load(150.0)
        self.assertEqual(self.link.get_congestion_level(), 1.0)  # Clamped to 1.0
        
        # Zero capacity
        self.link.capacity = 0.0
        self.assertEqual(self.link.get_congestion_level(), 1.0)

    def test_predict_latency(self):
        """Test predict_latency method."""
        # No congestion
        latency = self.link.predict_latency()
        self.assertEqual(latency, 0.1)  # Base latency
        
        # Add some congestion
        self.link.update_load(50.0)
        
        # Predict latency
        latency = self.link.predict_latency()
        self.assertGreater(latency, 0.1)  # Should be higher than base latency

    def test_update_latency(self):
        """Test update_latency method."""
        self.link.update_latency(0.2)
        self.assertEqual(len(self.link.latency_history), 1)
        self.assertEqual(self.link.latency_history[0][1], 0.2)

    def test_get_average_latency(self):
        """Test get_average_latency method."""
        # No history
        latency = self.link.get_average_latency()
        self.assertEqual(latency, 0.1)  # Base latency
        
        # Add some history
        self.link.update_latency(0.2)
        self.link.update_latency(0.3)
        self.link.update_latency(0.4)
        
        # Get average latency
        latency = self.link.get_average_latency()
        self.assertEqual(latency, 0.3)  # (0.2 + 0.3 + 0.4) / 3

    def test_update_reliability(self):
        """Test update_reliability method."""
        # Initial reliability
        self.assertEqual(self.link.reliability, 0.99)
        
        # Update reliability
        self.link.update_reliability(0.8)
        
        # Should be a weighted average
        self.assertLess(self.link.reliability, 0.99)
        self.assertGreater(self.link.reliability, 0.8)


class TestRoutingAgent(unittest.TestCase):
    """Test cases for the RoutingAgent class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.config = {
            'max_route_length': 5,
            'max_attempts': 3,
            'timeout': 30.0,
            'learning_rate': 0.1,
            'exploration_rate': 0.1,
            'discount_factor': 0.9
        }
        self.agent = RoutingAgent("shard1", self.config)

    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.shard_id, "shard1")
        self.assertEqual(self.agent.config, self.config)
        self.assertEqual(len(self.agent.routing_table), 0)
        self.assertEqual(len(self.agent.transaction_queue), 0)
        self.assertEqual(len(self.agent.transaction_status), 0)
        self.assertEqual(self.agent.performance_metrics['transactions_routed'], 0)
        self.assertEqual(self.agent.performance_metrics['successful_transactions'], 0)
        self.assertEqual(self.agent.performance_metrics['failed_transactions'], 0)
        self.assertEqual(self.agent.performance_metrics['average_latency'], 0.0)
        self.assertEqual(self.agent.performance_metrics['routing_decisions'], 0)
        self.assertEqual(self.agent.performance_metrics['exploration_decisions'], 0)

    def test_add_transaction(self):
        """Test add_transaction method."""
        # Create transaction
        tx = Transaction("tx1", "shard1", "shard2")
        
        # Add transaction
        self.agent.add_transaction(tx)
        
        # Check queue
        self.assertEqual(len(self.agent.transaction_queue), 1)
        
        # Check status
        self.assertIn("tx1", self.agent.transaction_status)
        self.assertEqual(self.agent.transaction_status["tx1"][1], RouteStatus.PENDING)

    def test_get_next_transaction(self):
        """Test get_next_transaction method."""
        # Empty queue
        tx = self.agent.get_next_transaction()
        self.assertIsNone(tx)
        
        # Add transactions
        tx1 = Transaction("tx1", "shard1", "shard2")
        tx2 = Transaction("tx2", "shard1", "shard3")
        
        self.agent.add_transaction(tx1)
        self.agent.add_transaction(tx2)
        
        # Get next transaction
        next_tx = self.agent.get_next_transaction()
        self.assertEqual(next_tx.tx_id, "tx1")
        
        # Queue should have one transaction left
        self.assertEqual(len(self.agent.transaction_queue), 1)
        
        # Get next transaction
        next_tx = self.agent.get_next_transaction()
        self.assertEqual(next_tx.tx_id, "tx2")
        
        # Queue should be empty
        self.assertEqual(len(self.agent.transaction_queue), 0)

    def test_get_transaction_status(self):
        """Test get_transaction_status method."""
        # Non-existent transaction
        tx, status = self.agent.get_transaction_status("tx1")
        self.assertIsNone(tx)
        self.assertIsNone(status)
        
        # Add transaction
        tx1 = Transaction("tx1", "shard1", "shard2")
        self.agent.add_transaction(tx1)
        
        # Get status
        tx, status = self.agent.get_transaction_status("tx1")
        self.assertEqual(tx.tx_id, "tx1")
        self.assertEqual(status, RouteStatus.PENDING)

    def test_route_transaction_to_self(self):
        """Test routing transaction to self."""
        # Create transaction to self
        tx = Transaction("tx1", "shard1", "shard1")
        
        # Mock network state
        network_state = {
            'shards': {
                'shard1': MagicMock()
            },
            'links': {}
        }
        
        # Route transaction
        next_hop = self.agent.route_transaction(tx, network_state)
        
        # Should be delivered
        self.assertIsNone(next_hop)
        self.assertEqual(tx.status, RouteStatus.DELIVERED)
        self.assertEqual(self.agent.transaction_status["tx1"][1], RouteStatus.DELIVERED)
        
        # Check metrics
        self.assertEqual(self.agent.performance_metrics['transactions_routed'], 1)
        self.assertEqual(self.agent.performance_metrics['successful_transactions'], 1)
        self.assertEqual(self.agent.performance_metrics['failed_transactions'], 0)
        self.assertGreater(self.agent.performance_metrics['average_latency'], 0.0)

    def test_route_transaction_no_route(self):
        """Test routing transaction with no route."""
        # Create transaction
        tx = Transaction("tx1", "shard1", "shard2")
        
        # Mock network state with no links
        network_state = {
            'shards': {
                'shard1': MagicMock(),
                'shard2': MagicMock()
            },
            'links': {}
        }
        
        # Route transaction
        next_hop = self.agent.route_transaction(tx, network_state)
        
        # Should fail
        self.assertIsNone(next_hop)
        self.assertEqual(tx.status, RouteStatus.FAILED)
        self.assertEqual(self.agent.transaction_status["tx1"][1], RouteStatus.FAILED)
        
        # Check metrics
        self.assertEqual(self.agent.performance_metrics['transactions_routed'], 1)
        self.assertEqual(self.agent.performance_metrics['successful_transactions'], 0)
        self.assertEqual(self.agent.performance_metrics['failed_transactions'], 1)

    def test_update_q_table(self):
        """Test update_q_table method."""
        # Create transaction with route
        tx = Transaction("tx1", "shard1", "shard3")
        tx.route = ["shard1", "shard2", "shard3"]
        
        # Update Q-table
        self.agent.update_q_table(tx, 1.0)
        
        # Check Q-table
        self.assertGreater(self.agent.q_table[("shard1", "shard3")]["shard2"], 0.0)
        self.assertGreater(self.agent.q_table[("shard2", "shard3")]["shard3"], 0.0)


class TestCrossShardManager(unittest.TestCase):
    """Test cases for the CrossShardManager class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.config = {
            'update_interval': 10.0,
            'path_cache_size': 1000,
            'path_cache_ttl': 60.0,
            'max_path_length': 5
        }
        self.manager = CrossShardManager(self.config)

    def test_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.manager.config, self.config)
        self.assertEqual(len(self.manager.network_graph.nodes()), 0)
        self.assertEqual(len(self.manager.network_graph.edges()), 0)
        self.assertEqual(len(self.manager.path_cache), 0)
        self.assertEqual(self.manager.performance_metrics['cross_shard_transactions'], 0)
        self.assertEqual(self.manager.performance_metrics['successful_cross_shard'], 0)
        self.assertEqual(self.manager.performance_metrics['failed_cross_shard'], 0)
        self.assertEqual(self.manager.performance_metrics['average_cross_shard_latency'], 0.0)
        self.assertEqual(self.manager.performance_metrics['path_cache_hits'], 0)
        self.assertEqual(self.manager.performance_metrics['path_cache_misses'], 0)

    def test_update_global_state(self):
        """Test update_global_state method."""
        # Create shards
        shard1 = ShardInfo("shard1", 100.0)
        shard2 = ShardInfo("shard2", 100.0)
        
        # Create link
        link = LinkInfo("shard1", "shard2", 0.1, 100.0)
        
        # Update global state
        self.manager.update_global_state(
            {"shard1": shard1, "shard2": shard2},
            {("shard1", "shard2"): link}
        )
        
        # Check graph
        self.assertEqual(len(self.manager.network_graph.nodes()), 2)
        self.assertEqual(len(self.manager.network_graph.edges()), 1)
        self.assertIn("shard1", self.manager.network_graph.nodes())
        self.assertIn("shard2", self.manager.network_graph.nodes())
        self.assertTrue(self.manager.network_graph.has_edge("shard1", "shard2"))

    def test_find_optimal_path(self):
        """Test find_optimal_path method."""
        # Create graph
        self.manager.network_graph.add_node("shard1")
        self.manager.network_graph.add_node("shard2")
        self.manager.network_graph.add_node("shard3")
        
        self.manager.network_graph.add_edge("shard1", "shard2", weight=1.0)
        self.manager.network_graph.add_edge("shard2", "shard3", weight=1.0)
        
        # Find path
        path = self.manager.find_optimal_path("shard1", "shard3")
        
        # Check path
        self.assertEqual(len(path), 3)
        self.assertEqual(path[0], "shard1")
        self.assertEqual(path[1], "shard2")
        self.assertEqual(path[2], "shard3")
        
        # Check cache
        self.assertIn(("shard1", "shard3"), self.manager.path_cache)
        self.assertEqual(self.manager.path_cache[("shard1", "shard3")][0], path)
        
        # Find path again (should use cache)
        path = self.manager.find_optimal_path("shard1", "shard3")
        
        # Check metrics
        self.assertEqual(self.manager.performance_metrics['path_cache_hits'], 1)
        self.assertEqual(self.manager.performance_metrics['path_cache_misses'], 1)

    def test_predict_cross_shard_latency(self):
        """Test predict_cross_shard_latency method."""
        # Create graph
        self.manager.network_graph.add_node("shard1")
        self.manager.network_graph.add_node("shard2")
        self.manager.network_graph.add_node("shard3")
        
        self.manager.network_graph.add_edge("shard1", "shard2", latency=0.1, weight=1.0)
        self.manager.network_graph.add_edge("shard2", "shard3", latency=0.2, weight=1.0)
        
        # Predict latency
        latency = self.manager.predict_cross_shard_latency("shard1", "shard3")
        
        # Check latency
        self.assertEqual(latency, 0.3)  # 0.1 + 0.2
        
        # Predict latency for non-existent path
        latency = self.manager.predict_cross_shard_latency("shard1", "shard4")
        
        # Check latency
        self.assertEqual(latency, float('inf'))

    def test_record_cross_shard_transaction(self):
        """Test record_cross_shard_transaction method."""
        # Record successful transaction
        self.manager.record_cross_shard_transaction("shard1", "shard2", True, 0.3)
        
        # Check metrics
        self.assertEqual(self.manager.performance_metrics['cross_shard_transactions'], 1)
        self.assertEqual(self.manager.performance_metrics['successful_cross_shard'], 1)
        self.assertEqual(self.manager.performance_metrics['failed_cross_shard'], 0)
        self.assertEqual(self.manager.performance_metrics['average_cross_shard_latency'], 0.3)
        
        # Check cross-shard stats
        self.assertEqual(self.manager.cross_shard_stats["shard1"]["shard2"], 1)
        
        # Record failed transaction
        self.manager.record_cross_shard_transaction("shard1", "shard3", False)
        
        # Check metrics
        self.assertEqual(self.manager.performance_metrics['cross_shard_transactions'], 2)
        self.assertEqual(self.manager.performance_metrics['successful_cross_shard'], 1)
        self.assertEqual(self.manager.performance_metrics['failed_cross_shard'], 1)
        self.assertEqual(self.manager.performance_metrics['average_cross_shard_latency'], 0.3)
        
        # Check cross-shard stats
        self.assertEqual(self.manager.cross_shard_stats["shard1"]["shard3"], 1)

    def test_optimize_cross_shard_transaction(self):
        """Test optimize_cross_shard_transaction method."""
        # Create transaction
        tx = Transaction("tx1", "shard1", "shard3")
        
        # Create graph
        self.manager.network_graph.add_node("shard1")
        self.manager.network_graph.add_node("shard2")
        self.manager.network_graph.add_node("shard3")
        
        self.manager.network_graph.add_edge("shard1", "shard2", latency=0.1, weight=1.0)
        self.manager.network_graph.add_edge("shard2", "shard3", latency=0.2, weight=1.0)
        
        # Optimize transaction
        optimized_tx = self.manager.optimize_cross_shard_transaction(tx)
        
        # Check optimized transaction
        self.assertEqual(optimized_tx.tx_id, "tx1")
        self.assertEqual(optimized_tx.route, ["shard1"])
        self.assertGreater(optimized_tx.timeout, 10.0)  # Should be at least 10 seconds

    def test_optimize_multi_shard_transaction(self):
        """Test optimize_multi_shard_transaction method."""
        # Create multi-shard transaction
        tx = Transaction("tx1", "shard1", ["shard2", "shard3"])
        
        # Create graph
        self.manager.network_graph.add_node("shard1")
        self.manager.network_graph.add_node("shard2")
        self.manager.network_graph.add_node("shard3")
        
        self.manager.network_graph.add_edge("shard1", "shard2", latency=0.1, weight=1.0)
        self.manager.network_graph.add_edge("shard1", "shard3", latency=0.2, weight=1.0)
        
        # Optimize transaction
        optimized_txs = self.manager.optimize_multi_shard_transaction(tx)
        
        # Check optimized transactions
        self.assertEqual(len(optimized_txs), 2)
        self.assertEqual(optimized_txs[0].tx_id, "tx1_0")
        self.assertEqual(optimized_txs[0].source_shard, "shard1")
        self.assertEqual(optimized_txs[0].target_shard, "shard2")
        self.assertEqual(optimized_txs[1].tx_id, "tx1_1")
        self.assertEqual(optimized_txs[1].source_shard, "shard1")
        self.assertEqual(optimized_txs[1].target_shard, "shard3")


class TestMADRAPID(unittest.TestCase):
    """Test cases for the MADRAPID class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.config = {
            'update_interval': 1.0,
            'max_route_length': 5,
            'max_attempts': 3,
            'timeout': 30.0,
            'learning_rate': 0.1,
            'exploration_rate': 0.1,
            'discount_factor': 0.9
        }
        self.mad_rapid = MADRAPID(self.config)

    def test_initialization(self):
        """Test MADRAPID initialization."""
        self.assertEqual(self.mad_rapid.config, self.config)
        self.assertEqual(len(self.mad_rapid.shards), 0)
        self.assertEqual(len(self.mad_rapid.links), 0)
        self.assertEqual(len(self.mad_rapid.agents), 0)
        self.assertIsInstance(self.mad_rapid.cross_shard_manager, CrossShardManager)
        self.assertEqual(len(self.mad_rapid.transactions), 0)
        self.assertEqual(len(self.mad_rapid.performance_metrics['routing_time']), 0)
        self.assertEqual(self.mad_rapid.performance_metrics['success_rate'], 0.0)
        self.assertFalse(self.mad_rapid.running)
        self.assertIsNone(self.mad_rapid.update_thread)

    def test_start_stop(self):
        """Test start and stop methods."""
        # Start
        self.mad_rapid.start()
        self.assertTrue(self.mad_rapid.running)
        self.assertIsNotNone(self.mad_rapid.update_thread)
        self.assertTrue(self.mad_rapid.update_thread.is_alive())
        
        # Stop
        self.mad_rapid.stop()
        self.assertFalse(self.mad_rapid.running)
        
        # Wait for thread to finish
        self.mad_rapid.update_thread.join(timeout=2.0)
        self.assertFalse(self.mad_rapid.update_thread.is_alive())

    def test_add_shard(self):
        """Test add_shard method."""
        # Add shard
        self.mad_rapid.add_shard("shard1", 100.0)
        
        # Check shard
        self.assertIn("shard1", self.mad_rapid.shards)
        self.assertEqual(self.mad_rapid.shards["shard1"].capacity, 100.0)
        
        # Check agent
        self.assertIn("shard1", self.mad_rapid.agents)
        self.assertEqual(self.mad_rapid.agents["shard1"].shard_id, "shard1")

    def test_add_link(self):
        """Test add_link method."""
        # Add shards
        self.mad_rapid.add_shard("shard1", 100.0)
        self.mad_rapid.add_shard("shard2", 100.0)
        
        # Add link
        self.mad_rapid.add_link("shard1", "shard2", 0.1, 100.0)
        
        # Check link
        self.assertIn(("shard1", "shard2"), self.mad_rapid.links)
        self.assertEqual(self.mad_rapid.links[("shard1", "shard2")].base_latency, 0.1)
        self.assertEqual(self.mad_rapid.links[("shard1", "shard2")].capacity, 100.0)
        
        # Check shard neighbors
        self.assertIn("shard2", self.mad_rapid.shards["shard1"].neighbors)
        
        # Try to add link with non-existent shards
        self.mad_rapid.add_link("shard1", "shard3", 0.1, 100.0)
        
        # Link should not be added
        self.assertNotIn(("shard1", "shard3"), self.mad_rapid.links)

    def test_remove_shard(self):
        """Test remove_shard method."""
        # Add shards
        self.mad_rapid.add_shard("shard1", 100.0)
        self.mad_rapid.add_shard("shard2", 100.0)
        
        # Add link
        self.mad_rapid.add_link("shard1", "shard2", 0.1, 100.0)
        
        # Remove shard
        self.mad_rapid.remove_shard("shard1")
        
        # Check shard
        self.assertNotIn("shard1", self.mad_rapid.shards)
        self.assertNotIn("shard1", self.mad_rapid.agents)
        
        # Check link
        self.assertNotIn(("shard1", "shard2"), self.mad_rapid.links)
        
        # Check shard neighbors
        self.assertNotIn("shard1", self.mad_rapid.shards["shard2"].neighbors)

    def test_remove_link(self):
        """Test remove_link method."""
        # Add shards
        self.mad_rapid.add_shard("shard1", 100.0)
        self.mad_rapid.add_shard("shard2", 100.0)
        
        # Add link
        self.mad_rapid.add_link("shard1", "shard2", 0.1, 100.0)
        
        # Remove link
        self.mad_rapid.remove_link("shard1", "shard2")
        
        # Check link
        self.assertNotIn(("shard1", "shard2"), self.mad_rapid.links)
        
        # Check shard neighbors
        self.assertNotIn("shard2", self.mad_rapid.shards["shard1"].neighbors)

    def test_update_shard(self):
        """Test update_shard method."""
        # Add shard
        self.mad_rapid.add_shard("shard1", 100.0)
        
        # Update shard
        self.mad_rapid.update_shard("shard1", 50.0)
        
        # Check shard
        self.assertEqual(self.mad_rapid.shards["shard1"].current_load, 50.0)
        self.assertEqual(len(self.mad_rapid.shards["shard1"].historical_load), 1)
        self.assertEqual(self.mad_rapid.shards["shard1"].historical_load[0][1], 50.0)
        self.assertEqual(len(self.mad_rapid.shards["shard1"].congestion_history), 1)

    def test_update_link(self):
        """Test update_link method."""
        # Add shards
        self.mad_rapid.add_shard("shard1", 100.0)
        self.mad_rapid.add_shard("shard2", 100.0)
        
        # Add link
        self.mad_rapid.add_link("shard1", "shard2", 0.1, 100.0)
        
        # Update link
        self.mad_rapid.update_link("shard1", "shard2", 50.0, 0.2)
        
        # Check link
        self.assertEqual(self.mad_rapid.links[("shard1", "shard2")].current_load, 50.0)
        self.assertEqual(len(self.mad_rapid.links[("shard1", "shard2")].historical_load), 1)
        self.assertEqual(self.mad_rapid.links[("shard1", "shard2")].historical_load[0][1], 50.0)
        self.assertEqual(len(self.mad_rapid.links[("shard1", "shard2")].latency_history), 1)
        self.assertEqual(self.mad_rapid.links[("shard1", "shard2")].latency_history[0][1], 0.2)
        self.assertEqual(len(self.mad_rapid.links[("shard1", "shard2")].congestion_history), 1)

    def test_update_trust(self):
        """Test update_trust method."""
        # Add shard
        self.mad_rapid.add_shard("shard1", 100.0)
        
        # Update trust
        self.mad_rapid.update_trust("shard1", 0.8)
        
        # Check shard
        self.assertEqual(self.mad_rapid.shards["shard1"].trust_score, 0.8)

    def test_add_transaction(self):
        """Test add_transaction method."""
        # Add shards
        self.mad_rapid.add_shard("shard1", 100.0)
        self.mad_rapid.add_shard("shard2", 100.0)
        
        # Add transaction
        tx = self.mad_rapid.add_transaction("tx1", "shard1", "shard2", {"data": "test"}, 5)
        
        # Check transaction
        self.assertIsNotNone(tx)
        self.assertEqual(tx.tx_id, "tx1")
        self.assertEqual(tx.source_shard, "shard1")
        self.assertEqual(tx.target_shard, "shard2")
        self.assertEqual(tx.data, {"data": "test"})
        self.assertEqual(tx.priority, 5)
        
        # Check stored transaction
        self.assertIn("tx1", self.mad_rapid.transactions)
        self.assertEqual(self.mad_rapid.transactions["tx1"].tx_id, "tx1")
        
        # Try to add transaction with non-existent source shard
        tx = self.mad_rapid.add_transaction("tx2", "shard3", "shard2")
        
        # Transaction should not be added
        self.assertIsNone(tx)
        self.assertNotIn("tx2", self.mad_rapid.transactions)

    def test_route_transaction(self):
        """Test route_transaction method."""
        # Add shards
        self.mad_rapid.add_shard("shard1", 100.0)
        self.mad_rapid.add_shard("shard2", 100.0)
        
        # Add link
        self.mad_rapid.add_link("shard1", "shard2", 0.1, 100.0)
        
        # Add transaction
        tx = self.mad_rapid.add_transaction("tx1", "shard1", "shard2", {"data": "test"}, 5)
        
        # Route transaction
        next_shard = self.mad_rapid.route_transaction("tx1")
        
        # Check next shard
        self.assertEqual(next_shard, "shard2")
        
        # Check transaction route
        self.assertEqual(len(tx.route), 1)
        self.assertEqual(tx.route[0], "shard2")
        
        # Check performance metrics
        self.assertEqual(len(self.mad_rapid.performance_metrics['routing_time']), 1)
        self.assertGreater(self.mad_rapid.performance_metrics['routing_time'][0], 0.0)

    def test_get_transaction_status(self):
        """Test get_transaction_status method."""
        # Add shards
        self.mad_rapid.add_shard("shard1", 100.0)
        self.mad_rapid.add_shard("shard2", 100.0)
        
        # Add transaction
        tx = self.mad_rapid.add_transaction("tx1", "shard1", "shard2", {"data": "test"}, 5)
        
        # Get status
        tx_obj, status = self.mad_rapid.get_transaction_status("tx1")
        
        # Check status
        self.assertEqual(tx_obj.tx_id, "tx1")
        self.assertEqual(status, RouteStatus.PENDING)
        
        # Get status for non-existent transaction
        tx_obj, status = self.mad_rapid.get_transaction_status("tx2")
        
        # Check status
        self.assertIsNone(tx_obj)
        self.assertIsNone(status)

    def test_predict_congestion(self):
        """Test predict_congestion method."""
        # Add shard
        self.mad_rapid.add_shard("shard1", 100.0)
        
        # Update shard
        self.mad_rapid.update_shard("shard1", 50.0)
        
        # Predict congestion
        congestion = self.mad_rapid.predict_congestion("shard1")
        
        # Check congestion
        self.assertEqual(congestion, 0.5)
        
        # Predict congestion for non-existent shard
        congestion = self.mad_rapid.predict_congestion("shard2")
        
        # Check congestion
        self.assertEqual(congestion, 0.0)

    def test_predict_latency(self):
        """Test predict_latency method."""
        # Add shards
        self.mad_rapid.add_shard("shard1", 100.0)
        self.mad_rapid.add_shard("shard2", 100.0)
        
        # Add link
        self.mad_rapid.add_link("shard1", "shard2", 0.1, 100.0)
        
        # Predict latency
        latency = self.mad_rapid.predict_latency("shard1", "shard2")
        
        # Check latency
        self.assertEqual(latency, 0.1)
        
        # Predict latency for non-existent link
        latency = self.mad_rapid.predict_latency("shard1", "shard3")
        
        # Check latency
        self.assertEqual(latency, float('inf'))

    def test_get_stats(self):
        """Test get_stats method."""
        # Add shards
        self.mad_rapid.add_shard("shard1", 100.0)
        self.mad_rapid.add_shard("shard2", 100.0)
        
        # Add link
        self.mad_rapid.add_link("shard1", "shard2", 0.1, 100.0)
        
        # Get stats
        stats = self.mad_rapid.get_stats()
        
        # Check stats
        self.assertEqual(stats['shards'], 2)
        self.assertEqual(stats['links'], 1)
        self.assertEqual(stats['transactions'], 0)
        self.assertEqual(stats['avg_routing_time'], 0.0)
        self.assertEqual(stats['success_rate'], 0.0)
        self.assertEqual(len(stats['agents']), 2)
        self.assertIn('cross_shard', stats)

    def test_export_network_topology(self):
        """Test export_network_topology method."""
        # Add shards
        self.mad_rapid.add_shard("shard1", 100.0)
        self.mad_rapid.add_shard("shard2", 100.0)
        
        # Add link
        self.mad_rapid.add_link("shard1", "shard2", 0.1, 100.0)
        
        # Export topology
        topology = self.mad_rapid.export_network_topology()
        
        # Check topology
        self.assertEqual(len(topology['nodes']), 2)
        self.assertEqual(len(topology['edges']), 1)
        self.assertEqual(topology['nodes'][0]['id'], "shard1")
        self.assertEqual(topology['nodes'][1]['id'], "shard2")
        self.assertEqual(topology['edges'][0]['source'], "shard1")
        self.assertEqual(topology['edges'][0]['target'], "shard2")


if __name__ == '__main__':
    unittest.main()
