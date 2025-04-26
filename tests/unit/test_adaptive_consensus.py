#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the Adaptive Consensus Selector.
Tests the standardized consensus protocol interfaces, Bayesian decision tree,
and protocol transition mechanism.
"""

import os
import sys
import unittest
import time
import threading
from unittest.mock import MagicMock, patch
import numpy as np

# Add parent directory to path to import QTrust modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.consensus.adaptive_consensus import (
    ConsensusType, ConsensusProtocol, PBFTConsensus, HotStuffConsensus,
    TendermintConsensus, RaftConsensus, PoAConsensus, ConsensusFactory,
    BayesianDecisionTree, AdaptiveConsensusSelector
)


class TestConsensusProtocol(unittest.TestCase):
    """Test cases for the ConsensusProtocol base class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.config = {'test_key': 'test_value'}
        self.protocol = ConsensusProtocol(self.config)

    def test_initialization(self):
        """Test protocol initialization."""
        self.assertEqual(self.protocol.config, self.config)
        self.assertFalse(self.protocol.running)
        self.assertEqual(self.protocol.metrics['throughput'], 0)
        self.assertEqual(self.protocol.metrics['latency'], 0)
        self.assertEqual(self.protocol.metrics['byzantine_tolerance'], 0)
        self.assertEqual(self.protocol.metrics['resource_usage'], 0)
        self.assertEqual(self.protocol.metrics['transaction_count'], 0)
        self.assertEqual(self.protocol.metrics['success_rate'], 0)

    def test_start_stop(self):
        """Test start and stop methods."""
        # Start protocol
        self.protocol.start()
        self.assertTrue(self.protocol.running)
        self.assertGreater(self.protocol.start_time, 0)
        
        # Stop protocol
        self.protocol.stop()
        self.assertFalse(self.protocol.running)

    def test_get_metrics(self):
        """Test get_metrics method."""
        # Set some metrics
        self.protocol.metrics['throughput'] = 100
        self.protocol.metrics['latency'] = 50
        
        # Get metrics
        metrics = self.protocol.get_metrics()
        
        # Check metrics
        self.assertEqual(metrics['throughput'], 100)
        self.assertEqual(metrics['latency'], 50)
        
        # Check that we got a copy
        metrics['throughput'] = 200
        self.assertEqual(self.protocol.metrics['throughput'], 100)

    def test_update_metrics(self):
        """Test update_metrics method."""
        # Update metrics
        new_metrics = {
            'throughput': 100,
            'latency': 50,
            'success_rate': 0.9
        }
        self.protocol.update_metrics(new_metrics)
        
        # Check metrics
        self.assertEqual(self.protocol.metrics['throughput'], 100)
        self.assertEqual(self.protocol.metrics['latency'], 50)
        self.assertEqual(self.protocol.metrics['success_rate'], 0.9)
        
        # Check that other metrics were not changed
        self.assertEqual(self.protocol.metrics['byzantine_tolerance'], 0)

    def test_get_state(self):
        """Test get_state method."""
        # Start protocol
        self.protocol.start()
        
        # Wait a moment
        time.sleep(0.01)
        
        # Get state
        state = self.protocol.get_state()
        
        # Check state
        self.assertTrue(state['running'])
        self.assertGreater(state['uptime'], 0)
        self.assertEqual(state['metrics'], self.protocol.get_metrics())
        
        # Stop protocol
        self.protocol.stop()

    def test_prepare_transition(self):
        """Test prepare_transition method."""
        # Create target protocol
        target_protocol = ConsensusProtocol()
        
        # Prepare transition
        transition_state = self.protocol.prepare_transition(target_protocol)
        
        # Check transition state
        self.assertEqual(transition_state['source_protocol'], self.protocol.__class__.__name__)
        self.assertEqual(transition_state['target_protocol'], target_protocol.__class__.__name__)
        self.assertEqual(transition_state['pending_transactions'], [])
        self.assertEqual(transition_state['state'], self.protocol.get_state())

    def test_apply_transition(self):
        """Test apply_transition method."""
        # Create transition state
        transition_state = {
            'source_protocol': 'TestProtocol',
            'target_protocol': self.protocol.__class__.__name__,
            'pending_transactions': [],
            'state': {
                'running': True,
                'uptime': 10,
                'metrics': {
                    'throughput': 100,
                    'latency': 50
                }
            }
        }
        
        # Apply transition
        self.protocol.apply_transition(transition_state)
        
        # Nothing to check since the base implementation doesn't do much


class TestConsensusImplementations(unittest.TestCase):
    """Test cases for the concrete consensus protocol implementations."""

    def test_pbft_consensus(self):
        """Test PBFT consensus protocol."""
        # Create PBFT consensus
        config = {
            'validators': ['node1', 'node2', 'node3', 'node4'],
            'f': 1,
            'primary': 0
        }
        pbft = PBFTConsensus(config)
        
        # Check initialization
        self.assertEqual(pbft.validators, config['validators'])
        self.assertEqual(pbft.f, config['f'])
        self.assertEqual(pbft.primary, config['primary'])
        self.assertEqual(pbft.metrics['byzantine_tolerance'], 0.25)  # 1/4
        
        # Start protocol
        pbft.start()
        
        # Process transaction
        result = pbft.process_transaction({'data': 'test'})
        
        # Check result
        self.assertTrue(result['success'])
        self.assertEqual(result['sequence_number'], 1)
        self.assertEqual(result['view_number'], 0)
        self.assertIn('latency', result)
        
        # Check metrics
        self.assertEqual(pbft.metrics['transaction_count'], 1)
        self.assertGreater(pbft.metrics['latency'], 0)
        self.assertEqual(pbft.metrics['success_rate'], 1.0)
        
        # Stop protocol
        pbft.stop()

    def test_hotstuff_consensus(self):
        """Test HotStuff consensus protocol."""
        # Create HotStuff consensus
        config = {
            'validators': ['node1', 'node2', 'node3', 'node4'],
            'f': 1,
            'leader': 0
        }
        hotstuff = HotStuffConsensus(config)
        
        # Check initialization
        self.assertEqual(hotstuff.validators, config['validators'])
        self.assertEqual(hotstuff.f, config['f'])
        self.assertEqual(hotstuff.leader, config['leader'])
        self.assertEqual(hotstuff.metrics['byzantine_tolerance'], 0.25)  # 1/4
        
        # Start protocol
        hotstuff.start()
        
        # Process transaction
        result = hotstuff.process_transaction({'data': 'test'})
        
        # Check result
        self.assertTrue(result['success'])
        self.assertEqual(result['sequence_number'], 1)
        self.assertEqual(result['view_number'], 0)
        self.assertIn('latency', result)
        
        # Check metrics
        self.assertEqual(hotstuff.metrics['transaction_count'], 1)
        self.assertGreater(hotstuff.metrics['latency'], 0)
        self.assertEqual(hotstuff.metrics['success_rate'], 1.0)
        
        # Stop protocol
        hotstuff.stop()

    def test_tendermint_consensus(self):
        """Test Tendermint consensus protocol."""
        # Create Tendermint consensus
        config = {
            'validators': ['node1', 'node2', 'node3', 'node4'],
            'f': 1,
            'proposer': 0
        }
        tendermint = TendermintConsensus(config)
        
        # Check initialization
        self.assertEqual(tendermint.validators, config['validators'])
        self.assertEqual(tendermint.f, config['f'])
        self.assertEqual(tendermint.proposer, config['proposer'])
        self.assertEqual(tendermint.metrics['byzantine_tolerance'], 0.25)  # 1/4
        
        # Start protocol
        tendermint.start()
        
        # Process transaction
        result = tendermint.process_transaction({'data': 'test'})
        
        # Check result
        self.assertTrue(result['success'])
        self.assertEqual(result['height'], 1)
        self.assertEqual(result['round'], 0)
        self.assertIn('latency', result)
        
        # Check metrics
        self.assertEqual(tendermint.metrics['transaction_count'], 1)
        self.assertGreater(tendermint.metrics['latency'], 0)
        self.assertEqual(tendermint.metrics['success_rate'], 1.0)
        
        # Stop protocol
        tendermint.stop()

    def test_raft_consensus(self):
        """Test Raft consensus protocol."""
        # Create Raft consensus
        config = {
            'nodes': ['node1', 'node2', 'node3', 'node4', 'node5'],
            'leader': 0
        }
        raft = RaftConsensus(config)
        
        # Check initialization
        self.assertEqual(raft.nodes, config['nodes'])
        self.assertEqual(raft.leader, config['leader'])
        self.assertEqual(raft.metrics['byzantine_tolerance'], 0)  # Raft is not Byzantine fault-tolerant
        
        # Start protocol
        raft.start()
        
        # Process transaction
        result = raft.process_transaction({'data': 'test'})
        
        # Check result
        self.assertTrue(result['success'])
        self.assertEqual(result['term'], 0)
        self.assertEqual(result['log_index'], 0)
        self.assertEqual(result['commit_index'], 0)
        self.assertIn('latency', result)
        
        # Check metrics
        self.assertEqual(raft.metrics['transaction_count'], 1)
        self.assertGreater(raft.metrics['latency'], 0)
        self.assertEqual(raft.metrics['success_rate'], 1.0)
        
        # Stop protocol
        raft.stop()

    def test_poa_consensus(self):
        """Test PoA consensus protocol."""
        # Create PoA consensus
        config = {
            'authorities': ['auth1', 'auth2', 'auth3'],
            'current_authority': 0
        }
        poa = PoAConsensus(config)
        
        # Check initialization
        self.assertEqual(poa.authorities, config['authorities'])
        self.assertEqual(poa.current_authority, config['current_authority'])
        self.assertEqual(poa.metrics['byzantine_tolerance'], 0.5)  # PoA can tolerate up to 50% malicious authorities
        
        # Start protocol
        poa.start()
        
        # Process transaction
        result = poa.process_transaction({'data': 'test'})
        
        # Check result
        self.assertTrue(result['success'])
        self.assertEqual(result['authority'], 'auth1')
        self.assertEqual(result['block_number'], 1)
        self.assertIn('latency', result)
        
        # Check metrics
        self.assertEqual(poa.metrics['transaction_count'], 1)
        self.assertGreater(poa.metrics['latency'], 0)
        self.assertEqual(poa.metrics['success_rate'], 1.0)
        
        # Check authority rotation
        self.assertEqual(poa.current_authority, 1)
        
        # Stop protocol
        poa.stop()


class TestConsensusFactory(unittest.TestCase):
    """Test cases for the ConsensusFactory."""

    def test_create_consensus(self):
        """Test create_consensus method."""
        # Create PBFT consensus
        pbft = ConsensusFactory.create_consensus(ConsensusType.PBFT)
        self.assertIsInstance(pbft, PBFTConsensus)
        
        # Create HotStuff consensus
        hotstuff = ConsensusFactory.create_consensus(ConsensusType.HOTSTUFF)
        self.assertIsInstance(hotstuff, HotStuffConsensus)
        
        # Create Tendermint consensus
        tendermint = ConsensusFactory.create_consensus(ConsensusType.TENDERMINT)
        self.assertIsInstance(tendermint, TendermintConsensus)
        
        # Create Raft consensus
        raft = ConsensusFactory.create_consensus(ConsensusType.RAFT)
        self.assertIsInstance(raft, RaftConsensus)
        
        # Create PoA consensus
        poa = ConsensusFactory.create_consensus(ConsensusType.POA)
        self.assertIsInstance(poa, PoAConsensus)
        
        # Test with config
        config = {'test_key': 'test_value'}
        pbft_with_config = ConsensusFactory.create_consensus(ConsensusType.PBFT, config)
        self.assertEqual(pbft_with_config.config, config)
        
        # Test invalid consensus type
        with self.assertRaises(ValueError):
            ConsensusFactory.create_consensus("invalid_type")


class TestBayesianDecisionTree(unittest.TestCase):
    """Test cases for the BayesianDecisionTree."""

    def setUp(self):
        """Set up test environment before each test."""
        self.decision_tree = BayesianDecisionTree()

    def test_initialization(self):
        """Test decision tree initialization."""
        # Check model structure
        self.assertEqual(len(self.decision_tree.model.nodes()), 5)
        self.assertEqual(len(self.decision_tree.model.edges()), 4)
        
        # Check CPDs
        self.assertEqual(self.decision_tree.network_cpd.variable, 'network_condition')
        self.assertEqual(self.decision_tree.security_cpd.variable, 'security_risk')
        self.assertEqual(self.decision_tree.transaction_cpd.variable, 'transaction_complexity')
        self.assertEqual(self.decision_tree.shard_cpd.variable, 'shard_size')
        self.assertEqual(self.decision_tree.consensus_cpd.variable, 'optimal_consensus')
        
        # Check learning data
        self.assertEqual(len(self.decision_tree.learning_data), 0)

    def test_predict(self):
        """Test predict method."""
        # Predict with default model
        consensus_type = self.decision_tree.predict(1, 1, 1, 1)
        
        # Check that we got a valid consensus type
        self.assertIsInstance(consensus_type, ConsensusType)
        
        # Predict with different conditions
        consensus_type_low_risk = self.decision_tree.predict(1, 0, 1, 1)
        consensus_type_high_risk = self.decision_tree.predict(1, 2, 1, 1)
        
        # These might be the same with the default model, but we're just testing the method

    def test_update(self):
        """Test update method."""
        # Update with some data
        self.decision_tree.update(1, 1, 1, 1, ConsensusType.PBFT, 0.9)
        
        # Check learning data
        self.assertEqual(len(self.decision_tree.learning_data), 1)
        self.assertEqual(self.decision_tree.learning_data[0]['network_condition'], 1)
        self.assertEqual(self.decision_tree.learning_data[0]['security_risk'], 1)
        self.assertEqual(self.decision_tree.learning_data[0]['transaction_complexity'], 1)
        self.assertEqual(self.decision_tree.learning_data[0]['shard_size'], 1)
        self.assertEqual(self.decision_tree.learning_data[0]['consensus_type'], ConsensusType.PBFT)
        self.assertEqual(self.decision_tree.learning_data[0]['performance'], 0.9)
        
        # Add more data to trigger CPD update
        for i in range(10):
            self.decision_tree.update(1, 1, 1, 1, ConsensusType.PBFT, 0.9)
        
        # Check learning data
        self.assertEqual(len(self.decision_tree.learning_data), 11)


class TestAdaptiveConsensusSelector(unittest.TestCase):
    """Test cases for the AdaptiveConsensusSelector."""

    def setUp(self):
        """Set up test environment before each test."""
        self.config = {
            'default_consensus': 'pbft',
            'transition_blocks': 10,
            'metrics_window': 100,
            'learning_rate': 0.1,
            'update_interval': 1,  # 1 second for faster testing
        }
        self.selector = AdaptiveConsensusSelector(self.config)

    def test_initialization(self):
        """Test selector initialization."""
        # Check config
        self.assertEqual(self.selector.config['default_consensus'], 'pbft')
        self.assertEqual(self.selector.config['transition_blocks'], 10)
        self.assertEqual(self.selector.config['metrics_window'], 100)
        self.assertEqual(self.selector.config['learning_rate'], 0.1)
        self.assertEqual(self.selector.config['update_interval'], 1)
        
        # Check decision tree
        self.assertIsInstance(self.selector.decision_tree, BayesianDecisionTree)
        
        # Check consensus protocols
        self.assertEqual(len(self.selector.consensus_protocols), 5)
        for consensus_type in ConsensusType:
            self.assertIn(consensus_type, self.selector.consensus_protocols)
            self.assertIsInstance(self.selector.consensus_protocols[consensus_type], ConsensusProtocol)
        
        # Check metrics
        self.assertEqual(len(self.selector.metrics), 5)
        for consensus_type in ConsensusType:
            self.assertIn(consensus_type, self.selector.metrics)
            self.assertEqual(len(self.selector.metrics[consensus_type]), 4)
            self.assertIn('throughput', self.selector.metrics[consensus_type])
            self.assertIn('latency', self.selector.metrics[consensus_type])
            self.assertIn('success_rate', self.selector.metrics[consensus_type])
            self.assertIn('resource_usage', self.selector.metrics[consensus_type])
        
        # Check providers
        self.assertIsNone(self.selector.network_state_provider)
        self.assertIsNone(self.selector.trust_provider)
        
        # Check running flag
        self.assertFalse(self.selector.running)
        
        # Check update thread
        self.assertIsNone(self.selector.update_thread)
        
        # Check current conditions
        self.assertEqual(self.selector.current_conditions['network_condition'], 1)
        self.assertEqual(self.selector.current_conditions['security_risk'], 1)
        self.assertEqual(self.selector.current_conditions['transaction_complexity'], 1)
        self.assertEqual(self.selector.current_conditions['shard_size'], 1)
        
        # Check transition state
        self.assertIsNone(self.selector.transition_state)
        self.assertFalse(self.selector.transition_in_progress)

    def test_start_stop(self):
        """Test start and stop methods."""
        # Start selector
        self.selector.start()
        self.assertTrue(self.selector.running)
        self.assertIsNotNone(self.selector.update_thread)
        self.assertTrue(self.selector.update_thread.is_alive())
        
        # Stop selector
        self.selector.stop()
        self.assertFalse(self.selector.running)
        
        # Wait for thread to finish
        self.selector.update_thread.join(timeout=2.0)
        self.assertFalse(self.selector.update_thread.is_alive())

    def test_register_providers(self):
        """Test provider registration."""
        # Create mock providers
        network_provider = MagicMock()
        trust_provider = MagicMock()
        
        # Register providers
        self.selector.register_network_state_provider(network_provider)
        self.selector.register_trust_provider(trust_provider)
        
        # Check providers
        self.assertEqual(self.selector.network_state_provider, network_provider)
        self.assertEqual(self.selector.trust_provider, trust_provider)

    def test_update_metrics(self):
        """Test update_metrics method."""
        # Update metrics
        metrics = {
            'throughput': 100,
            'latency': 50,
            'success_rate': 0.9,
            'resource_usage': 0.5
        }
        self.selector.update_metrics(ConsensusType.PBFT, metrics)
        
        # Check metrics
        for metric_name, value in metrics.items():
            self.assertEqual(len(self.selector.metrics[ConsensusType.PBFT][metric_name]), 1)
            self.assertEqual(self.selector.metrics[ConsensusType.PBFT][metric_name][0], value)
        
        # Update again
        self.selector.update_metrics(ConsensusType.PBFT, metrics)
        
        # Check metrics
        for metric_name, value in metrics.items():
            self.assertEqual(len(self.selector.metrics[ConsensusType.PBFT][metric_name]), 2)
            self.assertEqual(self.selector.metrics[ConsensusType.PBFT][metric_name][1], value)
        
        # Update with different metrics
        new_metrics = {
            'throughput': 200,
            'latency': 25,
            'success_rate': 0.95,
            'resource_usage': 0.6
        }
        self.selector.update_metrics(ConsensusType.PBFT, new_metrics)
        
        # Check metrics
        for metric_name, value in new_metrics.items():
            self.assertEqual(len(self.selector.metrics[ConsensusType.PBFT][metric_name]), 3)
            self.assertEqual(self.selector.metrics[ConsensusType.PBFT][metric_name][2], value)

    def test_get_consensus(self):
        """Test get_consensus method."""
        # Mock decision tree predict method
        self.selector.decision_tree.predict = MagicMock(return_value=ConsensusType.PBFT)
        
        # Get consensus
        consensus = self.selector.get_consensus(1, 1, 1, 1)
        
        # Check consensus
        self.assertIsInstance(consensus, PBFTConsensus)
        
        # Check that predict was called
        self.selector.decision_tree.predict.assert_called_once_with(1, 1, 1, 1)
        
        # Mock different return value
        self.selector.decision_tree.predict = MagicMock(return_value=ConsensusType.HOTSTUFF)
        
        # Get consensus
        consensus = self.selector.get_consensus(1, 1, 1, 1)
        
        # Check consensus
        self.assertIsInstance(consensus, HotStuffConsensus)

    def test_transition_protocol(self):
        """Test transition_protocol method."""
        # Start protocols
        for protocol in self.selector.consensus_protocols.values():
            protocol.start()
        
        # Transition from PBFT to HotStuff
        success = self.selector.transition_protocol(ConsensusType.PBFT, ConsensusType.HOTSTUFF)
        
        # Check success
        self.assertTrue(success)
        
        # Check transition state
        self.assertIsNotNone(self.selector.transition_state)
        self.assertEqual(self.selector.transition_state['source_protocol'], 'PBFTConsensus')
        self.assertEqual(self.selector.transition_state['target_protocol'], 'HotStuffConsensus')
        
        # Check that source protocol was stopped
        self.assertFalse(self.selector.consensus_protocols[ConsensusType.PBFT].running)
        
        # Check that target protocol is running
        self.assertTrue(self.selector.consensus_protocols[ConsensusType.HOTSTUFF].running)
        
        # Stop protocols
        for protocol in self.selector.consensus_protocols.values():
            protocol.stop()

    def test_get_status(self):
        """Test get_status method."""
        # Update metrics
        metrics = {
            'throughput': 100,
            'latency': 50,
            'success_rate': 0.9,
            'resource_usage': 0.5
        }
        self.selector.update_metrics(ConsensusType.PBFT, metrics)
        
        # Get status
        status = self.selector.get_status()
        
        # Check status
        self.assertFalse(status['running'])
        self.assertEqual(status['current_conditions'], self.selector.current_conditions)
        self.assertFalse(status['transition_in_progress'])
        
        # Check metrics
        self.assertEqual(status['metrics']['pbft']['throughput'], 100)
        self.assertEqual(status['metrics']['pbft']['latency'], 50)
        self.assertEqual(status['metrics']['pbft']['success_rate'], 0.9)
        self.assertEqual(status['metrics']['pbft']['resource_usage'], 0.5)
        
        # Start selector
        self.selector.start()
        
        # Get status
        status = self.selector.get_status()
        
        # Check status
        self.assertTrue(status['running'])
        
        # Stop selector
        self.selector.stop()


if __name__ == '__main__':
    unittest.main()
