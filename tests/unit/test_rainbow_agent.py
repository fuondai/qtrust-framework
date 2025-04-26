#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unit tests for the Rainbow DQN agent.
Tests all seven components of the Rainbow DQN implementation.
"""

import os
import sys
import unittest
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent directory to path to import QTrust modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.agents.rainbow_agent import RainbowDQNAgent, RainbowDQN, NoisyLinear, PrioritizedReplayBuffer


class TestRainbowDQN(unittest.TestCase):
    """Test cases for the Rainbow DQN implementation."""

    def setUp(self):
        """Set up test environment before each test."""
        self.state_dim = 10
        self.action_dim = 4
        self.config = {
            'hidden_dim': 64,
            'atom_size': 51,
            'v_min': -10,
            'v_max': 10,
            'noisy_std': 0.5,
            'buffer_size': 1000,
            'batch_size': 32,
            'gamma': 0.99,
            'learning_rate': 0.0001,
            'target_update': 100,
            'n_step': 3,
            'alpha': 0.6,
            'beta': 0.4,
            'beta_increment': 0.001,
            'epsilon': 1e-6
        }
        self.agent = RainbowDQNAgent(self.state_dim, self.action_dim, self.config)

    def test_noisy_linear_layer(self):
        """Test the noisy linear layer implementation."""
        layer = NoisyLinear(10, 5, 0.5)
        
        # Test initialization
        self.assertEqual(layer.in_features, 10)
        self.assertEqual(layer.out_features, 5)
        self.assertEqual(layer.std_init, 0.5)
        
        # Test forward pass
        x = np.random.randn(2, 10)
        y = layer.forward(x)
        self.assertEqual(y.shape, (2, 5))
        
        # Test noise reset
        old_weight_epsilon = layer.weight_epsilon.copy()
        old_bias_epsilon = layer.bias_epsilon.copy()
        layer.reset_noise()
        self.assertFalse(np.allclose(old_weight_epsilon, layer.weight_epsilon))
        self.assertFalse(np.allclose(old_bias_epsilon, layer.bias_epsilon))

    def test_rainbow_dqn_network(self):
        """Test the Rainbow DQN network architecture."""
        network = RainbowDQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=64,
            atom_size=51,
            v_min=-10,
            v_max=10,
            noisy_std=0.5
        )
        
        # Test initialization
        self.assertEqual(network.atom_size, 51)
        self.assertEqual(network.action_dim, self.action_dim)
        
        # Test forward pass
        x = np.random.randn(2, self.state_dim)
        q_dist = network.forward(x)
        self.assertEqual(q_dist.shape, (2, self.action_dim, 51))
        
        # Test that output is a valid probability distribution
        self.assertTrue(np.allclose(q_dist.sum(axis=2), np.ones((2, self.action_dim))))
        
        # Test noise reset
        for layer in network.layers:
            if isinstance(layer, NoisyLinear):
                layer.weight_epsilon.fill(0)
                layer.bias_epsilon.fill(0)
        
        network.reset_noise()
        
        for layer in network.layers:
            if isinstance(layer, NoisyLinear):
                self.assertFalse(np.allclose(layer.weight_epsilon, np.zeros_like(layer.weight_epsilon)))
                self.assertFalse(np.allclose(layer.bias_epsilon, np.zeros_like(layer.bias_epsilon)))

    def test_prioritized_replay_buffer(self):
        """Test the prioritized experience replay buffer."""
        buffer = PrioritizedReplayBuffer(
            capacity=100,
            alpha=0.6,
            beta=0.4,
            beta_increment=0.001,
            epsilon=1e-6,
            n_step=3,
            gamma=0.99
        )
        
        # Test initialization
        self.assertEqual(buffer.capacity, 100)
        self.assertEqual(buffer.alpha, 0.6)
        self.assertEqual(buffer.beta, 0.4)
        self.assertEqual(buffer.n_step, 3)
        
        # Test adding transitions
        for i in range(5):
            state = np.random.rand(self.state_dim)
            next_state = np.random.rand(self.state_dim)
            action = np.random.randint(0, self.action_dim)
            reward = np.random.rand()
            done = False
            
            buffer.add(state, action, reward, next_state, done)
        
        # Test n-step buffer
        self.assertEqual(len(buffer.n_step_buffer), 3)  # 5 added, 2 processed
        
        # Add more transitions to fill n-step buffer
        for i in range(3):
            state = np.random.rand(self.state_dim)
            next_state = np.random.rand(self.state_dim)
            action = np.random.randint(0, self.action_dim)
            reward = np.random.rand()
            done = False
            
            buffer.add(state, action, reward, next_state, done)
        
        # Test buffer size (3 transitions processed with n_step=3)
        # We expect 6 entries in the buffer based on the implementation
        self.assertEqual(len(buffer), 6)
        
        # Test sampling
        batch_size = 4
        (states, actions, rewards, next_states, dones, weights, indices,
         n_step_rewards, n_step_next_states, n_step_dones) = buffer.sample(batch_size)
        
        self.assertEqual(states.shape, (batch_size, self.state_dim))
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(next_states.shape, (batch_size, self.state_dim))
        self.assertEqual(dones.shape, (batch_size,))
        self.assertEqual(weights.shape, (batch_size,))
        self.assertEqual(len(indices), batch_size)
        self.assertEqual(n_step_rewards.shape, (batch_size,))
        self.assertEqual(n_step_next_states.shape, (batch_size, self.state_dim))
        self.assertEqual(n_step_dones.shape, (batch_size,))
        
        # Test priority update
        new_priorities = np.random.rand(batch_size)
        buffer.update_priorities(indices, new_priorities)
        
        # Test beta increment
        old_beta = buffer.beta
        for _ in range(10):
            buffer.sample(batch_size)
        self.assertGreater(buffer.beta, old_beta)

    def test_agent_initialization(self):
        """Test the agent initialization."""
        # Test network initialization
        self.assertIsInstance(self.agent.online_net, RainbowDQN)
        self.assertIsInstance(self.agent.target_net, RainbowDQN)
        
        # Test optimizer initialization
        self.assertEqual(self.agent.learning_rate, self.config['learning_rate'])
        
        # Test memory initialization
        self.assertIsInstance(self.agent.memory, PrioritizedReplayBuffer)
        self.assertEqual(self.agent.memory.capacity, self.config['buffer_size'])
        
        # Test other parameters
        self.assertEqual(self.agent.gamma, self.config['gamma'])
        self.assertEqual(self.agent.n_step, self.config['n_step'])
        self.assertEqual(self.agent.batch_size, self.config['batch_size'])
        self.assertEqual(self.agent.target_update, self.config['target_update'])

    def test_select_action(self):
        """Test action selection."""
        state = np.random.rand(self.state_dim)
        
        # Test action selection in train mode
        action = self.agent.select_action(state)
        self.assertTrue(0 <= action < self.action_dim)
        
        # Test action selection in eval mode
        action = self.agent.select_action(state, evaluate=True)
        self.assertTrue(0 <= action < self.action_dim)

    def test_multi_objective_reward(self):
        """Test the multi-objective reward function."""
        # Create mock state objects
        state = MagicMock()
        state.throughput = 100
        state.latency = 50
        state.byzantine_nodes = 5
        state.total_nodes = 100
        state.resource_usage = 0.5
        state.congestion_level = 0.3
        state.byzantine_ratio = 0.05
        
        next_state = MagicMock()
        next_state.throughput = 120
        next_state.latency = 40
        next_state.byzantine_nodes = 4
        next_state.total_nodes = 100
        next_state.resource_usage = 0.6
        next_state.throughput = 120
        
        # Test reward calculation
        reward = self.agent.calculate_reward(state, 0, next_state)
        self.assertIsInstance(reward, float)
        
        # Test with different network conditions
        state.congestion_level = 0.9  # High congestion
        reward_high_congestion = self.agent.calculate_reward(state, 0, next_state)
        
        state.congestion_level = 0.3  # Normal congestion
        state.byzantine_ratio = 0.3  # High Byzantine ratio
        reward_high_byzantine = self.agent.calculate_reward(state, 0, next_state)
        
        state.byzantine_ratio = 0.05  # Normal Byzantine ratio
        state.resource_usage = 0.95  # High resource usage
        reward_high_resource = self.agent.calculate_reward(state, 0, next_state)
        
        # Verify that weights are adjusted based on network conditions
        self.assertNotEqual(reward_high_congestion, reward_high_byzantine)
        self.assertNotEqual(reward_high_congestion, reward_high_resource)
        self.assertNotEqual(reward_high_byzantine, reward_high_resource)

    def test_store_transition(self):
        """Test storing transitions in the replay buffer."""
        # Mock the add method
        self.agent.memory.add = MagicMock()
        
        # Store a transition
        state = np.random.rand(self.state_dim)
        next_state = np.random.rand(self.state_dim)
        action = np.random.randint(0, self.action_dim)
        reward = np.random.rand()
        done = False
        
        self.agent.store_transition(state, action, reward, next_state, done)
        
        # Verify that add was called with the correct arguments
        self.agent.memory.add.assert_called_once_with(state, action, reward, next_state, done)

    def test_update(self):
        """Test the agent update method."""
        # Mock the sample method to return a batch of transitions
        batch_size = self.config['batch_size']
        
        # Add enough samples to memory
        states = np.random.rand(batch_size, self.state_dim)
        actions = np.random.randint(0, self.action_dim, size=batch_size)
        rewards = np.random.rand(batch_size)
        next_states = np.random.rand(batch_size, self.state_dim)
        dones = np.zeros(batch_size)
        weights = np.ones(batch_size)
        indices = np.arange(batch_size)
        n_step_rewards = np.random.rand(batch_size)
        n_step_next_states = np.random.rand(batch_size, self.state_dim)
        n_step_dones = np.zeros(batch_size)
        
        # Setup mocks
        self.agent.memory = MagicMock()
        self.agent.memory.__len__.return_value = batch_size
        self.agent.memory.sample.return_value = (
            states, actions, rewards, next_states, dones, weights, indices,
            n_step_rewards, n_step_next_states, n_step_dones
        )
        
        # Also mock other methods used in update to ensure they don't affect the test
        self.agent.online_net = MagicMock()
        self.agent.target_net = MagicMock()
        
        # Mock Q-values
        q_values = np.random.rand(batch_size, self.action_dim)
        self.agent.online_net.get_q_values.return_value = q_values
        self.agent.target_net.get_q_values.return_value = np.random.rand(batch_size, self.action_dim)
        
        # Set up a simple update function to replace the complex one
        def mock_update_online_network(loss):
            pass
        
        self.agent._update_online_network = mock_update_online_network
        
        # Call update method
        loss = self.agent.update()
        
        # Verify sample was called
        self.agent.memory.sample.assert_called_once_with(batch_size)
        
        # Verify loss is a float
        self.assertIsInstance(loss, float)

    def test_save_load(self):
        """Test saving and loading the agent."""
        # Create a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            
            # Save the agent
            self.agent.save(temp_path)
            
            # Create a new agent
            new_agent = RainbowDQNAgent(self.state_dim, self.action_dim, self.config)
            
            # Load the saved agent
            new_agent.load(temp_path)
            
            # Verify that the parameters are the same
            # Check feature layers
            self.assertTrue(np.allclose(
                self.agent.online_net.feature_layer1['weights'],
                new_agent.online_net.feature_layer1['weights']
            ))
            self.assertTrue(np.allclose(
                self.agent.online_net.feature_layer1['bias'],
                new_agent.online_net.feature_layer1['bias']
            ))
            
            # Check value layers
            self.assertTrue(np.allclose(
                self.agent.online_net.value_layer1.weight_mu,
                new_agent.online_net.value_layer1.weight_mu
            ))
            
            # Check advantage layers
            self.assertTrue(np.allclose(
                self.agent.online_net.advantage_layer1.weight_mu,
                new_agent.online_net.advantage_layer1.weight_mu
            ))
            
            # Clean up
            import os
            if os.path.exists(f"{temp_path}.npy"):
                os.remove(f"{temp_path}.npy")

    def test_trust_provider_integration(self):
        """Test integration with trust provider."""
        # Create a mock trust provider
        trust_provider = MagicMock()
        trust_provider.get_security_level.return_value = 0.8
        
        # Register the trust provider
        self.agent.register_trust_provider(trust_provider)
        
        # Verify that the trust provider was registered
        self.assertEqual(self.agent.trust_provider, trust_provider)
        
        # Create mock state
        state = MagicMock()
        
        # Test security level calculation
        security_level = self.agent._calculate_security_level(state)
        
        # Verify that the trust provider was used
        trust_provider.get_security_level.assert_called_once_with(state)
        self.assertEqual(security_level, 0.8)


if __name__ == '__main__':
    unittest.main()
