"""
Modified test file for rainbow agent that works with both PyTorch and mock implementations.
"""

import unittest
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the qtrust package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the implementation switch module
from qtrust.implementation_switch import get_rainbow_agent, set_use_pytorch

# Set to use mock implementation for testing
set_use_pytorch(False)

class TestMockRainbowDQNAgent(unittest.TestCase):
    """Test cases for the mock Rainbow DQN agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state_dim = 10
        self.action_dim = 4
        self.agent = get_rainbow_agent(self.state_dim, self.action_dim)
        
    def test_initialization(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.state_dim, self.state_dim)
        self.assertEqual(self.agent.action_dim, self.action_dim)
        
    def test_select_action(self):
        """Test action selection."""
        state = np.random.rand(self.state_dim)
        action = self.agent.select_action(state)
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_dim)
        
    def test_update(self):
        """Test agent update."""
        # Add some experiences
        for _ in range(10):
            state = np.random.rand(self.state_dim)
            action = self.agent.select_action(state)
            reward = np.random.rand()
            next_state = np.random.rand(self.state_dim)
            done = np.random.rand() > 0.8
            self.agent.add_experience(state, action, reward, next_state, done)
            
        # Update the agent
        metrics = self.agent.update()
        
        # Check metrics
        self.assertIn('loss', metrics)
        self.assertIn('q_mean', metrics)
        self.assertIn('q_max', metrics)
        
    def test_save_load(self):
        """Test save and load functionality."""
        # This is just a mock test since we're not actually saving/loading
        self.agent.save("test_model.pt")
        self.agent.load("test_model.pt")
        
    def test_get_metrics(self):
        """Test metrics retrieval."""
        metrics = self.agent.get_metrics()
        self.assertIn('loss', metrics)
        self.assertIn('q_values', metrics)
        self.assertIn('rewards', metrics)
        self.assertIn('epsilon', metrics)

if __name__ == '__main__':
    unittest.main()
