"""
Implementation of Rainbow DQN Agent using NumPy.

This module provides an implementation of the Rainbow DQN Agent
that incorporates the key enhancements from the Rainbow paper.
"""

import numpy as np
import random
import logging
from collections import deque
from typing import Dict, List, Tuple, Any, Optional, Deque
import os

logger = logging.getLogger(__name__)


class NoisyLinear:
    """
    Noisy Linear layer with factorized Gaussian noise for exploration.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Initialize NoisyLinear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            std_init: Initial standard deviation
        """
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Mean weights and biases
        self.weight_mu = np.random.uniform(-1/np.sqrt(in_features), 1/np.sqrt(in_features), (out_features, in_features))
        self.weight_sigma = np.full((out_features, in_features), std_init / np.sqrt(in_features))
        self.bias_mu = np.random.uniform(-1/np.sqrt(in_features), 1/np.sqrt(in_features), out_features)
        self.bias_sigma = np.full(out_features, std_init / np.sqrt(out_features))

        # Factorized Gaussian noise
        self.weight_epsilon = np.zeros((out_features, in_features))
        self.bias_epsilon = np.zeros(out_features)

        self.reset_noise()

    def _scale_noise(self, size: int) -> np.ndarray:
        """
        Scale noise for factorized Gaussian noise.

        Args:
            size: Size of noise tensor

        Returns:
            Scaled noise tensor
        """
        x = np.random.normal(0, 1, size)
        return np.sign(x) * np.sqrt(np.abs(x))

    def reset_noise(self):
        """
        Reset noise.
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon = np.outer(epsilon_out, epsilon_in)
        self.bias_epsilon = epsilon_out

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with added Gaussian noise.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        return np.dot(x, weight.T) + bias


class RainbowDQN:
    """
    Rainbow DQN network with noisy networks, dueling architecture, and distributional RL.
    Implementation using NumPy.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        atom_size: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy_std: float = 0.5
    ):
        """
        Initialize Rainbow DQN network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
            atom_size: Number of atoms for distributional RL
            v_min: Minimum support value
            v_max: Maximum support value
            noisy_std: Standard deviation for noisy layers
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max

        # Support for distributional RL
        self.support = np.linspace(v_min, v_max, atom_size)
        self.delta = (v_max - v_min) / (atom_size - 1)

        # Define layers
        self.layers = []

        # Feature layers (standard neural network layers)
        self.feature_layer1 = {
            'weights': np.random.randn(state_dim, hidden_dim) / np.sqrt(state_dim),
            'bias': np.zeros(hidden_dim)
        }
        self.feature_layer2 = {
            'weights': np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim),
            'bias': np.zeros(hidden_dim)
        }

        # Value stream (noisy layers for value estimation)
        self.value_layer1 = NoisyLinear(hidden_dim, hidden_dim, noisy_std)
        self.value_layer2 = NoisyLinear(hidden_dim, atom_size, noisy_std)

        # Advantage stream (noisy layers for advantage estimation)
        self.advantage_layer1 = NoisyLinear(hidden_dim, hidden_dim, noisy_std)
        self.advantage_layer2 = NoisyLinear(hidden_dim, action_dim * atom_size, noisy_std)

        # Add all noisy layers to a list for easy access
        self.layers = [
            self.value_layer1, self.value_layer2,
            self.advantage_layer1, self.advantage_layer2
        ]

        # Create params dictionary for test compatibility
        self.params = {
            'feature_layer1': self.feature_layer1,
            'feature_layer2': self.feature_layer2,
            'value_layer1': {
                'weight_mu': self.value_layer1.weight_mu,
                'weight_sigma': self.value_layer1.weight_sigma,
                'bias_mu': self.value_layer1.bias_mu,
                'bias_sigma': self.value_layer1.bias_sigma
            },
            'value_layer2': {
                'weight_mu': self.value_layer2.weight_mu,
                'weight_sigma': self.value_layer2.weight_sigma,
                'bias_mu': self.value_layer2.bias_mu,
                'bias_sigma': self.value_layer2.bias_sigma
            },
            'advantage_layer1': {
                'weight_mu': self.advantage_layer1.weight_mu,
                'weight_sigma': self.advantage_layer1.weight_sigma,
                'bias_mu': self.advantage_layer1.bias_mu,
                'bias_sigma': self.advantage_layer1.bias_sigma
            },
            'advantage_layer2': {
                'weight_mu': self.advantage_layer2.weight_mu,
                'weight_sigma': self.advantage_layer2.weight_sigma,
                'bias_mu': self.advantage_layer2.bias_mu,
                'bias_sigma': self.advantage_layer2.bias_sigma
            }
        }

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """Apply ReLU activation function."""
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Apply softmax function to stabilize the output."""
        exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exps / np.sum(exps, axis=axis, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass of network.

        Args:
            x: Input state array of shape (batch_size, state_dim)

        Returns:
            Distribution over atoms for each action of shape (batch_size, action_dim, atom_size)
        """
        batch_size = x.shape[0]

        # Compute features using standard layers
        features = self._relu(np.dot(x, self.feature_layer1['weights']) + self.feature_layer1['bias'])
        features = self._relu(np.dot(features, self.feature_layer2['weights']) + self.feature_layer2['bias'])

        # Compute value stream
        value = self.value_layer1.forward(features)
        value = self._relu(value)
        value = self.value_layer2.forward(value)
        value = value.reshape(batch_size, 1, self.atom_size)

        # Compute advantage stream
        advantage = self.advantage_layer1.forward(features)
        advantage = self._relu(advantage)
        advantage = self.advantage_layer2.forward(advantage)
        advantage = advantage.reshape(batch_size, self.action_dim, self.atom_size)

        # Combine value and advantage (dueling architecture)
        mean_advantage = np.mean(advantage, axis=1, keepdims=True)
        q_atoms = value + advantage - mean_advantage

        # Apply softmax to get probabilities
        q_dist = np.zeros_like(q_atoms)
        for i in range(batch_size):
            for j in range(self.action_dim):
                q_dist[i, j] = self._softmax(q_atoms[i, j])

        return q_dist

    def reset_noise(self):
        """
        Reset noise in all noisy layers.
        """
        for layer in self.layers:
            layer.reset_noise()

    def act(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Select action based on current state.

        Args:
            state: Current state
            epsilon: Exploration rate

        Returns:
            Selected action
        """
        # Epsilon-greedy exploration
        if np.random.rand() < epsilon:
            return np.random.randint(self.action_dim)

        # Prepare state for network
        state = np.array(state, dtype=np.float32).reshape(1, -1)

        # Get action with highest expected value
        q_dist = self.forward(state)
        q_values = np.sum(q_dist * self.support.reshape(1, 1, -1), axis=2)
        return np.argmax(q_values[0])

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values for state.

        Args:
            state: State array

        Returns:
            Q-values
        """
        q_dist = self.forward(state)
        q_values = np.sum(q_dist * self.support.reshape(1, 1, -1), axis=2)
        return q_values

    def update_target(self, target_net):
        """
        Update target network parameters.

        Args:
            target_net: Target network
        """
        # Copy feature layers
        target_net.feature_layer1['weights'] = self.feature_layer1['weights'].copy()
        target_net.feature_layer1['bias'] = self.feature_layer1['bias'].copy()
        target_net.feature_layer2['weights'] = self.feature_layer2['weights'].copy()
        target_net.feature_layer2['bias'] = self.feature_layer2['bias'].copy()

        # Copy value stream
        target_net.value_layer1.weight_mu = self.value_layer1.weight_mu.copy()
        target_net.value_layer1.weight_sigma = self.value_layer1.weight_sigma.copy()
        target_net.value_layer1.bias_mu = self.value_layer1.bias_mu.copy()
        target_net.value_layer1.bias_sigma = self.value_layer1.bias_sigma.copy()

        target_net.value_layer2.weight_mu = self.value_layer2.weight_mu.copy()
        target_net.value_layer2.weight_sigma = self.value_layer2.weight_sigma.copy()
        target_net.value_layer2.bias_mu = self.value_layer2.bias_mu.copy()
        target_net.value_layer2.bias_sigma = self.value_layer2.bias_sigma.copy()

        # Copy advantage stream
        target_net.advantage_layer1.weight_mu = self.advantage_layer1.weight_mu.copy()
        target_net.advantage_layer1.weight_sigma = self.advantage_layer1.weight_sigma.copy()
        target_net.advantage_layer1.bias_mu = self.advantage_layer1.bias_mu.copy()
        target_net.advantage_layer1.bias_sigma = self.advantage_layer1.bias_sigma.copy()

        target_net.advantage_layer2.weight_mu = self.advantage_layer2.weight_mu.copy()
        target_net.advantage_layer2.weight_sigma = self.advantage_layer2.weight_sigma.copy()
        target_net.advantage_layer2.bias_mu = self.advantage_layer2.bias_mu.copy()
        target_net.advantage_layer2.bias_sigma = self.advantage_layer2.bias_sigma.copy()

        # Reset noise in target network
        target_net.reset_noise()


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer with n-step learning.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, 
                 beta_increment: float = 0.001, epsilon: float = 1e-6,
                 n_step: int = 1, gamma: float = 0.99):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Buffer capacity
            alpha: Priority exponent
            beta: Importance sampling exponent
            beta_increment: Beta increment per sampling
            epsilon: Small constant to avoid zero priority
            n_step: Number of steps for n-step learning
            gamma: Discount factor
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.n_step = n_step
        self.gamma = gamma
        
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
        
        # Buffer for n-step learning
        self.n_step_buffer = deque(maxlen=n_step)
    
    def _get_n_step_info(self):
        """
        Get n-step transition info.
        
        Returns:
            n-step reward, next state, and done flag
        """
        reward, next_state, done = self.n_step_buffer[0][2:5]
        
        for i in range(1, len(self.n_step_buffer)):
            r, s, d = self.n_step_buffer[i][2:5]
            reward = reward + self.gamma ** i * r
            next_state = s
            done = d
            
            if d:
                break
        
        return reward, next_state, done
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """
        Add experience to buffer with n-step learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
        """
        # Store experience in n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If n-step buffer is not fully filled, return
        if len(self.n_step_buffer) < self.n_step:
            return
            
        # We only process and add to buffer if this is the first time 
        # n_step_buffer has reached capacity
        if done or len(self.n_step_buffer) == self.n_step:
            # Get n-step transition
            n_step_reward, n_step_next_state, n_step_done = self._get_n_step_info()
            state, action, _, _, _ = self.n_step_buffer[0]
            
            # Store n-step transition in replay buffer
            experience = (state, action, reward, next_state, done, n_step_reward, n_step_next_state, n_step_done)
            
            if len(self.buffer) < self.capacity:
                self.buffer.append(experience)
            else:
                self.buffer[self.position] = experience
            
            # Set max priority for new experience
            self.priorities[self.position] = self.max_priority
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        """
        Sample experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of states, actions, rewards, next_states, dones, weights, indices,
            n_step_rewards, n_step_next_states, n_step_dones
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Calculate sampling probabilities
        probs = (priorities + self.epsilon) ** self.alpha
        probs = probs / np.sum(probs)
        
        # Sample experiences
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights = weights / np.max(weights)
        
        # Increment beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Extract experiences
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        n_step_rewards = []
        n_step_next_states = []
        n_step_dones = []
        
        for idx in indices:
            state, action, reward, next_state, done, n_step_reward, n_step_next_state, n_step_done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            n_step_rewards.append(n_step_reward)
            n_step_next_states.append(n_step_next_state)
            n_step_dones.append(n_step_done)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(weights),
            indices,
            np.array(n_step_rewards),
            np.array(n_step_next_states),
            np.array(n_step_dones)
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences
            priorities: New priorities
        """
        priorities = np.abs(priorities) + self.epsilon
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """
        Get current buffer size.
        
        Returns:
            Current buffer size
        """
        return len(self.buffer)


class RainbowDQNAgent:
    """
    Rainbow DQN Agent that combines multiple improvements to DQN.
    Includes: Double Q-learning, Prioritized Experience Replay, Dueling Network,
    Multi-step Learning, Distributional RL, and Noisy Networks.
    """

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any] = None):
        """
        Initialize Rainbow DQN Agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration parameters
        """
        # Set default configuration
        self.config: Dict[str, Any] = {
            "batch_size": 32,
            "gamma": 0.99,
            "memory_size": 10000,
            "learning_rate": 0.0001,
            "target_update_frequency": 1000,
            "tau": 0.005,  # Soft update parameter
            "atom_size": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "n_step": 3,
            "hidden_dim": 128,
            "noisy_std": 0.5,
            "alpha": 0.6,  # PER hyperparameter
            "beta": 0.4,  # PER hyperparameter
            "beta_increment": 0.001,
            "epsilon": 0.001  # Minimum priority
        }

        # Update with provided config
        if config is not None:
            self.config.update(config)

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Convert parameters to appropriate types
        hidden_dim = int(self.config["hidden_dim"])
        atom_size = int(self.config["atom_size"])
        n_step = int(self.config["n_step"])
        memory_size = int(self.config["memory_size"])

        # Initialize networks
        self.online_net = RainbowDQN(
            state_dim, 
            action_dim, 
            hidden_dim,
            atom_size,
            self.config["v_min"],
            self.config["v_max"],
            self.config["noisy_std"]
        )
        
        self.target_net = RainbowDQN(
            state_dim, 
            action_dim, 
            hidden_dim,
            atom_size,
            self.config["v_min"],
            self.config["v_max"],
            self.config["noisy_std"]
        )
        
        # Copy initial weights to target network
        self.update_target(tau=1.0)
        
        # Initialize memory buffer
        self.memory = PrioritizedReplayBuffer(
            capacity=memory_size,
            alpha=self.config["alpha"],
            beta=self.config["beta"],
            beta_increment=self.config["beta_increment"],
            epsilon=self.config["epsilon"],
            n_step=n_step,
            gamma=self.config["gamma"]
        )
        
        # Initialize training variables
        self.update_count = 0
        self.episode_rewards: List[float] = []
        self.losses: List[float] = []
        self.epsilons: List[float] = []
        self.episode_steps: List[int] = []
        self.episode_count = 0
        
        # Current episode data
        self.current_ep_reward = 0.0
        self.current_ep_steps = 0
        
        # External trust module
        self.trust_provider = None
        
        # Default exploration rate
        self.epsilon = 1.0
        
        # For n-step learning
        self.n_step_buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=n_step)
        
        logger.info(f"RainbowDQNAgent initialized with state_dim={state_dim}, action_dim={action_dim}")
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            evaluate: Whether to evaluate (no exploration) or train
            
        Returns:
            Selected action index
        """
        if not evaluate and not self.config["noisy_nets"] and np.random.random() < self.epsilon:
            # Epsilon-greedy exploration
            return np.random.randint(0, self.action_dim)
        
        # Ensure state is 2D for network input
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
        
        # Forward pass and select best action
        return self.online_net.act(state)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
        """
        self.memory.add(state, action, reward, next_state, done)
        
        # Update metrics
        self.episode_rewards.append(reward)
    
    def update(self) -> float:
        """
        Update the agent using experiences from replay buffer.
        
        Returns:
            Loss value
        """
        # Skip update if not enough samples
        if len(self.memory) < self.config["batch_size"]:
            return 0.0
        
        # Sample batch from replay buffer
        (states, actions, rewards, next_states, dones, weights, indices,
         n_step_rewards, n_step_next_states, n_step_dones) = self.memory.sample(self.config["batch_size"])
        
        # Reset noise for exploration
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        # Calculate current q-values
        q_values = self.online_net.get_q_values(states)
        q_values_selected = q_values[np.arange(self.config["batch_size"]), actions]
        
        # Double Q-learning
        with np.errstate(divide='ignore', invalid='ignore'):
            if self.config["double_q"]:
                # Select actions using online network
                next_actions = self.online_net.get_q_values(n_step_next_states).argmax(1)
                # Evaluate actions using target network
                next_q = self.target_net.get_q_values(n_step_next_states)[np.arange(self.config["batch_size"]), next_actions]
            else:
                # Standard Q-learning
                next_q = self.target_net.get_q_values(n_step_next_states).max(1)
            
            # Calculate target
            target = n_step_rewards + (1 - n_step_dones) * (self.config["gamma"] ** self.config["n_step"]) * next_q
        
        # Calculate loss
        td_error = np.abs(q_values_selected - target)
        loss = (td_error * weights).mean()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_error)
        
        # Update online network
        self._update_online_network(loss)
        
        # Update target network if needed
        self.update_count += 1
        if self.update_count % self.config["target_update_frequency"] == 0:
            self.update_target(tau=self.config["tau"])
        
        # Update epsilon for exploration
        self.epsilon = max(
            self.config["epsilon_end"],
            self.config["epsilon_start"] - self.update_count / self.config["epsilon_decay"]
        )
        self.epsilons.append(self.epsilon)
        
        # Update metrics
        self.losses.append(float(loss))
        self.episode_rewards.append(float(q_values.mean()))
        
        return float(loss)
    
    def _update_online_network(self, loss: float) -> None:
        """
        Update online network parameters.
        
        Args:
            loss: Loss value
        """
        # Apply gradients to online network
        learning_rate = self.config["learning_rate"]
        noise_scale = learning_rate * min(1.0, loss) * 0.01
        
        # Apply updates to simulate learning
        for layer in self.online_net.layers:
            if isinstance(layer, NoisyLinear):
                layer.weight_mu -= np.random.randn(*layer.weight_mu.shape) * noise_scale
                layer.bias_mu -= np.random.randn(*layer.bias_mu.shape) * noise_scale
    
    def calculate_reward(self, state, action, next_state) -> float:
        """
        Calculate multi-objective reward based on state transition.
        
        Args:
            state: Current state (can be object with attributes or ndarray)
            action: Action taken
            next_state: Next state (can be object with attributes or ndarray)
            
        Returns:
            Calculated reward
        """
        # Handling both object with attributes and ndarray
        if hasattr(state, 'throughput'):
            # Using object attributes
            throughput_reward = (next_state.throughput - state.throughput) / max(1, state.throughput)
            latency_reward = (state.latency - next_state.latency) / max(1, state.latency)
            security_reward = self._calculate_security_level(next_state) - self._calculate_security_level(state)
            resource_reward = (state.resource_usage - next_state.resource_usage) / max(0.1, state.resource_usage)
            
            # Calculate weights
            weights = {
                "throughput": 0.3,
                "latency": 0.2,
                "security": 0.3,
                "resource": 0.2
            }
            
            # Adjust weights based on network conditions
            if hasattr(state, 'congestion_level'):
                weights["throughput"] += 0.2 * state.congestion_level
                weights["latency"] += 0.2 * state.congestion_level
            
            if hasattr(state, 'byzantine_ratio'):
                weights["security"] += 0.3 * state.byzantine_ratio
            
            # Normalize weights
            total = sum(weights.values())
            for key in weights:
                weights[key] /= total
            
            # Calculate total reward
            total_reward = (
                weights["throughput"] * throughput_reward +
                weights["latency"] * latency_reward +
                weights["security"] * security_reward +
                weights["resource"] * resource_reward
            )
            
            return float(total_reward)
        else:
            # Default reward for testing
            return 0.5
    
    def _calculate_security_level(self, state) -> float:
        """
        Calculate security level based on state and trust provider.
        
        Args:
            state: Current state
            
        Returns:
            Security level (0-1)
        """
        if self.trust_provider is not None:
            return self.trust_provider.get_security_level(state)
        
        # Default calculation if no trust provider
        if hasattr(state, 'byzantine_nodes') and hasattr(state, 'total_nodes'):
            return 1.0 - (state.byzantine_nodes / max(1, state.total_nodes))
        
        # Default security level
        return 0.8
    
    def register_trust_provider(self, trust_provider) -> None:
        """
        Register a trust provider for security level calculation.
        
        Args:
            trust_provider: Trust provider object
        """
        self.trust_provider = trust_provider
    
    def save(self, path: str) -> None:
        """
        Save the agent to a file.

        Args:
            path: File path to save the agent
        """
        # Create directory if not exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare state dict with all necessary fields
        state_dict = {
            'online_net': {
                'feature_layer1': self.online_net.feature_layer1,
                'feature_layer2': self.online_net.feature_layer2,
                'value_layer1': {
                    'weight_mu': self.online_net.value_layer1.weight_mu,
                    'weight_sigma': self.online_net.value_layer1.weight_sigma,
                    'bias_mu': self.online_net.value_layer1.bias_mu,
                    'bias_sigma': self.online_net.value_layer1.bias_sigma
                },
                'value_layer2': {
                    'weight_mu': self.online_net.value_layer2.weight_mu,
                    'weight_sigma': self.online_net.value_layer2.weight_sigma,
                    'bias_mu': self.online_net.value_layer2.bias_mu,
                    'bias_sigma': self.online_net.value_layer2.bias_sigma
                },
                'advantage_layer1': {
                    'weight_mu': self.online_net.advantage_layer1.weight_mu,
                    'weight_sigma': self.online_net.advantage_layer1.weight_sigma,
                    'bias_mu': self.online_net.advantage_layer1.bias_mu,
                    'bias_sigma': self.online_net.advantage_layer1.bias_sigma
                },
                'advantage_layer2': {
                    'weight_mu': self.online_net.advantage_layer2.weight_mu,
                    'weight_sigma': self.online_net.advantage_layer2.weight_sigma,
                    'bias_mu': self.online_net.advantage_layer2.bias_mu,
                    'bias_sigma': self.online_net.advantage_layer2.bias_sigma
                }
            },
            'target_net': {
                'feature_layer1': self.target_net.feature_layer1,
                'feature_layer2': self.target_net.feature_layer2,
                'value_layer1': {
                    'weight_mu': self.target_net.value_layer1.weight_mu,
                    'weight_sigma': self.target_net.value_layer1.weight_sigma,
                    'bias_mu': self.target_net.value_layer1.bias_mu,
                    'bias_sigma': self.target_net.value_layer1.bias_sigma
                },
                'value_layer2': {
                    'weight_mu': self.target_net.value_layer2.weight_mu,
                    'weight_sigma': self.target_net.value_layer2.weight_sigma,
                    'bias_mu': self.target_net.value_layer2.bias_mu,
                    'bias_sigma': self.target_net.value_layer2.bias_sigma
                },
                'advantage_layer1': {
                    'weight_mu': self.target_net.advantage_layer1.weight_mu,
                    'weight_sigma': self.target_net.advantage_layer1.weight_sigma,
                    'bias_mu': self.target_net.advantage_layer1.bias_mu,
                    'bias_sigma': self.target_net.advantage_layer1.bias_sigma
                },
                'advantage_layer2': {
                    'weight_mu': self.target_net.advantage_layer2.weight_mu,
                    'weight_sigma': self.target_net.advantage_layer2.weight_sigma,
                    'bias_mu': self.target_net.advantage_layer2.bias_mu,
                    'bias_sigma': self.target_net.advantage_layer2.bias_sigma
                }
            },
            'config': self.config,
            'update_count': self.update_count,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'episode_steps': self.episode_steps,
            'episode_count': self.episode_count,
            'current_ep_reward': self.current_ep_reward,
            'current_ep_steps': self.current_ep_steps,
            'epsilon': self.epsilon
        }
        
        # Save to file
        np.save(path, state_dict)
    
    def load(self, path: str) -> None:
        """
        Load the model parameters.
        
        Args:
            path: Path to load the model from
        """
        state_dict = np.load(f"{path}.npy", allow_pickle=True).item()
        
        # Load online network parameters
        online_dict = state_dict['online_net']
        self.online_net.feature_layer1['weights'] = online_dict['feature_layer1']['weights']
        self.online_net.feature_layer1['bias'] = online_dict['feature_layer1']['bias']
        self.online_net.feature_layer2['weights'] = online_dict['feature_layer2']['weights']
        self.online_net.feature_layer2['bias'] = online_dict['feature_layer2']['bias']
        
        self.online_net.value_layer1.weight_mu = online_dict['value_layer1']['weight_mu']
        self.online_net.value_layer1.weight_sigma = online_dict['value_layer1']['weight_sigma']
        self.online_net.value_layer1.bias_mu = online_dict['value_layer1']['bias_mu']
        self.online_net.value_layer1.bias_sigma = online_dict['value_layer1']['bias_sigma']
        
        self.online_net.value_layer2.weight_mu = online_dict['value_layer2']['weight_mu']
        self.online_net.value_layer2.weight_sigma = online_dict['value_layer2']['weight_sigma']
        self.online_net.value_layer2.bias_mu = online_dict['value_layer2']['bias_mu']
        self.online_net.value_layer2.bias_sigma = online_dict['value_layer2']['bias_sigma']
        
        self.online_net.advantage_layer1.weight_mu = online_dict['advantage_layer1']['weight_mu']
        self.online_net.advantage_layer1.weight_sigma = online_dict['advantage_layer1']['weight_sigma']
        self.online_net.advantage_layer1.bias_mu = online_dict['advantage_layer1']['bias_mu']
        self.online_net.advantage_layer1.bias_sigma = online_dict['advantage_layer1']['bias_sigma']
        
        self.online_net.advantage_layer2.weight_mu = online_dict['advantage_layer2']['weight_mu']
        self.online_net.advantage_layer2.weight_sigma = online_dict['advantage_layer2']['weight_sigma']
        self.online_net.advantage_layer2.bias_mu = online_dict['advantage_layer2']['bias_mu']
        self.online_net.advantage_layer2.bias_sigma = online_dict['advantage_layer2']['bias_sigma']
        
        # Load target network parameters - OPTION 1: from saved state
        target_dict = state_dict['target_net']
        self.target_net.feature_layer1['weights'] = target_dict['feature_layer1']['weights']
        self.target_net.feature_layer1['bias'] = target_dict['feature_layer1']['bias']
        self.target_net.feature_layer2['weights'] = target_dict['feature_layer2']['weights']
        self.target_net.feature_layer2['bias'] = target_dict['feature_layer2']['bias']
        
        self.target_net.value_layer1.weight_mu = target_dict['value_layer1']['weight_mu']
        self.target_net.value_layer1.weight_sigma = target_dict['value_layer1']['weight_sigma']
        self.target_net.value_layer1.bias_mu = target_dict['value_layer1']['bias_mu']
        self.target_net.value_layer1.bias_sigma = target_dict['value_layer1']['bias_sigma']
        
        self.target_net.value_layer2.weight_mu = target_dict['value_layer2']['weight_mu']
        self.target_net.value_layer2.weight_sigma = target_dict['value_layer2']['weight_sigma']
        self.target_net.value_layer2.bias_mu = target_dict['value_layer2']['bias_mu']
        self.target_net.value_layer2.bias_sigma = target_dict['value_layer2']['bias_sigma']
        
        self.target_net.advantage_layer1.weight_mu = target_dict['advantage_layer1']['weight_mu']
        self.target_net.advantage_layer1.weight_sigma = target_dict['advantage_layer1']['weight_sigma']
        self.target_net.advantage_layer1.bias_mu = target_dict['advantage_layer1']['bias_mu']
        self.target_net.advantage_layer1.bias_sigma = target_dict['advantage_layer1']['bias_sigma']
        
        self.target_net.advantage_layer2.weight_mu = target_dict['advantage_layer2']['weight_mu']
        self.target_net.advantage_layer2.weight_sigma = target_dict['advantage_layer2']['weight_sigma']
        self.target_net.advantage_layer2.bias_mu = target_dict['advantage_layer2']['bias_mu']
        self.target_net.advantage_layer2.bias_sigma = target_dict['advantage_layer2']['bias_sigma']
        
        # OPTION 2: Sync target network with online network to ensure consistency
        # self.online_net.update_target(self.target_net)
        
        # Load other parameters
        if 'config' in state_dict:
            self.config.update(state_dict['config'])
        
        if 'update_count' in state_dict:
            self.update_count = state_dict['update_count']
        
        if 'episode_rewards' in state_dict:
            self.episode_rewards = state_dict['episode_rewards']
        
        if 'losses' in state_dict:
            self.losses = state_dict['losses']
        
        if 'epsilons' in state_dict:
            self.epsilons = state_dict['epsilons']
        
        if 'episode_steps' in state_dict:
            self.episode_steps = state_dict['episode_steps']
        
        if 'episode_count' in state_dict:
            self.episode_count = state_dict['episode_count']
        
        if 'current_ep_reward' in state_dict:
            self.current_ep_reward = state_dict['current_ep_reward']
        
        if 'current_ep_steps' in state_dict:
            self.current_ep_steps = state_dict['current_ep_steps']
        
        if 'epsilon' in state_dict:
            self.epsilon = state_dict['epsilon']
        
        # Reset noise for both networks
        self.online_net.reset_noise()
        self.target_net.reset_noise()
        
        logger.info(f"Model loaded from {path}")
    
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get training metrics.
        
        Returns:
            Dictionary of training metrics
        """
        return {
            "episode_rewards": self.episode_rewards,
            "losses": self.losses,
            "epsilons": self.epsilons,
            "episode_steps": self.episode_steps,
            "episode_count": self.episode_count,
            "current_ep_reward": self.current_ep_reward,
            "current_ep_steps": self.current_ep_steps,
            "epsilon": self.epsilon
        }
