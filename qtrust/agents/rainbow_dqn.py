import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Any, Optional

class NoisyLinear(nn.Module):
    """
    Noisy Linear layer with factorized Gaussian noise.
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Initialize NoisyLinear layer.
        
        Args:
            in_features: Number of input features
            out_features: Number of output features
            std_init: Initial standard deviation
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Mean weights and biases
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        
        # Register buffer for random noise
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """
        Reset trainable network parameters.
        """
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """
        Scale noise for factorized Gaussian noise.
        
        Args:
            size: Size of noise tensor
            
        Returns:
            Scaled noise tensor
        """
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def reset_noise(self):
        """
        Reset noise.
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with added Gaussian noise.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Buffer capacity
            alpha: Priority exponent
            beta: Importance sampling exponent
            beta_increment: Beta increment per sampling
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        # Set max priority for new experience
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of states, actions, rewards, next_states, dones, indices, weights
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
        
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
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
        
        for idx in indices:
            state, action, reward, next_state, done = self.buffer[idx]
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update priorities for sampled experiences.
        
        Args:
            indices: Indices of experiences
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """
        Get buffer length.
        
        Returns:
            Current buffer length
        """
        return len(self.buffer)


class RainbowDQN(nn.Module):
    """
    Rainbow DQN network with noisy networks, dueling architecture, and distributional RL.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        atom_size: int = 51,
        support_min: float = -10.0,
        support_max: float = 10.0,
        hidden_dim: int = 128,
        noisy_std: float = 0.5
    ):
        """
        Initialize Rainbow DQN network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            atom_size: Number of atoms for distributional RL
            support_min: Minimum support value
            support_max: Maximum support value
            hidden_dim: Hidden layer dimension
            noisy_std: Standard deviation for noisy layers
        """
        super(RainbowDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.atom_size = atom_size
        self.support_min = support_min
        self.support_max = support_max
        
        # Support for distributional RL
        self.register_buffer(
            'support',
            torch.linspace(support_min, support_max, atom_size)
        )
        self.delta = (support_max - support_min) / (atom_size - 1)
        
        # Common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, atom_size, noisy_std)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim * atom_size, noisy_std)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Distribution over atoms for each action
        """
        batch_size = x.size(0)
        
        # Compute features
        features = self.feature_layer(x)
        
        # Compute value and advantage
        value = self.value_stream(features).view(batch_size, 1, self.atom_size)
        advantage = self.advantage_stream(features).view(batch_size, self.action_dim, self.atom_size)
        
        # Combine value and advantage (dueling architecture)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        # Apply softmax to get probabilities
        q_dist = F.softmax(q_atoms, dim=2)
        
        return q_dist
    
    def reset_noise(self):
        """
        Reset noise in all noisy layers.
        """
        # Reset noise in value stream
        for layer in self.value_stream:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()
        
        # Reset noise in advantage stream
        for layer in self.advantage_stream:
            if isinstance(layer, NoisyLinear):
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
        
        # Convert state to tensor
        state = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action with highest expected value
        with torch.no_grad():
            q_dist = self(state)
            q_values = (q_dist * self.support).sum(dim=2)
            return q_values.argmax(dim=1).item()
    
    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get Q-values for state.
        
        Args:
            state: State tensor
            
        Returns:
            Q-values
        """
        q_dist = self(state)
        q_values = (q_dist * self.support).sum(dim=2)
        return q_values
    
    def update_target(self, target_net):
        """
        Update target network parameters.
        
        Args:
            target_net: Target network
        """
        target_net.load_state_dict(self.state_dict())
    
    def support_projection(
        self,
        target_dist: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        """
        Project distributed return onto support.
        
        Args:
            target_dist: Target distribution
            rewards: Reward tensor
            dones: Done flags
            gamma: Discount factor
            
        Returns:
            Projected distribution
        """
        batch_size = rewards.size(0)
        
        # Compute projected atoms
        rewards = rewards.unsqueeze(1).expand(batch_size, self.atom_size)
        dones = dones.unsqueeze(1).expand(batch_size, self.atom_size)
        support = self.support.unsqueeze(0).expand(batch_size, self.atom_size)
        
        # Compute projected values
        projected_atoms = rewards + gamma * support * (1 - dones)
        projected_atoms = torch.clamp(projected_atoms, self.support_min, self.support_max)
        
        # Compute indices and offsets
        delta_z = (self.support_max - self.support_min) / (self.atom_size - 1)
        b = (projected_atoms - self.support_min) / delta_z
        
        lower_bound = b.floor().long()
        upper_bound = b.ceil().long()
        
        # Handle case where projected_atoms == support_max
        upper_bound = torch.min(
            upper_bound, torch.tensor(self.atom_size - 1, device=upper_bound.device)
        )
        
        # Initialize return distribution
        projected_dist = torch.zeros((batch_size, self.atom_size), device=target_dist.device)
        
        # Distribute probability
        for idx in range(batch_size):
            if dones[idx][0]:
                # If episode ends, only reward matters
                offset = (rewards[idx] - self.support_min) / delta_z
                index = offset.floor().long()
                index = torch.clamp(index, 0, self.atom_size - 1)
                projected_dist[idx, index] += 1.0
            else:
                # Distribute probability mass
                for j in range(self.atom_size):
                    tz = projected_atoms[idx][j]
                    l, u = lower_bound[idx][j], upper_bound[idx][j]
                    
                    # Handle exact match
                    if l == u:
                        projected_dist[idx, l] += target_dist[idx, j]
                    else:
                        # Distribute probability mass
                        lb_weight = (u - b[idx][j])
                        ub_weight = (b[idx][j] - l)
                        
                        projected_dist[idx, l] += target_dist[idx, j] * lb_weight
                        projected_dist[idx, u] += target_dist[idx, j] * ub_weight
        
        return projected_dist 