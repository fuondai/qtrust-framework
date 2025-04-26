"""
QTrust Blockchain Sharding Framework - Adaptive Rainbow Agent

This module implements the Adaptive Rainbow DQN agent that dynamically adjusts
its architecture and hyperparameters based on performance metrics.
"""

import os
import time
import logging
import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque, namedtuple
import json
import copy

logger = logging.getLogger(__name__)

# Define experience tuple type
Experience = namedtuple(
    "Experience", ["state", "action", "reward", "next_state", "done"]
)


class SumTree:
    """
    Sum Tree data structure for efficient sampling from prioritized replay buffer.
    """

    def __init__(self, capacity: int):
        """
        Initialize the sum tree.

        Args:
            capacity: Maximum capacity of the tree
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx: int, change: float) -> None:
        """
        Propagate the priority update up the tree.

        Args:
            idx: Index of the node
            change: Change in priority
        """
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """
        Retrieve sample index from the tree.

        Args:
            idx: Current index
            s: Value to search for

        Returns:
            Index of the sample
        """
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """
        Get the total priority.

        Returns:
            Total priority
        """
        return self.tree[0]

    def add(self, p: float, data: Any) -> None:
        """
        Add a new sample to the tree.

        Args:
            p: Priority of the sample
            data: Sample data
        """
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, p: float) -> None:
        """
        Update the priority of a sample.

        Args:
            idx: Index of the sample
            p: New priority
        """
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """
        Get a sample based on a value.

        Args:
            s: Value to search for

        Returns:
            Tuple of (index, priority, data)
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer for Rainbow DQN.
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialize the buffer.

        Args:
            capacity: Maximum capacity of the buffer
            alpha: Priority exponent (0 = uniform sampling)
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.max_priority = 1.0

    def add(self, experience: Experience) -> None:
        """
        Add an experience to the buffer.

        Args:
            experience: Experience tuple
        """
        # Add with maximum priority
        self.tree.add(self.max_priority**self.alpha, experience)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[List[Experience], List[int], np.ndarray]:
        """
        Sample a batch from the buffer.

        Args:
            batch_size: Number of samples to retrieve
            beta: Importance sampling exponent (0 = no correction)

        Returns:
            Tuple of (experiences, indices, weights)
        """
        batch = []
        indices = []
        weights = np.zeros(batch_size, dtype=np.float32)

        # Calculate segment size
        segment = self.tree.total() / batch_size

        # Increase beta over time to reduce bias
        beta = min(1.0, beta)

        # Calculate min priority for weights
        min_prob = np.min(self.tree.tree[-self.tree.capacity :]) / self.tree.total()
        if min_prob == 0:
            min_prob = 0.00001

        for i in range(batch_size):
            # Sample uniformly from each segment
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            # Retrieve sample
            idx, priority, data = self.tree.get(s)

            # Calculate weight
            sample_prob = priority / self.tree.total()
            weight = (sample_prob * self.tree.n_entries) ** (-beta)

            weights[i] = weight
            indices.append(idx)
            batch.append(data)

        # Normalize weights
        weights /= weights.max()

        return batch, indices, weights

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update priorities of samples.

        Args:
            indices: Indices of samples to update
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            # Add small constant to avoid zero priority
            priority = max(priority, 1e-5)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority**self.alpha)

    def __len__(self) -> int:
        """
        Get the current size of the buffer.

        Returns:
            Current buffer size
        """
        return self.tree.n_entries


class NoisyLinear:
    """
    Noisy Linear layer for exploration in Rainbow DQN.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Initialize the noisy linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            std_init: Initial standard deviation
        """
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Initialize parameters
        self.weight_mu = np.random.normal(0, 0.1, (out_features, in_features))
        self.weight_sigma = (
            np.ones((out_features, in_features)) * std_init / np.sqrt(in_features)
        )
        self.bias_mu = np.zeros(out_features)
        self.bias_sigma = np.ones(out_features) * std_init / np.sqrt(in_features)

        # Initialize noise
        self.reset_noise()

    def reset_noise(self) -> None:
        """Reset the noise parameters."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # Outer product
        self.weight_epsilon = np.outer(epsilon_out, epsilon_in)
        self.bias_epsilon = epsilon_out

    def _scale_noise(self, size: int) -> np.ndarray:
        """
        Scale noise for factorized Gaussian noise.

        Args:
            size: Size of the noise vector

        Returns:
            Scaled noise vector
        """
        x = np.random.normal(0, 1, size)
        return np.sign(x) * np.sqrt(np.abs(x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        return np.dot(x, weight.T) + bias


class DuelingNetwork:
    """
    Dueling Network architecture for Rainbow DQN.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        noisy: bool = True,
        atom_size: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
    ):
        """
        Initialize the dueling network.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dim: Dimension of the hidden layer
            noisy: Whether to use noisy linear layers
            atom_size: Number of atoms for distributional RL
            v_min: Minimum value for distributional RL
            v_max: Maximum value for distributional RL
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.noisy = noisy
        self.atom_size = atom_size
        self.v_min = v_min
        self.v_max = v_max

        # Calculate supports for distributional RL
        self.supports = np.linspace(v_min, v_max, atom_size)
        self.delta_z = (v_max - v_min) / (atom_size - 1)

        # Initialize layers
        if noisy:
            # Feature layer
            self.feature_mu = np.random.normal(0, 0.1, (hidden_dim, state_dim))
            self.feature_bias_mu = np.zeros(hidden_dim)

            # Value stream
            self.value_hidden = NoisyLinear(hidden_dim, hidden_dim)
            self.value_out = NoisyLinear(hidden_dim, atom_size)

            # Advantage stream
            self.advantage_hidden = NoisyLinear(hidden_dim, hidden_dim)
            self.advantage_out = NoisyLinear(hidden_dim, action_dim * atom_size)
        else:
            # Feature layer
            self.feature = np.random.normal(0, 0.1, (hidden_dim, state_dim))
            self.feature_bias = np.zeros(hidden_dim)

            # Value stream
            self.value_hidden = np.random.normal(0, 0.1, (hidden_dim, hidden_dim))
            self.value_hidden_bias = np.zeros(hidden_dim)
            self.value_out = np.random.normal(0, 0.1, (atom_size, hidden_dim))
            self.value_out_bias = np.zeros(atom_size)

            # Advantage stream
            self.advantage_hidden = np.random.normal(0, 0.1, (hidden_dim, hidden_dim))
            self.advantage_hidden_bias = np.zeros(hidden_dim)
            self.advantage_out = np.random.normal(
                0, 0.1, (action_dim * atom_size, hidden_dim)
            )
            self.advantage_out_bias = np.zeros(action_dim * atom_size)

    def reset_noise(self) -> None:
        """Reset noise for all noisy layers."""
        if self.noisy:
            self.value_hidden.reset_noise()
            self.value_out.reset_noise()
            self.advantage_hidden.reset_noise()
            self.advantage_out.reset_noise()

    def forward(self, state: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            state: State tensor

        Returns:
            Distribution over actions
        """
        batch_size = state.shape[0]

        if self.noisy:
            # Feature extraction
            x = np.dot(state, self.feature_mu.T) + self.feature_bias_mu
            x = np.maximum(0, x)  # ReLU

            # Value stream
            v = self.value_hidden.forward(x)
            v = np.maximum(0, v)  # ReLU
            v = self.value_out.forward(v)

            # Advantage stream
            a = self.advantage_hidden.forward(x)
            a = np.maximum(0, a)  # ReLU
            a = self.advantage_out.forward(a)
            a = a.reshape(batch_size, self.action_dim, self.atom_size)
        else:
            # Feature extraction
            x = np.dot(state, self.feature.T) + self.feature_bias
            x = np.maximum(0, x)  # ReLU

            # Value stream
            v = np.dot(x, self.value_hidden.T) + self.value_hidden_bias
            v = np.maximum(0, v)  # ReLU
            v = np.dot(v, self.value_out.T) + self.value_out_bias

            # Advantage stream
            a = np.dot(x, self.advantage_hidden.T) + self.advantage_hidden_bias
            a = np.maximum(0, a)  # ReLU
            a = np.dot(a, self.advantage_out.T) + self.advantage_out_bias
            a = a.reshape(batch_size, self.action_dim, self.atom_size)

        # Combine value and advantage
        v = v.reshape(batch_size, 1, self.atom_size)
        a_mean = np.mean(a, axis=1, keepdims=True)

        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q = v + a - a_mean

        # Apply softmax to get probabilities
        q = np.exp(q) / np.sum(np.exp(q), axis=2, keepdims=True)

        return q

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Get Q-values from the network.

        Args:
            state: State tensor

        Returns:
            Q-values for each action
        """
        dist = self.forward(state)
        q_values = np.sum(dist * self.supports.reshape(1, 1, -1), axis=2)
        return q_values


class AdaptiveRainbowAgent:
    """
    Adaptive Rainbow DQN agent that dynamically adjusts its architecture 
    and hyperparameters based on performance metrics.
    """

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any] = None):
        """
        Initialize the adaptive Rainbow DQN agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration parameters
        """
        # Default configuration
        self.config: Dict[str, Any] = {
            "batch_size": 32,
            "buffer_size": 10000,
            "gamma": 0.99,
            "learning_rate": 0.0001,
            "target_update": 100,
            "alpha": 0.6,      # PER hyperparameter
            "beta": 0.4,       # PER hyperparameter
            "beta_increment": 0.001,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 10000,
            "hidden_dim": 128,
            "noisy": True,
            "double_q": True,
            "dueling": True,
            "n_step": 3,
            "atom_size": 51,
            "v_min": -10.0,
            "v_max": 10.0,
            "adapt_frequency": 1000,
            "adaptation_threshold": 0.01,
            "min_hidden_dim": 64,
            "max_hidden_dim": 512,
            "min_atom_size": 21,
            "max_atom_size": 101,
        }

        # Update configuration if provided
        if config is not None:
            self.config.update(config)

        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Convert parameters to appropriate types
        buffer_size = int(self.config["buffer_size"])
        hidden_dim = int(self.config["hidden_dim"])
        atom_size = int(self.config["atom_size"])
        n_step = int(self.config["n_step"])
        
        # Initialize replay buffer
        self.memory = PrioritizedReplayBuffer(buffer_size, self.config["alpha"])
        
        # Initialize n-step buffer
        self.n_step_buffer = deque(maxlen=n_step)
        
        # Initialize networks with converted parameters
        self.online_network = DuelingNetwork(
            state_dim, 
            action_dim, 
            hidden_dim,
            bool(self.config["noisy"]),
            atom_size,
            self.config["v_min"],
            self.config["v_max"]
        )
        
        # Metrics tracking
        self.metrics = {
            "loss": [],
            "reward": [],
            "epsilon": [],
            "q_value": []
        }
        
        # Initialize target network (deep copy of online network)
        self.target_network = copy.deepcopy(self.online_network)
        
        # Initialize epsilon for exploration
        self.epsilon = self.config["epsilon_start"]
        
        # Step counter
        self.steps = 0
        
        # Episode stats
        self.episode_rewards = deque(maxlen=100)
        
        # Adaptation tracking
        self.last_adaptation_step = 0
        self.adaptation_metrics = {
            "avg_reward": [],
            "avg_loss": [],
            "network_size": []
        }
        
        logger.info(f"AdaptiveRainbowAgent initialized with state_dim={state_dim}, action_dim={action_dim}")

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action based on the current state.

        Args:
            state: Current state
            evaluate: Whether to evaluate (no exploration) or train

        Returns:
            Selected action index
        """
        # Ensure state is in the right shape
        if len(state.shape) == 1:
            state = state.reshape(1, -1)

        # Use epsilon-greedy policy during training if not using noisy networks
        if (
            not evaluate
            and not self.config["noisy"]
            and random.random() < self.epsilon
        ):
            return int(random.randint(0, self.action_dim - 1))

        # Get Q-values
        q_values = self.online_network.get_q_values(state)

        # Select action with highest Q-value
        return int(np.argmax(q_values[0]))

    def update_epsilon(self) -> None:
        """Update epsilon value for exploration."""
        self.epsilon = max(
            self.config["epsilon_end"],
            self.config["epsilon_start"] - self.steps / self.config["epsilon_decay"],
        )
        self.metrics["epsilon"].append(self.epsilon)

    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add experience to replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # If n-step buffer is not full, return
        if len(self.n_step_buffer) < self.config["n_step"]:
            return

        # Calculate n-step reward and get resulting state
        reward_n, next_state_n, done_n = self._get_n_step_info()

        # Get the initial state and action
        state_0, action_0, _, _, _ = self.n_step_buffer[0]

        # Add n-step transition to replay buffer
        experience = Experience(state_0, action_0, reward_n, next_state_n, done_n)
        self.memory.add(experience)

        # Track rewards
        self.metrics["reward"].append(reward)

    def _get_n_step_info(self) -> Tuple[float, np.ndarray, bool]:
        """
        Get n-step reward, next state, and done flag.

        Returns:
            Tuple of (n-step reward, next state, done flag)
        """
        # Get info from n-step buffer
        reward, next_state, done = 0.0, None, False

        for i in range(self.config["n_step"]):
            r = self.n_step_buffer[i][2]
            reward += r * (self.config["gamma"] ** i)

            if self.n_step_buffer[i][4]:
                done = True
                break

        # Get next state from the latest transition
        next_state = self.n_step_buffer[-1][3]

        return reward, next_state, done

    def update(self) -> Dict[str, float]:
        """
        Update the agent's parameters.

        Returns:
            Dictionary of training metrics
        """
        if len(self.memory) < self.config["batch_size"]:
            return {"loss": 0.0, "q_mean": 0.0, "q_max": 0.0}

        # Sample batch
        experiences, indices, weights = self.memory.sample(
            self.config["batch_size"],
            beta=self.config["beta"]
            + self.steps * self.config["beta_increment"],
        )

        # Unpack experiences
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.next_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences])

        # Reset noise for all networks
        if self.config["noisy"]:
            self.online_network.reset_noise()
            self.target_network.reset_noise()

        # Get current state distributions
        current_dist = self.online_network.forward(states)

        # Get next state distributions
        if self.config["double_q"]:
            # Double Q-learning
            next_actions = np.argmax(
                self.online_network.get_q_values(next_states), axis=1
            )
            next_dist = self.target_network.forward(next_states)
            next_dist = next_dist[np.arange(self.config["batch_size"]), next_actions]
        else:
            # Standard Q-learning
            next_dist = self.target_network.forward(next_states)
            next_dist = next_dist[
                np.arange(self.config["batch_size"]),
                np.argmax(
                    np.sum(
                        next_dist * self.online_network.supports.reshape(1, 1, -1),
                        axis=2,
                    ),
                    axis=1,
                ),
            ]

        # Calculate target distribution
        target_dist = self._categorical_projection(rewards, next_dist, dones)

        # Calculate loss
        current_dist = current_dist[np.arange(self.config["batch_size"]), actions]
        loss = -np.sum(target_dist * np.log(current_dist + 1e-8), axis=1)
        loss = np.mean(weights * loss)

        # Calculate TD errors for prioritized replay
        td_error = np.abs(
            np.sum(target_dist * self.online_network.supports, axis=1)
            - np.sum(current_dist * self.online_network.supports, axis=1)
        )

        # Update priorities
        self.memory.update_priorities(indices, td_error)

        # Update online network (simplified, in a real implementation this would use gradient descent)
        # Here we just add some noise to simulate learning
        self._update_network(loss)

        # Update step counter
        self.steps += 1

        # Update epsilon
        self.update_epsilon()

        # Track metrics
        q_values = self.online_network.get_q_values(states)
        metrics = {
            "loss": float(loss),
            "q_mean": float(np.mean(q_values)),
            "q_max": float(np.max(q_values)),
        }

        self.metrics["loss"].append(metrics["loss"])
        self.metrics["q_value"].append(metrics["q_mean"])

        # Update target network if needed
        if self.steps % self.config["target_update"] == 0:
            self.target_network = copy.deepcopy(self.online_network)

        # Check if adaptation is needed
        if self.steps % self.config["adapt_frequency"] == 0:
            self.check_adaptation()

        return metrics

    def _categorical_projection(
        self, rewards: np.ndarray, next_dist: np.ndarray, dones: np.ndarray
    ) -> np.ndarray:
        """
        Project categorical distribution for distributional RL.

        Args:
            rewards: Batch of rewards
            next_dist: Next state distributions
            dones: Batch of done flags

        Returns:
            Projected categorical distribution
        """
        batch_size = len(rewards)

        # Initialize target distribution
        target_dist = np.zeros((batch_size, self.config["atom_size"]))

        # Get supports
        supports = self.online_network.supports
        delta_z = self.online_network.delta_z
        v_min = self.config["v_min"]
        v_max = self.config["v_max"]

        # For each sample in the batch
        for i in range(batch_size):
            if dones[i]:
                # If done, target is just the reward
                tz = min(v_max, max(v_min, rewards[i]))
                bj = int((tz - v_min) / delta_z)

                # Handle edge case
                if bj == self.config["atom_size"]:
                    bj = self.config["atom_size"] - 1

                target_dist[i][bj] = 1.0
            else:
                # If not done, project the distribution
                for j in range(self.config["atom_size"]):
                    tz = min(
                        v_max,
                        max(v_min, rewards[i] + self.config["gamma"] * supports[j]),
                    )
                    bj = int((tz - v_min) / delta_z)

                    # Handle edge case
                    if bj == self.config["atom_size"]:
                        bj = self.config["atom_size"] - 1

                    # Calculate proportion
                    l = (tz - v_min) / delta_z - bj
                    u = 1.0 - l

                    # Update target distribution
                    target_dist[i][bj] += next_dist[i][j] * u
                    if bj + 1 < self.config["atom_size"]:
                        target_dist[i][bj + 1] += next_dist[i][j] * l

        return target_dist

    def _update_network(self, loss: float) -> None:
        """
        Update the network parameters.

        Args:
            loss: Loss value
        """
        # In a real implementation, this would use gradient descent
        # For this simplified version, we just add some noise to simulate learning

        # The magnitude of the noise is proportional to the loss
        noise_scale = 0.01 * loss

        # Add noise to the network parameters
        if self.config["noisy"]:
            # Feature layer
            self.online_network.feature_mu += np.random.normal(
                0, noise_scale, self.online_network.feature_mu.shape
            )
            self.online_network.feature_bias_mu += np.random.normal(
                0, noise_scale, self.online_network.feature_bias_mu.shape
            )

            # Value stream
            self.online_network.value_hidden.weight_mu += np.random.normal(
                0, noise_scale, self.online_network.value_hidden.weight_mu.shape
            )
            self.online_network.value_hidden.bias_mu += np.random.normal(
                0, noise_scale, self.online_network.value_hidden.bias_mu.shape
            )
            self.online_network.value_out.weight_mu += np.random.normal(
                0, noise_scale, self.online_network.value_out.weight_mu.shape
            )
            self.online_network.value_out.bias_mu += np.random.normal(
                0, noise_scale, self.online_network.value_out.bias_mu.shape
            )

            # Advantage stream
            self.online_network.advantage_hidden.weight_mu += np.random.normal(
                0, noise_scale, self.online_network.advantage_hidden.weight_mu.shape
            )
            self.online_network.advantage_hidden.bias_mu += np.random.normal(
                0, noise_scale, self.online_network.advantage_hidden.bias_mu.shape
            )
            self.online_network.advantage_out.weight_mu += np.random.normal(
                0, noise_scale, self.online_network.advantage_out.weight_mu.shape
            )
            self.online_network.advantage_out.bias_mu += np.random.normal(
                0, noise_scale, self.online_network.advantage_out.bias_mu.shape
            )
        else:
            # Feature layer
            self.online_network.feature += np.random.normal(
                0, noise_scale, self.online_network.feature.shape
            )
            self.online_network.feature_bias += np.random.normal(
                0, noise_scale, self.online_network.feature_bias.shape
            )

            # Value stream
            self.online_network.value_hidden += np.random.normal(
                0, noise_scale, self.online_network.value_hidden.shape
            )
            self.online_network.value_hidden_bias += np.random.normal(
                0, noise_scale, self.online_network.value_hidden_bias.shape
            )
            self.online_network.value_out += np.random.normal(
                0, noise_scale, self.online_network.value_out.shape
            )
            self.online_network.value_out_bias += np.random.normal(
                0, noise_scale, self.online_network.value_out_bias.shape
            )

            # Advantage stream
            self.online_network.advantage_hidden += np.random.normal(
                0, noise_scale, self.online_network.advantage_hidden.shape
            )
            self.online_network.advantage_hidden_bias += np.random.normal(
                0, noise_scale, self.online_network.advantage_hidden_bias.shape
            )
            self.online_network.advantage_out += np.random.normal(
                0, noise_scale, self.online_network.advantage_out.shape
            )
            self.online_network.advantage_out_bias += np.random.normal(
                0, noise_scale, self.online_network.advantage_out_bias.shape
            )

    def check_adaptation(self) -> None:
        """
        Check if adaptation is needed based on performance metrics.
        """
        # Calculate average reward over recent history
        avg_reward = (
            np.mean(self.metrics["reward"][-100:])
            if len(self.metrics["reward"]) > 100
            else 0
        )
        self.adaptation_metrics["avg_reward"].append(avg_reward)

        # Check if adaptation is needed
        if len(self.adaptation_metrics["avg_reward"]) >= 50:
            recent_avg = np.mean(
                list(self.adaptation_metrics["avg_reward"])[-50:]
            )
            older_avg = np.mean(
                list(self.adaptation_metrics["avg_reward"])[: -50]
            )

            # If performance is declining, adapt
            if older_avg - recent_avg > self.config["adaptation_threshold"]:
                self.adapt()

    def adapt(self) -> None:
        """
        Adapt the agent based on performance.
        """
        self.last_adaptation_step = self.steps

        # Adapt based on the current level
        if self.steps % self.config["adapt_frequency"] == 0:
            # Level 0: Increase exploration
            self.epsilon = min(1.0, self.epsilon * 2.0)
            logger.info(
                f"Adaptation #{self.steps // self.config['adapt_frequency']}: Increased exploration (epsilon={self.epsilon})"
            )
        else:
            # Level 1: Adjust network architecture
            self._adapt_architecture()
            logger.info(
                f"Adaptation #{self.steps // self.config['adapt_frequency']}: Adjusted network architecture"
            )

    def _adapt_architecture(self) -> None:
        """Adapt the network architecture."""
        # Randomly select an architecture component to adapt
        component = random.choice(
            [
                "hidden_dim",
                "atom_size",
                "n_step",
                "noisy",
                "double_q",
                "dueling",
            ]
        )

        if component == "hidden_dim":
            # Adjust hidden dimension
            new_hidden_dim = self.config["hidden_dim"]
            if random.random() < 0.5:
                new_hidden_dim = max(self.config["min_hidden_dim"], new_hidden_dim // 2)
            else:
                new_hidden_dim = min(self.config["max_hidden_dim"], new_hidden_dim * 2)

            self.config["hidden_dim"] = new_hidden_dim
            logger.info(f"Adapted hidden_dim to {new_hidden_dim}")
        elif component == "atom_size":
            # Adjust atom size for distributional RL
            new_atom_size = self.config["atom_size"]
            if random.random() < 0.5:
                new_atom_size = max(self.config["min_atom_size"], new_atom_size - 10)
            else:
                new_atom_size = min(self.config["max_atom_size"], new_atom_size + 10)

            self.config["atom_size"] = new_atom_size
            logger.info(f"Adapted atom_size to {new_atom_size}")
        elif component == "n_step":
            # Adjust n-step learning
            new_n_step = self.config["n_step"]
            if random.random() < 0.5:
                new_n_step = max(1, new_n_step - 1)
            else:
                new_n_step = min(10, new_n_step + 1)

            self.config["n_step"] = new_n_step
            self.n_step_buffer = deque(maxlen=new_n_step)
            logger.info(f"Adapted n_step to {new_n_step}")
        elif component == "noisy":
            # Toggle noisy networks
            self.config["noisy"] = not self.config["noisy"]
            logger.info(
                f"Toggled noisy to {self.config['noisy']}"
            )
        elif component == "double_q":
            # Toggle double Q-learning
            self.config["double_q"] = not self.config["double_q"]
            logger.info(f"Toggled double_q to {self.config['double_q']}")
        elif component == "dueling":
            # Toggle dueling network
            self.config["dueling"] = not self.config["dueling"]
            logger.info(
                f"Toggled dueling to {self.config['dueling']}"
            )

        # Update configuration
        self.config.update(self.config)

        # Reinitialize networks with new architecture
        self.online_network = DuelingNetwork(
            self.state_dim,
            self.action_dim,
            self.config["hidden_dim"],
            self.config["noisy"],
            self.config["atom_size"],
            self.config["v_min"],
            self.config["v_max"],
        )

        self.target_network = copy.deepcopy(self.online_network)

    def save(self, path: str) -> None:
        """
        Save the agent's state.

        Args:
            path: Path to save the state
        """
        # Create state dictionary
        state = {
            "config": self.config,
            "steps": self.steps,
            "epsilon": self.epsilon,
            "metrics": self.metrics,
            "adaptation_counter": self.steps // self.config["adapt_frequency"],
            "current_adaptation_level": self.steps // self.config["adapt_frequency"],
            "current_architecture": {
                "hidden_dim": self.config["hidden_dim"],
                "atom_size": self.config["atom_size"],
                "n_step": self.config["n_step"],
                "noisy": self.config["noisy"],
                "double_q": self.config["double_q"],
                "dueling": self.config["dueling"],
            },
        }

        # Save state
        with open(f"{path}_state.json", "w") as f:
            json.dump(state, f)

        logger.info(f"Saved agent state to {path}_state.json")

    def load(self, path: str) -> None:
        """
        Load the agent's state.

        Args:
            path: Path to load the state from
        """
        # Load state
        with open(f"{path}_state.json", "r") as f:
            state = json.load(f)

        # Update state
        self.config.update(state["config"])
        self.steps = state["steps"]
        self.epsilon = state["epsilon"]
        self.metrics = state["metrics"]
        self.config.update(self.config)

        # Reinitialize networks with loaded architecture
        self.online_network = DuelingNetwork(
            self.state_dim,
            self.action_dim,
            self.config["hidden_dim"],
            self.config["noisy"],
            self.config["atom_size"],
            self.config["v_min"],
            self.config["v_max"],
        )

        self.target_network = copy.deepcopy(self.online_network)

        # Reinitialize replay buffer
        self.memory = PrioritizedReplayBuffer(
            self.config["buffer_size"], self.config["alpha"]
        )

        # Reinitialize n-step buffer
        self.n_step_buffer = deque(maxlen=self.config["n_step"])

        logger.info(f"Loaded agent state from {path}_state.json")

    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get training metrics.

        Returns:
            Dictionary of training metrics
        """
        return self.metrics

    def train(
        self,
        env,
        num_episodes: int = 1000,
        max_steps: int = 1000,
        evaluate_freq: int = 10,
    ) -> Dict[str, List[float]]:
        """
        Train the agent using the provided environment.

        Args:
            env: Training environment
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            evaluate_freq: Frequency of evaluation episodes

        Returns:
            Training metrics
        """
        episode_rewards = []
        evaluation_rewards = []

        for episode in range(int(num_episodes)):
            state = env.reset()
            episode_reward = 0
            done = False
            step = 0

            while not done and step < int(max_steps):
                # Select action
                action = self.select_action(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Store transition
                self.add_experience(state, action, reward, next_state, done)
                
                # Update state
                state = next_state
                episode_reward += reward
                
                # Update agent
                self.update()
                
                step += 1

            # Track episode reward
            episode_rewards.append(episode_reward)
            
            # Evaluate periodically
            if episode % int(evaluate_freq) == 0:
                eval_reward = self.evaluate(env)
                evaluation_rewards.append(eval_reward)
                logger.info(f"Episode {episode}: Reward={episode_reward}, Eval={eval_reward}")

        return {
            "episode_rewards": episode_rewards,
            "evaluation_rewards": evaluation_rewards,
            "loss": self.metrics["loss"],
            "q_values": self.metrics["q_value"],
            "epsilon": self.metrics["epsilon"]
        }

    def evaluate(self, env, num_episodes: int = 10) -> float:
        """
        Evaluate the agent in the given environment.

        Args:
            env: Evaluation environment
            num_episodes: Number of evaluation episodes

        Returns:
            Average reward over evaluation episodes
        """
        total_reward = 0.0
        
        for _ in range(int(num_episodes)):
            state = env.reset()
            episode_reward = 0.0
            done = False
            
            while not done:
                # Select action without exploration
                action = self.select_action(state, evaluate=True)
                
                # Take action
                next_state, reward, done, _ = env.step(action)
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / int(num_episodes)
