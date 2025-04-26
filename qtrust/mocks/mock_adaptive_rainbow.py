#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Mock Adaptive Rainbow Agent
This module implements a PyTorch-free version of the adaptive rainbow agent
for testing purposes.
"""

import numpy as np
import random
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)


class MockAdaptiveRainbowAgent:
    """
    Mock implementation of Adaptive Rainbow DQN Agent for testing.

    This class provides a simplified implementation that mimics the behavior
    of the actual Adaptive Rainbow DQN Agent without requiring PyTorch.
    """

    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any] = None):
        """
        Initialize the mock Adaptive Rainbow DQN agent.

        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            config: Configuration parameters
        """
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Default configuration
        self.config = {
            "learning_rate": 0.0001,
            "gamma": 0.99,
            "batch_size": 32,
            "replay_buffer_size": 10000,
            "target_update_frequency": 100,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 10000,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_beta_increment": 0.001,
            "dueling_network": True,
            "double_q": True,
            "noisy_nets": False,
            "n_step": 3,
            "v_min": -10.0,
            "v_max": 10.0,
            "atom_size": 51,
            "adaptation_frequency": 1000,
            "adaptation_threshold": 0.1,
        }

        # Update configuration if provided
        if config:
            self.config.update(config)

        # Initialize replay buffer
        self.replay_buffer = []
        self.priorities = []

        # Initialize step counter
        self.steps = 0

        # Initialize epsilon for exploration
        self.epsilon = self.config["epsilon_start"]

        # Initialize mock network weights
        self.online_weights = np.random.randn(state_dim, action_dim)
        self.target_weights = self.online_weights.copy()

        # Initialize training metrics
        self.metrics = {
            "loss": [0.0],  # Initialize with default values
            "q_values": [0.0],
            "rewards": [0.0],
            "epsilon": [self.epsilon],
        }

        # Adaptation metrics
        self.performance_history = deque(maxlen=100)
        self.adaptation_counter = 0
        self.current_adaptation_level = 0

        logger.info(
            f"Initialized MockAdaptiveRainbowAgent with state_dim={state_dim}, action_dim={action_dim}"
        )

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        Select an action based on the current state.

        Args:
            state: Current state
            evaluate: Whether to evaluate (no exploration) or train

        Returns:
            Selected action index
        """
        # Use epsilon-greedy policy during training
        if not evaluate and random.random() < self.epsilon:
            return int(random.randint(0, self.action_dim - 1))

        # Mock Q-value computation
        q_values = np.dot(state, self.online_weights)
        return int(np.argmax(q_values))

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
        # Add experience to replay buffer
        experience = (state, action, reward, next_state, done)

        if len(self.replay_buffer) >= self.config["replay_buffer_size"]:
            self.replay_buffer.pop(0)
            self.priorities.pop(0)

        self.replay_buffer.append(experience)

        # Initialize with max priority
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

        # Track rewards
        self.metrics["rewards"].append(reward)

    def sample_batch(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch from the replay buffer.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights)
        """
        batch_size = min(self.config["batch_size"], len(self.replay_buffer))

        # Convert priorities to probabilities
        probs = np.array(self.priorities) ** self.config["prioritized_replay_alpha"]
        probs /= probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(len(self.replay_buffer), batch_size, p=probs)

        # Calculate importance sampling weights
        beta = min(
            1.0,
            self.config["prioritized_replay_beta"]
            + self.steps * self.config["prioritized_replay_beta_increment"],
        )
        weights = (len(self.replay_buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        # Extract experiences
        batch = [self.replay_buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones)
        weights = np.array(weights)

        return states, actions, rewards, next_states, dones, weights

    def update(self) -> Dict[str, float]:
        """
        Update the agent's parameters.

        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.config["batch_size"]:
            return {"loss": 0.0, "q_mean": 0.0, "q_max": 0.0}

        # Sample batch
        states, actions, rewards, next_states, dones, weights = self.sample_batch()

        # Mock loss computation
        current_q = np.sum(
            np.dot(states, self.online_weights) * np.eye(self.action_dim)[actions],
            axis=1,
        )

        # Mock target computation
        if self.config["double_q"]:
            # Double Q-learning
            next_actions = np.argmax(np.dot(next_states, self.online_weights), axis=1)
            next_q = np.sum(
                np.dot(next_states, self.target_weights)
                * np.eye(self.action_dim)[next_actions],
                axis=1,
            )
        else:
            # Standard Q-learning
            next_q = np.max(np.dot(next_states, self.target_weights), axis=1)

        # Calculate target Q-values
        target_q = rewards + (1 - dones) * self.config["gamma"] * next_q

        # Mock TD error
        td_error = np.abs(current_q - target_q)

        # Mock loss
        loss = np.mean(weights * (td_error**2))

        # Update priorities
        for idx, error in zip(range(len(td_error)), td_error):
            self.priorities[idx] = (
                error + 1e-6
            )  # Add small constant to avoid zero priority

        # Mock network update (just add some noise to weights)
        update_noise = np.random.randn(self.state_dim, self.action_dim) * 0.01
        self.online_weights -= update_noise * loss

        # Update step counter
        self.steps += 1

        # Update epsilon
        self.update_epsilon()

        # Track metrics
        metrics = {
            "loss": float(loss),
            "q_mean": float(np.mean(current_q)),
            "q_max": float(np.max(current_q)),
        }

        self.metrics["loss"].append(metrics["loss"])
        self.metrics["q_values"].append(metrics["q_mean"])

        # Update target network if needed
        if self.steps % self.config["target_update_frequency"] == 0:
            self.target_weights = self.online_weights.copy()

        # Check if adaptation is needed
        if self.steps % self.config["adaptation_frequency"] == 0:
            self.check_adaptation()

        return metrics

    def check_adaptation(self) -> None:
        """
        Check if adaptation is needed based on performance metrics.
        """
        # In a real implementation, this would analyze performance and adapt the agent
        # For testing, we just randomly decide to adapt

        # Calculate average reward over recent history
        avg_reward = (
            np.mean(self.metrics["rewards"][-100:])
            if len(self.metrics["rewards"]) > 100
            else 0
        )
        self.performance_history.append(avg_reward)

        # Check if adaptation is needed
        if len(self.performance_history) >= 10:
            recent_avg = np.mean(list(self.performance_history)[-10:])
            older_avg = np.mean(list(self.performance_history)[:-10])

            # If performance is declining, adapt
            if older_avg - recent_avg > self.config["adaptation_threshold"]:
                self.adapt()

    def adapt(self) -> None:
        """
        Adapt the agent based on performance.
        """
        # In a real implementation, this would modify the agent's architecture or hyperparameters
        # For testing, we just log the adaptation

        self.adaptation_counter += 1
        self.current_adaptation_level = (self.current_adaptation_level + 1) % 3

        # Mock adaptation by slightly modifying the weights
        adaptation_noise = np.random.randn(self.state_dim, self.action_dim) * 0.05
        self.online_weights += adaptation_noise
        self.target_weights = self.online_weights.copy()

        logger.info(
            f"Adapted agent to level {self.current_adaptation_level} (adaptation #{self.adaptation_counter})"
        )

    def save(self, path: str) -> None:
        """
        Mock save method.

        Args:
            path: Path to save the model
        """
        logger.info(f"Mock saving adaptive model to {path}")

    def load(self, path: str) -> None:
        """
        Mock load method.

        Args:
            path: Path to load the model from
        """
        logger.info(f"Mock loading adaptive model from {path}")

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
        Train the agent on the given environment.

        Args:
            env: Environment to train on
            num_episodes: Number of episodes to train for
            max_steps: Maximum steps per episode
            evaluate_freq: Frequency of evaluation

        Returns:
            Dictionary of training metrics
        """
        episode_rewards = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps):
                # Select action
                action = self.select_action(state)

                # Take action
                next_state, reward, done, _ = env.step(action)

                # Add experience
                self.add_experience(state, action, reward, next_state, done)

                # Update agent
                self.update()

                # Update state and reward
                state = next_state
                episode_reward += reward

                if done:
                    break

            # Track episode reward
            episode_rewards.append(episode_reward)

            # Evaluate if needed
            if episode % evaluate_freq == 0:
                eval_reward = self.evaluate(env, num_episodes=5)
                logger.info(
                    f"Episode {episode}: Training reward = {episode_reward}, Evaluation reward = {eval_reward}"
                )

        return {
            "episode_rewards": episode_rewards,
            "loss": self.metrics["loss"],
            "q_values": self.metrics["q_values"],
            "epsilon": self.metrics["epsilon"],
        }

    def evaluate(self, env, num_episodes: int = 10) -> float:
        """
        Evaluate the agent on the given environment.

        Args:
            env: Environment to evaluate on
            num_episodes: Number of episodes to evaluate for

        Returns:
            Average reward over episodes
        """
        total_reward = 0

        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False

            while not done:
                # Select action (no exploration)
                action = self.select_action(state, evaluate=True)

                # Take action
                next_state, reward, done, _ = env.step(action)

                # Update state and reward
                state = next_state
                episode_reward += reward

            total_reward += episode_reward

        return total_reward / num_episodes
