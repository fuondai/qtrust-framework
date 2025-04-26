#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Privacy-Preserving Federated Learning
This module implements federated learning with privacy-preserving model aggregation,
partial model sharing, and secure gradient exchange.
"""

import os
import time
import threading
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict, deque
from typing import Dict, List, Tuple, Set, Optional, Any, Callable, Union
import copy
import hashlib
import json

from ..crypto_utils import CryptoManager
from ..secure_channel import SecureChannel


class DifferentialPrivacy:
    """
    Implements differential privacy techniques for model updates
    """

    def __init__(
        self, epsilon: float = 1.0, delta: float = 1e-5, clip_norm: float = 1.0
    ):
        """
        Initialize differential privacy

        Args:
            epsilon: Privacy budget (lower = more privacy)
            delta: Probability of privacy breach
            clip_norm: Maximum L2 norm for gradient clipping
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm

    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add calibrated noise to tensor for differential privacy

        Args:
            tensor: Input tensor

        Returns:
            Tensor with added noise
        """
        # Calculate sensitivity based on clipping
        sensitivity = 2.0 * self.clip_norm

        # Calculate noise scale using Gaussian mechanism
        noise_scale = (
            sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon
        )

        # Generate Gaussian noise
        noise = torch.randn_like(tensor) * noise_scale

        # Add noise to tensor
        noisy_tensor = tensor + noise

        return noisy_tensor

    def clip_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Clip gradients to limit sensitivity

        Args:
            gradients: List of gradient tensors

        Returns:
            Clipped gradients
        """
        # Calculate total L2 norm
        total_norm = 0.0
        for grad in gradients:
            if grad is not None:
                total_norm += grad.norm(2).item() ** 2
        total_norm = math.sqrt(total_norm)

        # Calculate scaling factor
        scale = self.clip_norm / (total_norm + 1e-6)
        if scale < 1.0:
            # Apply clipping
            clipped_gradients = [
                grad * scale if grad is not None else None for grad in gradients
            ]
            return clipped_gradients
        else:
            # No clipping needed
            return gradients

    def privatize_model_update(self, model_update: OrderedDict) -> OrderedDict:
        """
        Apply differential privacy to model update

        Args:
            model_update: Model parameter update

        Returns:
            Privatized model update
        """
        privatized_update = OrderedDict()

        for key, param in model_update.items():
            # Add noise to parameter
            privatized_update[key] = self.add_noise(param)

        return privatized_update


class SecureAggregation:
    """
    Implements secure aggregation protocol for federated learning
    """

    def __init__(
        self, node_id: str, crypto_manager: CryptoManager, config: Dict[str, Any] = None
    ):
        """
        Initialize secure aggregation

        Args:
            node_id: Node identifier
            crypto_manager: Cryptographic manager
            config: Configuration dictionary
        """
        self.node_id = node_id
        self.crypto = crypto_manager
        self.config = config or {}

        # Default configuration
        self.threshold = self.config.get(
            "threshold", 0.7
        )  # Minimum fraction of nodes required
        self.timeout = self.config.get("timeout", 60.0)  # Timeout in seconds

        # Secure channel for communication
        self.secure_channel = SecureChannel(
            node_id, crypto_manager, self.config.get("channel_config", {})
        )

        # State for aggregation rounds
        self.rounds = {}  # round_id -> round_state

        # Lock for thread safety
        self.lock = threading.RLock()

    def start(self):
        """
        Start secure aggregation
        """
        self.secure_channel.start()

        # Register message handlers
        self.secure_channel.register_handler(
            "mask_commitment", self._handle_mask_commitment
        )
        self.secure_channel.register_handler("masked_input", self._handle_masked_input)
        self.secure_channel.register_handler("mask_reveal", self._handle_mask_reveal)

    def stop(self):
        """
        Stop secure aggregation
        """
        self.secure_channel.stop()

    def aggregate(
        self, round_id: str, local_model: OrderedDict, peers: List[str]
    ) -> Optional[OrderedDict]:
        """
        Perform secure aggregation with peers

        Args:
            round_id: Unique identifier for aggregation round
            local_model: Local model parameters
            peers: List of peer node identifiers

        Returns:
            Aggregated model parameters or None if aggregation failed
        """
        with self.lock:
            # Initialize round state
            self.rounds[round_id] = {
                "peers": set(peers),
                "local_model": local_model,
                "masks": {},  # peer_id -> mask
                "mask_commitments": {},  # peer_id -> commitment
                "masked_inputs": {},  # peer_id -> masked_input
                "revealed_masks": {},  # peer_id -> revealed_mask
                "start_time": time.time(),
                "completed": False,
                "result": None,
                "event": threading.Event(),
            }

        try:
            # Phase 1: Generate and share mask commitments
            if not self._phase_mask_commitment(round_id, peers):
                return None

            # Phase 2: Share masked inputs
            if not self._phase_masked_input(round_id, local_model, peers):
                return None

            # Phase 3: Reveal masks
            if not self._phase_mask_reveal(round_id, peers):
                return None

            # Wait for aggregation to complete
            round_state = self.rounds[round_id]
            if not round_state["event"].wait(self.timeout):
                print(f"Secure aggregation timed out for round {round_id}")
                return None

            # Return aggregated result
            return round_state["result"]

        except Exception as e:
            print(f"Error in secure aggregation: {e}")
            return None
        finally:
            # Clean up round state
            with self.lock:
                if round_id in self.rounds:
                    del self.rounds[round_id]

    def _phase_mask_commitment(self, round_id: str, peers: List[str]) -> bool:
        """
        Phase 1: Generate and share mask commitments

        Args:
            round_id: Round identifier
            peers: List of peer node identifiers

        Returns:
            True if phase completed successfully, False otherwise
        """
        # Generate random mask for each peer
        masks = {}
        for peer_id in peers:
            # Generate random mask with same structure as model
            mask = OrderedDict()
            for key, param in self.rounds[round_id]["local_model"].items():
                mask[key] = torch.randn_like(param)

            masks[peer_id] = mask

        # Store masks
        with self.lock:
            self.rounds[round_id]["masks"] = masks

        # Generate commitments for masks
        for peer_id, mask in masks.items():
            # Serialize mask
            mask_bytes = self._serialize_model(mask)

            # Generate commitment
            commitment = self.crypto.hash_data(mask_bytes)

            # Send commitment to peer
            message = {"round_id": round_id, "commitment": commitment}

            self.secure_channel.send_message(peer_id, "mask_commitment", message)

        # Wait for commitments from all peers
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            with self.lock:
                received = len(self.rounds[round_id]["mask_commitments"])
                total = len(peers)

                if received >= total * self.threshold:
                    return True

            time.sleep(0.1)

        print(f"Timeout waiting for mask commitments in round {round_id}")
        return False

    def _phase_masked_input(
        self, round_id: str, local_model: OrderedDict, peers: List[str]
    ) -> bool:
        """
        Phase 2: Share masked inputs

        Args:
            round_id: Round identifier
            local_model: Local model parameters
            peers: List of peer node identifiers

        Returns:
            True if phase completed successfully, False otherwise
        """
        # Calculate masked input
        masked_input = OrderedDict()

        for key, param in local_model.items():
            # Start with local parameter
            masked_param = param.clone()

            # Add masks for each peer
            for peer_id in peers:
                if peer_id in self.rounds[round_id]["masks"]:
                    mask = self.rounds[round_id]["masks"][peer_id][key]
                    masked_param += mask

            masked_input[key] = masked_param

        # Send masked input to all peers
        for peer_id in peers:
            message = {
                "round_id": round_id,
                "masked_input": self._serialize_model(masked_input),
            }

            self.secure_channel.send_message(peer_id, "masked_input", message)

        # Wait for masked inputs from all peers
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            with self.lock:
                received = len(self.rounds[round_id]["masked_inputs"])
                total = len(peers)

                if received >= total * self.threshold:
                    return True

            time.sleep(0.1)

        print(f"Timeout waiting for masked inputs in round {round_id}")
        return False

    def _phase_mask_reveal(self, round_id: str, peers: List[str]) -> bool:
        """
        Phase 3: Reveal masks

        Args:
            round_id: Round identifier
            peers: List of peer node identifiers

        Returns:
            True if phase completed successfully, False otherwise
        """
        # Send mask reveals to all peers
        for peer_id in peers:
            if peer_id in self.rounds[round_id]["mask_commitments"]:
                # Get mask for this peer
                mask = self.rounds[round_id]["masks"].get(peer_id)

                if mask:
                    message = {
                        "round_id": round_id,
                        "mask": self._serialize_model(mask),
                    }

                    self.secure_channel.send_message(peer_id, "mask_reveal", message)

        # Wait for mask reveals from all peers
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            with self.lock:
                received = len(self.rounds[round_id]["revealed_masks"])
                total = len(peers)

                if received >= total * self.threshold:
                    # Perform final aggregation
                    self._finalize_aggregation(round_id)
                    return True

            time.sleep(0.1)

        print(f"Timeout waiting for mask reveals in round {round_id}")
        return False

    def _finalize_aggregation(self, round_id: str) -> None:
        """
        Finalize aggregation by combining masked inputs and removing masks

        Args:
            round_id: Round identifier
        """
        with self.lock:
            round_state = self.rounds[round_id]

            # Get participating peers
            participating_peers = set(round_state["masked_inputs"].keys()) & set(
                round_state["revealed_masks"].keys()
            )

            if not participating_peers:
                print(f"No participating peers in round {round_id}")
                round_state["completed"] = True
                round_state["event"].set()
                return

            # Initialize aggregated model
            aggregated_model = OrderedDict()
            for key, param in round_state["local_model"].items():
                aggregated_model[key] = torch.zeros_like(param)

            # Add all masked inputs
            for peer_id in participating_peers:
                masked_input = self._deserialize_model(
                    round_state["masked_inputs"][peer_id]
                )

                for key in aggregated_model:
                    if key in masked_input:
                        aggregated_model[key] += masked_input[key]

            # Add local masked input
            for key, param in round_state["local_model"].items():
                # Start with local parameter
                masked_param = param.clone()

                # Add masks for each peer
                for peer_id in participating_peers:
                    if peer_id in round_state["masks"]:
                        mask = round_state["masks"][peer_id][key]
                        masked_param += mask

                aggregated_model[key] += masked_param

            # Remove all masks
            for peer_id in participating_peers:
                # Remove masks from this peer
                if peer_id in round_state["revealed_masks"]:
                    revealed_mask = self._deserialize_model(
                        round_state["revealed_masks"][peer_id]
                    )

                    for key in aggregated_model:
                        if key in revealed_mask:
                            aggregated_model[key] -= revealed_mask[key]

                # Remove local masks for this peer
                if peer_id in round_state["masks"]:
                    for key in aggregated_model:
                        if key in round_state["masks"][peer_id]:
                            aggregated_model[key] -= round_state["masks"][peer_id][key]

            # Average the result
            num_participants = len(participating_peers) + 1  # +1 for local model
            for key in aggregated_model:
                aggregated_model[key] /= num_participants

            # Store result
            round_state["result"] = aggregated_model
            round_state["completed"] = True
            round_state["event"].set()

    def _handle_mask_commitment(self, peer_id: str, message: Dict[str, Any]) -> None:
        """
        Handle mask commitment message

        Args:
            peer_id: Peer node identifier
            message: Message data
        """
        round_id = message.get("round_id")
        commitment = message.get("commitment")

        if not round_id or not commitment:
            return

        with self.lock:
            if round_id in self.rounds:
                self.rounds[round_id]["mask_commitments"][peer_id] = commitment

    def _handle_masked_input(self, peer_id: str, message: Dict[str, Any]) -> None:
        """
        Handle masked input message

        Args:
            peer_id: Peer node identifier
            message: Message data
        """
        round_id = message.get("round_id")
        masked_input = message.get("masked_input")

        if not round_id or not masked_input:
            return

        with self.lock:
            if round_id in self.rounds:
                self.rounds[round_id]["masked_inputs"][peer_id] = masked_input

    def _handle_mask_reveal(self, peer_id: str, message: Dict[str, Any]) -> None:
        """
        Handle mask reveal message

        Args:
            peer_id: Peer node identifier
            message: Message data
        """
        round_id = message.get("round_id")
        mask = message.get("mask")

        if not round_id or not mask:
            return

        with self.lock:
            if round_id in self.rounds:
                # Verify mask against commitment
                if peer_id in self.rounds[round_id]["mask_commitments"]:
                    commitment = self.rounds[round_id]["mask_commitments"][peer_id]
                    mask_hash = self.crypto.hash_data(mask)

                    if mask_hash == commitment:
                        self.rounds[round_id]["revealed_masks"][peer_id] = mask
                    else:
                        print(
                            f"Mask verification failed for peer {peer_id} in round {round_id}"
                        )

    def _serialize_model(self, model: OrderedDict) -> bytes:
        """
        Serialize model parameters to bytes

        Args:
            model: Model parameters

        Returns:
            Serialized model bytes
        """
        serialized = {}
        for key, param in model.items():
            serialized[key] = param.cpu().numpy().tolist()

        return json.dumps(serialized).encode("utf-8")

    def _deserialize_model(self, serialized: bytes) -> OrderedDict:
        """
        Deserialize model parameters from bytes

        Args:
            serialized: Serialized model bytes

        Returns:
            Deserialized model parameters
        """
        data = json.loads(serialized.decode("utf-8"))
        model = OrderedDict()

        for key, param_list in data.items():
            param_array = np.array(param_list, dtype=np.float32)
            model[key] = torch.from_numpy(param_array)

        return model


class HomomorphicEncryption:
    """
    Implements simplified homomorphic encryption for model updates

    Note: This is a simplified implementation for demonstration purposes.
    In a production environment, use a proper homomorphic encryption library.
    """

    def __init__(self, key_size: int = 2048):
        """
        Initialize homomorphic encryption

        Args:
            key_size: Key size in bits
        """
        self.key_size = key_size
        self.public_key = None
        self.private_key = None

        # Generate keys
        self._generate_keys()

    def _generate_keys(self):
        """Generate encryption keys"""
        # In a real implementation, this would use a proper homomorphic encryption library
        # For demonstration, we'll use a simplified approach

        # Generate a random scaling factor as the private key
        self.private_key = random.uniform(0.5, 2.0)

        # Public key is the inverse of the private key
        self.public_key = 1.0 / self.private_key

    def encrypt(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Encrypt a tensor

        Args:
            tensor: Input tensor

        Returns:
            Encrypted tensor
        """
        # In a real implementation, this would use proper homomorphic encryption
        # For demonstration, we'll use a simplified approach with scaling and noise

        # Add small random noise
        noise = torch.randn_like(tensor) * 0.01

        # Scale the tensor (this allows for approximate addition)
        encrypted = tensor * self.public_key + noise

        return encrypted

    def decrypt(self, encrypted_tensor: torch.Tensor) -> torch.Tensor:
        """
        Decrypt a tensor

        Args:
            encrypted_tensor: Encrypted tensor

        Returns:
            Decrypted tensor
        """
        # Scale back using private key
        decrypted = encrypted_tensor * self.private_key

        return decrypted

    def aggregate_encrypted(
        self, encrypted_tensors: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Aggregate encrypted tensors

        Args:
            encrypted_tensors: List of encrypted tensors

        Returns:
            Aggregated encrypted tensor
        """
        # In homomorphic encryption, addition is preserved
        # So we can add encrypted tensors directly
        if not encrypted_tensors:
            return None

        result = torch.zeros_like(encrypted_tensors[0])
        for tensor in encrypted_tensors:
            result += tensor

        return result


class FederatedModel(nn.Module):
    """
    Base class for federated learning models
    """

    def __init__(self):
        """Initialize federated model"""
        super(FederatedModel, self).__init__()

    def get_parameters(self) -> OrderedDict:
        """
        Get model parameters

        Returns:
            Model parameters
        """
        return OrderedDict(self.named_parameters())

    def set_parameters(self, parameters: OrderedDict) -> None:
        """
        Set model parameters

        Args:
            parameters: Model parameters
        """
        for name, param in self.named_parameters():
            if name in parameters:
                param.data.copy_(parameters[name].data)

    def get_gradients(self) -> OrderedDict:
        """
        Get model gradients

        Returns:
            Model gradients
        """
        gradients = OrderedDict()
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        return gradients

    def set_gradients(self, gradients: OrderedDict) -> None:
        """
        Set model gradients

        Args:
            gradients: Model gradients
        """
        for name, param in self.named_parameters():
            if name in gradients:
                if param.grad is None:
                    param.grad = gradients[name].clone()
                else:
                    param.grad.copy_(gradients[name])


class FederatedLearningClient:
    """
    Client for federated learning
    """

    def __init__(
        self,
        node_id: str,
        model: FederatedModel,
        crypto_manager: CryptoManager,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize federated learning client

        Args:
            node_id: Node identifier
            model: Federated model
            crypto_manager: Cryptographic manager
            config: Configuration dictionary
        """
        self.node_id = node_id
        self.model = model
        self.crypto = crypto_manager
        self.config = config or {}

        # Default configuration
        self.local_epochs = self.config.get("local_epochs", 1)
        self.batch_size = self.config.get("batch_size", 32)
        self.learning_rate = self.config.get("learning_rate", 0.01)
        self.privacy_budget = self.config.get("privacy_budget", 1.0)
        self.use_differential_privacy = self.config.get(
            "use_differential_privacy", True
        )
        self.use_secure_aggregation = self.config.get("use_secure_aggregation", True)
        self.use_homomorphic_encryption = self.config.get(
            "use_homomorphic_encryption", False
        )

        # Create optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)

        # Create differential privacy module
        self.dp = DifferentialPrivacy(
            epsilon=self.privacy_budget,
            delta=self.config.get("privacy_delta", 1e-5),
            clip_norm=self.config.get("gradient_clip_norm", 1.0),
        )

        # Create secure aggregation module
        self.secure_aggregation = SecureAggregation(
            node_id, crypto_manager, self.config.get("secure_aggregation_config", {})
        )

        # Create homomorphic encryption module
        if self.use_homomorphic_encryption:
            self.homomorphic = HomomorphicEncryption(
                key_size=self.config.get("homomorphic_key_size", 2048)
            )
        else:
            self.homomorphic = None

        # Training data
        self.train_data = None

        # Round state
        self.current_round = 0
        self.round_history = []

    def start(self):
        """
        Start federated learning client
        """
        if self.use_secure_aggregation:
            self.secure_aggregation.start()

    def stop(self):
        """
        Stop federated learning client
        """
        if self.use_secure_aggregation:
            self.secure_aggregation.stop()

    def set_train_data(self, train_data: Any) -> None:
        """
        Set training data

        Args:
            train_data: Training data
        """
        self.train_data = train_data

    def train_local(self, epochs: Optional[int] = None) -> Dict[str, float]:
        """
        Train model on local data

        Args:
            epochs: Number of local epochs

        Returns:
            Training metrics
        """
        if self.train_data is None:
            return {"loss": 0.0}

        epochs = epochs or self.local_epochs

        # Set model to training mode
        self.model.train()

        # Training metrics
        metrics = {"loss": 0.0, "accuracy": 0.0, "samples": 0}

        # Train for specified number of epochs
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            # Create data loader
            data_loader = torch.utils.data.DataLoader(
                self.train_data, batch_size=self.batch_size, shuffle=True
            )

            # Train on batches
            for batch_idx, (data, target) in enumerate(data_loader):
                # Zero gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.model(data)

                # Calculate loss
                loss = F.cross_entropy(output, target)

                # Backward pass
                loss.backward()

                # Apply differential privacy if enabled
                if self.use_differential_privacy:
                    # Get gradients
                    gradients = [param.grad for param in self.model.parameters()]

                    # Clip gradients
                    clipped_gradients = self.dp.clip_gradients(gradients)

                    # Apply clipped gradients
                    for param, grad in zip(self.model.parameters(), clipped_gradients):
                        if grad is not None:
                            param.grad = grad

                # Update parameters
                self.optimizer.step()

                # Update metrics
                epoch_loss += loss.item() * data.size(0)
                pred = output.argmax(dim=1, keepdim=True)
                epoch_correct += pred.eq(target.view_as(pred)).sum().item()
                epoch_samples += data.size(0)

            # Calculate epoch metrics
            epoch_loss /= epoch_samples
            epoch_accuracy = epoch_correct / epoch_samples

            # Update overall metrics
            metrics["loss"] += epoch_loss
            metrics["accuracy"] += epoch_accuracy
            metrics["samples"] += epoch_samples

        # Average metrics over epochs
        metrics["loss"] /= epochs
        metrics["accuracy"] /= epochs
        metrics["samples"] /= epochs

        return metrics

    def get_model_update(self) -> OrderedDict:
        """
        Get model update for federated learning

        Returns:
            Model parameters
        """
        # Get model parameters
        parameters = self.model.get_parameters()

        # Apply differential privacy if enabled
        if self.use_differential_privacy:
            parameters = self.dp.privatize_model_update(parameters)

        # Apply homomorphic encryption if enabled
        if self.use_homomorphic_encryption and self.homomorphic:
            encrypted_parameters = OrderedDict()
            for name, param in parameters.items():
                encrypted_parameters[name] = self.homomorphic.encrypt(param)

            return encrypted_parameters

        return parameters

    def aggregate_updates(
        self,
        round_id: str,
        peers: List[str],
        updates: Optional[List[OrderedDict]] = None,
    ) -> Optional[OrderedDict]:
        """
        Aggregate model updates from peers

        Args:
            round_id: Round identifier
            peers: List of peer node identifiers
            updates: List of model updates from peers (if not using secure aggregation)

        Returns:
            Aggregated model update
        """
        # Use secure aggregation if enabled
        if self.use_secure_aggregation:
            # Get local model parameters
            local_model = self.model.get_parameters()

            # Perform secure aggregation
            aggregated = self.secure_aggregation.aggregate(round_id, local_model, peers)

            return aggregated

        # Otherwise, perform simple aggregation
        if not updates:
            return None

        # Get local model parameters
        local_update = self.get_model_update()

        # Add local update to list
        all_updates = [local_update] + updates

        # Initialize aggregated update
        aggregated = OrderedDict()
        for name, param in local_update.items():
            aggregated[name] = torch.zeros_like(param)

        # Sum all updates
        for update in all_updates:
            for name, param in update.items():
                if name in aggregated:
                    aggregated[name] += param

        # Average updates
        for name in aggregated:
            aggregated[name] /= len(all_updates)

        return aggregated

    def apply_update(self, update: OrderedDict) -> None:
        """
        Apply aggregated update to model

        Args:
            update: Aggregated model update
        """
        # Decrypt update if using homomorphic encryption
        if self.use_homomorphic_encryption and self.homomorphic:
            decrypted_update = OrderedDict()
            for name, param in update.items():
                decrypted_update[name] = self.homomorphic.decrypt(param)

            update = decrypted_update

        # Apply update to model
        self.model.set_parameters(update)

        # Increment round counter
        self.current_round += 1


class FederatedLearningServer:
    """
    Server for federated learning
    """

    def __init__(
        self,
        node_id: str,
        model: FederatedModel,
        crypto_manager: CryptoManager,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize federated learning server

        Args:
            node_id: Node identifier
            model: Federated model
            crypto_manager: Cryptographic manager
            config: Configuration dictionary
        """
        self.node_id = node_id
        self.model = model
        self.crypto = crypto_manager
        self.config = config or {}

        # Default configuration
        self.rounds = self.config.get("rounds", 10)
        self.min_clients = self.config.get("min_clients", 2)
        self.client_fraction = self.config.get("client_fraction", 1.0)
        self.timeout = self.config.get("timeout", 60.0)

        # Secure channel for communication
        self.secure_channel = SecureChannel(
            node_id, crypto_manager, self.config.get("channel_config", {})
        )

        # Client registry
        self.clients = set()

        # Round state
        self.current_round = 0
        self.round_updates = {}  # round_id -> {client_id: update}
        self.round_metrics = {}  # round_id -> {client_id: metrics}

        # Lock for thread safety
        self.lock = threading.RLock()

    def start(self):
        """
        Start federated learning server
        """
        self.secure_channel.start()

        # Register message handlers
        self.secure_channel.register_handler(
            "register_client", self._handle_register_client
        )
        self.secure_channel.register_handler("model_update", self._handle_model_update)
        self.secure_channel.register_handler(
            "training_metrics", self._handle_training_metrics
        )

    def stop(self):
        """
        Stop federated learning server
        """
        self.secure_channel.stop()

    def register_client(self, client_id: str) -> None:
        """
        Register a client

        Args:
            client_id: Client identifier
        """
        with self.lock:
            self.clients.add(client_id)

    def run_federated_learning(self, rounds: Optional[int] = None) -> Dict[str, Any]:
        """
        Run federated learning process

        Args:
            rounds: Number of rounds

        Returns:
            Training results
        """
        rounds = rounds or self.rounds

        # Results
        results = {"rounds": [], "final_model": None, "client_metrics": {}}

        # Run for specified number of rounds
        for round_idx in range(rounds):
            # Generate round ID
            round_id = f"round_{self.current_round}"

            # Select clients for this round
            selected_clients = self._select_clients()

            if len(selected_clients) < self.min_clients:
                print(f"Not enough clients for round {round_id}")
                continue

            # Initialize round state
            with self.lock:
                self.round_updates[round_id] = {}
                self.round_metrics[round_id] = {}

            # Send model to clients
            self._send_model_to_clients(round_id, selected_clients)

            # Wait for updates from clients
            updates_received = self._wait_for_updates(round_id, selected_clients)

            if updates_received < self.min_clients:
                print(f"Not enough updates received for round {round_id}")
                continue

            # Aggregate updates
            aggregated_update = self._aggregate_updates(round_id)

            if aggregated_update is None:
                print(f"Failed to aggregate updates for round {round_id}")
                continue

            # Apply aggregated update to global model
            self.model.set_parameters(aggregated_update)

            # Evaluate global model
            eval_metrics = self._evaluate_global_model()

            # Store round results
            round_results = {
                "round": self.current_round,
                "clients": len(selected_clients),
                "updates_received": updates_received,
                "eval_metrics": eval_metrics,
                "client_metrics": self.round_metrics.get(round_id, {}),
            }

            results["rounds"].append(round_results)

            # Increment round counter
            self.current_round += 1

        # Store final model
        results["final_model"] = self.model.get_parameters()

        # Aggregate client metrics
        for round_id, metrics in self.round_metrics.items():
            for client_id, client_metrics in metrics.items():
                if client_id not in results["client_metrics"]:
                    results["client_metrics"][client_id] = []

                results["client_metrics"][client_id].append(client_metrics)

        return results

    def _select_clients(self) -> List[str]:
        """
        Select clients for current round

        Returns:
            List of selected client identifiers
        """
        with self.lock:
            # Calculate number of clients to select
            num_clients = max(
                self.min_clients, int(len(self.clients) * self.client_fraction)
            )

            # Select random clients
            if num_clients >= len(self.clients):
                return list(self.clients)
            else:
                return random.sample(list(self.clients), num_clients)

    def _send_model_to_clients(self, round_id: str, clients: List[str]) -> None:
        """
        Send current model to selected clients

        Args:
            round_id: Round identifier
            clients: List of client identifiers
        """
        # Get model parameters
        model_params = self.model.get_parameters()

        # Serialize model parameters
        serialized_model = self._serialize_model(model_params)

        # Send to each client
        for client_id in clients:
            message = {"round_id": round_id, "model": serialized_model}

            self.secure_channel.send_message(client_id, "global_model", message)

    def _wait_for_updates(self, round_id: str, clients: List[str]) -> int:
        """
        Wait for updates from clients

        Args:
            round_id: Round identifier
            clients: List of client identifiers

        Returns:
            Number of updates received
        """
        start_time = time.time()

        while time.time() - start_time < self.timeout:
            with self.lock:
                updates_received = len(self.round_updates.get(round_id, {}))

                # Check if we have enough updates
                if updates_received >= self.min_clients:
                    return updates_received

            time.sleep(0.1)

        # Return number of updates received after timeout
        with self.lock:
            return len(self.round_updates.get(round_id, {}))

    def _aggregate_updates(self, round_id: str) -> Optional[OrderedDict]:
        """
        Aggregate model updates from clients

        Args:
            round_id: Round identifier

        Returns:
            Aggregated model update
        """
        with self.lock:
            updates = self.round_updates.get(round_id, {})

            if not updates:
                return None

            # Initialize aggregated update with zeros
            first_update = next(iter(updates.values()))
            aggregated = OrderedDict()

            for name, param in first_update.items():
                aggregated[name] = torch.zeros_like(param)

            # Sum all updates
            for client_id, update in updates.items():
                for name, param in update.items():
                    if name in aggregated:
                        aggregated[name] += param

            # Average updates
            for name in aggregated:
                aggregated[name] /= len(updates)

            return aggregated

    def _evaluate_global_model(self) -> Dict[str, float]:
        """
        Evaluate global model

        Returns:
            Evaluation metrics
        """
        # In a real implementation, this would evaluate on a validation set
        # For demonstration, we'll return placeholder metrics
        return {"loss": random.uniform(0.1, 0.5), "accuracy": random.uniform(0.7, 0.95)}

    def _handle_register_client(self, client_id: str, message: Dict[str, Any]) -> None:
        """
        Handle client registration message

        Args:
            client_id: Client identifier
            message: Message data
        """
        self.register_client(client_id)

    def _handle_model_update(self, client_id: str, message: Dict[str, Any]) -> None:
        """
        Handle model update message

        Args:
            client_id: Client identifier
            message: Message data
        """
        round_id = message.get("round_id")
        serialized_update = message.get("update")

        if not round_id or not serialized_update:
            return

        # Deserialize update
        update = self._deserialize_model(serialized_update)

        # Store update
        with self.lock:
            if round_id in self.round_updates:
                self.round_updates[round_id][client_id] = update

    def _handle_training_metrics(self, client_id: str, message: Dict[str, Any]) -> None:
        """
        Handle training metrics message

        Args:
            client_id: Client identifier
            message: Message data
        """
        round_id = message.get("round_id")
        metrics = message.get("metrics")

        if not round_id or not metrics:
            return

        # Store metrics
        with self.lock:
            if round_id in self.round_metrics:
                self.round_metrics[round_id][client_id] = metrics

    def _serialize_model(self, model: OrderedDict) -> bytes:
        """
        Serialize model parameters to bytes

        Args:
            model: Model parameters

        Returns:
            Serialized model bytes
        """
        serialized = {}
        for key, param in model.items():
            serialized[key] = param.cpu().numpy().tolist()

        return json.dumps(serialized).encode("utf-8")

    def _deserialize_model(self, serialized: bytes) -> OrderedDict:
        """
        Deserialize model parameters from bytes

        Args:
            serialized: Serialized model bytes

        Returns:
            Deserialized model parameters
        """
        data = json.loads(serialized.decode("utf-8"))
        model = OrderedDict()

        for key, param_list in data.items():
            param_array = np.array(param_list, dtype=np.float32)
            model[key] = torch.from_numpy(param_array)

        return model


class PrivacyPreservingFederatedLearning:
    """
    Main class for privacy-preserving federated learning
    """

    def __init__(
        self,
        node_id: str,
        is_server: bool,
        crypto_manager: CryptoManager,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize privacy-preserving federated learning

        Args:
            node_id: Node identifier
            is_server: Whether this node is a server
            crypto_manager: Cryptographic manager
            config: Configuration dictionary
        """
        self.node_id = node_id
        self.is_server = is_server
        self.crypto = crypto_manager
        self.config = config or {}

        # Create model
        self.model = self._create_model()

        # Create client or server
        if is_server:
            self.server = FederatedLearningServer(
                node_id,
                self.model,
                crypto_manager,
                self.config.get("server_config", {}),
            )
            self.client = None
        else:
            self.client = FederatedLearningClient(
                node_id,
                self.model,
                crypto_manager,
                self.config.get("client_config", {}),
            )
            self.server = None

    def start(self):
        """
        Start federated learning
        """
        if self.is_server and self.server:
            self.server.start()
        elif self.client:
            self.client.start()

    def stop(self):
        """
        Stop federated learning
        """
        if self.is_server and self.server:
            self.server.stop()
        elif self.client:
            self.client.stop()

    def run_server(self, rounds: Optional[int] = None) -> Dict[str, Any]:
        """
        Run federated learning server

        Args:
            rounds: Number of rounds

        Returns:
            Training results
        """
        if not self.is_server or not self.server:
            raise ValueError("This node is not configured as a server")

        return self.server.run_federated_learning(rounds)

    def run_client(
        self, train_data: Any, epochs: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Run federated learning client

        Args:
            train_data: Training data
            epochs: Number of local epochs

        Returns:
            Training metrics
        """
        if self.is_server or not self.client:
            raise ValueError("This node is not configured as a client")

        # Set training data
        self.client.set_train_data(train_data)

        # Train local model
        return self.client.train_local(epochs)

    def get_model(self) -> FederatedModel:
        """
        Get current model

        Returns:
            Current model
        """
        return self.model

    def _create_model(self) -> FederatedModel:
        """
        Create federated learning model

        Returns:
            Federated model
        """
        # Get model configuration
        model_config = self.config.get("model_config", {})
        model_type = model_config.get("type", "mlp")

        if model_type == "mlp":
            return self._create_mlp_model(model_config)
        elif model_type == "cnn":
            return self._create_cnn_model(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _create_mlp_model(self, config: Dict[str, Any]) -> FederatedModel:
        """
        Create MLP model

        Args:
            config: Model configuration

        Returns:
            MLP model
        """
        input_dim = config.get("input_dim", 784)
        hidden_dim = config.get("hidden_dim", 128)
        output_dim = config.get("output_dim", 10)

        class MLPModel(FederatedModel):
            def __init__(self):
                super(MLPModel, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)

            def forward(self, x):
                x = x.view(-1, input_dim)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        return MLPModel()

    def _create_cnn_model(self, config: Dict[str, Any]) -> FederatedModel:
        """
        Create CNN model

        Args:
            config: Model configuration

        Returns:
            CNN model
        """
        input_channels = config.get("input_channels", 1)
        output_dim = config.get("output_dim", 10)

        class CNNModel(FederatedModel):
            def __init__(self):
                super(CNNModel, self).__init__()
                self.conv1 = nn.Conv2d(
                    input_channels, 32, kernel_size=3, stride=1, padding=1
                )
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
                self.fc1 = nn.Linear(64 * 7 * 7, 128)
                self.fc2 = nn.Linear(128, output_dim)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2)
                x = x.view(-1, 64 * 7 * 7)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x

        return CNNModel()


class HierarchicalFederatedLearning(PrivacyPreservingFederatedLearning):
    """
    Hierarchical Federated Learning with Privacy Preservation.
    
    This class extends the PrivacyPreservingFederatedLearning by implementing
    a hierarchical structure for federated learning across shards, enabling
    more efficient model aggregation and improved privacy guarantees.
    """
    
    def __init__(
        self,
        node_id: str,
        is_server: bool,
        crypto_manager: CryptoManager,
        config: Dict[str, Any] = None,
    ):
        """
        Initialize hierarchical federated learning.
        
        Args:
            node_id: Node identifier
            is_server: Whether this node is a server
            crypto_manager: Cryptographic manager
            config: Configuration parameters
        """
        super().__init__(node_id, is_server, crypto_manager, config)
        
        # Default hierarchical learning configuration
        self.hierarchical_config = {
            "levels": 3,  # Number of hierarchical levels
            "aggregation_thresholds": [0.5, 0.7, 0.9],  # Min fraction of nodes required at each level
            "privacy_budgets": [0.3, 0.5, 1.0],  # Privacy budget for each level
            "level_timeouts": [30, 60, 120],  # Timeout in seconds for each level
        }
        
        # Update configuration if provided
        if config and "hierarchical" in config:
            self.hierarchical_config.update(config["hierarchical"])
            
        # Initialize hierarchy
        self.node_level = 0  # Default level
        self.parent_node = None
        self.child_nodes = set()
        self.level_managers = []  # Managers for each level
        
        # Initialize metrics
        self.hierarchical_metrics = {
            "level_convergence": [0.0] * self.hierarchical_config["levels"],
            "level_participation": [0.0] * self.hierarchical_config["levels"],
            "level_privacy_cost": [0.0] * self.hierarchical_config["levels"],
        }
        
    def start(self):
        """Start hierarchical federated learning."""
        super().start()
        
        # Initialize level managers
        for level in range(self.hierarchical_config["levels"]):
            # Create secure aggregation for this level with appropriate threshold
            level_aggr = SecureAggregation(
                self.node_id,
                self.crypto_manager,
                {
                    "threshold": self.hierarchical_config["aggregation_thresholds"][level],
                    "timeout": self.hierarchical_config["level_timeouts"][level],
                }
            )
            
            # Create differential privacy for this level
            level_dp = DifferentialPrivacy(
                epsilon=self.hierarchical_config["privacy_budgets"][level],
                delta=1e-5,
                clip_norm=1.0
            )
            
            self.level_managers.append({
                "aggregator": level_aggr,
                "privacy": level_dp
            })
            
        # Start all aggregators
        for level in range(self.hierarchical_config["levels"]):
            self.level_managers[level]["aggregator"].start()
    
    def stop(self):
        """Stop hierarchical federated learning."""
        # Stop all aggregators
        for level in range(self.hierarchical_config["levels"]):
            if "aggregator" in self.level_managers[level]:
                self.level_managers[level]["aggregator"].stop()
                
        super().stop()
    
    def set_hierarchy_position(self, level: int, parent: Optional[str] = None, children: Optional[List[str]] = None):
        """
        Set the position of this node in the hierarchy.
        
        Args:
            level: Hierarchy level (0 = leaf, higher = closer to root)
            parent: Parent node ID (None if root)
            children: List of child node IDs
        """
        self.node_level = level
        self.parent_node = parent
        self.child_nodes = set(children) if children else set()
    
    def add_child_node(self, node_id: str):
        """
        Add a child node to this node.
        
        Args:
            node_id: Child node ID
        """
        self.child_nodes.add(node_id)
    
    def remove_child_node(self, node_id: str):
        """
        Remove a child node from this node.
        
        Args:
            node_id: Child node ID
        """
        if node_id in self.child_nodes:
            self.child_nodes.remove(node_id)
    
    def run_hierarchical_learning(self, rounds: Optional[int] = None, data: Any = None) -> Dict[str, Any]:
        """
        Run hierarchical federated learning.
        
        Args:
            rounds: Number of rounds to run
            data: Training data
            
        Returns:
            Metrics from training
        """
        if rounds is None:
            rounds = self.config.get("default_rounds", 10)
            
        metrics = {
            "rounds": [],
            "accuracy": [],
            "loss": [],
            "convergence": [],
            "privacy_cost": [],
            "hierarchical": self.hierarchical_metrics.copy()
        }
        
        # If this is a leaf node (level 0), train with local data
        if self.node_level == 0:
            if data is not None:
                local_metrics = self.run_client(data)
                metrics.update(local_metrics)
        # If this is an intermediate or root node, coordinate aggregation
        else:
            # Select the appropriate level manager
            level_manager = self.level_managers[self.node_level - 1]
            
            for round_num in range(rounds):
                round_id = f"hierarchical_round_{round_num}"
                
                # Wait for updates from children
                received_updates = self._collect_child_updates(round_id)
                
                # Privatize and aggregate updates
                if received_updates:
                    # Apply differential privacy to updates
                    privatized_updates = [
                        level_manager["privacy"].privatize_model_update(update)
                        for update in received_updates
                    ]
                    
                    # Securely aggregate updates
                    aggregated_update = level_manager["aggregator"].aggregate(
                        round_id,
                        self.get_model().get_parameters(),
                        list(self.child_nodes)
                    )
                    
                    if aggregated_update:
                        # Apply the aggregated update
                        if self.parent_node:
                            # If not root, send to parent
                            self._send_update_to_parent(round_id, aggregated_update)
                        else:
                            # If root, apply to global model
                            self.server.apply_global_update(aggregated_update)
                            
                            # Evaluate and record metrics
                            eval_metrics = self.server._evaluate_global_model()
                            metrics["rounds"].append(round_num)
                            metrics["accuracy"].append(eval_metrics.get("accuracy", 0.0))
                            metrics["loss"].append(eval_metrics.get("loss", 0.0))
                
                # Record participation metrics
                if self.child_nodes:
                    participation = len(received_updates) / len(self.child_nodes)
                    metrics["hierarchical"]["level_participation"][self.node_level - 1] = participation
        
        return metrics
                
    def _collect_child_updates(self, round_id: str) -> List[OrderedDict]:
        """
        Collect model updates from child nodes.
        
        Args:
            round_id: Round identifier
            
        Returns:
            List of model updates
        """
        # Initialize update collection
        updates = []
        
        # Set timeout for collection
        timeout = self.hierarchical_config["level_timeouts"][self.node_level - 1]
        start_time = time.time()
        
        # Record which children have reported
        reported_children = set()
        
        # Wait for updates from children
        while (time.time() - start_time < timeout and 
               len(reported_children) < len(self.child_nodes)):
            
            # Check for new updates (implementation depends on messaging system)
            new_updates = self._check_for_updates(round_id)
            
            for child_id, update in new_updates.items():
                if child_id in self.child_nodes and child_id not in reported_children:
                    updates.append(update)
                    reported_children.add(child_id)
            
            # Throttle checking
            time.sleep(0.1)
            
        return updates
    
    def _check_for_updates(self, round_id: str) -> Dict[str, OrderedDict]:
        """
        Check for new updates from child nodes.
        
        Args:
            round_id: Round identifier
            
        Returns:
            Dictionary mapping node IDs to updates
        """
        # Mock implementation - would be implemented with actual messaging
        # In a real implementation, this would check a message queue
        return {}
    
    def _send_update_to_parent(self, round_id: str, update: OrderedDict) -> bool:
        """
        Send update to parent node.
        
        Args:
            round_id: Round identifier
            update: Model update
            
        Returns:
            Success flag
        """
        if not self.parent_node:
            return False
            
        # Mock implementation - would be implemented with actual messaging
        # In a real implementation, this would send a message to the parent
        return True
