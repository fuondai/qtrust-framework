#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive Consensus Selector for QTrust Blockchain Sharding Framework
Version: 3.0.0

This module implements the enhanced Adaptive Consensus Selector with:
1. Standardized consensus protocol interfaces
2. Decision tree with Bayesian network for protocol selection
3. Smooth protocol transition mechanism
"""

import os
import sys
import time
import logging
import threading
import numpy as np
import json
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from collections import defaultdict
import networkx as nx
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class ConsensusType(Enum):
    """Consensus protocol types."""

    PBFT = "pbft"
    HOTSTUFF = "hotstuff"
    TENDERMINT = "tendermint"
    RAFT = "raft"
    POA = "poa"


class ConsensusProtocol:
    """Base class for all consensus protocols."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the consensus protocol.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.running = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.metrics = {
            "throughput": 0,
            "latency": 0,
            "byzantine_tolerance": 0,
            "resource_usage": 0,
            "transaction_count": 0,
            "success_rate": 0,
        }
        self.start_time = 0

    def start(self) -> None:
        """Start the consensus protocol."""
        self.running = True
        self.start_time = time.time()
        self.logger.info(f"{self.__class__.__name__} started")

    def stop(self) -> None:
        """Stop the consensus protocol."""
        self.running = False
        self.logger.info(f"{self.__class__.__name__} stopped")

    def process_transaction(self, transaction: Any) -> Dict[str, Any]:
        """
        Process a transaction.

        Args:
            transaction: Transaction to process

        Returns:
            Transaction result
        """
        raise NotImplementedError("Subclasses must implement process_transaction")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get consensus metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update consensus metrics.

        Args:
            metrics: New metrics
        """
        self.metrics.update(metrics)

    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the consensus protocol.

        Returns:
            State dictionary
        """
        return {
            "running": self.running,
            "uptime": time.time() - self.start_time if self.start_time > 0 else 0,
            "metrics": self.get_metrics(),
        }

    def prepare_transition(
        self, target_protocol: "ConsensusProtocol"
    ) -> Dict[str, Any]:
        """
        Prepare for transition to another consensus protocol.

        Args:
            target_protocol: Target consensus protocol

        Returns:
            Transition state
        """
        return {
            "source_protocol": self.__class__.__name__,
            "target_protocol": target_protocol.__class__.__name__,
            "pending_transactions": [],
            "state": self.get_state(),
        }

    def apply_transition(self, transition_state: Dict[str, Any]) -> None:
        """
        Apply transition state from another consensus protocol.

        Args:
            transition_state: Transition state
        """
        if transition_state is None:
            return
            
        # Process any pending transactions
        pending_transactions = transition_state.get("pending_transactions", [])
        for transaction in pending_transactions:
            self.process_transaction(transaction)


class PBFTConsensus(ConsensusProtocol):
    """Practical Byzantine Fault Tolerance consensus protocol."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PBFT consensus protocol.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.view_number = 0
        self.sequence_number = 0
        self.primary = 0
        self.validators = []
        self.f = 0  # Maximum number of faulty nodes
        self.prepare_messages = defaultdict(set)
        self.commit_messages = defaultdict(set)

        # Initialize from config
        if config:
            self.validators = config.get("validators", [])
            self.f = config.get("f", (len(self.validators) - 1) // 3)
            self.primary = config.get("primary", 0)

        # Set Byzantine tolerance
        self.metrics["byzantine_tolerance"] = (
            self.f / len(self.validators) if self.validators else 0
        )

    def process_transaction(self, transaction: Any) -> Dict[str, Any]:
        """
        Process a transaction using PBFT.

        Args:
            transaction: Transaction to process

        Returns:
            Transaction result
        """
        if not self.running:
            return {"success": False, "error": "Consensus not running"}

        start_time = time.time()

        # Simulate PBFT phases
        # 1. Pre-prepare phase
        self.sequence_number += 1

        # 2. Prepare phase
        prepare_count = self._simulate_prepare_phase(transaction)

        # 3. Commit phase
        commit_count = self._simulate_commit_phase(transaction)

        # 4. Reply phase
        success = commit_count >= 2 * self.f + 1

        # Update metrics
        latency = time.time() - start_time
        self.metrics["transaction_count"] += 1
        self.metrics["latency"] = (
            self.metrics["latency"] * (self.metrics["transaction_count"] - 1) + latency
        ) / self.metrics["transaction_count"]
        self.metrics["success_rate"] = (
            self.metrics["success_rate"] * (self.metrics["transaction_count"] - 1)
            + (1 if success else 0)
        ) / self.metrics["transaction_count"]

        return {
            "success": success,
            "sequence_number": self.sequence_number,
            "view_number": self.view_number,
            "latency": latency,
        }

    def _simulate_prepare_phase(self, transaction: Any) -> int:
        """
        Simulate the prepare phase.

        Args:
            transaction: Transaction to process

        Returns:
            Number of prepare messages
        """
        # Thêm một độ trễ nhỏ để đảm bảo latency > 0
        time.sleep(0.001)
        
        # Simulate prepare messages from validators
        prepare_count = 0
        for i in range(len(self.validators)):
            # Simulate Byzantine behavior for some validators
            if i % 10 == 0 and self.config.get("simulate_byzantine", False):
                continue

            self.prepare_messages[self.sequence_number].add(i)
            prepare_count += 1

        return prepare_count

    def _simulate_commit_phase(self, transaction: Any) -> int:
        """
        Simulate the commit phase.

        Args:
            transaction: Transaction to process

        Returns:
            Number of commit messages
        """
        # Thêm một độ trễ nhỏ để đảm bảo latency > 0
        time.sleep(0.001)
        
        # Only proceed to commit if we have enough prepare messages
        if len(self.prepare_messages[self.sequence_number]) < 2 * self.f + 1:
            return 0

        # Simulate commit messages from validators
        commit_count = 0
        for i in range(len(self.validators)):
            # Simulate Byzantine behavior for some validators
            if i % 10 == 0 and self.config.get("simulate_byzantine", False):
                continue

            self.commit_messages[self.sequence_number].add(i)
            commit_count += 1

        return commit_count


class HotStuffConsensus(ConsensusProtocol):
    """HotStuff consensus protocol."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the HotStuff consensus protocol.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.view_number = 0
        self.sequence_number = 0
        self.leader = 0
        self.validators = []
        self.f = 0  # Maximum number of faulty nodes
        self.votes = defaultdict(set)

        # Initialize from config
        if config:
            self.validators = config.get("validators", [])
            self.f = config.get("f", (len(self.validators) - 1) // 3)
            self.leader = config.get("leader", 0)

        # Set Byzantine tolerance
        self.metrics["byzantine_tolerance"] = (
            self.f / len(self.validators) if self.validators else 0
        )

    def process_transaction(self, transaction: Any) -> Dict[str, Any]:
        """
        Process a transaction using HotStuff.

        Args:
            transaction: Transaction to process

        Returns:
            Transaction result
        """
        if not self.running:
            return {"success": False, "error": "Consensus not running"}

        start_time = time.time()

        # Simulate HotStuff phases
        self.sequence_number += 1

        # Simulate the three-phase voting
        prepare_votes = self._simulate_voting_phase("prepare")
        pre_commit_votes = self._simulate_voting_phase("pre-commit")
        commit_votes = self._simulate_voting_phase("commit")

        # Check if we have enough votes in all phases
        success = (
            prepare_votes >= 2 * self.f + 1
            and pre_commit_votes >= 2 * self.f + 1
            and commit_votes >= 2 * self.f + 1
        )

        # Update metrics
        latency = time.time() - start_time
        self.metrics["transaction_count"] += 1
        self.metrics["latency"] = (
            self.metrics["latency"] * (self.metrics["transaction_count"] - 1) + latency
        ) / self.metrics["transaction_count"]
        self.metrics["success_rate"] = (
            self.metrics["success_rate"] * (self.metrics["transaction_count"] - 1)
            + (1 if success else 0)
        ) / self.metrics["transaction_count"]

        return {
            "success": success,
            "sequence_number": self.sequence_number,
            "view_number": self.view_number,
            "latency": latency,
        }

    def _simulate_voting_phase(self, phase: str) -> int:
        """
        Simulate a voting phase.

        Args:
            phase: Voting phase name

        Returns:
            Number of votes
        """
        # Thêm một độ trễ nhỏ để đảm bảo latency > 0
        time.sleep(0.001)
        
        vote_count = 0
        for i in range(len(self.validators)):
            # Simulate Byzantine behavior for some validators
            if i % 10 == 0 and self.config.get("simulate_byzantine", False):
                continue

            self.votes[f"{phase}_{self.sequence_number}"].add(i)
            vote_count += 1

        return vote_count


class TendermintConsensus(ConsensusProtocol):
    """Tendermint consensus protocol."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Tendermint consensus protocol.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.height = 0
        self.round = 0
        self.proposer = 0
        self.validators = []
        self.f = 0  # Maximum number of faulty nodes
        self.prevote_messages = defaultdict(set)
        self.precommit_messages = defaultdict(set)

        # Initialize from config
        if config:
            self.validators = config.get("validators", [])
            self.f = config.get("f", (len(self.validators) - 1) // 3)
            self.proposer = config.get("proposer", 0)

        # Set Byzantine tolerance
        self.metrics["byzantine_tolerance"] = (
            self.f / len(self.validators) if self.validators else 0
        )

    def process_transaction(self, transaction: Any) -> Dict[str, Any]:
        """
        Process a transaction using Tendermint.

        Args:
            transaction: Transaction to process

        Returns:
            Transaction result
        """
        if not self.running:
            return {"success": False, "error": "Consensus not running"}

        start_time = time.time()

        # Simulate Tendermint phases
        self.height += 1

        # 1. Propose phase
        # Proposer creates a block with the transaction

        # 2. Prevote phase
        prevote_count = self._simulate_prevote_phase(transaction)

        # 3. Precommit phase
        precommit_count = self._simulate_precommit_phase(transaction)

        # 4. Commit phase
        success = precommit_count >= 2 * self.f + 1

        # Update metrics
        latency = time.time() - start_time
        self.metrics["transaction_count"] += 1
        self.metrics["latency"] = (
            self.metrics["latency"] * (self.metrics["transaction_count"] - 1) + latency
        ) / self.metrics["transaction_count"]
        self.metrics["success_rate"] = (
            self.metrics["success_rate"] * (self.metrics["transaction_count"] - 1)
            + (1 if success else 0)
        ) / self.metrics["transaction_count"]

        return {
            "success": success,
            "height": self.height,
            "round": self.round,
            "latency": latency,
        }

    def _simulate_prevote_phase(self, transaction: Any) -> int:
        """
        Simulate the prevote phase.

        Args:
            transaction: Transaction to process

        Returns:
            Number of prevote messages
        """
        # Thêm một độ trễ nhỏ để đảm bảo latency > 0
        time.sleep(0.001)
        
        # Simulate prevote messages from validators
        prevote_count = 0
        for i in range(len(self.validators)):
            # Simulate Byzantine behavior for some validators
            if i % 10 == 0 and self.config.get("simulate_byzantine", False):
                continue

            self.prevote_messages[self.height].add(i)
            prevote_count += 1

        return prevote_count

    def _simulate_precommit_phase(self, transaction: Any) -> int:
        """
        Simulate the precommit phase.

        Args:
            transaction: Transaction to process

        Returns:
            Number of precommit messages
        """
        # Thêm một độ trễ nhỏ để đảm bảo latency > 0
        time.sleep(0.001)
        
        # Only proceed to precommit if we have enough prevote messages
        if len(self.prevote_messages[self.height]) < 2 * self.f + 1:
            return 0

        # Simulate precommit messages from validators
        precommit_count = 0
        for i in range(len(self.validators)):
            # Simulate Byzantine behavior for some validators
            if i % 10 == 0 and self.config.get("simulate_byzantine", False):
                continue

            self.precommit_messages[self.height].add(i)
            precommit_count += 1

        return precommit_count


class RaftConsensus(ConsensusProtocol):
    """Raft consensus protocol."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Raft consensus protocol.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.term = 0
        self.leader = 0
        self.nodes = []
        self.log = []
        self.commit_index = 0

        # Initialize from config
        if config:
            self.nodes = config.get("nodes", [])
            self.leader = config.get("leader", 0)

        # Set Byzantine tolerance
        self.metrics["byzantine_tolerance"] = 0  # Raft is not Byzantine fault-tolerant

    def process_transaction(self, transaction: Any) -> Dict[str, Any]:
        """
        Process a transaction using Raft.

        Args:
            transaction: Transaction to process

        Returns:
            Transaction result
        """
        if not self.running:
            return {"success": False, "error": "Consensus not running"}

        start_time = time.time()

        # Simulate Raft phases
        # 1. Leader appends transaction to log
        self.log.append({"term": self.term, "transaction": transaction})
        log_index = len(self.log) - 1

        # 2. Leader sends AppendEntries to followers
        success = self._simulate_append_entries(log_index)

        # 3. If majority of nodes respond, commit the transaction
        if success:
            self.commit_index = log_index

        # Update metrics
        latency = time.time() - start_time
        self.metrics["transaction_count"] += 1
        self.metrics["latency"] = (
            self.metrics["latency"] * (self.metrics["transaction_count"] - 1) + latency
        ) / self.metrics["transaction_count"]
        self.metrics["success_rate"] = (
            self.metrics["success_rate"] * (self.metrics["transaction_count"] - 1)
            + (1 if success else 0)
        ) / self.metrics["transaction_count"]

        return {
            "success": success,
            "term": self.term,
            "log_index": log_index,
            "commit_index": self.commit_index,
            "latency": latency,
        }

    def _simulate_append_entries(self, log_index: int) -> bool:
        """
        Simulate the AppendEntries RPC.

        Args:
            log_index: Index of the log entry

        Returns:
            True if majority of nodes acknowledge, False otherwise
        """
        # Thêm một độ trễ nhỏ để đảm bảo latency > 0
        time.sleep(0.001)
        
        # Simulate responses from nodes
        ack_count = 1  # Leader already acknowledges
        for i in range(len(self.nodes)):
            if i == self.leader:
                continue  # Skip leader

            # Simulate node failure
            if i % 5 == 0 and self.config.get("simulate_failures", False):
                continue

            ack_count += 1

        # Check if we have majority
        return ack_count > len(self.nodes) / 2


class PoAConsensus(ConsensusProtocol):
    """Proof of Authority consensus protocol."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the PoA consensus protocol.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.authorities = []
        self.current_authority = 0
        self.block_number = 0

        # Initialize from config
        if config:
            self.authorities = config.get("authorities", [])
            self.current_authority = config.get("current_authority", 0)

        # Set Byzantine tolerance
        self.metrics["byzantine_tolerance"] = (
            0.5  # PoA can tolerate up to 50% malicious authorities
        )

    def process_transaction(self, transaction: Any) -> Dict[str, Any]:
        """
        Process a transaction using PoA.

        Args:
            transaction: Transaction to process

        Returns:
            Transaction result
        """
        if not self.running:
            return {"success": False, "error": "Consensus not running"}

        start_time = time.time()

        # Simulate PoA validation
        validation_result = self._simulate_authority_validation(transaction)

        if not validation_result:
            self.metrics["transaction_count"] += 1
            self.metrics["success_rate"] = (
                self.metrics["success_rate"] * (self.metrics["transaction_count"] - 1)
                + 0
            ) / self.metrics["transaction_count"]

            return {
                "success": False,
                "error": "Transaction rejected by authority validation",
            }

        # Simulate PoA finalization
        # PoA has single-step finalization that is quite fast
        time.sleep(0.05)  # Simulate block inclusion time

        # Calculate latency
        latency = time.time() - start_time

        # Update metrics
        self.metrics["transaction_count"] += 1
        self.metrics["latency"] = (
            self.metrics["latency"] * (self.metrics["transaction_count"] - 1) + latency
        ) / self.metrics["transaction_count"]
        self.metrics["success_rate"] = (
            self.metrics["success_rate"] * (self.metrics["transaction_count"] - 1) + 1
        ) / self.metrics["transaction_count"]

        return {
            "success": True,
            "latency": latency,
            "block_number": int(time.time()),  # Simulated block number
        }

    def _simulate_authority_validation(self, transaction: Any) -> bool:
        """
        Simulate authority validation.

        Args:
            transaction: Transaction to process

        Returns:
            True if validation succeeds, False otherwise
        """
        # Thêm một độ trễ nhỏ để đảm bảo latency > 0
        time.sleep(0.001)
        
        # Simulate validation
        # In a real implementation, this would check the transaction signature
        # and validate against business rules

        # Simulate authority failure
        if self.current_authority % 10 == 0 and self.config.get(
            "simulate_failures", False
        ):
            return False

        return True


class ConsensusFactory:
    """Factory for creating consensus protocol instances."""

    @staticmethod
    def create_consensus(
        consensus_type: ConsensusType, config: Dict[str, Any] = None
    ) -> ConsensusProtocol:
        """
        Create a consensus protocol instance.

        Args:
            consensus_type: Type of consensus protocol to create
            config: Configuration dictionary

        Returns:
            Consensus protocol instance
        """
        # Use empty dict if config is None
        actual_config = config if config is not None else {}
        
        if consensus_type == ConsensusType.PBFT:
            return PBFTConsensus(actual_config)
        elif consensus_type == ConsensusType.HOTSTUFF:
            return HotStuffConsensus(actual_config)
        elif consensus_type == ConsensusType.TENDERMINT:
            return TendermintConsensus(actual_config)
        elif consensus_type == ConsensusType.RAFT:
            return RaftConsensus(actual_config)
        elif consensus_type == ConsensusType.POA:
            return PoAConsensus(actual_config)
        else:
            raise ValueError(f"Unknown consensus type: {consensus_type}")


class BayesianDecisionTree:
    """Bayesian network-based decision tree for consensus protocol selection."""

    def __init__(self):
        """Initialize the Bayesian decision tree."""
        # Create Bayesian network
        self.model = DiscreteBayesianNetwork(
            [
                ("network_condition", "optimal_consensus"),
                ("security_risk", "optimal_consensus"),
                ("transaction_complexity", "optimal_consensus"),
                ("shard_size", "optimal_consensus"),
            ]
        )

        # Define CPDs (Conditional Probability Distributions)
        # Network condition CPD
        self.network_cpd = TabularCPD(
            variable="network_condition",
            variable_card=3,  # Low, Medium, High
            values=[[0.33], [0.33], [0.34]],
        )

        # Security risk CPD
        self.security_cpd = TabularCPD(
            variable="security_risk",
            variable_card=3,  # Low, Medium, High
            values=[[0.33], [0.33], [0.34]],
        )

        # Transaction complexity CPD
        self.transaction_cpd = TabularCPD(
            variable="transaction_complexity",
            variable_card=3,  # Low, Medium, High
            values=[[0.33], [0.33], [0.34]],
        )

        # Shard size CPD
        self.shard_cpd = TabularCPD(
            variable="shard_size",
            variable_card=3,  # Small, Medium, Large
            values=[[0.33], [0.33], [0.34]],
        )

        # Optimal consensus CPD
        # This is a complex CPD with 3^4 = 81 combinations
        # We'll initialize with equal probabilities and update based on learning
        self.consensus_cpd = TabularCPD(
            variable="optimal_consensus",
            variable_card=5,  # PBFT, HotStuff, Tendermint, Raft, PoA
            values=np.ones((5, 3 * 3 * 3 * 3)) / 5,  # Equal probabilities initially
            evidence=[
                "network_condition",
                "security_risk",
                "transaction_complexity",
                "shard_size",
            ],
            evidence_card=[3, 3, 3, 3],
        )

        # Add CPDs to the model
        self.model.add_cpds(
            self.network_cpd,
            self.security_cpd,
            self.transaction_cpd,
            self.shard_cpd,
            self.consensus_cpd,
        )

        # Check if the model is valid
        assert self.model.check_model()

        # Create inference engine
        self.inference = VariableElimination(self.model)

        # Initialize learning data
        self.learning_data = []

    def predict(
        self,
        network_condition: int,
        security_risk: int,
        transaction_complexity: int,
        shard_size: int,
    ) -> ConsensusType:
        """
        Predict the optimal consensus protocol.

        Args:
            network_condition: Network condition (0=Low, 1=Medium, 2=High)
            security_risk: Security risk (0=Low, 1=Medium, 2=High)
            transaction_complexity: Transaction complexity (0=Low, 1=Medium, 2=High)
            shard_size: Shard size (0=Small, 1=Medium, 2=Large)

        Returns:
            Optimal consensus type
        """
        # Query the model
        evidence = {
            "network_condition": network_condition,
            "security_risk": security_risk,
            "transaction_complexity": transaction_complexity,
            "shard_size": shard_size,
        }

        result = self.inference.query(
            variables=["optimal_consensus"], evidence=evidence
        )

        # Get the most probable consensus type
        probabilities = result.values
        consensus_index = np.argmax(probabilities)

        # Map index to consensus type
        consensus_types = list(ConsensusType)
        return consensus_types[consensus_index]

    def update(
        self,
        network_condition: int,
        security_risk: int,
        transaction_complexity: int,
        shard_size: int,
        consensus_type: ConsensusType,
        performance: float,
    ) -> None:
        """
        Update the model based on performance feedback.

        Args:
            network_condition: Network condition (0=Low, 1=Medium, 2=High)
            security_risk: Security risk (0=Low, 1=Medium, 2=High)
            transaction_complexity: Transaction complexity (0=Low, 1=Medium, 2=High)
            shard_size: Shard size (0=Small, 1=Medium, 2=Large)
            consensus_type: Consensus type used
            performance: Performance score (0.0 to 1.0)
        """
        # Add to learning data
        self.learning_data.append(
            {
                "network_condition": network_condition,
                "security_risk": security_risk,
                "transaction_complexity": transaction_complexity,
                "shard_size": shard_size,
                "consensus_type": consensus_type,
                "performance": performance,
            }
        )

        # If we have enough data, update the CPD
        if len(self.learning_data) >= 10:
            self._update_cpd()

    def _update_cpd(self) -> None:
        """Update the CPD based on learning data."""
        # Create a matrix to hold the updated probabilities
        values = np.zeros((5, 3 * 3 * 3 * 3))

        # Group data by evidence combination
        evidence_groups = defaultdict(list)
        for data in self.learning_data:
            key = (
                data["network_condition"],
                data["security_risk"],
                data["transaction_complexity"],
                data["shard_size"],
            )
            evidence_groups[key].append(data)

        # Update probabilities for each evidence combination
        for key, group in evidence_groups.items():
            # Calculate index in the CPD
            idx = key[0] * 27 + key[1] * 9 + key[2] * 3 + key[3]

            # Calculate performance for each consensus type
            consensus_performance = defaultdict(list)
            for data in group:
                consensus_idx = list(ConsensusType).index(data["consensus_type"])
                consensus_performance[consensus_idx].append(data["performance"])

            # Calculate average performance for each consensus type
            avg_performance = {}
            for consensus_idx, performances in consensus_performance.items():
                avg_performance[consensus_idx] = sum(performances) / len(performances)

            # If we have data for all consensus types, update based on performance
            if len(avg_performance) == 5:
                total_performance = sum(avg_performance.values())
                for consensus_idx, performance in avg_performance.items():
                    values[consensus_idx, idx] = performance / total_performance
            # Otherwise, update based on available data
            elif avg_performance:
                # Set equal probabilities for types with no data
                missing_types = 5 - len(avg_performance)
                equal_prob = 0.2 / missing_types if missing_types > 0 else 0

                # Set probabilities based on performance for types with data
                total_performance = sum(avg_performance.values())
                for consensus_idx in range(5):
                    if consensus_idx in avg_performance:
                        values[consensus_idx, idx] = 0.8 * (
                            avg_performance[consensus_idx] / total_performance
                        )
                    else:
                        values[consensus_idx, idx] = equal_prob
            # If no data for this combination, use equal probabilities
            else:
                for consensus_idx in range(5):
                    values[consensus_idx, idx] = 0.2

        # Update the CPD
        self.consensus_cpd.values = values

        # Update the model
        self.model.add_cpds(self.consensus_cpd)

        # Check if the model is valid
        assert self.model.check_model()

        # Update inference engine
        self.inference = VariableElimination(self.model)


class AdaptiveConsensusSelector:
    """
    Adaptive Consensus Selector that chooses appropriate consensus
    protocols based on network conditions, security risks, and transaction patterns.
    Uses a Bayesian decision tree for intelligent selection.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adaptive consensus selector.

        Args:
            config: Configuration dictionary
        """
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Default configuration
        self.config: Dict[str, Any] = {
            "update_interval": 60,  # seconds
            "adaptation_threshold": 0.2,
            "default_consensus": "pbft",
            "transition_wait_time": 10,  # seconds
            "metrics_window_size": 100,
            "learning_rate": 0.1,
        }
        
        # Update configuration if provided
        if config is not None:
            self.config.update(config)
            
        # Initialize decision tree
        self.decision_tree = BayesianDecisionTree()
        
        # Active consensus protocols per shard
        self.active_consensus: Dict[str, ConsensusProtocol] = {}
        
        # Available consensus protocol instances
        self.consensus_protocols: Dict[ConsensusType, ConsensusProtocol] = {}
        
        # Protocol metrics for performance tracking
        self.metrics: Dict[ConsensusType, List[Dict[str, float]]] = {
            ct: [] for ct in ConsensusType
        }
        
        # State providers
        self.network_state_provider = None
        self.trust_provider = None
        
        # Thread for periodic updates
        self.update_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Historical performance for learning
        self.performance_history: Dict[ConsensusType, List[float]] = {
            ct: [] for ct in ConsensusType
        }
        
        # Initialize default consensus protocols
        self._initialize_default_protocols()
        
        self.logger.info("AdaptiveConsensusSelector initialized")

    def start(self) -> None:
        """Start the consensus selector."""
        if self.running:
            self.logger.warning("AdaptiveConsensusSelector already running")
            return
            
        self.running = True
        
        # Start the update thread
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info("AdaptiveConsensusSelector started")

    def stop(self) -> None:
        """Stop the adaptive consensus selector."""
        self.running = False

        # Wait for update thread to finish
        if self.update_thread:
            self.update_thread.join(timeout=5.0)

        self.logger.info("Adaptive consensus selector stopped")

    def register_network_state_provider(self, provider) -> None:
        """
        Register a network state provider.

        Args:
            provider: Network state provider
        """
        self.network_state_provider = provider
        self.logger.info("Network state provider registered")

    def register_trust_provider(self, provider) -> None:
        """
        Register a trust provider.

        Args:
            provider: Trust provider
        """
        self.trust_provider = provider
        self.logger.info("Trust provider registered")

    def update_metrics(
        self, consensus_type: ConsensusType, metrics: Dict[str, Any]
    ) -> None:
        """
        Update metrics for a consensus protocol.

        Args:
            consensus_type: Consensus type
            metrics: Metrics dictionary
        """
        with self.update_lock:
            # Add metrics to window
            for metric_name in [
                "throughput",
                "latency",
                "success_rate",
                "resource_usage",
            ]:
                if metric_name in metrics:
                    self.metrics[consensus_type][metric_name].append(
                        metrics[metric_name]
                    )

                    # Trim window if needed
                    if (
                        len(self.metrics[consensus_type][metric_name])
                        > self.config["metrics_window_size"]
                    ):
                        self.metrics[consensus_type][metric_name].pop(0)

    def get_current_protocol(self, shard_id: str) -> ConsensusProtocol:
        """
        Get the current consensus protocol for a shard.

        Args:
            shard_id: Shard ID

        Returns:
            Consensus protocol instance
        """
        # Get current conditions for the shard
        network_condition = self._get_network_condition(shard_id)
        security_risk = self._get_security_risk(shard_id)
        transaction_complexity = self._get_transaction_complexity(shard_id)
        shard_size = self._get_shard_size(shard_id)

        # Update current conditions
        self.current_conditions = {
            "network_condition": network_condition,
            "security_risk": security_risk,
            "transaction_complexity": transaction_complexity,
            "shard_size": shard_size,
        }

        # Get optimal consensus type
        consensus_type = self.decision_tree.predict(
            network_condition, security_risk, transaction_complexity, shard_size
        )

        # Return consensus protocol
        return self.consensus_protocols[consensus_type]

    def get_consensus(
        self,
        network_condition: int,
        security_risk: int,
        transaction_complexity: int,
        shard_size: int = 1,
    ) -> ConsensusProtocol:
        """
        Get the optimal consensus protocol for given conditions.

        Args:
            network_condition: Network condition (0=Low, 1=Medium, 2=High)
            security_risk: Security risk (0=Low, 1=Medium, 2=High)
            transaction_complexity: Transaction complexity (0=Low, 1=Medium, 2=High)
            shard_size: Shard size (0=Small, 1=Medium, 2=Large)

        Returns:
            Consensus protocol instance
        """
        # Get optimal consensus type
        consensus_type = self.decision_tree.predict(
            network_condition, security_risk, transaction_complexity, shard_size
        )

        # Return consensus protocol
        return self.consensus_protocols[consensus_type]

    def transition_protocol(
        self, source_type: ConsensusType, target_type: ConsensusType
    ) -> bool:
        """
        Transition from one consensus protocol to another.

        Args:
            source_type: Source consensus protocol type
            target_type: Target consensus protocol type

        Returns:
            Whether the transition was successful
        """
        if source_type == target_type:
            return True  # No transition needed
            
        self.logger.info(f"Transitioning from {source_type.value} to {target_type.value}")
        
        try:
            # Get source and target protocol instances
            source_protocol = self.consensus_protocols.get(source_type)
            target_protocol = self.consensus_protocols.get(target_type)
            
            # Create protocols if they don't exist
            if source_protocol is None:
                source_config = self.config.get(source_type.value, {})
                source_protocol = ConsensusFactory.create_consensus(source_type, source_config)
                self.consensus_protocols[source_type] = source_protocol
                
            if target_protocol is None:
                target_config = self.config.get(target_type.value, {})
                target_protocol = ConsensusFactory.create_consensus(target_type, target_config)
                self.consensus_protocols[target_type] = target_protocol
                
            # Prepare transition state from source
            transition_state = source_protocol.prepare_transition(target_protocol)
            
            # Apply transition state to target
            target_protocol.apply_transition(transition_state)
            
            # Wait for transition to complete
            time.sleep(float(self.config["transition_wait_time"]))
            
            # Update active consensus for affected shards
            for shard_id, protocol in self.active_consensus.items():
                if protocol == source_protocol:
                    self.active_consensus[shard_id] = target_protocol
                    
            self.logger.info(f"Transition from {source_type.value} to {target_type.value} completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Transition error: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the adaptive consensus selector.

        Returns:
            Status dictionary
        """
        return {
            "running": self.running,
            "current_conditions": self.current_conditions,
            "transition_in_progress": self.transition_in_progress,
            "metrics": {
                consensus_type.value: {
                    metric_name: np.mean(values) if values else 0
                    for metric_name, values in metrics.items()
                }
                for consensus_type, metrics in self.metrics.items()
            },
        }

    def _update_loop(self) -> None:
        """Periodic update loop for consensus adaptation."""
        while self.running:
            try:
                # Update the decision tree
                self._update_decision_tree()
                
                # Sleep for the configured interval
                time.sleep(float(self.config["update_interval"]))
            except Exception as e:
                self.logger.error(f"Error in update loop: {e}")
                # Sleep for a short time to avoid tight loop in case of errors
                time.sleep(5.0)

    def _update_decision_tree(self) -> None:
        """Update the decision tree based on metrics."""
        with self.update_lock:
            # Calculate performance for each consensus type
            for consensus_type in ConsensusType:
                metrics = self.metrics[consensus_type]

                # Skip if we don't have enough metrics
                if not all(len(values) > 0 for values in metrics.values()):
                    continue

                # Calculate performance score
                throughput = (
                    np.mean(metrics["throughput"]) if metrics["throughput"] else 0
                )
                latency = np.mean(metrics["latency"]) if metrics["latency"] else 0
                success_rate = (
                    np.mean(metrics["success_rate"]) if metrics["success_rate"] else 0
                )
                resource_usage = (
                    np.mean(metrics["resource_usage"])
                    if metrics["resource_usage"]
                    else 0
                )

                # Normalize metrics
                max_throughput = 10000  # Transactions per second
                max_latency = 1000  # Milliseconds

                norm_throughput = min(1.0, throughput / max_throughput)
                norm_latency = 1.0 - min(1.0, latency / max_latency)

                # Calculate performance score (weighted average)
                performance = (
                    0.4 * norm_throughput
                    + 0.3 * norm_latency
                    + 0.2 * success_rate
                    + 0.1 * (1.0 - resource_usage)
                )

                # Update decision tree
                self.decision_tree.update(
                    self.current_conditions["network_condition"],
                    self.current_conditions["security_risk"],
                    self.current_conditions["transaction_complexity"],
                    self.current_conditions["shard_size"],
                    consensus_type,
                    performance,
                )

    def _get_network_condition(self, shard_id: str) -> int:
        """
        Get the network condition for a shard.

        Args:
            shard_id: Shard ID

        Returns:
            Network condition (0=Low, 1=Medium, 2=High)
        """
        if self.network_state_provider:
            congestion = self.network_state_provider.get_shard_congestion(shard_id)
            if congestion < 0.3:
                return 0  # Low congestion
            elif congestion < 0.7:
                return 1  # Medium congestion
            else:
                return 2  # High congestion

        return 1  # Default to medium

    def _get_security_risk(self, shard_id: str) -> int:
        """
        Get the security risk for a shard.

        Args:
            shard_id: Shard ID

        Returns:
            Security risk (0=Low, 1=Medium, 2=High)
        """
        if self.trust_provider:
            trust = self.trust_provider.get_trust_score(shard_id)
            if trust > 0.7:
                return 0  # Low risk
            elif trust > 0.4:
                return 1  # Medium risk
            else:
                return 2  # High risk

        return 1  # Default to medium

    def _get_transaction_complexity(self, shard_id: str) -> int:
        """
        Get the transaction complexity for a shard.

        Args:
            shard_id: Shard ID

        Returns:
            Transaction complexity (0=Low, 1=Medium, 2=High)
        """
        # In a real implementation, this would analyze recent transactions
        # For now, return a default value
        return 1  # Default to medium

    def _get_shard_size(self, shard_id: str) -> int:
        """
        Get the size of a shard.

        Args:
            shard_id: Shard ID

        Returns:
            Shard size (0=Small, 1=Medium, 2=Large)
        """
        # In a real implementation, this would get the number of nodes in the shard
        # For now, return a default value
        return 1  # Default to medium

    def _initialize_default_protocols(self) -> None:
        """Initialize default consensus protocols."""
        for consensus_type in ConsensusType:
            protocol_config = self.config.get(consensus_type.value, {})
            self.consensus_protocols[consensus_type] = (
                ConsensusFactory.create_consensus(consensus_type, protocol_config)
            )
