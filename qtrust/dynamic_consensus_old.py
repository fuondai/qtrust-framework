#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Pipelined Consensus
This module implements a pipelined consensus protocol to overlap communication and computation.
"""

import time
import threading
import queue
import hashlib
import random
from typing import Dict, List, Tuple, Set, Optional, Any, Callable

from ..common.async_utils import AsyncProcessor, AsyncEvent, AsyncResult, AsyncBarrier
from ..common.serialization import SerializationManager
from .crypto_utils import CryptoManager
from .state_trie import AccountState


class PipelinedConsensus:
    """
    Implements a pipelined consensus protocol that overlaps communication and computation
    to reduce latency in the consensus process.
    """

    # Pipeline stages
    STAGE_PROPOSAL = 0
    STAGE_VALIDATION = 1
    STAGE_PRE_COMMIT = 2
    STAGE_COMMIT = 3

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the pipelined consensus protocol.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Default configuration
        self.pipeline_size = self.config.get("pipeline_size", 4)
        self.batch_size = self.config.get("batch_size", 100)
        self.validation_threads = self.config.get("validation_threads", 4)
        self.early_validation = self.config.get("early_validation", True)
        self.speculative_execution = self.config.get("speculative_execution", True)
        self.timeout = self.config.get("timeout", 5.0)

        # Pipeline data structures
        self.pipeline = [[] for _ in range(4)]  # One list per stage
        self.pipeline_lock = threading.RLock()

        # Transaction pools
        self.pending_txs = queue.Queue()
        self.validated_txs = queue.Queue()
        self.pre_committed_txs = queue.Queue()
        self.committed_txs = queue.Queue()

        # Async processor for validation
        self.async_processor = AsyncProcessor(num_workers=self.validation_threads)

        # Serialization manager
        self.serialization = SerializationManager()

        # Cryptographic manager
        self.crypto = CryptoManager(self.config.get("crypto", {}))

        # Account state for balance verification
        self.account_state = AccountState()

        # Node state
        self.node_id = self.config.get("node_id", f"node_{random.randint(1000, 9999)}")
        self.is_leader = self.config.get("is_leader", False)
        self.validators = self.config.get("validators", [])
        self.leader_id = self.config.get("leader_id", None)

        # Consensus state
        self.current_height = 0
        self.current_round = 0
        self.last_committed_block = None

        # Events
        self.new_proposal_event = AsyncEvent()
        self.new_validation_event = AsyncEvent()
        self.new_pre_commit_event = AsyncEvent()
        self.new_commit_event = AsyncEvent()

        # Running flag
        self.running = False
        self.threads = []

    def start(self):
        """
        Start the pipelined consensus protocol.
        """
        if self.running:
            return

        self.running = True
        self.async_processor.start()

        # Start pipeline stage threads
        self.threads = []

        proposal_thread = threading.Thread(target=self._proposal_loop)
        proposal_thread.daemon = True
        proposal_thread.start()
        self.threads.append(proposal_thread)

        validation_thread = threading.Thread(target=self._validation_loop)
        validation_thread.daemon = True
        validation_thread.start()
        self.threads.append(validation_thread)

        pre_commit_thread = threading.Thread(target=self._pre_commit_loop)
        pre_commit_thread.daemon = True
        pre_commit_thread.start()
        self.threads.append(pre_commit_thread)

        commit_thread = threading.Thread(target=self._commit_loop)
        commit_thread.daemon = True
        commit_thread.start()
        self.threads.append(commit_thread)

    def stop(self):
        """
        Stop the pipelined consensus protocol.
        """
        self.running = False

        for thread in self.threads:
            thread.join(timeout=2.0)

        self.async_processor.stop()
        self.threads = []

    def add_transaction(self, tx: Dict[str, Any]) -> bool:
        """
        Add a transaction to the pending pool.

        Args:
            tx: Transaction data

        Returns:
            True if the transaction was added, False otherwise
        """
        try:
            self.pending_txs.put(tx, block=False)
            return True
        except queue.Full:
            return False

    def add_transactions(self, txs: List[Dict[str, Any]]) -> int:
        """
        Add multiple transactions to the pending pool.

        Args:
            txs: List of transaction data

        Returns:
            Number of transactions added
        """
        count = 0
        for tx in txs:
            if self.add_transaction(tx):
                count += 1
        return count

    def get_committed_transactions(self, max_count: int = 100) -> List[Dict[str, Any]]:
        """
        Get committed transactions.

        Args:
            max_count: Maximum number of transactions to return

        Returns:
            List of committed transactions
        """
        result = []
        for _ in range(max_count):
            try:
                tx = self.committed_txs.get(block=False)
                result.append(tx)
                self.committed_txs.task_done()
            except queue.Empty:
                break
        return result

    def get_current_height(self) -> int:
        """
        Get the current blockchain height.

        Returns:
            Current height
        """
        return self.current_height

    def get_last_committed_block(self) -> Optional[Dict[str, Any]]:
        """
        Get the last committed block.

        Returns:
            Last committed block or None if no blocks have been committed
        """
        return self.last_committed_block

    def _proposal_loop(self):
        """
        Main loop for the proposal stage.
        """
        while self.running:
            try:
                # Check if we can add a new proposal to the pipeline
                with self.pipeline_lock:
                    if len(self.pipeline[self.STAGE_PROPOSAL]) >= self.pipeline_size:
                        # Pipeline full, wait
                        time.sleep(0.01)
                        continue

                # Collect transactions for a new proposal
                txs = []
                for _ in range(self.batch_size):
                    try:
                        tx = self.pending_txs.get(block=False)
                        txs.append(tx)
                        self.pending_txs.task_done()
                    except queue.Empty:
                        break

                if not txs:
                    # No transactions available, wait
                    time.sleep(0.01)
                    continue

                # Create a new proposal
                proposal = self._create_proposal(txs)

                # Add to pipeline
                with self.pipeline_lock:
                    self.pipeline[self.STAGE_PROPOSAL].append(proposal)

                # Notify validation stage
                self.new_proposal_event.set()
                self.new_proposal_event.clear()

                # If early validation is enabled, start validation immediately
                if self.early_validation:
                    self._start_early_validation(proposal)

            except Exception as e:
                print(f"Error in proposal loop: {e}")
                time.sleep(0.1)

    def _validation_loop(self):
        """
        Main loop for the validation stage.
        """
        while self.running:
            try:
                # Wait for new proposals
                if not self.new_proposal_event.wait(timeout=0.1):
                    continue

                # Process proposals
                with self.pipeline_lock:
                    proposals = self.pipeline[self.STAGE_PROPOSAL].copy()
                    if not proposals:
                        continue

                # Validate proposals
                for proposal in proposals:
                    if not self._is_proposal_validated(proposal):
                        self._validate_proposal(proposal)

                # Move validated proposals to the next stage
                with self.pipeline_lock:
                    for proposal in self.pipeline[self.STAGE_PROPOSAL].copy():
                        if self._is_proposal_validated(proposal):
                            self.pipeline[self.STAGE_PROPOSAL].remove(proposal)
                            self.pipeline[self.STAGE_VALIDATION].append(proposal)

                            # Add transactions to validated pool
                            for tx in proposal["transactions"]:
                                self.validated_txs.put(tx)

                # Notify pre-commit stage
                if self.pipeline[self.STAGE_VALIDATION]:
                    self.new_validation_event.set()
                    self.new_validation_event.clear()

            except Exception as e:
                print(f"Error in validation loop: {e}")
                time.sleep(0.1)

    def _pre_commit_loop(self):
        """
        Main loop for the pre-commit stage.
        """
        while self.running:
            try:
                # Wait for validated proposals
                if not self.new_validation_event.wait(timeout=0.1):
                    continue

                # Process validated proposals
                with self.pipeline_lock:
                    proposals = self.pipeline[self.STAGE_VALIDATION].copy()
                    if not proposals:
                        continue

                # Pre-commit proposals
                for proposal in proposals:
                    if not self._is_proposal_pre_committed(proposal):
                        self._pre_commit_proposal(proposal)

                # Move pre-committed proposals to the next stage
                with self.pipeline_lock:
                    for proposal in self.pipeline[self.STAGE_VALIDATION].copy():
                        if self._is_proposal_pre_committed(proposal):
                            self.pipeline[self.STAGE_VALIDATION].remove(proposal)
                            self.pipeline[self.STAGE_PRE_COMMIT].append(proposal)

                            # Add transactions to pre-committed pool
                            for tx in proposal["transactions"]:
                                self.pre_committed_txs.put(tx)

                # Notify commit stage
                if self.pipeline[self.STAGE_PRE_COMMIT]:
                    self.new_pre_commit_event.set()
                    self.new_pre_commit_event.clear()

                # If speculative execution is enabled, start commit immediately
                if self.speculative_execution:
                    self._start_speculative_commit()

            except Exception as e:
                print(f"Error in pre-commit loop: {e}")
                time.sleep(0.1)

    def _commit_loop(self):
        """
        Main loop for the commit stage.
        """
        while self.running:
            try:
                # Wait for pre-committed proposals
                if not self.new_pre_commit_event.wait(timeout=0.1):
                    continue

                # Process pre-committed proposals
                with self.pipeline_lock:
                    proposals = self.pipeline[self.STAGE_PRE_COMMIT].copy()
                    if not proposals:
                        continue

                # Commit proposals
                for proposal in proposals:
                    if not self._is_proposal_committed(proposal):
                        self._commit_proposal(proposal)

                # Move committed proposals out of the pipeline
                with self.pipeline_lock:
                    for proposal in self.pipeline[self.STAGE_PRE_COMMIT].copy():
                        if self._is_proposal_committed(proposal):
                            self.pipeline[self.STAGE_PRE_COMMIT].remove(proposal)

                            # Update consensus state
                            self.current_height = proposal["height"]
                            self.last_committed_block = proposal

                            # Add transactions to committed pool
                            for tx in proposal["transactions"]:
                                self.committed_txs.put(tx)

                # Notify listeners
                self.new_commit_event.set()
                self.new_commit_event.clear()

            except Exception as e:
                print(f"Error in commit loop: {e}")
                time.sleep(0.1)

    def _create_proposal(self, txs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a new proposal.

        Args:
            txs: List of transactions

        Returns:
            Proposal data
        """
        # Create proposal
        proposal = {
            "height": self.current_height + 1,
            "round": self.current_round,
            "timestamp": time.time(),
            "proposer": self.node_id,
            "transactions": txs,
            "prev_block_hash": self._get_prev_block_hash(),
            "merkle_root": self._compute_merkle_root(txs),
            "signature": None,
            "validators": {},
            "pre_commits": {},
            "commits": {},
        }

        # Sign the proposal
        proposal["signature"] = self._sign_proposal(proposal)

        return proposal

    def _validate_proposal(self, proposal: Dict[str, Any]) -> bool:
        """
        Validate a proposal.

        Args:
            proposal: Proposal data

        Returns:
            True if validation was successful, False otherwise
        """
        # Validate proposal structure
        if not self._validate_proposal_structure(proposal):
            return False

        # Validate proposal signature
        if not self._validate_proposal_signature(proposal):
            return False

        # Validate transactions
        valid_txs = self._validate_transactions(proposal["transactions"])

        # Add validation result
        proposal["validators"][self.node_id] = {
            "valid": valid_txs == len(proposal["transactions"]),
            "timestamp": time.time(),
            "signature": self._sign_validation(proposal, valid_txs),
        }

        return proposal["validators"][self.node_id]["valid"]

    def _pre_commit_proposal(self, proposal: Dict[str, Any]) -> bool:
        """
        Pre-commit a proposal.

        Args:
            proposal: Proposal data

        Returns:
            True if pre-commit was successful, False otherwise
        """
        # Check if we have enough validations
        valid_validations = 0
        for validator_id, validation in proposal["validators"].items():
            if validation["valid"] and self._validate_validation_signature(
                proposal, validator_id, validation
            ):
                valid_validations += 1

        # Need 2/3 of validators to agree
        if valid_validations < len(self.validators) * 2 // 3:
            return False

        # Add pre-commit
        proposal["pre_commits"][self.node_id] = {
            "timestamp": time.time(),
            "signature": self._sign_pre_commit(proposal),
        }

        return True

    def _commit_proposal(self, proposal: Dict[str, Any]) -> bool:
        """
        Commit a proposal.

        Args:
            proposal: Proposal data

        Returns:
            True if commit was successful, False otherwise
        """
        # Check if we have enough pre-commits
        valid_pre_commits = 0
        for validator_id, pre_commit in proposal["pre_commits"].items():
            if self._validate_pre_commit_signature(proposal, validator_id, pre_commit):
                valid_pre_commits += 1

        # Need 2/3 of validators to agree
        if valid_pre_commits < len(self.validators) * 2 // 3:
            return False

        # Add commit
        proposal["commits"][self.node_id] = {
            "timestamp": time.time(),
            "signature": self._sign_commit(proposal),
        }

        return True

    def _is_proposal_validated(self, proposal: Dict[str, Any]) -> bool:
        """
        Check if a proposal has been validated by this node.

        Args:
            proposal: Proposal data

        Returns:
            True if the proposal has been validated, False otherwise
        """
        return self.node_id in proposal["validators"]

    def _is_proposal_pre_committed(self, proposal: Dict[str, Any]) -> bool:
        """
        Check if a proposal has been pre-committed by this node.

        Args:
            proposal: Proposal data

        Returns:
            True if the proposal has been pre-committed, False otherwise
        """
        return self.node_id in proposal["pre_commits"]

    def _is_proposal_committed(self, proposal: Dict[str, Any]) -> bool:
        """
        Check if a proposal has been committed by this node.

        Args:
            proposal: Proposal data

        Returns:
            True if the proposal has been committed, False otherwise
        """
        return self.node_id in proposal["commits"]

    def _start_early_validation(self, proposal: Dict[str, Any]):
        """
        Start early validation of a proposal.

        Args:
            proposal: Proposal data
        """
        # Submit validation tasks to async processor
        for tx in proposal["transactions"]:
            self.async_processor.submit_task(self._validate_transaction, tx)

    def _start_speculative_commit(self):
        """
        Start speculative commit of proposals.
        """
        with self.pipeline_lock:
            for proposal in self.pipeline[self.STAGE_PRE_COMMIT]:
                if not self._is_proposal_committed(proposal):
                    # Check if we have enough pre-commits
                    valid_pre_commits = 0
                    for validator_id, pre_commit in proposal["pre_commits"].items():
                        if self._validate_pre_commit_signature(
                            proposal, validator_id, pre_commit
                        ):
                            valid_pre_commits += 1

                    # If we have enough pre-commits from other nodes, we can speculatively commit
                    if valid_pre_commits >= len(self.validators) * 2 // 3 - 1:
                        self._commit_proposal(proposal)

    def _validate_proposal_structure(self, proposal: Dict[str, Any]) -> bool:
        """
        Validate the structure of a proposal.

        Args:
            proposal: Proposal data

        Returns:
            True if the structure is valid, False otherwise
        """
        # Check required fields
        required_fields = [
            "height",
            "round",
            "timestamp",
            "proposer",
            "transactions",
            "prev_block_hash",
            "merkle_root",
            "signature",
        ]

        for field in required_fields:
            if field not in proposal:
                return False

        # Check height
        if proposal["height"] != self.current_height + 1:
            return False

        # Check timestamp
        if proposal["timestamp"] > time.time() + 5.0:  # Allow 5 seconds clock skew
            return False

        # Check previous block hash
        if proposal["prev_block_hash"] != self._get_prev_block_hash():
            return False

        # Check merkle root
        if proposal["merkle_root"] != self._compute_merkle_root(
            proposal["transactions"]
        ):
            return False

        return True

    def _validate_proposal_signature(self, proposal: Dict[str, Any]) -> bool:
        """
        Validate the signature of a proposal.

        Args:
            proposal: Proposal data

        Returns:
            True if the signature is valid, False otherwise
        """
        # Create a copy of the proposal without the signature for verification
        proposal_copy = proposal.copy()
        signature = proposal_copy.pop("signature")

        # Verify the signature using the proposer's public key
        return self.crypto.verify_signature(
            proposal_copy, signature, proposal["proposer"]
        )

    def _validate_transactions(self, txs: List[Dict[str, Any]]) -> int:
        """
        Validate a list of transactions.

        Args:
            txs: List of transactions

        Returns:
            Number of valid transactions
        """
        valid_count = 0
        for tx in txs:
            if self._validate_transaction(tx):
                valid_count += 1
        return valid_count

    def _validate_transaction(self, tx: Dict[str, Any]) -> bool:
        """
        Validate a single transaction.

        Args:
            tx: Transaction data

        Returns:
            True if the transaction is valid, False otherwise
        """
        # Check required fields
        required_fields = [
            "sender",
            "receiver",
            "amount",
            "nonce",
            "timestamp",
            "signature",
        ]

        for field in required_fields:
            if field not in tx:
                return False

        # Check timestamp (allow 5 seconds clock skew)
        if tx["timestamp"] > time.time() + 5.0:
            return False

        # Validate sender and receiver addresses
        if not self.crypto.validate_address(
            tx["sender"]
        ) or not self.crypto.validate_address(tx["receiver"]):
            return False

        # Verify transaction signature
        tx_copy = tx.copy()
        signature = tx_copy.pop("signature")
        if not self.crypto.verify_signature(tx_copy, signature, tx["sender"]):
            return False

        # Verify transaction against account state
        is_valid, _ = self.account_state.verify_transaction(tx)
        if not is_valid:
            return False

        return True

    def _validate_validation_signature(
        self, proposal: Dict[str, Any], validator_id: str, validation: Dict[str, Any]
    ) -> bool:
        """
        Validate the signature of a validation.

        Args:
            proposal: Proposal data
            validator_id: Validator ID
            validation: Validation data

        Returns:
            True if the signature is valid, False otherwise
        """
        # Create validation data for signature verification
        validation_data = {
            "proposal_height": proposal["height"],
            "proposal_round": proposal["round"],
            "valid": validation["valid"],
            "timestamp": validation["timestamp"],
        }

        # Verify the signature using the validator's public key
        return self.crypto.verify_signature(
            validation_data, validation["signature"], validator_id
        )

    def _validate_pre_commit_signature(
        self, proposal: Dict[str, Any], validator_id: str, pre_commit: Dict[str, Any]
    ) -> bool:
        """
        Validate the signature of a pre-commit.

        Args:
            proposal: Proposal data
            validator_id: Validator ID
            pre_commit: Pre-commit data

        Returns:
            True if the signature is valid, False otherwise
        """
        # Create pre-commit data for signature verification
        pre_commit_data = {
            "proposal_height": proposal["height"],
            "proposal_round": proposal["round"],
            "timestamp": pre_commit["timestamp"],
        }

        # Verify the signature using the validator's public key
        return self.crypto.verify_signature(
            pre_commit_data, pre_commit["signature"], validator_id
        )

    def _get_prev_block_hash(self) -> str:
        """
        Get the hash of the previous block.

        Returns:
            Hash of the previous block
        """
        if self.last_committed_block is None:
            return "0" * 64  # Genesis block

        # Hash the previous block using the crypto manager
        return self.crypto.hash_data(self.last_committed_block)

    def _compute_merkle_root(self, txs: List[Dict[str, Any]]) -> str:
        """
        Compute the Merkle root of a list of transactions.

        Args:
            txs: List of transactions

        Returns:
            Merkle root hash
        """
        # Use the crypto manager to compute the Merkle root
        return self.crypto.compute_merkle_root(txs)

    def _sign_proposal(self, proposal: Dict[str, Any]) -> str:
        """
        Sign a proposal.

        Args:
            proposal: Proposal data

        Returns:
            Signature
        """
        # Create a copy of the proposal without the signature for signing
        proposal_copy = proposal.copy()
        proposal_copy["signature"] = None

        # Sign the proposal using the crypto manager
        return self.crypto.sign_data(proposal_copy)

    def _sign_validation(self, proposal: Dict[str, Any], valid_txs: int) -> str:
        """
        Sign a validation result.

        Args:
            proposal: Proposal data
            valid_txs: Number of valid transactions

        Returns:
            Signature
        """
        # Create validation data for signing
        validation_data = {
            "proposal_height": proposal["height"],
            "proposal_round": proposal["round"],
            "valid": valid_txs == len(proposal["transactions"]),
            "timestamp": time.time(),
        }

        # Sign the validation data using the crypto manager
        return self.crypto.sign_data(validation_data)

    def _sign_pre_commit(self, proposal: Dict[str, Any]) -> str:
        """
        Sign a pre-commit.

        Args:
            proposal: Proposal data

        Returns:
            Signature
        """
        # Create pre-commit data for signing
        pre_commit_data = {
            "proposal_height": proposal["height"],
            "proposal_round": proposal["round"],
            "timestamp": time.time(),
        }

        # Sign the pre-commit data using the crypto manager
        return self.crypto.sign_data(pre_commit_data)

    def _sign_commit(self, proposal: Dict[str, Any]) -> str:
        """
        Sign a commit.

        Args:
            proposal: Proposal data

        Returns:
            Signature
        """
        # Create commit data for signing
        commit_data = {
            "proposal_height": proposal["height"],
            "proposal_round": proposal["round"],
            "timestamp": time.time(),
        }

        # Sign the commit data using the crypto manager
        return self.crypto.sign_data(commit_data)

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current pipeline.

        Returns:
            Dictionary with pipeline statistics
        """
        with self.pipeline_lock:
            stats = {
                "proposal_stage": len(self.pipeline[self.STAGE_PROPOSAL]),
                "validation_stage": len(self.pipeline[self.STAGE_VALIDATION]),
                "pre_commit_stage": len(self.pipeline[self.STAGE_PRE_COMMIT]),
                "commit_stage": 0,  # Committed proposals are removed from pipeline
                "pending_txs": self.pending_txs.qsize(),
                "validated_txs": self.validated_txs.qsize(),
                "pre_committed_txs": self.pre_committed_txs.qsize(),
                "committed_txs": self.committed_txs.qsize(),
                "current_height": self.current_height,
                "current_round": self.current_round,
            }

            return stats
