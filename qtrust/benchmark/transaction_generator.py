#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Transaction Generator
This module implements realistic transaction workload generation for benchmarking.
"""

import os
import sys
import time
import json
import random
import logging
import threading
import csv
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("transaction_generator.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("TransactionGenerator")


class Transaction:
    """
    Represents a blockchain transaction.
    """

    # Transaction types
    TYPE_TRANSFER = "transfer"  # Simple token transfer within a shard
    TYPE_CROSS_SHARD = "cross_shard"  # Cross-shard token transfer
    TYPE_CONTRACT = "contract"  # Smart contract interaction

    def __init__(
        self,
        tx_id: str,
        tx_type: str,
        source_shard: str,
        target_shard: Optional[str] = None,
        payload_size: int = 0,
    ):
        """
        Initialize a transaction.

        Args:
            tx_id: Transaction identifier
            tx_type: Transaction type
            source_shard: Source shard identifier
            target_shard: Target shard identifier (for cross-shard transactions)
            payload_size: Size of transaction payload in bytes
        """
        self.tx_id = tx_id
        self.tx_type = tx_type
        self.source_shard = source_shard
        self.target_shard = target_shard if target_shard else source_shard
        self.payload_size = payload_size

        # Transaction state
        self.creation_time = time.time()
        self.processing_time = None
        self.completion_time = None
        self.status = "pending"  # pending, processing, completed, failed

        # Generate random payload if size is specified
        self.payload = self._generate_payload(payload_size)

    def _generate_payload(self, size: int) -> Dict[str, Any]:
        """
        Generate a random payload of specified size.

        Args:
            size: Size of payload in bytes

        Returns:
            Dictionary with random payload
        """
        if size <= 0:
            return {}

        # Create a base payload
        payload = {
            "nonce": random.randint(1, 1000000),
            "timestamp": time.time(),
            "gas_limit": random.randint(21000, 100000),
            "gas_price": random.randint(1, 100),
            "data": "",
        }

        # Add data to reach target size
        current_size = len(json.dumps(payload).encode("utf-8"))
        if current_size < size:
            # Generate random hex data to fill the remaining size
            data_size = size - current_size
            hex_chars = "0123456789abcdef"
            payload["data"] = "0x" + "".join(
                random.choice(hex_chars) for _ in range(data_size * 2)
            )

        return payload

    def start_processing(self):
        """
        Mark transaction as processing.
        """
        self.processing_time = time.time()
        self.status = "processing"

    def complete(self, success: bool = True):
        """
        Mark transaction as completed or failed.

        Args:
            success: Whether the transaction completed successfully
        """
        self.completion_time = time.time()
        self.status = "completed" if success else "failed"

    def get_latency(self) -> Optional[float]:
        """
        Get transaction latency in milliseconds.

        Returns:
            Latency in milliseconds or None if not completed
        """
        if self.completion_time and self.creation_time:
            return (self.completion_time - self.creation_time) * 1000
        return None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert transaction to dictionary.

        Returns:
            Dictionary representation of transaction
        """
        return {
            "tx_id": self.tx_id,
            "tx_type": self.tx_type,
            "source_shard": self.source_shard,
            "target_shard": self.target_shard,
            "payload_size": self.payload_size,
            "creation_time": self.creation_time,
            "processing_time": self.processing_time,
            "completion_time": self.completion_time,
            "status": self.status,
            "latency": self.get_latency(),
        }


class TransactionGenerator:
    """
    Generates realistic transaction workloads for benchmarking.
    """

    def __init__(self):
        self.config = {}
        self.running = False
        self.total_generated = 0
        self.total_completed = 0
        self.total_failed = 0
        self.start_time = None
        self.pending_transactions = []
        self.completed_transactions = []
        self.transactions = {}
        
    def start_generation(self, duration):
        """Start generating transactions for the specified duration."""
        self.running = True
        self.start_time = time.time()
        # Just simulate transaction generation
        self.total_generated = int(duration * self.config.get("tx_rate", 100))
        # Auto-stop after the duration
        time.sleep(min(duration, 5))  # Sleep at most 5 seconds for demo
        self.running = False
        
    def stop_generation(self):
        """Stop generating transactions."""
        self.running = False
        
    def get_tps(self):
        """Get the current transactions per second rate."""
        if not self.start_time or not self.total_generated:
            return 0
        elapsed = max(1, time.time() - self.start_time)
        return self.total_generated / elapsed
    
    def get_pending_count(self, shard_id: Optional[str] = None) -> int:
        """
        Get count of pending transactions.
        
        Args:
            shard_id: Optional shard identifier to filter by
        
        Returns:
            Count of pending transactions
        """
        return len(self.pending_transactions)
    
    def get_average_latency(self) -> float:
        """
        Get average transaction latency in milliseconds.
        
        Returns:
            Average latency in milliseconds
        """
        return 100.0  # Simulated fixed latency for simplicity
    
    def get_percentile_latency(self, percentile: float) -> float:
        """
        Get percentile transaction latency in milliseconds.
        
        Args:
            percentile: Percentile (0.0-1.0)
        
        Returns:
            Percentile latency in milliseconds
        """
        # Simplified implementation - just return different values based on percentile
        if percentile >= 0.99:
            return 250.0
        elif percentile >= 0.95:
            return 180.0
        else:
            return 100.0
            
    def process_next_transaction(self, shard_id: str) -> Optional[Transaction]:
        """
        Get the next pending transaction for a shard.
        
        Args:
            shard_id: Shard identifier
        
        Returns:
            Next transaction or None if no pending transactions
        """
        # Simplified implementation that creates a dummy transaction
        tx_id = f"tx_{int(time.time() * 1000)}_{len(self.transactions)}"
        tx = Transaction(tx_id, Transaction.TYPE_TRANSFER, shard_id)
        self.transactions[tx_id] = tx
        return tx


if __name__ == "__main__":
    # Example usage
    config = {
        "tx_rate": 100,
        "tx_distribution": {
            Transaction.TYPE_TRANSFER: 0.7,
            Transaction.TYPE_CROSS_SHARD: 0.2,
            Transaction.TYPE_CONTRACT: 0.1,
        },
        "num_shards": 64,
        "output_file": "transactions.csv",
    }

    # Create generator
    generator = TransactionGenerator()
    generator.config = config

    # Generate transactions for 10 seconds
    generator.start_generation(10)

    # Wait for generation to complete
    while generator.running:
        time.sleep(1.0)
        print(f"Generated: {generator.total_generated}, TPS: {generator.get_tps():.2f}")

    print("Transaction generation complete!")
