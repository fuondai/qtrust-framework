#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - State Trie
This module implements a simple in-memory state trie for balance verification.
"""

import hashlib
from typing import Dict, Any, Optional, Tuple


class StateTrie:
    """
    Implements a simple in-memory state trie for balance verification.
    Uses a Patricia Merkle Trie structure for efficient state storage.
    """

    def __init__(self):
        """
        Initialize the state trie.
        """
        self.root = {}
        self.root_hash = self._hash_node(self.root)

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the trie.

        Args:
            key: Key to look up

        Returns:
            Value or None if key not found
        """
        path = self._key_to_path(key)
        node = self.root

        for nibble in path:
            if nibble not in node:
                return None
            node = node[nibble]

            if isinstance(node, dict) and "_value" in node:
                return node["_value"]
            elif not isinstance(node, dict):
                return node

        return None

    def put(self, key: str, value: Any) -> None:
        """
        Put a value into the trie.

        Args:
            key: Key to store
            value: Value to store
        """
        path = self._key_to_path(key)
        self._put_node(self.root, path, value)
        self.root_hash = self._hash_node(self.root)

    def delete(self, key: str) -> bool:
        """
        Delete a key from the trie.

        Args:
            key: Key to delete

        Returns:
            True if key was deleted, False if key not found
        """
        path = self._key_to_path(key)
        result = self._delete_node(self.root, path)
        self.root_hash = self._hash_node(self.root)
        return result

    def get_root_hash(self) -> str:
        """
        Get the root hash of the trie.

        Returns:
            Root hash as a hexadecimal string
        """
        return self.root_hash

    def _key_to_path(self, key: str) -> list:
        """
        Convert a key to a path of nibbles.

        Args:
            key: Key to convert

        Returns:
            List of nibbles (hex digits)
        """
        # Hash the key to get a fixed-length path
        key_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()

        # Convert to nibbles (individual hex digits)
        return [int(c, 16) for c in key_hash]

    def _put_node(self, node: dict, path: list, value: Any) -> None:
        """
        Recursively put a value into a node.

        Args:
            node: Current node
            path: Remaining path
            value: Value to store
        """
        if not path:
            node["_value"] = value
            return

        nibble = path[0]
        remaining_path = path[1:]

        if nibble not in node:
            if not remaining_path:
                node[nibble] = value
            else:
                node[nibble] = {}
                self._put_node(node[nibble], remaining_path, value)
        else:
            if isinstance(node[nibble], dict):
                self._put_node(node[nibble], remaining_path, value)
            else:
                # Convert leaf to branch
                old_value = node[nibble]
                node[nibble] = {"_value": old_value}
                if remaining_path:
                    self._put_node(node[nibble], remaining_path, value)
                else:
                    node[nibble]["_value"] = value

    def _delete_node(self, node: dict, path: list) -> bool:
        """
        Recursively delete a value from a node.

        Args:
            node: Current node
            path: Remaining path

        Returns:
            True if value was deleted, False otherwise
        """
        if not path:
            if "_value" in node:
                del node["_value"]
                return True
            return False

        nibble = path[0]
        remaining_path = path[1:]

        if nibble not in node:
            return False

        if not remaining_path:
            if nibble in node:
                del node[nibble]
                return True
            return False

        if isinstance(node[nibble], dict):
            result = self._delete_node(node[nibble], remaining_path)

            # Clean up empty nodes
            if result and not node[nibble]:
                del node[nibble]

            return result

        return False

    def _hash_node(self, node: Any) -> str:
        """
        Hash a node.

        Args:
            node: Node to hash

        Returns:
            Hash as a hexadecimal string
        """
        if isinstance(node, dict):
            # Sort keys for deterministic hashing
            serialized = str(sorted(node.items()))
        else:
            serialized = str(node)

        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class AccountState:
    """
    Manages account state including balances and nonces.
    """

    def __init__(self):
        """
        Initialize the account state.
        """
        self.state_trie = StateTrie()
        self.accounts = {}  # Cache for faster access

    def get_balance(self, address: str) -> int:
        """
        Get the balance of an account.

        Args:
            address: Account address

        Returns:
            Account balance (0 if account does not exist)
        """
        if address in self.accounts:
            return self.accounts[address].get("balance", 0)

        account_data = self.state_trie.get(address)
        if account_data:
            self.accounts[address] = account_data
            return account_data.get("balance", 0)

        return 0

    def get_nonce(self, address: str) -> int:
        """
        Get the nonce of an account.

        Args:
            address: Account address

        Returns:
            Account nonce (0 if account does not exist)
        """
        if address in self.accounts:
            return self.accounts[address].get("nonce", 0)

        account_data = self.state_trie.get(address)
        if account_data:
            self.accounts[address] = account_data
            return account_data.get("nonce", 0)

        return 0

    def set_balance(self, address: str, balance: int) -> None:
        """
        Set the balance of an account.

        Args:
            address: Account address
            balance: New balance
        """
        account_data = self.state_trie.get(address) or {}
        account_data["balance"] = balance

        self.state_trie.put(address, account_data)
        self.accounts[address] = account_data

    def set_nonce(self, address: str, nonce: int) -> None:
        """
        Set the nonce of an account.

        Args:
            address: Account address
            nonce: New nonce
        """
        account_data = self.state_trie.get(address) or {}
        account_data["nonce"] = nonce

        self.state_trie.put(address, account_data)
        self.accounts[address] = account_data

    def increment_nonce(self, address: str) -> int:
        """
        Increment the nonce of an account.

        Args:
            address: Account address

        Returns:
            New nonce value
        """
        nonce = self.get_nonce(address) + 1
        self.set_nonce(address, nonce)
        return nonce

    def transfer(self, sender: str, receiver: str, amount: int) -> bool:
        """
        Transfer funds between accounts.

        Args:
            sender: Sender address
            receiver: Receiver address
            amount: Amount to transfer

        Returns:
            True if transfer was successful, False otherwise
        """
        sender_balance = self.get_balance(sender)

        if sender_balance < amount:
            return False

        receiver_balance = self.get_balance(receiver)

        self.set_balance(sender, sender_balance - amount)
        self.set_balance(receiver, receiver_balance + amount)
        self.increment_nonce(sender)

        return True

    def get_state_root(self) -> str:
        """
        Get the root hash of the state trie.

        Returns:
            Root hash as a hexadecimal string
        """
        return self.state_trie.get_root_hash()

    def verify_transaction(self, tx: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Verify a transaction against the current state.

        Args:
            tx: Transaction data

        Returns:
            Tuple of (is_valid, reason)
        """
        # Check required fields
        required_fields = ["sender", "receiver", "amount", "nonce", "signature"]
        for field in required_fields:
            if field not in tx:
                return False, f"Missing required field: {field}"

        # Check sender balance
        sender_balance = self.get_balance(tx["sender"])
        if sender_balance < tx["amount"]:
            return False, "Insufficient balance"

        # Check nonce
        sender_nonce = self.get_nonce(tx["sender"])
        if tx["nonce"] != sender_nonce + 1:
            return (
                False,
                f"Invalid nonce: expected {sender_nonce + 1}, got {tx['nonce']}",
            )

        return True, "Transaction is valid"
