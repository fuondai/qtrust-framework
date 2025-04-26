"""
Crypto utilities for QTrust framework.

This module provides cryptographic utilities for the QTrust blockchain
sharding framework, including encryption, signatures, and hashing.
"""

import os
import hashlib
import base64
import json
import time
import uuid
from typing import Dict, List, Tuple, Optional, Any, Union

from Crypto.PublicKey import ECC, RSA
from Crypto.Signature import DSS, pkcs1_15
from Crypto.Hash import SHA256, SHA3_256
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad


class CryptoManager:
    """
    Cryptographic utility manager for QTrust framework.

    This class provides methods for key generation, encryption, decryption,
    signing, verification, and hashing operations.
    """

    def __init__(self, key_size: int = 2048, curve: str = "P-256"):
        """
        Initialize a crypto manager.

        Args:
            key_size: RSA key size in bits
            curve: Elliptic curve to use for ECC
        """
        self.key_size = key_size
        self.curve = curve
        self.rsa_keys: Dict[str, RSA.RsaKey] = {}
        self.ecc_keys: Dict[str, ECC.EccKey] = {}

    def generate_rsa_keypair(self, key_id: Optional[str] = None) -> str:
        """
        Generate an RSA key pair.

        Args:
            key_id: Optional identifier for the key pair

        Returns:
            Key ID for the generated key pair
        """
        if key_id is None:
            key_id = str(uuid.uuid4())

        key = RSA.generate(self.key_size)
        self.rsa_keys[key_id] = key
        return key_id

    def generate_ecc_keypair(self, key_id: Optional[str] = None) -> str:
        """
        Generate an ECC key pair.

        Args:
            key_id: Optional identifier for the key pair

        Returns:
            Key ID for the generated key pair
        """
        if key_id is None:
            key_id = str(uuid.uuid4())

        key = ECC.generate(curve=self.curve)
        self.ecc_keys[key_id] = key
        return key_id

    def export_rsa_public_key(self, key_id: str, format: str = "PEM") -> str:
        """
        Export an RSA public key.

        Args:
            key_id: Identifier for the key pair
            format: Export format ('PEM' or 'DER')

        Returns:
            Public key in the specified format

        Raises:
            ValueError: If the key ID is not found
        """
        if key_id not in self.rsa_keys:
            raise ValueError(f"RSA key ID {key_id} not found")

        key = self.rsa_keys[key_id]
        public_key = key.publickey()

        if format.upper() == "PEM":
            key_data = public_key.export_key(format="PEM")
            return key_data if isinstance(key_data, str) else key_data.decode("utf-8")
        elif format.upper() == "DER":
            key_data = public_key.export_key(format="DER")
            return base64.b64encode(
                key_data if isinstance(key_data, bytes) else key_data.encode("utf-8")
            ).decode("utf-8")
        else:
            raise ValueError(f"Unsupported format: {format}")

    def export_ecc_public_key(self, key_id: str, format: str = "PEM") -> str:
        """
        Export an ECC public key.

        Args:
            key_id: Identifier for the key pair
            format: Export format ('PEM' or 'DER')

        Returns:
            Public key in the specified format

        Raises:
            ValueError: If the key ID is not found
        """
        if key_id not in self.ecc_keys:
            raise ValueError(f"ECC key ID {key_id} not found")

        key = self.ecc_keys[key_id]
        public_key = key.public_key()

        if format.upper() == "PEM":
            key_data = public_key.export_key(format="PEM")
            return key_data if isinstance(key_data, str) else key_data.decode("utf-8")
        elif format.upper() == "DER":
            key_data = public_key.export_key(format="DER")
            return base64.b64encode(
                key_data if isinstance(key_data, bytes) else key_data.encode("utf-8")
            ).decode("utf-8")
        else:
            raise ValueError(f"Unsupported format: {format}")

    def rsa_encrypt(self, key_id: str, data: Union[str, bytes]) -> str:
        """
        Encrypt data using RSA.

        Args:
            key_id: Identifier for the key pair
            data: Data to encrypt

        Returns:
            Base64-encoded encrypted data

        Raises:
            ValueError: If the key ID is not found
        """
        if key_id not in self.rsa_keys:
            raise ValueError(f"RSA key ID {key_id} not found")

        key = self.rsa_keys[key_id]
        public_key = key.publickey()

        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Create cipher and encrypt
        cipher = PKCS1_OAEP.new(public_key)
        encrypted = cipher.encrypt(data)

        return base64.b64encode(encrypted).decode("utf-8")

    def rsa_decrypt(self, key_id: str, encrypted_data: str) -> bytes:
        """
        Decrypt data using RSA.

        Args:
            key_id: Identifier for the key pair
            encrypted_data: Base64-encoded encrypted data

        Returns:
            Decrypted data as bytes

        Raises:
            ValueError: If the key ID is not found
        """
        if key_id not in self.rsa_keys:
            raise ValueError(f"RSA key ID {key_id} not found")

        key = self.rsa_keys[key_id]

        # Decode base64
        encrypted = base64.b64decode(encrypted_data)

        # Create cipher and decrypt
        cipher = PKCS1_OAEP.new(key)
        decrypted = cipher.decrypt(encrypted)

        return decrypted

    def aes_encrypt(
        self, data: Union[str, bytes], key: Optional[bytes] = None
    ) -> Dict[str, str]:
        """
        Encrypt data using AES.

        Args:
            data: Data to encrypt
            key: Optional AES key (generated if not provided)

        Returns:
            Dictionary containing base64-encoded key, IV, and encrypted data
        """
        # Generate key if not provided
        if key is None:
            key = get_random_bytes(32)  # 256-bit key

        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Generate IV and create cipher
        iv = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_CBC, iv)

        # Pad and encrypt
        padded_data = pad(data, AES.block_size)
        encrypted = cipher.encrypt(padded_data)

        return {
            "key": base64.b64encode(key).decode("utf-8"),
            "iv": base64.b64encode(iv).decode("utf-8"),
            "data": base64.b64encode(encrypted).decode("utf-8"),
        }

    def aes_decrypt(self, encrypted_data: Dict[str, str]) -> bytes:
        """
        Decrypt data using AES.

        Args:
            encrypted_data: Dictionary containing base64-encoded key, IV, and encrypted data

        Returns:
            Decrypted data as bytes

        Raises:
            ValueError: If the encrypted data is invalid
        """
        # Check required fields
        required_fields = ["key", "iv", "data"]
        for field in required_fields:
            if field not in encrypted_data:
                raise ValueError(f"Missing required field: {field}")

        # Decode base64
        key = base64.b64decode(encrypted_data["key"])
        iv = base64.b64decode(encrypted_data["iv"])
        data = base64.b64decode(encrypted_data["data"])

        # Create cipher and decrypt
        cipher = AES.new(key, AES.MODE_CBC, iv)
        decrypted_padded = cipher.decrypt(data)

        # Unpad and return
        return unpad(decrypted_padded, AES.block_size)

    def rsa_sign(self, key_id: str, data: Union[str, bytes]) -> str:
        """
        Sign data using RSA.

        Args:
            key_id: Identifier for the key pair
            data: Data to sign

        Returns:
            Base64-encoded signature

        Raises:
            ValueError: If the key ID is not found
        """
        if key_id not in self.rsa_keys:
            raise ValueError(f"RSA key ID {key_id} not found")

        key = self.rsa_keys[key_id]

        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Create hash and sign
        h = SHA256.new(data)
        signature = pkcs1_15.new(key).sign(h)

        return base64.b64encode(signature).decode("utf-8")

    def rsa_verify(self, key_id: str, data: Union[str, bytes], signature: str) -> bool:
        """
        Verify an RSA signature.

        Args:
            key_id: Identifier for the key pair
            data: Original data
            signature: Base64-encoded signature

        Returns:
            True if the signature is valid, False otherwise
        """
        if key_id not in self.rsa_keys:
            raise ValueError(f"RSA key ID {key_id} not found")

        key = self.rsa_keys[key_id]
        public_key = key.publickey()

        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Decode signature
        sig_bytes = base64.b64decode(signature)

        # Create hash and verify
        h = SHA256.new(data)

        try:
            pkcs1_15.new(public_key).verify(h, sig_bytes)
            return True
        except (ValueError, TypeError):
            return False

    def ecc_sign(self, key_id: str, data: Union[str, bytes]) -> str:
        """
        Sign data using ECC.

        Args:
            key_id: Identifier for the key pair
            data: Data to sign

        Returns:
            Base64-encoded signature

        Raises:
            ValueError: If the key ID is not found
        """
        if key_id not in self.ecc_keys:
            raise ValueError(f"ECC key ID {key_id} not found")

        key = self.ecc_keys[key_id]

        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Create hash and sign
        h = SHA256.new(data)
        signer = DSS.new(key, "fips-186-3")
        signature = signer.sign(h)

        return base64.b64encode(signature).decode("utf-8")

    def ecc_verify(self, key_id: str, data: Union[str, bytes], signature: str) -> bool:
        """
        Verify an ECC signature.

        Args:
            key_id: Identifier for the key pair
            data: Original data
            signature: Base64-encoded signature

        Returns:
            True if the signature is valid, False otherwise
        """
        if key_id not in self.ecc_keys:
            raise ValueError(f"ECC key ID {key_id} not found")

        key = self.ecc_keys[key_id]

        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode("utf-8")

        # Decode signature
        sig_bytes = base64.b64decode(signature)

        # Create hash and verify
        h = SHA256.new(data)
        verifier = DSS.new(key, "fips-186-3")

        try:
            verifier.verify(h, sig_bytes)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def hash_sha256(data: Union[str, bytes]) -> str:
        """
        Compute SHA-256 hash of data.

        Args:
            data: Data to hash

        Returns:
            Hexadecimal hash string
        """
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode("utf-8")

        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hash_sha3_256(data: Union[str, bytes]) -> str:
        """
        Compute SHA3-256 hash of data.

        Args:
            data: Data to hash

        Returns:
            Hexadecimal hash string
        """
        # Convert string to bytes if necessary
        if isinstance(data, str):
            data = data.encode("utf-8")

        return hashlib.sha3_256(data).hexdigest()

    @staticmethod
    def hash_object(obj: Any) -> str:
        """
        Compute hash of a JSON-serializable object.

        Args:
            obj: Object to hash

        Returns:
            Hexadecimal hash string
        """
        # Convert to canonical JSON string
        json_str = json.dumps(obj, sort_keys=True)

        # Compute hash
        return CryptoManager.hash_sha256(json_str)


class CryptoUtils:
    """
    High-level cryptographic utilities for QTrust framework.

    This class provides methods for secure communication, transaction
    signing, and blockchain-specific cryptographic operations.
    """

    def __init__(self):
        """Initialize crypto utilities."""
        self.crypto_manager = CryptoManager()
        self.node_keys: Dict[str, str] = {}  # node_id -> key_id

    def generate_node_keys(self, node_id: str) -> Dict[str, str]:
        """
        Generate cryptographic keys for a node.

        Args:
            node_id: Unique identifier for the node

        Returns:
            Dictionary containing key information
        """
        # Generate RSA key pair
        rsa_key_id = self.crypto_manager.generate_rsa_keypair(f"{node_id}_rsa")

        # Generate ECC key pair
        ecc_key_id = self.crypto_manager.generate_ecc_keypair(f"{node_id}_ecc")

        # Store key IDs for the node
        self.node_keys[node_id] = {"rsa": rsa_key_id, "ecc": ecc_key_id}

        # Export public keys
        rsa_public = self.crypto_manager.export_rsa_public_key(rsa_key_id)
        ecc_public = self.crypto_manager.export_ecc_public_key(ecc_key_id)

        return {
            "node_id": node_id,
            "rsa_key_id": rsa_key_id,
            "ecc_key_id": ecc_key_id,
            "rsa_public_key": rsa_public,
            "ecc_public_key": ecc_public,
        }

    def sign_transaction(
        self, node_id: str, transaction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sign a transaction using a node's keys.

        Args:
            node_id: Unique identifier for the signing node
            transaction: Transaction data

        Returns:
            Transaction with added signature

        Raises:
            ValueError: If the node ID is not found
        """
        if node_id not in self.node_keys:
            raise ValueError(f"Node ID {node_id} not found")

        # Get key IDs for the node
        key_ids = self.node_keys[node_id]

        # Create a copy of the transaction
        signed_tx = transaction.copy()

        # Add timestamp if not present
        if "timestamp" not in signed_tx:
            signed_tx["timestamp"] = int(time.time() * 1000)

        # Add sender if not present
        if "sender" not in signed_tx:
            signed_tx["sender"] = node_id

        # Compute transaction hash
        tx_hash = self.crypto_manager.hash_object(signed_tx)
        signed_tx["hash"] = tx_hash

        # Sign the hash using ECC (faster and shorter signatures)
        signature = self.crypto_manager.ecc_sign(key_ids["ecc"], tx_hash)

        # Add signature to transaction
        signed_tx["signature"] = signature
        signed_tx["signature_type"] = "ecc"

        return signed_tx

    def verify_transaction(
        self, transaction: Dict[str, Any], node_id: Optional[str] = None
    ) -> bool:
        """
        Verify a transaction signature.

        Args:
            transaction: Signed transaction data
            node_id: Optional node ID to verify against

        Returns:
            True if the signature is valid, False otherwise
        """
        # Check required fields
        required_fields = ["hash", "signature", "signature_type", "sender"]
        for field in required_fields:
            if field not in transaction:
                return False

        # Get sender node ID
        sender_id = transaction["sender"]

        # If node_id is specified, check it matches the sender
        if node_id is not None and sender_id != node_id:
            return False

        # Check if we have keys for the sender
        if sender_id not in self.node_keys:
            return False

        # Get key IDs for the sender
        key_ids = self.node_keys[sender_id]

        # Get signature and hash
        signature = transaction["signature"]
        tx_hash = transaction["hash"]

        # Create a copy of the transaction without signature fields to verify hash
        tx_copy = transaction.copy()
        for field in ["hash", "signature", "signature_type"]:
            if field in tx_copy:
                del tx_copy[field]

        # Compute hash of the transaction data
        computed_hash = self.crypto_manager.hash_object(tx_copy)

        # Check if the hash matches
        if computed_hash != tx_hash:
            return False

        # Verify signature based on type
        if transaction["signature_type"] == "ecc":
            return self.crypto_manager.ecc_verify(key_ids["ecc"], tx_hash, signature)
        elif transaction["signature_type"] == "rsa":
            return self.crypto_manager.rsa_verify(key_ids["rsa"], tx_hash, signature)
        else:
            return False

    def encrypt_message(
        self, sender_id: str, recipient_id: str, message: Union[str, bytes]
    ) -> Dict[str, Any]:
        """
        Encrypt a message for secure communication between nodes.

        Args:
            sender_id: Unique identifier for the sending node
            recipient_id: Unique identifier for the receiving node
            message: Message to encrypt

        Returns:
            Dictionary containing encrypted message and metadata

        Raises:
            ValueError: If sender or recipient ID is not found
        """
        if sender_id not in self.node_keys:
            raise ValueError(f"Sender ID {sender_id} not found")

        if recipient_id not in self.node_keys:
            raise ValueError(f"Recipient ID {recipient_id} not found")

        # Get key IDs
        sender_keys = self.node_keys[sender_id]
        recipient_keys = self.node_keys[recipient_id]

        # Generate a random AES key for this message
        aes_key = get_random_bytes(32)

        # Encrypt the message with AES
        aes_result = self.crypto_manager.aes_encrypt(message, aes_key)

        # Encrypt the AES key with recipient's RSA public key
        encrypted_key = self.crypto_manager.rsa_encrypt(recipient_keys["rsa"], aes_key)

        # Sign the encrypted message hash
        message_hash = self.crypto_manager.hash_sha256(aes_result["data"])
        signature = self.crypto_manager.ecc_sign(sender_keys["ecc"], message_hash)

        # Create the final message
        encrypted_message = {
            "sender": sender_id,
            "recipient": recipient_id,
            "encrypted_key": encrypted_key,
            "iv": aes_result["iv"],
            "data": aes_result["data"],
            "hash": message_hash,
            "signature": signature,
            "timestamp": int(time.time() * 1000),
        }

        return encrypted_message

    def decrypt_message(self, encrypted_message: Dict[str, Any]) -> Optional[bytes]:
        """
        Decrypt a secure message.

        Args:
            encrypted_message: Encrypted message data

        Returns:
            Decrypted message as bytes, or None if verification fails

        Raises:
            ValueError: If the message format is invalid
        """
        # Check required fields
        required_fields = [
            "sender",
            "recipient",
            "encrypted_key",
            "iv",
            "data",
            "hash",
            "signature",
            "timestamp",
        ]
        for field in required_fields:
            if field not in encrypted_message:
                raise ValueError(f"Missing required field: {field}")

        # Get sender and recipient IDs
        sender_id = encrypted_message["sender"]
        recipient_id = encrypted_message["recipient"]

        # Check if we have keys for both
        if sender_id not in self.node_keys:
            raise ValueError(f"Sender ID {sender_id} not found")

        if recipient_id not in self.node_keys:
            raise ValueError(f"Recipient ID {recipient_id} not found")

        # Get key IDs
        sender_keys = self.node_keys[sender_id]
        recipient_keys = self.node_keys[recipient_id]

        # Verify the signature
        message_hash = encrypted_message["hash"]
        signature = encrypted_message["signature"]

        if not self.crypto_manager.ecc_verify(
            sender_keys["ecc"], message_hash, signature
        ):
            return None

        # Decrypt the AES key
        encrypted_key = encrypted_message["encrypted_key"]
        aes_key = self.crypto_manager.rsa_decrypt(recipient_keys["rsa"], encrypted_key)

        # Decrypt the message
        aes_data = {
            "key": base64.b64encode(aes_key).decode("utf-8"),
            "iv": encrypted_message["iv"],
            "data": encrypted_message["data"],
        }

        return self.crypto_manager.aes_decrypt(aes_data)

    def test_signature_scheme(self, scheme: str) -> Dict[str, Any]:
        """
        Test a signature scheme for security and performance.

        Args:
            scheme: Name of the signature scheme to test

        Returns:
            Dictionary containing test results

        Raises:
            ValueError: If the scheme is not supported
        """
        supported_schemes = ["dilithium", "falcon", "sphincs"]
        if scheme not in supported_schemes:
            raise ValueError(f"Unsupported signature scheme: {scheme}")

        # Placeholder for actual implementation
        # In a real implementation, this would test the scheme with
        # various parameters and return detailed results

        return {
            "scheme": scheme,
            "quantum_resistant": True,
            "signature_size_bytes": {
                "dilithium": 2420,
                "falcon": 1280,
                "sphincs": 41000,
            }.get(scheme, 0),
            "verification_time_ms": {
                "dilithium": 0.5,
                "falcon": 0.3,
                "sphincs": 13.5,
            }.get(scheme, 0),
            "security_level_bits": {
                "dilithium": 128,
                "falcon": 128,
                "sphincs": 192,
            }.get(scheme, 0),
        }

    def test_encryption_scheme(self, scheme: str) -> Dict[str, Any]:
        """
        Test an encryption scheme for security and performance.

        Args:
            scheme: Name of the encryption scheme to test

        Returns:
            Dictionary containing test results

        Raises:
            ValueError: If the scheme is not supported
        """
        supported_schemes = ["kyber", "ntru", "saber"]
        if scheme not in supported_schemes:
            raise ValueError(f"Unsupported encryption scheme: {scheme}")

        # Placeholder for actual implementation
        # In a real implementation, this would test the scheme with
        # various parameters and return detailed results

        return {
            "scheme": scheme,
            "quantum_resistant": True,
            "ciphertext_size_bytes": {"kyber": 1088, "ntru": 1138, "saber": 1088}.get(
                scheme, 0
            ),
            "decryption_time_ms": {"kyber": 0.12, "ntru": 0.18, "saber": 0.15}.get(
                scheme, 0
            ),
            "security_level_bits": {"kyber": 128, "ntru": 128, "saber": 128}.get(
                scheme, 0
            ),
        }

    def test_hash_function(self, hash_fn: str) -> Dict[str, Any]:
        """
        Test a hash function for security and performance.

        Args:
            hash_fn: Name of the hash function to test

        Returns:
            Dictionary containing test results

        Raises:
            ValueError: If the hash function is not supported
        """
        supported_hash_fns = ["sha3_256", "sha3_512", "blake2b"]
        if hash_fn not in supported_hash_fns:
            raise ValueError(f"Unsupported hash function: {hash_fn}")

        # Placeholder for actual implementation
        # In a real implementation, this would test the hash function with
        # various parameters and return detailed results

        return {
            "hash_function": hash_fn,
            "quantum_resistant": True,
            "digest_size_bytes": {"sha3_256": 32, "sha3_512": 64, "blake2b": 64}.get(
                hash_fn, 0
            ),
            "throughput_mbps": {"sha3_256": 800, "sha3_512": 500, "blake2b": 950}.get(
                hash_fn, 0
            ),
            "collision_resistance_bits": {
                "sha3_256": 128,
                "sha3_512": 256,
                "blake2b": 256,
            }.get(hash_fn, 0),
        }
