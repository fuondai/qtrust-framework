"""
Unit tests for crypto utilities.

This module contains unit tests for the cryptographic utilities
in the QTrust blockchain sharding framework.
"""

import os
import sys
import unittest
import json
import base64
from typing import Dict, List, Any

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.crypto_utils import CryptoManager, CryptoUtils


class TestCryptoManager(unittest.TestCase):
    """Unit tests for the CryptoManager class."""

    def setUp(self):
        """Set up test environment."""
        self.crypto_manager = CryptoManager(key_size=2048, curve='P-256')

    def test_rsa_keypair_generation(self):
        """Test RSA key pair generation."""
        # Generate key pair
        key_id = self.crypto_manager.generate_rsa_keypair("test_rsa")
        self.assertEqual(key_id, "test_rsa")
        self.assertIn(key_id, self.crypto_manager.rsa_keys)
        
        # Export public key
        public_key_pem = self.crypto_manager.export_rsa_public_key(key_id, format='PEM')
        self.assertIsInstance(public_key_pem, str)
        self.assertIn("BEGIN PUBLIC KEY", public_key_pem)
        
        public_key_der = self.crypto_manager.export_rsa_public_key(key_id, format='DER')
        self.assertIsInstance(public_key_der, str)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            self.crypto_manager.export_rsa_public_key(key_id, format='INVALID')
            
        # Test invalid key ID
        with self.assertRaises(ValueError):
            self.crypto_manager.export_rsa_public_key("invalid_key_id")

    def test_ecc_keypair_generation(self):
        """Test ECC key pair generation."""
        # Generate key pair
        key_id = self.crypto_manager.generate_ecc_keypair("test_ecc")
        self.assertEqual(key_id, "test_ecc")
        self.assertIn(key_id, self.crypto_manager.ecc_keys)
        
        # Export public key
        public_key_pem = self.crypto_manager.export_ecc_public_key(key_id, format='PEM')
        self.assertIsInstance(public_key_pem, str)
        self.assertIn("BEGIN PUBLIC KEY", public_key_pem)
        
        public_key_der = self.crypto_manager.export_ecc_public_key(key_id, format='DER')
        self.assertIsInstance(public_key_der, str)
        
        # Test invalid format
        with self.assertRaises(ValueError):
            self.crypto_manager.export_ecc_public_key(key_id, format='INVALID')
            
        # Test invalid key ID
        with self.assertRaises(ValueError):
            self.crypto_manager.export_ecc_public_key("invalid_key_id")

    def test_rsa_encryption_decryption(self):
        """Test RSA encryption and decryption."""
        # Generate key pair
        key_id = self.crypto_manager.generate_rsa_keypair("test_rsa_crypt")
        
        # Test with string data
        original_data = "Test encryption data"
        encrypted = self.crypto_manager.rsa_encrypt(key_id, original_data)
        self.assertIsInstance(encrypted, str)
        
        decrypted = self.crypto_manager.rsa_decrypt(key_id, encrypted)
        self.assertEqual(decrypted.decode('utf-8'), original_data)
        
        # Test with binary data
        binary_data = b"Binary test data \x00\x01\x02\x03"
        encrypted = self.crypto_manager.rsa_encrypt(key_id, binary_data)
        decrypted = self.crypto_manager.rsa_decrypt(key_id, encrypted)
        self.assertEqual(decrypted, binary_data)
        
        # Test invalid key ID
        with self.assertRaises(ValueError):
            self.crypto_manager.rsa_encrypt("invalid_key_id", "test")
            
        with self.assertRaises(ValueError):
            self.crypto_manager.rsa_decrypt("invalid_key_id", encrypted)

    def test_aes_encryption_decryption(self):
        """Test AES encryption and decryption."""
        # Test with string data and auto-generated key
        original_data = "Test AES encryption data"
        encrypted = self.crypto_manager.aes_encrypt(original_data)
        
        # Check encrypted data format
        self.assertIn('key', encrypted)
        self.assertIn('iv', encrypted)
        self.assertIn('data', encrypted)
        
        # Decrypt and verify
        decrypted = self.crypto_manager.aes_decrypt(encrypted)
        self.assertEqual(decrypted.decode('utf-8'), original_data)
        
        # Test with binary data and provided key
        binary_data = b"Binary AES test data \x00\x01\x02\x03"
        key = os.urandom(32)  # 256-bit key
        encrypted = self.crypto_manager.aes_encrypt(binary_data, key)
        decrypted = self.crypto_manager.aes_decrypt(encrypted)
        self.assertEqual(decrypted, binary_data)
        
        # Test with invalid encrypted data
        with self.assertRaises(ValueError):
            self.crypto_manager.aes_decrypt({'invalid': 'format'})

    def test_rsa_signing_verification(self):
        """Test RSA signing and verification."""
        # Generate key pair
        key_id = self.crypto_manager.generate_rsa_keypair("test_rsa_sign")
        
        # Test with string data
        data = "Test signing data"
        signature = self.crypto_manager.rsa_sign(key_id, data)
        self.assertIsInstance(signature, str)
        
        # Verify signature
        self.assertTrue(self.crypto_manager.rsa_verify(key_id, data, signature))
        
        # Verify with modified data
        self.assertFalse(self.crypto_manager.rsa_verify(key_id, data + "modified", signature))
        
        # Verify with invalid signature
        self.assertFalse(self.crypto_manager.rsa_verify(key_id, data, "invalid_signature"))
        
        # Test invalid key ID
        with self.assertRaises(ValueError):
            self.crypto_manager.rsa_sign("invalid_key_id", data)
            
        with self.assertRaises(ValueError):
            self.crypto_manager.rsa_verify("invalid_key_id", data, signature)

    def test_ecc_signing_verification(self):
        """Test ECC signing and verification."""
        # Generate key pair
        key_id = self.crypto_manager.generate_ecc_keypair("test_ecc_sign")
        
        # Test with string data
        data = "Test ECC signing data"
        signature = self.crypto_manager.ecc_sign(key_id, data)
        self.assertIsInstance(signature, str)
        
        # Verify signature
        self.assertTrue(self.crypto_manager.ecc_verify(key_id, data, signature))
        
        # Verify with modified data
        self.assertFalse(self.crypto_manager.ecc_verify(key_id, data + "modified", signature))
        
        # Verify with invalid signature
        self.assertFalse(self.crypto_manager.ecc_verify(key_id, data, "invalid_signature"))
        
        # Test invalid key ID
        with self.assertRaises(ValueError):
            self.crypto_manager.ecc_sign("invalid_key_id", data)
            
        with self.assertRaises(ValueError):
            self.crypto_manager.ecc_verify("invalid_key_id", data, signature)

    def test_hashing(self):
        """Test hash functions."""
        # Test SHA-256
        data = "Test hash data"
        hash_result = self.crypto_manager.hash_sha256(data)
        self.assertIsInstance(hash_result, str)
        self.assertEqual(len(hash_result), 64)  # 32 bytes = 64 hex chars
        
        # Test SHA3-256
        hash_result = self.crypto_manager.hash_sha3_256(data)
        self.assertIsInstance(hash_result, str)
        self.assertEqual(len(hash_result), 64)
        
        # Test object hashing
        test_obj = {
            "field1": "value1",
            "field2": 123,
            "nested": {
                "array": [1, 2, 3]
            }
        }
        hash_result = self.crypto_manager.hash_object(test_obj)
        self.assertIsInstance(hash_result, str)
        self.assertEqual(len(hash_result), 64)
        
        # Test determinism
        hash1 = self.crypto_manager.hash_object(test_obj)
        hash2 = self.crypto_manager.hash_object(test_obj)
        self.assertEqual(hash1, hash2)


class TestCryptoUtils(unittest.TestCase):
    """Unit tests for the CryptoUtils class."""

    def setUp(self):
        """Set up test environment."""
        self.crypto_utils = CryptoUtils()

    def test_node_key_generation(self):
        """Test node key generation."""
        # Generate keys for a node
        node_id = "test_node_1"
        key_info = self.crypto_utils.generate_node_keys(node_id)
        
        # Check key info
        self.assertEqual(key_info['node_id'], node_id)
        self.assertIn('rsa_key_id', key_info)
        self.assertIn('ecc_key_id', key_info)
        self.assertIn('rsa_public_key', key_info)
        self.assertIn('ecc_public_key', key_info)
        
        # Check node keys are stored
        self.assertIn(node_id, self.crypto_utils.node_keys)
        self.assertIn('rsa', self.crypto_utils.node_keys[node_id])
        self.assertIn('ecc', self.crypto_utils.node_keys[node_id])

    def test_transaction_signing_verification(self):
        """Test transaction signing and verification."""
        # Generate keys for a node
        node_id = "test_node_2"
        self.crypto_utils.generate_node_keys(node_id)
        
        # Create a transaction
        transaction = {
            "from": "account_1",
            "to": "account_2",
            "amount": 100,
            "fee": 1,
            "nonce": 12345
        }
        
        # Sign the transaction
        signed_tx = self.crypto_utils.sign_transaction(node_id, transaction)
        
        # Check signature fields
        self.assertIn('hash', signed_tx)
        self.assertIn('signature', signed_tx)
        self.assertIn('signature_type', signed_tx)
        self.assertEqual(signed_tx['signature_type'], 'ecc')
        self.assertEqual(signed_tx['sender'], node_id)
        
        # Verify the transaction
        self.assertTrue(self.crypto_utils.verify_transaction(signed_tx))
        self.assertTrue(self.crypto_utils.verify_transaction(signed_tx, node_id))
        
        # Verify with wrong node ID
        self.assertFalse(self.crypto_utils.verify_transaction(signed_tx, "wrong_node_id"))
        
        # Modify transaction and verify
        modified_tx = signed_tx.copy()
        modified_tx['amount'] = 200
        self.assertFalse(self.crypto_utils.verify_transaction(modified_tx))
        
        # Test with invalid node ID
        with self.assertRaises(ValueError):
            self.crypto_utils.sign_transaction("invalid_node_id", transaction)

    def test_secure_messaging(self):
        """Test secure messaging between nodes."""
        # Generate keys for two nodes
        sender_id = "sender_node"
        recipient_id = "recipient_node"
        self.crypto_utils.generate_node_keys(sender_id)
        self.crypto_utils.generate_node_keys(recipient_id)
        
        # Create a message
        original_message = "This is a secure test message"
        
        # Encrypt the message
        encrypted = self.crypto_utils.encrypt_message(sender_id, recipient_id, original_message)
        
        # Check encrypted message format
        self.assertEqual(encrypted['sender'], sender_id)
        self.assertEqual(encrypted['recipient'], recipient_id)
        self.assertIn('encrypted_key', encrypted)
        self.assertIn('iv', encrypted)
        self.assertIn('data', encrypted)
        self.assertIn('hash', encrypted)
        self.assertIn('signature', encrypted)
        self.assertIn('timestamp', encrypted)
        
        # Decrypt the message
        decrypted = self.crypto_utils.decrypt_message(encrypted)
        self.assertEqual(decrypted.decode('utf-8'), original_message)
        
        # Test with invalid sender
        with self.assertRaises(ValueError):
            self.crypto_utils.encrypt_message("invalid_sender", recipient_id, original_message)
            
        # Test with invalid recipient
        with self.assertRaises(ValueError):
            self.crypto_utils.encrypt_message(sender_id, "invalid_recipient", original_message)
            
        # Test with invalid message format
        with self.assertRaises(ValueError):
            self.crypto_utils.decrypt_message({'invalid': 'format'})

    def test_quantum_resistance(self):
        """Test quantum resistance features."""
        # Test signature schemes
        for scheme in ['dilithium', 'falcon', 'sphincs']:
            result = self.crypto_utils.test_signature_scheme(scheme)
            self.assertEqual(result['scheme'], scheme)
            self.assertTrue(result['quantum_resistant'])
            self.assertGreater(result['signature_size_bytes'], 0)
            self.assertGreater(result['verification_time_ms'], 0)
            self.assertGreater(result['security_level_bits'], 0)
            
        # Test with invalid scheme
        with self.assertRaises(ValueError):
            self.crypto_utils.test_signature_scheme("invalid_scheme")
            
        # Test encryption schemes
        for scheme in ['kyber', 'ntru', 'saber']:
            result = self.crypto_utils.test_encryption_scheme(scheme)
            self.assertEqual(result['scheme'], scheme)
            self.assertTrue(result['quantum_resistant'])
            self.assertGreater(result['ciphertext_size_bytes'], 0)
            self.assertGreater(result['decryption_time_ms'], 0)
            self.assertGreater(result['security_level_bits'], 0)
            
        # Test with invalid scheme
        with self.assertRaises(ValueError):
            self.crypto_utils.test_encryption_scheme("invalid_scheme")
            
        # Test hash functions
        for hash_fn in ['sha3_256', 'sha3_512', 'blake2b']:
            result = self.crypto_utils.test_hash_function(hash_fn)
            self.assertEqual(result['hash_function'], hash_fn)
            self.assertTrue(result['quantum_resistant'])
            self.assertGreater(result['digest_size_bytes'], 0)
            self.assertGreater(result['throughput_mbps'], 0)
            self.assertGreater(result['collision_resistance_bits'], 0)
            
        # Test with invalid hash function
        with self.assertRaises(ValueError):
            self.crypto_utils.test_hash_function("invalid_hash_fn")


if __name__ == '__main__':
    unittest.main()
