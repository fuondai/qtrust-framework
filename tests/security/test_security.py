"""
Security tests for QTrust framework.

This module contains security tests that verify the QTrust framework's
resistance to various attack vectors and security vulnerabilities.
"""

import os
import sys
import time
import unittest
import json
from typing import Dict, List, Any

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.qtrust_framework import QTrustFramework
from qtrust.trust.htdcm import HTDCM
from qtrust.crypto_utils import CryptoUtils


class TestSybilAttackResistance(unittest.TestCase):
    """Tests for resistance to Sybil attacks."""

    def setUp(self):
        """Set up test environment."""
        self.framework = QTrustFramework()
        self.htdcm = HTDCM()
        
        # Create results directory if it doesn't exist
        os.makedirs('security_results', exist_ok=True)

    def test_sybil_attack_detection(self):
        """Test detection of Sybil attack attempts."""
        # Configure test network
        config = {
            'shards': 8,
            'nodes_per_shard': 12,
            'sybil_nodes': 20,  # Simulated Sybil nodes
            'simulation_duration': 300  # seconds
        }
        
        # Run simulation with Sybil attack
        results = self.framework.run_security_simulation('sybil_attack', config)
        
        # Save results
        with open('security_results/sybil_attack_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Verify Sybil attack detection
        self.assertGreaterEqual(results['detection_rate'], 0.9)  # At least 90% detection rate
        self.assertLessEqual(results['false_positive_rate'], 0.05)  # Less than 5% false positives

    def test_sybil_attack_mitigation(self):
        """Test mitigation of Sybil attacks through trust mechanisms."""
        # Configure test network
        config = {
            'shards': 8,
            'nodes_per_shard': 12,
            'sybil_nodes': 20,  # Simulated Sybil nodes
            'simulation_duration': 300  # seconds
        }
        
        # Run simulation with Sybil attack mitigation
        results = self.framework.run_security_simulation('sybil_mitigation', config)
        
        # Save results
        with open('security_results/sybil_mitigation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Verify Sybil attack mitigation
        self.assertGreaterEqual(results['throughput_preservation'], 0.9)  # Maintain at least 90% throughput
        self.assertLessEqual(results['compromised_transactions'], 0.01)  # Less than 1% compromised transactions


class TestByzantineResistance(unittest.TestCase):
    """Tests for resistance to Byzantine behavior."""

    def setUp(self):
        """Set up test environment."""
        self.framework = QTrustFramework()
        
        # Create results directory if it doesn't exist
        os.makedirs('security_results', exist_ok=True)

    def test_byzantine_fault_tolerance(self):
        """Test Byzantine fault tolerance with varying percentages of Byzantine nodes."""
        results = {}
        
        for byzantine_percent in [0, 10, 20, 30, 33]:
            config = {
                'shards': 8,
                'nodes_per_shard': 12,
                'byzantine_percent': byzantine_percent,
                'simulation_duration': 300  # seconds
            }
            
            # Run simulation with Byzantine nodes
            simulation_results = self.framework.run_security_simulation('byzantine', config)
            results[f'byzantine_{byzantine_percent}'] = simulation_results
        
        # Save results
        with open('security_results/byzantine_tolerance_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Verify Byzantine fault tolerance
        # Should maintain consensus with up to 33% Byzantine nodes
        self.assertTrue(results['byzantine_30']['consensus_maintained'])
        
        # Should fail consensus with more than 33% Byzantine nodes
        self.assertFalse(results['byzantine_33']['consensus_maintained'])


class TestCrossShadAttackResistance(unittest.TestCase):
    """Tests for resistance to cross-shard attacks."""

    def setUp(self):
        """Set up test environment."""
        self.framework = QTrustFramework()
        
        # Create results directory if it doesn't exist
        os.makedirs('security_results', exist_ok=True)

    def test_cross_shard_double_spend(self):
        """Test resistance to cross-shard double-spend attacks."""
        # Configure test network
        config = {
            'shards': 16,
            'nodes_per_shard': 12,
            'attack_attempts': 100,
            'simulation_duration': 600  # seconds
        }
        
        # Run simulation with cross-shard double-spend attacks
        results = self.framework.run_security_simulation('cross_shard_double_spend', config)
        
        # Save results
        with open('security_results/cross_shard_double_spend_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Verify cross-shard double-spend prevention
        self.assertEqual(results['successful_attacks'], 0)  # No successful attacks
        self.assertGreaterEqual(results['detection_rate'], 0.99)  # At least 99% detection rate


class TestCryptographicSecurity(unittest.TestCase):
    """Tests for cryptographic security."""

    def setUp(self):
        """Set up test environment."""
        self.crypto_utils = CryptoUtils()
        
        # Create results directory if it doesn't exist
        os.makedirs('security_results', exist_ok=True)

    def test_quantum_resistance(self):
        """Test quantum resistance of cryptographic primitives."""
        # Test various cryptographic operations
        results = {
            'signature_schemes': {},
            'encryption_schemes': {},
            'hash_functions': {}
        }
        
        # Test signature schemes
        for scheme in ['dilithium', 'falcon', 'sphincs']:
            scheme_results = self.crypto_utils.test_signature_scheme(scheme)
            results['signature_schemes'][scheme] = scheme_results
            
        # Test encryption schemes
        for scheme in ['kyber', 'ntru', 'saber']:
            scheme_results = self.crypto_utils.test_encryption_scheme(scheme)
            results['encryption_schemes'][scheme] = scheme_results
            
        # Test hash functions
        for hash_fn in ['sha3_256', 'sha3_512', 'blake2b']:
            hash_results = self.crypto_utils.test_hash_function(hash_fn)
            results['hash_functions'][hash_fn] = hash_results
        
        # Save results
        with open('security_results/quantum_resistance_results.json', 'w') as f:
            json.dump(results, f, indent=2)
            
        # Verify quantum resistance
        for scheme, scheme_results in results['signature_schemes'].items():
            self.assertTrue(scheme_results['quantum_resistant'])
            
        for scheme, scheme_results in results['encryption_schemes'].items():
            self.assertTrue(scheme_results['quantum_resistant'])


if __name__ == '__main__':
    unittest.main()
