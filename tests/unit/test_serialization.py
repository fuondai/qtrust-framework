"""
Unit tests for serialization utilities.

This module contains unit tests for the serialization utilities
in the QTrust blockchain sharding framework.
"""

import os
import sys
import unittest
import json
import pickle
import gzip
import base64
from typing import Dict, List, Any

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.common.serialization import Serializer


class TestSerializer(unittest.TestCase):
    """Unit tests for the Serializer class."""

    def setUp(self):
        """Set up test environment."""
        self.test_data = {
            'string': 'test string',
            'integer': 42,
            'float': 3.14159,
            'boolean': True,
            'list': [1, 2, 3, 4, 5],
            'dict': {'a': 1, 'b': 2, 'c': 3},
            'nested': {
                'list': [{'x': 1}, {'y': 2}],
                'dict': {'a': [1, 2, 3]}
            }
        }

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        # Test serialization
        json_str = Serializer.to_json(self.test_data)
        self.assertIsInstance(json_str, str)
        
        # Test pretty serialization
        pretty_json = Serializer.to_json(self.test_data, pretty=True)
        self.assertIsInstance(pretty_json, str)
        self.assertGreater(len(pretty_json), len(json_str))
        
        # Test deserialization
        deserialized = Serializer.from_json(json_str)
        self.assertEqual(deserialized, self.test_data)
        
        # Test error handling
        with self.assertRaises(ValueError):
            Serializer.from_json("invalid json")

    def test_binary_serialization(self):
        """Test binary serialization and deserialization."""
        # Test serialization
        binary_data = Serializer.to_binary(self.test_data)
        self.assertIsInstance(binary_data, bytes)
        
        # Test deserialization
        deserialized = Serializer.from_binary(binary_data)
        self.assertEqual(deserialized, self.test_data)
        
        # Test error handling
        with self.assertRaises(ValueError):
            Serializer.from_binary(b"invalid binary data")

    def test_compression(self):
        """Test data compression and decompression."""
        # Test compression
        original_data = b"test data" * 100
        compressed = Serializer.compress(original_data)
        self.assertIsInstance(compressed, bytes)
        self.assertLess(len(compressed), len(original_data))
        
        # Test decompression
        decompressed = Serializer.decompress(compressed)
        self.assertEqual(decompressed, original_data)
        
        # Test error handling
        with self.assertRaises(ValueError):
            Serializer.decompress(b"invalid compressed data")

    def test_base64_encoding(self):
        """Test base64 encoding and decoding."""
        # Test encoding
        original_data = b"test data with special chars: !@#$%^&*()"
        encoded = Serializer.to_base64(original_data)
        self.assertIsInstance(encoded, str)
        
        # Test decoding
        decoded = Serializer.from_base64(encoded)
        self.assertEqual(decoded, original_data)
        
        # Test error handling
        with self.assertRaises(ValueError):
            Serializer.from_base64("invalid base64 data")

    def test_compact_serialization(self):
        """Test compact serialization and deserialization."""
        # Test serialization
        compact_str = Serializer.serialize_compact(self.test_data)
        self.assertIsInstance(compact_str, str)
        
        # Test deserialization
        deserialized = Serializer.deserialize_compact(compact_str)
        self.assertEqual(deserialized, self.test_data)
        
        # Test error handling
        with self.assertRaises(ValueError):
            Serializer.deserialize_compact("invalid compact data")

    def test_large_data_serialization(self):
        """Test serialization of large data structures."""
        # Create a large data structure
        large_data = {
            'array': list(range(10000)),
            'nested': {key: list(range(100)) for key in range(100)}
        }
        
        # Test compact serialization
        compact_str = Serializer.serialize_compact(large_data)
        deserialized = Serializer.deserialize_compact(compact_str)
        self.assertEqual(deserialized, large_data)


if __name__ == '__main__':
    unittest.main()
