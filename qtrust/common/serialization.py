"""
Serialization utilities for QTrust framework.

This module provides serialization and deserialization utilities for the QTrust
blockchain sharding framework, supporting various data formats and compression.
"""

import base64
import gzip
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union


class Serializer:
    """
    Serialization utility for QTrust data structures.

    This class provides methods for serializing and deserializing data in
    various formats, with support for compression and encryption.
    """

    @staticmethod
    def to_json(data: Any, pretty: bool = False) -> str:
        """
        Serialize data to JSON format.

        Args:
            data: Data to serialize
            pretty: Whether to format the JSON with indentation

        Returns:
            JSON string representation of the data
        """
        if pretty:
            return json.dumps(data, indent=2, sort_keys=True)
        return json.dumps(data)

    @staticmethod
    def from_json(json_str: str) -> Any:
        """
        Deserialize data from JSON format.

        Args:
            json_str: JSON string to deserialize

        Returns:
            Deserialized data

        Raises:
            ValueError: If the JSON string is invalid
        """
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    @staticmethod
    def to_binary(data: Any) -> bytes:
        """
        Serialize data to binary format using pickle.

        Args:
            data: Data to serialize

        Returns:
            Binary representation of the data
        """
        return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_binary(binary_data: bytes) -> Any:
        """
        Deserialize data from binary format.

        Args:
            binary_data: Binary data to deserialize

        Returns:
            Deserialized data

        Raises:
            ValueError: If the binary data is invalid
        """
        try:
            return pickle.loads(binary_data)
        except Exception as e:
            raise ValueError(f"Invalid binary data: {e}")

    @staticmethod
    def compress(data: bytes) -> bytes:
        """
        Compress binary data using gzip.

        Args:
            data: Binary data to compress

        Returns:
            Compressed binary data
        """
        return gzip.compress(data)

    @staticmethod
    def decompress(compressed_data: bytes) -> bytes:
        """
        Decompress binary data using gzip.

        Args:
            compressed_data: Compressed binary data

        Returns:
            Decompressed binary data

        Raises:
            ValueError: If the compressed data is invalid
        """
        try:
            return gzip.decompress(compressed_data)
        except Exception as e:
            raise ValueError(f"Invalid compressed data: {e}")

    @staticmethod
    def to_base64(data: bytes) -> str:
        """
        Encode binary data as base64 string.

        Args:
            data: Binary data to encode

        Returns:
            Base64-encoded string
        """
        return base64.b64encode(data).decode("utf-8")

    @staticmethod
    def from_base64(base64_str: str) -> bytes:
        """
        Decode base64 string to binary data.

        Args:
            base64_str: Base64-encoded string

        Returns:
            Decoded binary data

        Raises:
            ValueError: If the base64 string is invalid
        """
        try:
            return base64.b64decode(base64_str)
        except Exception as e:
            raise ValueError(f"Invalid base64 data: {e}")

    @classmethod
    def serialize_compact(cls, data: Any) -> str:
        """
        Serialize data to a compact string format.

        This method combines binary serialization, compression, and base64 encoding
        to create a compact string representation of the data.

        Args:
            data: Data to serialize

        Returns:
            Compact string representation of the data
        """
        binary_data = cls.to_binary(data)
        compressed_data = cls.compress(binary_data)
        return cls.to_base64(compressed_data)

    @classmethod
    def deserialize_compact(cls, compact_str: str) -> Any:
        """
        Deserialize data from a compact string format.

        This method reverses the process of serialize_compact, performing
        base64 decoding, decompression, and binary deserialization.

        Args:
            compact_str: Compact string to deserialize

        Returns:
            Deserialized data

        Raises:
            ValueError: If the compact string is invalid
        """
        try:
            binary_data = cls.from_base64(compact_str)
            decompressed_data = cls.decompress(binary_data)
            return cls.from_binary(decompressed_data)
        except Exception as e:
            raise ValueError(f"Failed to deserialize compact data: {e}")
