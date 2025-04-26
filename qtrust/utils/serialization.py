# src/qtrust/common/serialization.py

import json
import zlib
import pickle
import struct
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import protobuf

    HAS_PROTOBUF = True
except ImportError:
    HAS_PROTOBUF = False


class SerializationManager:
    """
    Manages serialization and deserialization of messages with optimized performance.
    Supports multiple serialization formats and compression for large payloads.
    """

    # Serialization formats
    FORMAT_JSON = 0
    FORMAT_PICKLE = 1
    FORMAT_PROTOBUF = 2

    # Compression thresholds and levels
    DEFAULT_COMPRESSION_THRESHOLD = 1024  # bytes
    DEFAULT_COMPRESSION_LEVEL = 6  # 0-9, higher means more compression but slower

    def __init__(
        self,
        default_format: int = FORMAT_JSON,
        compression_threshold: int = DEFAULT_COMPRESSION_THRESHOLD,
        compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    ):
        """
        Initialize the serialization manager.

        Args:
            default_format: Default serialization format to use
            compression_threshold: Size threshold in bytes for applying compression
            compression_level: Compression level (0-9)
        """
        self.default_format = default_format
        self.compression_threshold = compression_threshold
        self.compression_level = compression_level

        # Validate protobuf availability
        if default_format == self.FORMAT_PROTOBUF and not HAS_PROTOBUF:
            print("Warning: Protobuf not available, falling back to JSON")
            self.default_format = self.FORMAT_JSON

    def serialize(self, data: Any, format_type: Optional[int] = None) -> bytes:
        """
        Serialize data to bytes using the specified format.

        Args:
            data: Data to serialize
            format_type: Serialization format to use (defaults to self.default_format)

        Returns:
            Serialized data as bytes
        """
        if format_type is None:
            format_type = self.default_format

        # Serialize based on format
        if format_type == self.FORMAT_JSON:
            serialized = json.dumps(data).encode("utf-8")
        elif format_type == self.FORMAT_PICKLE:
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        elif format_type == self.FORMAT_PROTOBUF and HAS_PROTOBUF:
            # This would be implemented with actual protobuf code
            # For now, fall back to JSON
            serialized = json.dumps(data).encode("utf-8")
        else:
            # Default to JSON
            serialized = json.dumps(data).encode("utf-8")

        # Apply compression if needed
        if len(serialized) >= self.compression_threshold:
            compressed = zlib.compress(serialized, level=self.compression_level)

            # Only use compression if it actually reduces size
            if len(compressed) < len(serialized):
                # Format: [1-byte format][1-byte compressed flag][4-byte length][data]
                header = struct.pack("!BBI", format_type, 1, len(serialized))
                return header + compressed

        # Format: [1-byte format][1-byte compressed flag][4-byte length][data]
        header = struct.pack("!BBI", format_type, 0, len(serialized))
        return header + serialized

    def deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes to data.

        Args:
            data: Serialized data as bytes

        Returns:
            Deserialized data
        """
        # Extract header
        format_type, compressed, original_length = struct.unpack("!BBI", data[:6])
        payload = data[6:]

        # Decompress if needed
        if compressed:
            payload = zlib.decompress(payload)

        # Deserialize based on format
        if format_type == self.FORMAT_JSON:
            return json.loads(payload.decode("utf-8"))
        elif format_type == self.FORMAT_PICKLE:
            return pickle.loads(payload)
        elif format_type == self.FORMAT_PROTOBUF and HAS_PROTOBUF:
            # This would be implemented with actual protobuf code
            # For now, fall back to JSON
            return json.loads(payload.decode("utf-8"))
        else:
            # Default to JSON
            return json.loads(payload.decode("utf-8"))

    def serialize_incremental(
        self, base_data: Dict[str, Any], updates: Dict[str, Any]
    ) -> bytes:
        """
        Serialize only the updates to a base data structure.
        Useful for reducing bandwidth when sending updates to large data structures.

        Args:
            base_data: Original data structure
            updates: Only the fields that have changed

        Returns:
            Serialized incremental update as bytes
        """
        # Create a minimal update package
        update_package = {"type": "incremental", "updates": updates}

        return self.serialize(update_package)

    def apply_incremental(
        self, base_data: Dict[str, Any], serialized_updates: bytes
    ) -> Dict[str, Any]:
        """
        Apply incremental updates to a base data structure.

        Args:
            base_data: Original data structure to update
            serialized_updates: Serialized incremental update

        Returns:
            Updated data structure
        """
        # Deserialize the update package
        update_package = self.deserialize(serialized_updates)

        # Verify it's an incremental update
        if (
            not isinstance(update_package, dict)
            or update_package.get("type") != "incremental"
        ):
            raise ValueError("Not an incremental update package")

        # Apply updates
        updates = update_package.get("updates", {})
        result = base_data.copy()

        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                # Recursively update nested dictionaries
                result[key] = self._update_dict(result[key], value)
            else:
                # Direct update for non-dict values
                result[key] = value

        return result

    def _update_dict(
        self, base: Dict[str, Any], updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Helper method to recursively update nested dictionaries.

        Args:
            base: Base dictionary
            updates: Updates to apply

        Returns:
            Updated dictionary
        """
        result = base.copy()

        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(result.get(key), dict):
                # Recursively update nested dictionaries
                result[key] = self._update_dict(result[key], value)
            else:
                # Direct update for non-dict values
                result[key] = value

        return result

    def estimate_size(self, data: Any, format_type: Optional[int] = None) -> int:
        """
        Estimate the serialized size of data without actually serializing it.
        Useful for deciding whether to split large messages.

        Args:
            data: Data to estimate size for
            format_type: Serialization format to use

        Returns:
            Estimated size in bytes
        """
        # This is a rough estimate and will vary by format
        if isinstance(data, dict):
            # Estimate dictionary size
            size = 2  # {}
            for k, v in data.items():
                # Key + separator + value + comma
                size += len(str(k)) + 2 + self.estimate_size(v) + 1
        elif isinstance(data, list):
            # Estimate list size
            size = 2  # []
            for item in data:
                # Item + comma
                size += self.estimate_size(item) + 1
        elif isinstance(data, str):
            # String size plus quotes
            size = len(data) + 2
        elif isinstance(data, (int, float, bool, type(None))):
            # Simple types
            size = len(str(data))
        else:
            # Complex object, harder to estimate
            size = 100  # Arbitrary default

        return size

    def should_split_message(self, data: Any, max_size: int = 1024 * 1024) -> bool:
        """
        Determine if a message should be split based on estimated size.

        Args:
            data: Data to check
            max_size: Maximum message size in bytes

        Returns:
            True if message should be split, False otherwise
        """
        return self.estimate_size(data) > max_size

    def split_message(self, data: Any, max_size: int = 1024 * 1024) -> List[bytes]:
        """
        Split a large message into smaller chunks.

        Args:
            data: Data to split
            max_size: Maximum chunk size in bytes

        Returns:
            List of serialized chunks
        """
        if not self.should_split_message(data, max_size):
            # No need to split
            return [self.serialize(data)]

        # For dictionaries, split by keys
        if isinstance(data, dict):
            chunks = []
            current_chunk = {}
            current_size = 0

            for key, value in data.items():
                item_size = self.estimate_size({key: value})

                if item_size > max_size:
                    # Individual item too large, needs special handling
                    # For simplicity, we'll just include it as is
                    chunks.append(self.serialize({key: value}))
                elif current_size + item_size > max_size:
                    # Current chunk would exceed max size, start a new one
                    chunks.append(self.serialize(current_chunk))
                    current_chunk = {key: value}
                    current_size = item_size
                else:
                    # Add to current chunk
                    current_chunk[key] = value
                    current_size += item_size

            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(self.serialize(current_chunk))

            return chunks

        # For lists, split by items
        elif isinstance(data, list):
            chunks = []
            current_chunk = []
            current_size = 0

            for item in data:
                item_size = self.estimate_size(item)

                if item_size > max_size:
                    # Individual item too large, needs special handling
                    chunks.append(self.serialize([item]))
                elif current_size + item_size > max_size:
                    # Current chunk would exceed max size, start a new one
                    chunks.append(self.serialize(current_chunk))
                    current_chunk = [item]
                    current_size = item_size
                else:
                    # Add to current chunk
                    current_chunk.append(item)
                    current_size += item_size

            # Add the last chunk if not empty
            if current_chunk:
                chunks.append(self.serialize(current_chunk))

            return chunks

        # For other types, just serialize as is
        return [self.serialize(data)]

    def merge_chunks(self, chunks: List[bytes]) -> Any:
        """
        Merge chunks back into a single data structure.

        Args:
            chunks: List of serialized chunks

        Returns:
            Merged data structure
        """
        if not chunks:
            return None

        if len(chunks) == 1:
            return self.deserialize(chunks[0])

        # Deserialize all chunks
        deserialized = [self.deserialize(chunk) for chunk in chunks]

        # Determine the type of the original data
        if all(isinstance(d, dict) for d in deserialized):
            # Merge dictionaries
            result = {}
            for d in deserialized:
                result.update(d)
            return result
        elif all(isinstance(d, list) for d in deserialized):
            # Merge lists
            result = []
            for d in deserialized:
                result.extend(d)
            return result
        else:
            # Mixed types, just return as list
            return deserialized
