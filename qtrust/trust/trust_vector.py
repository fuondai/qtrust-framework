"""
Implementation of a TrustVector class for the QTrust framework.

This module provides a multi-dimensional trust vector implementation.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
import json
import hashlib

logger = logging.getLogger(__name__)


class TrustVector:
    """A vector of trust values across multiple dimensions."""

    def __init__(
        self,
        dimensions: List[str] = None,
        initial_value: float = 0.5,
        weights: Dict[str, float] = None,
        decay_rate: float = 0.99,
        min_value: float = 0.0,
        max_value: float = 1.0,
    ):
        """
        Initialize a trust vector.

        Args:
            dimensions: List of dimension names
            initial_value: Initial trust value for all dimensions
            weights: Dictionary of dimension weights
            decay_rate: Rate at which trust decays over time
            min_value: Minimum trust value
            max_value: Maximum trust value
        """
        self.dimensions = dimensions or ["reliability", "performance", "security"]
        self.min_value = min_value
        self.max_value = max_value
        self.decay_rate = decay_rate

        # Initialize trust values
        self.values = {dim: initial_value for dim in self.dimensions}
        
        # Initialize history
        self.history = {dim: [(0, initial_value)] for dim in self.dimensions}
        
        # Initialize dimension weights
        if weights:
            self.weights = {dim: weights.get(dim, 1.0) for dim in self.dimensions}
        else:
            self.weights = {dim: 1.0 for dim in self.dimensions}
            
        # Initialize update counts
        self.update_counts = {dim: 0 for dim in self.dimensions}
        
        # Track trends for each dimension
        self.trends = {dim: 0.0 for dim in self.dimensions}
        
        # Metadata
        self.metadata = {}
        
        logger.debug(f"Initialized TrustVector with dimensions: {self.dimensions}")

    def update_dimension(
        self, dimension: str, value: float, timestamp: Optional[float] = None
    ) -> bool:
        """
        Update trust value for a specific dimension.

        Args:
            dimension: The dimension to update
            value: New trust value
            timestamp: Optional timestamp for the update

        Returns:
            Whether the update was successful
        """
        if dimension not in self.dimensions:
            logger.warning(f"Dimension {dimension} not found in trust vector")
            return False

        # Clamp value to valid range
        clamped_value = max(min(value, self.max_value), self.min_value)
        
        # Use current time if timestamp not provided
        if timestamp is None:
            timestamp = len(self.history[dimension])
            
        # Update value
        self.values[dimension] = clamped_value
        
        # Update history
        self.history[dimension].append((timestamp, clamped_value))
        
        # Update count
        self.update_counts[dimension] += 1
        
        # Update trend
        self._update_trend(dimension)
        
        logger.debug(f"Updated dimension {dimension} to {clamped_value:.3f}")
        return True

    def update_all(self, values: Dict[str, float], timestamp: Optional[float] = None) -> bool:
        """
        Update multiple dimensions at once.

        Args:
            values: Dictionary of dimension values
            timestamp: Optional timestamp for the update

        Returns:
            Whether all updates were successful
        """
        success = True
        for dimension, value in values.items():
            if not self.update_dimension(dimension, value, timestamp):
                success = False
        return success

    def get_value(self, dimension: str) -> Optional[float]:
        """
        Get trust value for a specific dimension.

        Args:
            dimension: The dimension to get

        Returns:
            Trust value or None if dimension not found
        """
        return self.values.get(dimension)

    def get_weighted_average(self) -> float:
        """
        Calculate weighted average of all trust dimensions.

        Returns:
            Weighted average trust value
        """
        total_weight = sum(self.weights.values())
        if total_weight == 0:
            return 0.0
            
        weighted_sum = sum(
            self.values[dim] * self.weights[dim] for dim in self.dimensions
        )
        
        return weighted_sum / total_weight

    def decay(self, dimensions: List[str] = None) -> None:
        """
        Apply decay to trust values.

        Args:
            dimensions: Optional list of dimensions to decay (None for all)
        """
        dims_to_decay = dimensions or self.dimensions
        timestamp = max(
            len(self.history[dim]) for dim in self.dimensions
        )
        
        for dimension in dims_to_decay:
            if dimension in self.dimensions:
                # Apply decay
                self.values[dimension] *= self.decay_rate
                
                # Clamp to valid range
                self.values[dimension] = max(
                    min(self.values[dimension], self.max_value), self.min_value
                )
                
                # Update history
                self.history[dimension].append((timestamp, self.values[dimension]))
                
                # Update trend
                self._update_trend(dimension)
                
        logger.debug(f"Applied decay to dimensions: {dims_to_decay}")

    def _update_trend(self, dimension: str, window: int = 5) -> None:
        """
        Update trend for a dimension.

        Args:
            dimension: The dimension to update
            window: Number of recent values to consider
        """
        if dimension not in self.dimensions:
            return
            
        # Get recent history
        history = self.history[dimension]
        if len(history) < 2:
            self.trends[dimension] = 0.0
            return
            
        # Get last window entries
        recent = history[-min(window, len(history)):]
        timestamps = [t for t, _ in recent]
        values = [v for _, v in recent]
        
        # Calculate trend using linear regression
        if len(values) <= 1:
            self.trends[dimension] = 0.0
            return
            
        try:
            # Use numpy for calculation
            x = np.array(timestamps)
            y = np.array(values)
            
            if np.std(x) == 0:
                self.trends[dimension] = 0.0
                return
                
            # Calculate slope
            slope, _ = np.polyfit(x, y, 1)
            
            # Normalize to [-1, 1]
            max_possible_slope = (self.max_value - self.min_value) / (max(x) - min(x)) if max(x) > min(x) else 1.0
            normalized_slope = min(max(slope / max_possible_slope, -1.0), 1.0)
            
            self.trends[dimension] = normalized_slope
            
        except (np.linalg.LinAlgError, TypeError, ValueError) as e:
            logger.warning(f"Error calculating trend for {dimension}: {e}")
            self.trends[dimension] = 0.0

    def get_trend(self, dimension: str) -> Optional[float]:
        """
        Get trend for a specific dimension.

        Args:
            dimension: The dimension to get

        Returns:
            Trend value or None if dimension not found
        """
        return self.trends.get(dimension)

    def set_weight(self, dimension: str, weight: float) -> bool:
        """
        Set weight for a specific dimension.

        Args:
            dimension: The dimension to update
            weight: New weight value

        Returns:
            Whether the update was successful
        """
        if dimension not in self.dimensions:
            logger.warning(f"Dimension {dimension} not found in trust vector")
            return False
            
        self.weights[dimension] = max(0.0, weight)
        logger.debug(f"Set weight for dimension {dimension} to {weight:.3f}")
        return True

    def get_weight(self, dimension: str) -> Optional[float]:
        """
        Get weight for a specific dimension.

        Args:
            dimension: The dimension to get

        Returns:
            Weight value or None if dimension not found
        """
        return self.weights.get(dimension)

    def add_dimension(
        self, dimension: str, initial_value: float = 0.5, weight: float = 1.0
    ) -> bool:
        """
        Add a new dimension to the trust vector.

        Args:
            dimension: The dimension to add
            initial_value: Initial trust value
            weight: Weight value

        Returns:
            Whether the addition was successful
        """
        if dimension in self.dimensions:
            logger.warning(f"Dimension {dimension} already exists in trust vector")
            return False
            
        # Add dimension
        self.dimensions.append(dimension)
        
        # Initialize values
        self.values[dimension] = max(min(initial_value, self.max_value), self.min_value)
        self.weights[dimension] = max(0.0, weight)
        self.history[dimension] = [(0, self.values[dimension])]
        self.update_counts[dimension] = 0
        self.trends[dimension] = 0.0
        
        logger.debug(f"Added dimension {dimension} with initial value {initial_value:.3f}")
        return True

    def remove_dimension(self, dimension: str) -> bool:
        """
        Remove a dimension from the trust vector.

        Args:
            dimension: The dimension to remove

        Returns:
            Whether the removal was successful
        """
        if dimension not in self.dimensions:
            logger.warning(f"Dimension {dimension} not found in trust vector")
            return False
            
        # Remove dimension
        self.dimensions.remove(dimension)
        
        # Remove values
        del self.values[dimension]
        del self.weights[dimension]
        del self.history[dimension]
        del self.update_counts[dimension]
        del self.trends[dimension]
        
        logger.debug(f"Removed dimension {dimension}")
        return True

    def get_dimensions(self) -> List[str]:
        """
        Get all dimensions.

        Returns:
            List of dimension names
        """
        return self.dimensions.copy()

    def get_update_count(self, dimension: str) -> Optional[int]:
        """
        Get number of updates for a dimension.

        Args:
            dimension: The dimension to check

        Returns:
            Update count or None if dimension not found
        """
        return self.update_counts.get(dimension)

    def get_history(self, dimension: str) -> Optional[List[Tuple[float, float]]]:
        """
        Get history for a dimension.

        Args:
            dimension: The dimension to get

        Returns:
            History as list of (timestamp, value) tuples or None if dimension not found
        """
        return self.history.get(dimension, []).copy()

    def reset(self, value: Optional[float] = None) -> None:
        """
        Reset trust vector to initial state.

        Args:
            value: Optional value to reset all dimensions to
        """
        reset_value = value if value is not None else 0.5
        reset_value = max(min(reset_value, self.max_value), self.min_value)
        
        # Reset values
        for dimension in self.dimensions:
            self.values[dimension] = reset_value
            self.history[dimension] = [(0, reset_value)]
            self.update_counts[dimension] = 0
            self.trends[dimension] = 0.0
            
        logger.debug(f"Reset trust vector to {reset_value:.3f}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trust vector to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "dimensions": self.dimensions.copy(),
            "values": self.values.copy(),
            "weights": self.weights.copy(),
            "decay_rate": self.decay_rate,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "update_counts": self.update_counts.copy(),
            "trends": self.trends.copy(),
            "metadata": self.metadata.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrustVector":
        """
        Create trust vector from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            TrustVector instance
        """
        trust_vector = cls(
            dimensions=data.get("dimensions", []),
            initial_value=0.0,  # Will be overridden
            weights=data.get("weights", {}),
            decay_rate=data.get("decay_rate", 0.99),
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 1.0),
        )
        
        # Set values
        values = data.get("values", {})
        for dimension, value in values.items():
            if dimension in trust_vector.dimensions:
                trust_vector.values[dimension] = value
                
        # Set history if present
        history = data.get("history", {})
        for dimension, hist in history.items():
            if dimension in trust_vector.dimensions:
                trust_vector.history[dimension] = hist.copy()
                
        # Set update counts
        update_counts = data.get("update_counts", {})
        for dimension, count in update_counts.items():
            if dimension in trust_vector.dimensions:
                trust_vector.update_counts[dimension] = count
                
        # Set trends
        trends = data.get("trends", {})
        for dimension, trend in trends.items():
            if dimension in trust_vector.dimensions:
                trust_vector.trends[dimension] = trend
                
        # Set metadata
        trust_vector.metadata = data.get("metadata", {}).copy()
        
        return trust_vector

    def set_metadata(self, key: str, value: Any) -> None:
        """
        Set metadata.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str) -> Optional[Any]:
        """
        Get metadata.

        Args:
            key: Metadata key

        Returns:
            Metadata value or None if not found
        """
        return self.metadata.get(key)

    def merge(self, other: "TrustVector", weight: float = 0.5) -> "TrustVector":
        """
        Merge with another trust vector.

        Args:
            other: Other trust vector
            weight: Weight of other vector in merge (0.0 to 1.0)

        Returns:
            New merged trust vector
        """
        # Create new vector with union of dimensions
        all_dimensions = list(set(self.dimensions) | set(other.get_dimensions()))
        merged = TrustVector(
            dimensions=all_dimensions,
            initial_value=0.0,  # Will be overridden
            decay_rate=self.decay_rate,
            min_value=self.min_value,
            max_value=self.max_value,
        )
        
        # Merge values
        for dimension in all_dimensions:
            self_value = self.get_value(dimension) or 0.0
            other_value = other.get_value(dimension) or 0.0
            
            # Weighted average
            merged_value = self_value * (1 - weight) + other_value * weight
            merged.values[dimension] = merged_value
            
            # Merge weights (average)
            self_weight = self.get_weight(dimension) or 0.0
            other_weight = other.get_weight(dimension) or 0.0
            merged.weights[dimension] = (self_weight + other_weight) / 2
            
        return merged

    def is_trusted(self, threshold: float = 0.7) -> bool:
        """
        Check if this vector represents a trusted entity.

        Args:
            threshold: Trust threshold

        Returns:
            Whether the entity is trusted
        """
        return self.get_weighted_average() >= threshold

    def serialize(self) -> Dict:
        """
        Serialize the trust vector to a dictionary.

        Returns:
            Dictionary representation of the trust vector
        """
        return {
            "dimensions": self.dimensions,
            "weights": self.weights,
            "values": self.values,
            "decay_rate": self.decay_rate,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "update_counts": self.update_counts,
            "trends": self.trends,
            "metadata": self.metadata,
            "history": {dim: self.history[dim][-20:] for dim in self.dimensions},
        }

    @classmethod
    def deserialize(cls, data: Dict) -> "TrustVector":
        """
        Create a trust vector from a serialized dictionary.

        Args:
            data: Dictionary representation of a trust vector

        Returns:
            A new TrustVector instance
        """
        vector = cls(
            dimensions=data.get("dimensions", []),
            initial_value=0.0,  # Will be overridden
            weights=data.get("weights", {}),
            decay_rate=data.get("decay_rate", 0.99),
            min_value=data.get("min_value", 0.0),
            max_value=data.get("max_value", 1.0),
        )
        
        # Set values
        values = data.get("values", {})
        for dimension, value in values.items():
            if dimension in vector.dimensions:
                vector.values[dimension] = value
                
        # Set history
        history = data.get("history", {})
        for dimension, hist in history.items():
            if dimension in vector.dimensions:
                vector.history[dimension] = hist.copy()
                
        # Set update counts
        update_counts = data.get("update_counts", {})
        for dimension, count in update_counts.items():
            if dimension in vector.dimensions:
                vector.update_counts[dimension] = count
                
        # Set trends
        trends = data.get("trends", {})
        for dimension, trend in trends.items():
            if dimension in vector.dimensions:
                vector.trends[dimension] = trend
                
        # Set metadata
        vector.metadata = data.get("metadata", {}).copy()
        
        return vector

    def to_json(self) -> str:
        """
        Convert the trust vector to a JSON string.

        Returns:
            JSON string representation of the trust vector
        """
        return json.dumps(self.serialize())

    @classmethod
    def from_json(cls, json_str: str) -> "TrustVector":
        """
        Create a trust vector from a JSON string.

        Args:
            json_str: JSON string representation of a trust vector

        Returns:
            A new TrustVector instance
        """
        return cls.deserialize(json.loads(json_str))

    def get_fingerprint(self) -> str:
        """
        Generate a unique fingerprint for this trust vector.

        Returns:
            A hash string representing the trust vector state
        """
        serialized = json.dumps(self.serialize(), sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def __repr__(self) -> str:
        """String representation of the trust vector."""
        return f"TrustVector(dimensions={self.dimensions}, trust={self.get_weighted_average():.4f})"
