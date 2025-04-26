"""
THE implementation of HTDCM.
"""

import random
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set
import time
import networkx as nx

logger = logging.getLogger(__name__)


class HTDCM:
    """
    Mock implementation of Hierarchical Trust and Dynamic Consensus Management (HTDCM) for testing.

    This class provides a simplified implementation that mimics the behavior
    of the actual HTDCM without requiring dependencies.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the mock HTDCM.

        Args:
            config: Configuration parameters
        """
        # Default configuration
        self.config = {
            "trust_threshold": 0.7,
            "reputation_decay": 0.95,
            "initial_trust": 0.5,
            "trust_update_weight": 0.2,
        }

        # Update configuration if provided
        if config:
            self.config.update(config)

        # Initialize trust data structures
        self.node_trust = {}  # node_id -> trust_score
        self.shard_trust = {}  # shard_id -> trust_score
        self.node_to_shard = {}  # node_id -> shard_id
        self.shard_nodes = {}  # shard_id -> list of node_ids

        logger.info("Initialized MockHTDCM")

    def add_shard(self, shard_id: str) -> None:
        """
        Add a new shard to the system.

        Args:
            shard_id: Shard identifier
        """
        if shard_id not in self.shard_trust:
            self.shard_trust[shard_id] = self.config["initial_trust"]
            self.shard_nodes[shard_id] = []
            logger.info(f"Added shard {shard_id}")

    def add_node(self, node_id: str, shard_id: str = None) -> None:
        """
        Add a new node to a shard.

        Args:
            node_id: Node identifier
            shard_id: Shard identifier (optional, can be None for backward compatibility)
        """
        # For backward compatibility, if shard_id is not provided
        if shard_id is None:
            # If node already exists, use its existing shard
            if node_id in self.node_to_shard:
                shard_id = self.node_to_shard[node_id]
            # Otherwise, use a default shard or create one
            else:
                shard_id = "default_shard"

        if shard_id not in self.shard_trust:
            self.add_shard(shard_id)

        if node_id not in self.node_trust:
            self.node_trust[node_id] = self.config["initial_trust"]
            self.node_to_shard[node_id] = shard_id
            self.shard_nodes[shard_id].append(node_id)
            logger.info(f"Added node {node_id} to shard {shard_id}")

    def update_trust(self, node_id: str, behavior_score: float) -> None:
        """
        Update trust score for a node based on its behavior.

        Args:
            node_id: Node identifier
            behavior_score: Score representing node's behavior (0.0 to 1.0)
        """
        if node_id not in self.node_trust:
            logger.warning(f"Node {node_id} not found")
            return

        # Update node trust
        old_trust = self.node_trust[node_id]
        self.node_trust[node_id] = (
            1 - self.config["trust_update_weight"]
        ) * old_trust + self.config["trust_update_weight"] * behavior_score

        # Update shard trust
        shard_id = self.node_to_shard[node_id]
        self._update_shard_trust(shard_id)

        logger.info(
            f"Updated trust for node {node_id}: {old_trust:.3f} -> {self.node_trust[node_id]:.3f}"
        )

    def _update_shard_trust(self, shard_id: str) -> None:
        """
        Update trust score for a shard based on its nodes.

        Args:
            shard_id: Shard identifier
        """
        if shard_id not in self.shard_trust:
            logger.warning(f"Shard {shard_id} not found")
            return

        # Calculate average trust of nodes in the shard
        nodes = self.shard_nodes[shard_id]
        if not nodes:
            return

        avg_trust = sum(self.node_trust[node_id] for node_id in nodes) / len(nodes)
        self.shard_trust[shard_id] = avg_trust

    def get_trust(self, node_id: str) -> float:
        """
        Get trust score for a node.

        Args:
            node_id: Node identifier

        Returns:
            Trust score (0.0 to 1.0)
        """
        if node_id not in self.node_trust:
            logger.warning(f"Node {node_id} not found")
            return 0.0

        return self.node_trust[node_id]

    def get_shard_trust(self, shard_id: str) -> float:
        """
        Get trust score for a shard.

        Args:
            shard_id: Shard identifier

        Returns:
            Trust score (0.0 to 1.0)
        """
        if shard_id not in self.shard_trust:
            logger.warning(f"Shard {shard_id} not found")
            return 0.0

        return self.shard_trust[shard_id]

    def is_trusted(self, node_id: str) -> bool:
        """
        Check if a node is trusted.

        Args:
            node_id: Node identifier

        Returns:
            Whether the node is trusted
        """
        return self.get_trust(node_id) >= self.config["trust_threshold"]

    def is_shard_trusted(self, shard_id: str) -> bool:
        """
        Check if a shard is trusted.

        Args:
            shard_id: Shard identifier

        Returns:
            Whether the shard is trusted
        """
        return self.get_shard_trust(shard_id) >= self.config["trust_threshold"]

    def decay_trust(self) -> None:
        """Decay trust scores over time."""
        # Apply decay to all node trust scores
        for node_id in self.node_trust:
            self.node_trust[node_id] *= self.config["reputation_decay"]

        # Update all shard trust scores
        for shard_id in self.shard_trust:
            self._update_shard_trust(shard_id)

        logger.info("Applied trust decay")

    def get_trusted_nodes(self) -> List[str]:
        """
        Get list of trusted nodes.

        Returns:
            List of trusted node IDs
        """
        return [
            node_id
            for node_id, trust in self.node_trust.items()
            if trust >= self.config["trust_threshold"]
        ]

    def get_trusted_shards(self) -> List[str]:
        """
        Get list of trusted shards.

        Returns:
            List of trusted shard IDs
        """
        return [
            shard_id
            for shard_id, trust in self.shard_trust.items()
            if trust >= self.config["trust_threshold"]
        ]

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the system.

        Args:
            node_id: Node identifier

        Returns:
            Whether the node was removed successfully
        """
        if node_id not in self.node_trust:
            logger.warning(f"Node {node_id} not found")
            return False

        shard_id = self.node_to_shard[node_id]

        # Remove node
        del self.node_trust[node_id]
        del self.node_to_shard[node_id]
        self.shard_nodes[shard_id].remove(node_id)

        # Update shard trust
        self._update_shard_trust(shard_id)

        logger.info(f"Removed node {node_id} from shard {shard_id}")
        return True


class TrustScore:
    """
    Trust score with confidence and history tracking.
    """

    def __init__(self, score: float = 0.5, confidence: float = 0.5):
        """
        Initialize trust score.

        Args:
            score: Initial trust score (0.0 to 1.0)
            confidence: Confidence in the score (0.0 to 1.0)
        """
        self.score = max(0.0, min(1.0, score))
        self.confidence = max(0.0, min(1.0, confidence))
        self.last_update = time.time()
        self.history = [(self.score, self.confidence, self.last_update)]

    def update(self, score: float, confidence: float) -> None:
        """
        Update trust score.

        Args:
            score: New trust score (0.0 to 1.0)
            confidence: New confidence (0.0 to 1.0)
        """
        self.score = max(0.0, min(1.0, score))
        self.confidence = max(0.0, min(1.0, confidence))
        self.last_update = time.time()
        self.history.append((self.score, self.confidence, self.last_update))

        # Limit history to last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def decay(self, decay_factor: float) -> None:
        """
        Apply decay to trust score.

        Args:
            decay_factor: Factor to decay score and confidence by (0.0 to 1.0)
        """
        self.score *= decay_factor
        self.confidence *= decay_factor
        self.last_update = time.time()
        self.history.append((self.score, self.confidence, self.last_update))

        # Limit history to last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]

    def get_trend(self, window: int = 10) -> float:
        """
        Calculate trust score trend over a window.

        Args:
            window: Number of history entries to consider

        Returns:
            Trend value (-1.0 to 1.0)
        """
        if len(self.history) < 2:
            return 0.0

        # Get last 'window' entries or all if fewer
        history_window = self.history[-min(window, len(self.history)):]
        
        # Extract scores
        scores = [entry[0] for entry in history_window]
        
        # If all scores are the same, return 0.0
        if all(score == scores[0] for score in scores):
            return 0.0
            
        # Calculate trend using simple linear regression
        n = len(scores)
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(scores) / n
        
        # Calculate slope
        numerator = sum((x[i] - mean_x) * (scores[i] - mean_y) for i in range(n))
        denominator = sum((x[i] - mean_x) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
            
        slope = numerator / denominator
        
        # Normalize slope to (-1.0, 1.0)
        return max(-1.0, min(1.0, slope * n))


class TrustDimension:
    """
    Trust dimension with weight and decay rate.
    """

    def __init__(self, name: str, weight: float = 1.0, decay_rate: float = 0.95):
        """
        Initialize trust dimension.

        Args:
            name: Dimension name
            weight: Dimension weight in overall trust calculation
            decay_rate: Rate at which trust decays over time
        """
        self.name = name
        self.weight = max(0.0, weight)
        self.decay_rate = max(0.0, min(1.0, decay_rate))

    def get_name(self) -> str:
        """
        Get dimension name.

        Returns:
            Dimension name
        """
        return self.name

    def get_weight(self) -> float:
        """
        Get dimension weight.

        Returns:
            Dimension weight
        """
        return self.weight

    def get_decay_rate(self) -> float:
        """
        Get dimension decay rate.

        Returns:
            Dimension decay rate
        """
        return self.decay_rate

    def set_weight(self, weight: float) -> None:
        """
        Set dimension weight.

        Args:
            weight: New weight
        """
        self.weight = max(0.0, weight)

    def set_decay_rate(self, decay_rate: float) -> None:
        """
        Set dimension decay rate.

        Args:
            decay_rate: New decay rate
        """
        self.decay_rate = max(0.0, min(1.0, decay_rate))


class TrustEntity:
    """
    Entity with multi-dimensional trust scores.
    """

    def __init__(self, entity_id: str, entity_type: str):
        """
        Initialize trust entity.

        Args:
            entity_id: Entity identifier
            entity_type: Entity type (e.g., 'node', 'shard')
        """
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.trust_scores = {}  # dimension -> TrustScore
        self.verification_status = False
        self.metadata = {}  # Custom metadata
        self._aggregate_trust = None  # Cached aggregate trust

    def set_trust_score(
        self, dimension: str, score: float, confidence: float = 0.5
    ) -> None:
        """
        Set trust score for a dimension.

        Args:
            dimension: Trust dimension
            score: Trust score (0.0 to 1.0)
            confidence: Confidence in the score (0.0 to 1.0)
        """
        self.trust_scores[dimension] = TrustScore(score, confidence)
        self._aggregate_trust = None  # Invalidate cache

    def update_trust_score(
        self, dimension: str, score: float, confidence: float = 0.5
    ) -> None:
        """
        Update trust score for a dimension.

        Args:
            dimension: Trust dimension
            score: Trust score (0.0 to 1.0)
            confidence: Confidence in the score (0.0 to 1.0)
        """
        if dimension in self.trust_scores:
            self.trust_scores[dimension].update(score, confidence)
        else:
            self.set_trust_score(dimension, score, confidence)
        self._aggregate_trust = None  # Invalidate cache

    def get_trust_score(self, dimension: str) -> Optional[TrustScore]:
        """
        Get trust score for a dimension.

        Args:
            dimension: Trust dimension

        Returns:
            TrustScore or None if not found
        """
        return self.trust_scores.get(dimension)

    def get_trust_value(self, dimension: str) -> float:
        """
        Get trust value for a dimension.

        Args:
            dimension: Trust dimension

        Returns:
            Trust value (0.0 to 1.0) or 0.0 if not found
        """
        score = self.get_trust_score(dimension)
        return score.score if score else 0.0

    def get_dimensions(self) -> List[str]:
        """
        Get all trust dimensions.

        Returns:
            List of dimension names
        """
        return list(self.trust_scores.keys())

    def calculate_aggregate_trust(
        self, weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate aggregate trust across dimensions.

        Args:
            weights: Optional weights for each dimension

        Returns:
            Aggregate trust value (0.0 to 1.0)
        """
        if not self.trust_scores:
            return 0.0

        # Use provided weights or default to equal weights
        if weights is None:
            weights = {dim: 1.0 for dim in self.trust_scores}

        # Calculate weighted average
        total_weight = 0.0
        weighted_sum = 0.0

        for dimension, score in self.trust_scores.items():
            weight = weights.get(dimension, 0.0)
            if weight > 0:
                weighted_sum += score.score * weight
                total_weight += weight

        # Avoid division by zero
        if total_weight == 0:
            return 0.0

        # Cache result
        self._aggregate_trust = weighted_sum / total_weight
        return self._aggregate_trust

    def get_aggregate_trust(self) -> float:
        """
        Get cached aggregate trust or calculate if needed.

        Returns:
            Aggregate trust value (0.0 to 1.0)
        """
        if self._aggregate_trust is None:
            return self.calculate_aggregate_trust()
        return self._aggregate_trust

    def set_verified(self, status: bool) -> None:
        """
        Set verification status.

        Args:
            status: Verification status
        """
        self.verification_status = status

    def is_verified(self) -> bool:
        """
        Check if entity is verified.

        Returns:
            Verification status
        """
        return self.verification_status

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

    def get_all_metadata(self) -> Dict[str, Any]:
        """
        Get all metadata.

        Returns:
            Dictionary of metadata
        """
        return self.metadata.copy()


class TrustHierarchy:
    """
    Hierarchical trust management system.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize trust hierarchy.

        Args:
            config: Configuration parameters
        """
        # Default configuration
        self.config = {
            "trust_threshold": 0.7,
            "reputation_decay": 0.95,
            "initial_trust": 0.5,
        }

        # Update configuration if provided
        if config:
            self.config.update(config)

        # Initialize trust entities
        self.nodes = {}  # node_id -> TrustEntity
        self.shards = {}  # shard_id -> TrustEntity
        self.node_to_shard = {}  # node_id -> shard_id
        self.shard_nodes = {}  # shard_id -> set of node_ids

        # Initialize trust dimensions
        self.dimensions = {
            "performance": TrustDimension("performance", 1.0, 0.97),
            "uptime": TrustDimension("uptime", 1.0, 0.99),
            "consensus": TrustDimension("consensus", 1.5, 0.95),
            "transactions": TrustDimension("transactions", 1.2, 0.96),
            "security": TrustDimension("security", 2.0, 0.90),
        }

        # Initialize trust graph for PageRank-based trust
        self.trust_graph = nx.DiGraph()

    def add_node(self, node_id: str, shard_id: Optional[str] = None) -> None:
        """
        Add a node to the hierarchy.

        Args:
            node_id: Node identifier
            shard_id: Shard identifier (optional)
        """
        if node_id not in self.nodes:
            self.nodes[node_id] = TrustEntity(node_id, "node")
            self.trust_graph.add_node(node_id)

            # Initialize trust scores for each dimension
            for dim_name, dimension in self.dimensions.items():
                self.nodes[node_id].set_trust_score(
                    dim_name, self.config["initial_trust"], 0.5
                )

        # Assign to shard if provided
        if shard_id:
            self.assign_node_to_shard(node_id, shard_id)

    def add_shard(self, shard_id: str) -> None:
        """
        Add a shard to the hierarchy.

        Args:
            shard_id: Shard identifier
        """
        if shard_id not in self.shards:
            self.shards[shard_id] = TrustEntity(shard_id, "shard")
            self.trust_graph.add_node(shard_id)
            self.shard_nodes[shard_id] = set()

            # Initialize trust scores for each dimension
            for dim_name, dimension in self.dimensions.items():
                self.shards[shard_id].set_trust_score(
                    dim_name, self.config["initial_trust"], 0.5
                )

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the hierarchy.

        Args:
            node_id: Node identifier

        Returns:
            True if removed, False if not found
        """
        if node_id not in self.nodes:
            return False

        # Remove from shard if assigned
        if node_id in self.node_to_shard:
            shard_id = self.node_to_shard[node_id]
            if shard_id in self.shard_nodes:
                self.shard_nodes[shard_id].discard(node_id)
            del self.node_to_shard[node_id]

        # Remove from trust graph
        if self.trust_graph.has_node(node_id):
            self.trust_graph.remove_node(node_id)

        # Remove trust entity
        del self.nodes[node_id]

        return True

    def remove_shard(self, shard_id: str) -> bool:
        """
        Remove a shard from the hierarchy.

        Args:
            shard_id: Shard identifier

        Returns:
            True if removed, False if not found
        """
        if shard_id not in self.shards:
            return False

        # Remove all nodes assigned to this shard
        if shard_id in self.shard_nodes:
            for node_id in list(self.shard_nodes[shard_id]):
                if node_id in self.node_to_shard:
                    del self.node_to_shard[node_id]
            del self.shard_nodes[shard_id]

        # Remove from trust graph
        if self.trust_graph.has_node(shard_id):
            self.trust_graph.remove_node(shard_id)

        # Remove trust entity
        del self.shards[shard_id]

        return True

    def assign_node_to_shard(self, node_id: str, shard_id: str) -> bool:
        """
        Assign a node to a shard.

        Args:
            node_id: Node identifier
            shard_id: Shard identifier

        Returns:
            True if assigned, False if node or shard not found
        """
        if node_id not in self.nodes or shard_id not in self.shards:
            return False

        # Remove from previous shard if any
        if node_id in self.node_to_shard:
            prev_shard = self.node_to_shard[node_id]
            if prev_shard in self.shard_nodes:
                self.shard_nodes[prev_shard].discard(node_id)

        # Assign to new shard
        self.node_to_shard[node_id] = shard_id
        if shard_id not in self.shard_nodes:
            self.shard_nodes[shard_id] = set()
        self.shard_nodes[shard_id].add(node_id)

        # Add edge in trust graph
        self.trust_graph.add_edge(node_id, shard_id, weight=1.0)
        self.trust_graph.add_edge(shard_id, node_id, weight=0.5)

        return True

    def get_node_shard(self, node_id: str) -> Optional[str]:
        """
        Get the shard a node is assigned to.

        Args:
            node_id: Node identifier

        Returns:
            Shard identifier or None if not assigned
        """
        return self.node_to_shard.get(node_id)

    def get_shard_nodes(self, shard_id: str) -> Set[str]:
        """
        Get nodes assigned to a shard.

        Args:
            shard_id: Shard identifier

        Returns:
            Set of node identifiers
        """
        return self.shard_nodes.get(shard_id, set()).copy()

    def update_node_trust(
        self, node_id: str, dimension: str, score: float, confidence: float = 0.5
    ) -> bool:
        """
        Update trust score for a node.

        Args:
            node_id: Node identifier
            dimension: Trust dimension
            score: Trust score (0.0 to 1.0)
            confidence: Confidence in the score (0.0 to 1.0)

        Returns:
            True if updated, False if node not found
        """
        if node_id not in self.nodes:
            return False

        # Update node trust
        self.nodes[node_id].update_trust_score(dimension, score, confidence)

        # Update shard trust if assigned
        if node_id in self.node_to_shard:
            shard_id = self.node_to_shard[node_id]
            self._update_shard_trust(shard_id, dimension)

        # Update trust graph edge weights
        # More trusted nodes have higher edge weights
        if self.trust_graph.has_node(node_id):
            for neighbor in self.trust_graph.neighbors(node_id):
                self.trust_graph[node_id][neighbor]["weight"] = self.nodes[
                    node_id
                ].get_aggregate_trust()

        return True

    def _update_shard_trust(self, shard_id: str, dimension: str) -> None:
        """
        Update trust score for a shard based on its nodes.

        Args:
            shard_id: Shard identifier
            dimension: Trust dimension
        """
        if shard_id not in self.shards or shard_id not in self.shard_nodes:
            return

        nodes = self.shard_nodes[shard_id]
        if not nodes:
            return

        # Calculate average trust of nodes in the shard for this dimension
        total_trust = 0.0
        total_confidence = 0.0
        count = 0

        for node_id in nodes:
            if node_id in self.nodes:
                trust_score = self.nodes[node_id].get_trust_score(dimension)
                if trust_score:
                    total_trust += trust_score.score
                    total_confidence += trust_score.confidence
                    count += 1

        if count > 0:
            avg_trust = total_trust / count
            avg_confidence = total_confidence / count
            self.shards[shard_id].update_trust_score(dimension, avg_trust, avg_confidence)

            # Update trust graph edge weights
            if self.trust_graph.has_node(shard_id):
                for neighbor in self.trust_graph.neighbors(shard_id):
                    self.trust_graph[shard_id][neighbor]["weight"] = self.shards[
                        shard_id
                    ].get_aggregate_trust()

    def propagate_trust(self) -> None:
        """
        Propagate trust throughout the hierarchy.
        """
        # Update shard trust based on nodes
        for shard_id in self.shards:
            for dimension in self.dimensions:
                self._update_shard_trust(shard_id, dimension)

        # Calculate PageRank-based trust
        self._calculate_pagerank_trust()

    def _calculate_pagerank_trust(self, alpha: float = 0.85) -> None:
        """
        Calculate PageRank-based trust scores.

        Args:
            alpha: Damping factor for PageRank
        """
        if not self.trust_graph.nodes():
            return

        # Calculate PageRank
        pagerank = nx.pagerank(self.trust_graph, alpha=alpha)

        # Update entities with PageRank scores
        for node_id, score in pagerank.items():
            # Scale score to [0, 1]
            scaled_score = score * len(self.trust_graph)
            scaled_score = min(1.0, scaled_score)

            if node_id in self.nodes:
                self.nodes[node_id].set_metadata("pagerank", scaled_score)
            elif node_id in self.shards:
                self.shards[node_id].set_metadata("pagerank", scaled_score)

    def get_node_trust(self, node_id: str, dimension: Optional[str] = None) -> float:
        """
        Get trust score for a node.

        Args:
            node_id: Node identifier
            dimension: Trust dimension (if None, returns aggregate trust)

        Returns:
            Trust score (0.0 to 1.0)
        """
        if node_id not in self.nodes:
            return 0.0

        if dimension:
            return self.nodes[node_id].get_trust_value(dimension)
        else:
            return self.nodes[node_id].get_aggregate_trust()

    def get_shard_trust(self, shard_id: str, dimension: Optional[str] = None) -> float:
        """
        Get trust score for a shard.

        Args:
            shard_id: Shard identifier
            dimension: Trust dimension (if None, returns aggregate trust)

        Returns:
            Trust score (0.0 to 1.0)
        """
        if shard_id not in self.shards:
            return 0.0

        if dimension:
            return self.shards[shard_id].get_trust_value(dimension)
        else:
            return self.shards[shard_id].get_aggregate_trust()

    def is_node_trusted(self, node_id: str, threshold: Optional[float] = None) -> bool:
        """
        Check if a node is trusted.

        Args:
            node_id: Node identifier
            threshold: Trust threshold (if None, uses config threshold)

        Returns:
            True if trusted, False otherwise
        """
        if threshold is None:
            threshold = self.config["trust_threshold"]

        return self.get_node_trust(node_id) >= threshold

    def is_shard_trusted(self, shard_id: str, threshold: Optional[float] = None) -> bool:
        """
        Check if a shard is trusted.

        Args:
            shard_id: Shard identifier
            threshold: Trust threshold (if None, uses config threshold)

        Returns:
            True if trusted, False otherwise
        """
        if threshold is None:
            threshold = self.config["trust_threshold"]

        return self.get_shard_trust(shard_id) >= threshold

    def decay_trust(self) -> None:
        """
        Decay trust scores over time.
        """
        # Decay node trust
        for node_id, node in self.nodes.items():
            for dim_name, dimension in self.dimensions.items():
                trust_score = node.get_trust_score(dim_name)
                if trust_score:
                    trust_score.decay(dimension.get_decay_rate())

        # Decay shard trust
        for shard_id, shard in self.shards.items():
            for dim_name, dimension in self.dimensions.items():
                trust_score = shard.get_trust_score(dim_name)
                if trust_score:
                    trust_score.decay(dimension.get_decay_rate())

        # Recalculate propagated trust
        self.propagate_trust()

    def get_trust_rankings(
        self, entity_type: str = "node", dimension: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Get ranked list of entities by trust score.

        Args:
            entity_type: Entity type ('node' or 'shard')
            dimension: Trust dimension (if None, uses aggregate trust)

        Returns:
            List of (entity_id, trust_score) tuples, sorted by trust score
        """
        entities = self.nodes if entity_type == "node" else self.shards
        
        if dimension:
            rankings = [
                (entity_id, entity.get_trust_value(dimension))
                for entity_id, entity in entities.items()
            ]
        else:
            rankings = [
                (entity_id, entity.get_aggregate_trust())
                for entity_id, entity in entities.items()
            ]
            
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def detect_byzantine_nodes(
        self, threshold: Optional[float] = None
    ) -> List[str]:
        """
        Detect potentially Byzantine nodes based on trust scores.

        Args:
            threshold: Trust threshold (if None, uses config threshold)

        Returns:
            List of potentially Byzantine node identifiers
        """
        if threshold is None:
            threshold = self.config["trust_threshold"] / 2.0  # Lower threshold for Byzantine detection
            
        # Get nodes with low trust scores
        byzantine_nodes = []
        
        for node_id, node in self.nodes.items():
            # Check security dimension first if available
            security_trust = node.get_trust_value("security")
            if security_trust > 0 and security_trust < threshold:
                byzantine_nodes.append(node_id)
                continue
                
            # Then check overall trust
            if node.get_aggregate_trust() < threshold:
                byzantine_nodes.append(node_id)
                
        return byzantine_nodes

    def find_trusted_path(
        self, source_node: str, target_node: str, threshold: Optional[float] = None
    ) -> List[str]:
        """
        Find a trusted path between two nodes.

        Args:
            source_node: Source node identifier
            target_node: Target node identifier
            threshold: Trust threshold (if None, uses config threshold)

        Returns:
            List of node identifiers forming a trusted path, or empty list if none
        """
        if threshold is None:
            threshold = self.config["trust_threshold"]
            
        # Check if nodes exist
        if source_node not in self.nodes or target_node not in self.nodes:
            return []
            
        # Create a subgraph of trusted nodes
        trusted_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.get_aggregate_trust() >= threshold
        ]
        
        # If either source or target is not trusted, return empty path
        if source_node not in trusted_nodes or target_node not in trusted_nodes:
            return []
            
        # Try to find a path in the subgraph
        try:
            path = nx.shortest_path(
                self.trust_graph.subgraph(trusted_nodes),
                source=source_node,
                target=target_node
            )
            return path
        except (nx.NetworkXNoPath, nx.NetworkXError):
            return []
