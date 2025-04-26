"""
Implementation of a Hierarchical Trust Management System.

This module provides a hierarchical trust management system for the QTrust framework.
"""

import logging
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Set

logger = logging.getLogger(__name__)

class TrustScore:
    """Trust score with history and confidence tracking."""

    def __init__(self, score: float = 0.5, confidence: float = 0.5):
        """
        Initialize TrustScore.

        Args:
            score: Initial trust score (0.0 to 1.0)
            confidence: Initial confidence in the score (0.0 to 1.0)
        """
        self.score = score
        self.confidence = confidence
        self.history = [(score, confidence)]
        self.trend = 0.0

    def update(self, score: float, confidence: float) -> None:
        """
        Update trust score.

        Args:
            score: New trust score
            confidence: Confidence in the new score
        """
        # Weight new score by confidence
        weighted_old = self.score * (1 - confidence)
        weighted_new = score * confidence
        self.score = weighted_old + weighted_new

        # Update confidence (can only increase or stay the same)
        self.confidence = max(self.confidence, confidence)

        # Add to history
        self.history.append((self.score, self.confidence))

        # Update trend
        self.trend = self.get_trend()

    def decay(self, decay_factor: float) -> None:
        """
        Apply decay to trust score.

        Args:
            decay_factor: Factor to decay by (0.0 to 1.0)
        """
        self.score *= decay_factor
        self.confidence *= decay_factor
        self.history.append((self.score, self.confidence))
        self.trend = self.get_trend()

    def get_trend(self, window: int = 10) -> float:
        """
        Calculate trend of trust score.

        Args:
            window: Number of recent scores to consider

        Returns:
            Trend value (-1.0 to 1.0)
        """
        if len(self.history) < 2:
            return 0.0

        # Get recent scores
        recent = self.history[-min(window, len(self.history)):]
        scores = [s for s, _ in recent]

        # Calculate trend
        if len(scores) <= 1:
            return 0.0

        # Simple linear regression
        x = np.arange(len(scores))
        y = np.array(scores)
        A = np.vstack([x, np.ones(len(x))]).T
        try:
            m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            # Normalize trend between -1 and 1
            return max(min(m * window, 1.0), -1.0)
        except np.linalg.LinAlgError:
            return 0.0

    def get_score(self) -> float:
        """Get current trust score."""
        return self.score

    def get_confidence(self) -> float:
        """Get current confidence."""
        return self.confidence

    def get_history(self) -> List[Tuple[float, float]]:
        """Get history of trust scores."""
        return self.history.copy()


class TrustDimension:
    """Dimension of trust with weight and decay rate."""

    def __init__(self, name: str, weight: float = 1.0, decay_rate: float = 0.95):
        """
        Initialize trust dimension.

        Args:
            name: Dimension name
            weight: Weight in aggregate calculations
            decay_rate: Rate at which trust decays
        """
        self.name = name
        self.weight = max(0.0, min(weight, 1.0))
        self.decay_rate = max(0.0, min(decay_rate, 1.0))

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
        self.weight = max(0.0, min(weight, 1.0))

    def set_decay_rate(self, decay_rate: float) -> None:
        """
        Set dimension decay rate.

        Args:
            decay_rate: New decay rate
        """
        self.decay_rate = max(0.0, min(decay_rate, 1.0))


class TrustEntity:
    """Entity with multidimensional trust."""

    def __init__(self, entity_id: str, entity_type: str):
        """
        Initialize trust entity.

        Args:
            entity_id: Entity identifier
            entity_type: Entity type (e.g., "node", "shard")
        """
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.dimensions = {}  # dimension_name -> TrustScore
        self.verified = False
        self.verification_time = None
        self.metadata = {}

    def set_trust_score(
        self, dimension: str, score: float, confidence: float = 0.5
    ) -> None:
        """
        Set trust score for a dimension.

        Args:
            dimension: Trust dimension
            score: Trust score
            confidence: Confidence in the score
        """
        self.dimensions[dimension] = TrustScore(score, confidence)

    def update_trust_score(
        self, dimension: str, score: float, confidence: float = 0.5
    ) -> None:
        """
        Update trust score for a dimension.

        Args:
            dimension: Trust dimension
            score: Trust score
            confidence: Confidence in the score
        """
        if dimension not in self.dimensions:
            self.set_trust_score(dimension, score, confidence)
        else:
            self.dimensions[dimension].update(score, confidence)

    def get_trust_score(self, dimension: str) -> Optional[TrustScore]:
        """
        Get trust score for a dimension.

        Args:
            dimension: Trust dimension

        Returns:
            TrustScore or None if dimension not found
        """
        return self.dimensions.get(dimension)

    def get_trust_value(self, dimension: str) -> float:
        """
        Get trust value for a dimension.

        Args:
            dimension: Trust dimension

        Returns:
            Trust value or 0.0 if dimension not found
        """
        trust_score = self.get_trust_score(dimension)
        return trust_score.get_score() if trust_score else 0.0

    def get_dimensions(self) -> List[str]:
        """
        Get all trust dimensions.

        Returns:
            List of dimension names
        """
        return list(self.dimensions.keys())

    def calculate_aggregate_trust(
        self, weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate aggregate trust across dimensions.

        Args:
            weights: Optional weights for dimensions

        Returns:
            Aggregate trust value
        """
        if not self.dimensions:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for dimension, trust_score in self.dimensions.items():
            weight = weights.get(dimension, 1.0) if weights else 1.0
            total_weight += weight
            weighted_sum += trust_score.get_score() * weight

        if total_weight == 0.0:
            return 0.0

        return weighted_sum / total_weight

    def get_aggregate_trust(self) -> float:
        """
        Get aggregate trust across all dimensions.

        Returns:
            Aggregate trust value
        """
        return self.calculate_aggregate_trust()

    def set_verified(self, status: bool) -> None:
        """
        Set verification status.

        Args:
            status: Verification status
        """
        self.verified = status
        self.verification_time = None if not status else np.datetime64('now')

    def is_verified(self) -> bool:
        """
        Check if entity is verified.

        Returns:
            Verification status
        """
        return self.verified

    def get_verification_age(self) -> Optional[float]:
        """
        Get age of verification in seconds.

        Returns:
            Age in seconds or None if not verified
        """
        if not self.verified or self.verification_time is None:
            return None
        
        now = np.datetime64('now')
        return (now - self.verification_time) / np.timedelta64(1, 's')

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
            Metadata value or None if key not found
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
    """Hierarchical trust management system."""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize trust hierarchy.

        Args:
            config: Configuration parameters
        """
        # Default configuration
        self.config = {
            "trust_threshold": 0.7,
            "byzantine_threshold": 0.3,
            "initial_trust": 0.5,
            "initial_confidence": 0.5,
            "default_dimensions": ["transaction_success", "response_time", "uptime"],
            "dimension_weights": {
                "transaction_success": 0.5,
                "response_time": 0.3,
                "uptime": 0.2,
            },
        }

        # Update configuration if provided
        if config:
            self.config.update(config)

        # Initialize trust entities
        self.nodes = {}  # node_id -> TrustEntity
        self.shards = {}  # shard_id -> TrustEntity
        self.node_to_shard = {}  # node_id -> shard_id
        self.shard_to_nodes = {}  # shard_id -> Set[node_id]

        # Initialize trust graph for PageRank-based propagation
        self.trust_graph = nx.DiGraph()

        logger.info("Initialized TrustHierarchy")

    def add_node(self, node_id: str, shard_id: Optional[str] = None) -> None:
        """
        Add a node to the trust hierarchy.

        Args:
            node_id: Node identifier
            shard_id: Optional shard to assign node to
        """
        if node_id not in self.nodes:
            self.nodes[node_id] = TrustEntity(node_id, "node")
            
            # Initialize default dimensions
            for dimension in self.config["default_dimensions"]:
                self.nodes[node_id].set_trust_score(
                    dimension,
                    self.config["initial_trust"],
                    self.config["initial_confidence"],
                )
                
            # Add to trust graph
            self.trust_graph.add_node(node_id, type="node")
            
            logger.info(f"Added node {node_id} to trust hierarchy")
        
        # Assign to shard if provided
        if shard_id is not None:
            self.assign_node_to_shard(node_id, shard_id)

    def add_shard(self, shard_id: str) -> None:
        """
        Add a shard to the trust hierarchy.

        Args:
            shard_id: Shard identifier
        """
        if shard_id not in self.shards:
            self.shards[shard_id] = TrustEntity(shard_id, "shard")
            
            # Initialize default dimensions
            for dimension in self.config["default_dimensions"]:
                self.shards[shard_id].set_trust_score(
                    dimension,
                    self.config["initial_trust"],
                    self.config["initial_confidence"],
                )
                
            # Add to trust graph
            self.trust_graph.add_node(shard_id, type="shard")
            
            # Initialize node set
            self.shard_to_nodes[shard_id] = set()
            
            logger.info(f"Added shard {shard_id} to trust hierarchy")

    def remove_node(self, node_id: str) -> bool:
        """
        Remove a node from the trust hierarchy.

        Args:
            node_id: Node identifier

        Returns:
            Whether the node was removed successfully
        """
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not found in trust hierarchy")
            return False
            
        # Remove from shard
        if node_id in self.node_to_shard:
            shard_id = self.node_to_shard[node_id]
            if shard_id in self.shard_to_nodes:
                self.shard_to_nodes[shard_id].discard(node_id)
            del self.node_to_shard[node_id]
            
        # Remove from trust graph
        if self.trust_graph.has_node(node_id):
            self.trust_graph.remove_node(node_id)
            
        # Remove node
        del self.nodes[node_id]
        
        logger.info(f"Removed node {node_id} from trust hierarchy")
        return True

    def remove_shard(self, shard_id: str) -> bool:
        """
        Remove a shard from the trust hierarchy.

        Args:
            shard_id: Shard identifier

        Returns:
            Whether the shard was removed successfully
        """
        if shard_id not in self.shards:
            logger.warning(f"Shard {shard_id} not found in trust hierarchy")
            return False
            
        # Remove node-to-shard relationships
        if shard_id in self.shard_to_nodes:
            for node_id in list(self.shard_to_nodes[shard_id]):
                if node_id in self.node_to_shard:
                    del self.node_to_shard[node_id]
            del self.shard_to_nodes[shard_id]
            
        # Remove from trust graph
        if self.trust_graph.has_node(shard_id):
            self.trust_graph.remove_node(shard_id)
            
        # Remove shard
        del self.shards[shard_id]
        
        logger.info(f"Removed shard {shard_id} from trust hierarchy")
        return True

    def assign_node_to_shard(self, node_id: str, shard_id: str) -> bool:
        """
        Assign a node to a shard.

        Args:
            node_id: Node identifier
            shard_id: Shard identifier

        Returns:
            Whether the assignment was successful
        """
        # Ensure node and shard exist
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not found in trust hierarchy")
            return False
            
        if shard_id not in self.shards:
            logger.warning(f"Shard {shard_id} not found in trust hierarchy")
            return False
            
        # Remove from old shard if any
        if node_id in self.node_to_shard:
            old_shard_id = self.node_to_shard[node_id]
            if old_shard_id in self.shard_to_nodes:
                self.shard_to_nodes[old_shard_id].discard(node_id)
                
        # Assign to new shard
        self.node_to_shard[node_id] = shard_id
        if shard_id not in self.shard_to_nodes:
            self.shard_to_nodes[shard_id] = set()
        self.shard_to_nodes[shard_id].add(node_id)
        
        # Update trust graph
        if self.trust_graph.has_node(node_id) and self.trust_graph.has_node(shard_id):
            # Add edges in both directions with weights
            self.trust_graph.add_edge(node_id, shard_id, weight=0.7)
            self.trust_graph.add_edge(shard_id, node_id, weight=0.3)
            
        logger.info(f"Assigned node {node_id} to shard {shard_id}")
        return True

    def get_node_shard(self, node_id: str) -> Optional[str]:
        """
        Get the shard that a node is assigned to.

        Args:
            node_id: Node identifier

        Returns:
            Shard identifier or None if node not found or not assigned
        """
        return self.node_to_shard.get(node_id)

    def get_shard_nodes(self, shard_id: str) -> Set[str]:
        """
        Get the nodes assigned to a shard.

        Args:
            shard_id: Shard identifier

        Returns:
            Set of node identifiers
        """
        return self.shard_to_nodes.get(shard_id, set()).copy()

    def update_node_trust(
        self, node_id: str, dimension: str, score: float, confidence: float = 0.5
    ) -> bool:
        """
        Update trust score for a node.

        Args:
            node_id: Node identifier
            dimension: Trust dimension
            score: Trust score
            confidence: Confidence in the score

        Returns:
            Whether the update was successful
        """
        if node_id not in self.nodes:
            logger.warning(f"Node {node_id} not found in trust hierarchy")
            return False
            
        # Update node trust
        self.nodes[node_id].update_trust_score(dimension, score, confidence)
        
        # Update shard trust
        if node_id in self.node_to_shard:
            shard_id = self.node_to_shard[node_id]
            self._update_shard_trust(shard_id, dimension)
            
        logger.debug(f"Updated trust for node {node_id}, dimension {dimension}, score {score:.3f}")
        return True

    def _update_shard_trust(self, shard_id: str, dimension: str) -> None:
        """
        Update trust score for a shard based on its nodes.

        Args:
            shard_id: Shard identifier
            dimension: Trust dimension
        """
        if shard_id not in self.shards:
            logger.warning(f"Shard {shard_id} not found in trust hierarchy")
            return
            
        # Calculate average trust of nodes in the shard
        nodes = self.shard_to_nodes.get(shard_id, set())
        if not nodes:
            return
            
        # Calculate weighted average
        total_score = 0.0
        total_confidence = 0.0
        count = 0
        
        for node_id in nodes:
            if node_id in self.nodes:
                trust_score = self.nodes[node_id].get_trust_score(dimension)
                if trust_score:
                    total_score += trust_score.get_score() * trust_score.get_confidence()
                    total_confidence += trust_score.get_confidence()
                    count += 1
                    
        if count > 0 and total_confidence > 0:
            avg_score = total_score / total_confidence
            avg_confidence = total_confidence / count
            
            # Update shard trust
            self.shards[shard_id].update_trust_score(dimension, avg_score, avg_confidence)
            
            logger.debug(
                f"Updated trust for shard {shard_id}, dimension {dimension}, score {avg_score:.3f}"
            )

    def propagate_trust(self) -> None:
        """Propagate trust through the hierarchy."""
        # Update PageRank-based trust
        self._calculate_pagerank_trust()
        
        # Update shard trust scores for all dimensions
        for shard_id in self.shards:
            for dimension in self.config["default_dimensions"]:
                self._update_shard_trust(shard_id, dimension)
                
        logger.info("Propagated trust through hierarchy")

    def _calculate_pagerank_trust(self, alpha: float = 0.85) -> None:
        """
        Calculate PageRank-based trust scores.

        Args:
            alpha: Damping factor
        """
        # Skip if graph is empty
        if len(self.trust_graph) == 0:
            return
            
        try:
            # Calculate PageRank
            pagerank = nx.pagerank(self.trust_graph, alpha=alpha)
            
            # Update trust scores based on PageRank
            for entity_id, rank in pagerank.items():
                # Normalize rank to [0, 1]
                normalized_rank = rank * len(self.trust_graph)
                
                # Update node or shard
                if entity_id in self.nodes:
                    self.nodes[entity_id].set_metadata("pagerank", normalized_rank)
                elif entity_id in self.shards:
                    self.shards[entity_id].set_metadata("pagerank", normalized_rank)
                    
        except Exception as e:
            logger.error(f"Error calculating PageRank: {e}")

    def get_node(self, node_id: str) -> Optional[TrustEntity]:
        """
        Get a node entity.

        Args:
            node_id: Node identifier

        Returns:
            TrustEntity or None if not found
        """
        return self.nodes.get(node_id)

    def get_shard(self, shard_id: str) -> Optional[TrustEntity]:
        """
        Get a shard entity.

        Args:
            shard_id: Shard identifier

        Returns:
            TrustEntity or None if not found
        """
        return self.shards.get(shard_id)

    def get_node_trust(self, node_id: str, dimension: Optional[str] = None) -> float:
        """
        Get trust score for a node.

        Args:
            node_id: Node identifier
            dimension: Trust dimension (None for aggregate)

        Returns:
            Trust score or 0.0 if not found
        """
        if node_id not in self.nodes:
            return 0.0
            
        if dimension is None:
            return self.nodes[node_id].get_aggregate_trust()
        else:
            return self.nodes[node_id].get_trust_value(dimension)

    def get_shard_trust(self, shard_id: str, dimension: Optional[str] = None) -> float:
        """
        Get trust score for a shard.

        Args:
            shard_id: Shard identifier
            dimension: Trust dimension (None for aggregate)

        Returns:
            Trust score or 0.0 if not found
        """
        if shard_id not in self.shards:
            return 0.0
            
        if dimension is None:
            return self.shards[shard_id].get_aggregate_trust()
        else:
            return self.shards[shard_id].get_trust_value(dimension)

    def is_node_trusted(self, node_id: str, threshold: Optional[float] = None) -> bool:
        """
        Check if a node is trusted.

        Args:
            node_id: Node identifier
            threshold: Trust threshold (None for default)

        Returns:
            Whether the node is trusted
        """
        if threshold is None:
            threshold = self.config["trust_threshold"]
            
        return self.get_node_trust(node_id) >= threshold

    def is_shard_trusted(self, shard_id: str, threshold: Optional[float] = None) -> bool:
        """
        Check if a shard is trusted.

        Args:
            shard_id: Shard identifier
            threshold: Trust threshold (None for default)

        Returns:
            Whether the shard is trusted
        """
        if threshold is None:
            threshold = self.config["trust_threshold"]
            
        return self.get_shard_trust(shard_id) >= threshold

    def decay_trust(self) -> None:
        """Apply decay to all trust scores."""
        # Decay node trust
        for node_id, node in self.nodes.items():
            for dimension in node.get_dimensions():
                trust_score = node.get_trust_score(dimension)
                if trust_score:
                    decay_rate = 0.95  # Default decay rate
                    trust_score.decay(decay_rate)
                    
        # Decay shard trust
        for shard_id, shard in self.shards.items():
            for dimension in shard.get_dimensions():
                trust_score = shard.get_trust_score(dimension)
                if trust_score:
                    decay_rate = 0.99  # Slower decay for shards
                    trust_score.decay(decay_rate)
                    
        # Update shard trust based on nodes
        for shard_id in self.shards:
            for dimension in self.config["default_dimensions"]:
                self._update_shard_trust(shard_id, dimension)
                
        logger.info("Applied decay to all trust scores")

    def get_trust_rankings(
        self, entity_type: str = "node", dimension: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Get rankings of entities by trust score.

        Args:
            entity_type: Entity type ("node" or "shard")
            dimension: Trust dimension (None for aggregate)

        Returns:
            List of (entity_id, trust_score) tuples, sorted by score
        """
        if entity_type == "node":
            entities = self.nodes
        elif entity_type == "shard":
            entities = self.shards
        else:
            logger.warning(f"Unknown entity type: {entity_type}")
            return []
            
        # Get trust scores
        rankings = []
        for entity_id, entity in entities.items():
            if dimension is None:
                score = entity.get_aggregate_trust()
            else:
                score = entity.get_trust_value(dimension)
                
            rankings.append((entity_id, score))
            
        # Sort by score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings

    def detect_byzantine_nodes(
        self, threshold: Optional[float] = None
    ) -> List[str]:
        """
        Detect potentially Byzantine nodes based on trust scores.

        Args:
            threshold: Byzantine detection threshold (None for default)

        Returns:
            List of potentially Byzantine node identifiers
        """
        if threshold is None:
            threshold = self.config["byzantine_threshold"]
            
        byzantine_nodes = []
        
        for node_id, node in self.nodes.items():
            # Check aggregate trust
            if node.get_aggregate_trust() < threshold:
                byzantine_nodes.append(node_id)
                continue
                
            # Check individual dimensions
            for dimension in self.config["default_dimensions"]:
                score = node.get_trust_value(dimension)
                if score < threshold / 2:  # More strict check for individual dimensions
                    byzantine_nodes.append(node_id)
                    break
                    
        return byzantine_nodes

    def find_trusted_path(
        self, source_node: str, target_node: str, threshold: Optional[float] = None
    ) -> List[str]:
        """
        Find trusted path between nodes.

        Args:
            source_node: Source node identifier
            target_node: Target node identifier
            threshold: Trust threshold (None for default)

        Returns:
            List of node identifiers forming the path, or empty list if no path
        """
        if threshold is None:
            threshold = self.config["trust_threshold"]
            
        # Create a temporary graph with only trusted nodes and edges
        trusted_graph = nx.DiGraph()
        
        # Add trusted nodes
        for node_id, node in self.nodes.items():
            if node.get_aggregate_trust() >= threshold:
                trusted_graph.add_node(node_id)
                
        # Add edges based on shared shards
        for node1 in trusted_graph.nodes():
            for node2 in trusted_graph.nodes():
                if node1 == node2:
                    continue
                    
                # Check if nodes share a shard
                shard1 = self.get_node_shard(node1)
                shard2 = self.get_node_shard(node2)
                
                if shard1 is not None and shard1 == shard2:
                    # Add edge with weight based on trust
                    weight = 2.0 - (self.get_node_trust(node1) + self.get_node_trust(node2)) / 2
                    trusted_graph.add_edge(node1, node2, weight=weight)
                    
        # Find shortest path
        if source_node in trusted_graph and target_node in trusted_graph:
            try:
                path = nx.shortest_path(trusted_graph, source=source_node, target=target_node, weight="weight")
                return path
            except nx.NetworkXNoPath:
                return []
                
        return []
    
    def get_dimensions(self) -> List[str]:
        """
        Get all trust dimensions.

        Returns:
            List of dimension names
        """
        return list(self.config["default_dimensions"]) 