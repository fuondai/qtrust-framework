#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Dynamic Trust Thresholds
This module implements dynamic trust threshold adjustment based on network conditions.
"""

import time
import threading
import numpy as np
import math
from typing import Dict, List, Tuple, Set, Optional, Any, Callable

from ..common.async_utils import AsyncProcessor, AsyncEvent, AsyncCache


class DynamicTrustManager:
    """
    Implements dynamic trust threshold adjustment based on network conditions.
    Enhances trust score propagation efficiency and reduces false positive rates.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the dynamic trust manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Default configuration
        self.base_threshold = self.config.get("base_threshold", 0.7)
        self.min_threshold = self.config.get("min_threshold", 0.3)
        self.max_threshold = self.config.get("max_threshold", 0.9)
        self.adjustment_rate = self.config.get("adjustment_rate", 0.05)
        self.update_interval = self.config.get("update_interval", 60.0)  # seconds
        self.history_window = self.config.get(
            "history_window", 100
        )  # Number of events to keep
        self.dimension_weights = self.config.get("dimension_weights", {})
        self.default_dimension_weight = self.config.get("default_dimension_weight", 1.0)

        # Trust data structures
        self.nodes = {}  # node_id -> node_info
        self.trust_scores = {}  # (node_id1, node_id2) -> trust_score
        self.trust_dimensions = {}  # (node_id1, node_id2) -> {dimension -> score}
        self.trust_history = {}  # (node_id1, node_id2) -> [historical_scores]
        self.violation_history = {}  # node_id -> [violation_events]
        self.current_thresholds = {}  # node_id -> current_threshold

        # Trust propagation cache
        self.trust_cache = AsyncCache(max_size=10000, ttl=300.0)  # 5 minutes TTL

        # Async processor for trust calculations
        self.async_processor = AsyncProcessor(num_workers=4)

        # Lock for thread safety
        self.lock = threading.RLock()

        # Running flag
        self.running = False
        self.update_thread = None

    def start(self):
        """
        Start the dynamic trust manager.
        """
        if self.running:
            return

        self.running = True
        self.async_processor.start()

        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()

    def stop(self):
        """
        Stop the dynamic trust manager.
        """
        self.running = False

        if self.update_thread:
            self.update_thread.join(timeout=5.0)
            self.update_thread = None

        self.async_processor.stop()

    def _update_loop(self):
        """
        Background thread for periodic trust threshold updates.
        """
        while self.running:
            try:
                self.update_thresholds()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Error in trust threshold update: {e}")
                time.sleep(5.0)  # Shorter interval on error

    def add_node(self, node_id: str, node_info: Dict[str, Any]):
        """
        Add a node to the trust network.

        Args:
            node_id: Unique identifier for the node
            node_info: Node information including reputation, history, etc.
        """
        with self.lock:
            self.nodes[node_id] = node_info
            self.current_thresholds[node_id] = self.base_threshold
            self.violation_history[node_id] = []

    def remove_node(self, node_id: str):
        """
        Remove a node from the trust network.

        Args:
            node_id: Unique identifier for the node
        """
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]

            if node_id in self.current_thresholds:
                del self.current_thresholds[node_id]

            if node_id in self.violation_history:
                del self.violation_history[node_id]

            # Remove trust relationships
            keys_to_remove = []
            for key in self.trust_scores:
                if node_id in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                if key in self.trust_scores:
                    del self.trust_scores[key]
                if key in self.trust_dimensions:
                    del self.trust_dimensions[key]
                if key in self.trust_history:
                    del self.trust_history[key]

    def update_trust_score(
        self,
        source_id: str,
        target_id: str,
        score: float,
        dimensions: Optional[Dict[str, float]] = None,
    ):
        """
        Update the trust score from source to target.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            score: Overall trust score (0-1)
            dimensions: Optional dictionary of dimension-specific scores
        """
        with self.lock:
            if source_id not in self.nodes or target_id not in self.nodes:
                return

            # Update overall score
            self.trust_scores[(source_id, target_id)] = score

            # Update dimensions
            if dimensions:
                if (source_id, target_id) not in self.trust_dimensions:
                    self.trust_dimensions[(source_id, target_id)] = {}

                for dimension, dim_score in dimensions.items():
                    self.trust_dimensions[(source_id, target_id)][dimension] = dim_score

            # Update history
            if (source_id, target_id) not in self.trust_history:
                self.trust_history[(source_id, target_id)] = []

            self.trust_history[(source_id, target_id)].append((time.time(), score))

            # Trim history if needed
            if len(self.trust_history[(source_id, target_id)]) > self.history_window:
                self.trust_history[(source_id, target_id)] = self.trust_history[
                    (source_id, target_id)
                ][-self.history_window :]

            # Invalidate cache
            self.trust_cache.remove((source_id, target_id))

    def get_trust_score(self, source_id: str, target_id: str) -> float:
        """
        Get the trust score from source to target.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            Trust score (0-1) or 0.0 if not found
        """
        with self.lock:
            # Check cache first
            cached_score = self.trust_cache.get((source_id, target_id))
            if cached_score is not None:
                return cached_score

            # Direct trust
            if (source_id, target_id) in self.trust_scores:
                score = self.trust_scores[(source_id, target_id)]
                self.trust_cache.put((source_id, target_id), score)
                return score

            # Indirect trust (transitive)
            score = self._calculate_indirect_trust(source_id, target_id)
            self.trust_cache.put((source_id, target_id), score)
            return score

    def get_trust_dimensions(self, source_id: str, target_id: str) -> Dict[str, float]:
        """
        Get the dimension-specific trust scores from source to target.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            Dictionary of dimension-specific scores or empty dict if not found
        """
        with self.lock:
            return self.trust_dimensions.get((source_id, target_id), {}).copy()

    def get_current_threshold(self, node_id: str) -> float:
        """
        Get the current trust threshold for a node.

        Args:
            node_id: Node ID

        Returns:
            Current trust threshold (0-1) or base threshold if not found
        """
        with self.lock:
            return self.current_thresholds.get(node_id, self.base_threshold)

    def report_violation(
        self, node_id: str, violation_type: str, severity: float, evidence: Any = None
    ):
        """
        Report a trust violation by a node.

        Args:
            node_id: Node ID that committed the violation
            violation_type: Type of violation
            severity: Severity of violation (0-1)
            evidence: Optional evidence of the violation
        """
        with self.lock:
            if node_id not in self.nodes:
                return

            # Record violation
            violation = {
                "timestamp": time.time(),
                "type": violation_type,
                "severity": severity,
                "evidence": evidence,
            }

            self.violation_history[node_id].append(violation)

            # Trim history if needed
            if len(self.violation_history[node_id]) > self.history_window:
                self.violation_history[node_id] = self.violation_history[node_id][
                    -self.history_window :
                ]

            # Adjust threshold immediately for severe violations
            if severity > 0.7:
                self._adjust_threshold(node_id, 0.1)  # Increase threshold

    def update_thresholds(self):
        """
        Update trust thresholds for all nodes based on network conditions.
        """
        with self.lock:
            for node_id in self.nodes:
                self._update_node_threshold(node_id)

    def _update_node_threshold(self, node_id: str):
        """
        Update the trust threshold for a specific node.

        Args:
            node_id: Node ID
        """
        if node_id not in self.nodes:
            return

        # Calculate violation rate
        violation_rate = self._calculate_violation_rate(node_id)

        # Calculate network health
        network_health = self._calculate_network_health()

        # Calculate threshold adjustment
        adjustment = 0.0

        # Adjust based on violation rate (higher violations -> higher threshold)
        adjustment += violation_rate * self.adjustment_rate

        # Adjust based on network health (lower health -> higher threshold)
        adjustment += (1.0 - network_health) * self.adjustment_rate

        # Apply adjustment
        self._adjust_threshold(node_id, adjustment)

    def _adjust_threshold(self, node_id: str, adjustment: float):
        """
        Adjust the trust threshold for a node.

        Args:
            node_id: Node ID
            adjustment: Adjustment amount (positive increases threshold)
        """
        if node_id not in self.current_thresholds:
            self.current_thresholds[node_id] = self.base_threshold

        # Apply adjustment
        new_threshold = self.current_thresholds[node_id] + adjustment

        # Clamp to valid range
        new_threshold = max(self.min_threshold, min(self.max_threshold, new_threshold))

        # Update threshold
        self.current_thresholds[node_id] = new_threshold

    def _calculate_violation_rate(self, node_id: str) -> float:
        """
        Calculate the violation rate for a node.

        Args:
            node_id: Node ID

        Returns:
            Violation rate (0-1)
        """
        if node_id not in self.violation_history or not self.violation_history[node_id]:
            return 0.0

        # Count recent violations (last 24 hours)
        recent_time = time.time() - 86400  # 24 hours ago
        recent_violations = [
            v for v in self.violation_history[node_id] if v["timestamp"] >= recent_time
        ]

        if not recent_violations:
            return 0.0

        # Calculate weighted violation rate
        total_severity = sum(v["severity"] for v in recent_violations)
        return min(1.0, total_severity / 10.0)  # Cap at 1.0

    def _calculate_network_health(self) -> float:
        """
        Calculate overall network health.

        Returns:
            Network health score (0-1)
        """
        if not self.nodes:
            return 1.0  # Assume perfect health if no nodes

        # Calculate average trust score
        trust_scores = list(self.trust_scores.values())
        if not trust_scores:
            return 1.0  # Assume perfect health if no trust scores

        avg_trust = sum(trust_scores) / len(trust_scores)

        # Calculate recent violation rate across all nodes
        recent_time = time.time() - 3600  # 1 hour ago
        recent_violations = []

        for node_id, violations in self.violation_history.items():
            recent_violations.extend(
                [v for v in violations if v["timestamp"] >= recent_time]
            )

        violation_rate = len(recent_violations) / len(self.nodes) if self.nodes else 0.0

        # Combine metrics (70% trust, 30% violations)
        health = 0.7 * avg_trust + 0.3 * (1.0 - violation_rate)

        return max(0.0, min(1.0, health))

    def _calculate_indirect_trust(self, source_id: str, target_id: str) -> float:
        """
        Calculate indirect trust between source and target through intermediate nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID

        Returns:
            Indirect trust score (0-1)
        """
        # Find all nodes that source trusts and that trust target
        intermediates = []

        for node_id in self.nodes:
            if node_id == source_id or node_id == target_id:
                continue

            source_to_intermediate = self.trust_scores.get((source_id, node_id), 0.0)
            intermediate_to_target = self.trust_scores.get((node_id, target_id), 0.0)

            if source_to_intermediate > 0.0 and intermediate_to_target > 0.0:
                intermediates.append(
                    (node_id, source_to_intermediate, intermediate_to_target)
                )

        if not intermediates:
            return 0.0  # No path found

        # Calculate weighted average of transitive trust
        total_weight = 0.0
        total_score = 0.0

        for node_id, source_trust, target_trust in intermediates:
            # Weight is the product of the two trust scores
            weight = source_trust * target_trust

            # Transitive trust is the minimum of the two trust scores
            score = min(source_trust, target_trust)

            total_weight += weight
            total_score += weight * score

        if total_weight == 0.0:
            return 0.0

        return total_score / total_weight

    def propagate_trust(
        self,
        source_id: str,
        target_id: str,
        score: float,
        dimensions: Optional[Dict[str, float]] = None,
    ):
        """
        Propagate a trust update to other nodes.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            score: Trust score
            dimensions: Optional dimension-specific scores
        """
        # Update our own record first
        self.update_trust_score(source_id, target_id, score, dimensions)

        # In a real implementation, this would propagate to other nodes
        # For simulation, we'll just update our local state

        # Find nodes that trust the source
        trusting_nodes = []
        for (s, t), trust_score in self.trust_scores.items():
            if t == source_id and trust_score > self.get_current_threshold(s):
                trusting_nodes.append(s)

        # Propagate to trusting nodes with decay
        for node_id in trusting_nodes:
            node_trust_in_source = self.trust_scores.get((node_id, source_id), 0.0)
            propagated_score = score * node_trust_in_source * 0.8  # 20% decay

            # Only propagate if significant
            if propagated_score > 0.1:
                # In a real implementation, this would send to other nodes
                # For simulation, we'll just update our local state
                self.update_trust_score(node_id, target_id, propagated_score)

    def calculate_multidimensional_trust(
        self, source_id: str, target_id: str, context: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate trust score considering multiple dimensions and context.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            context: Optional context with dimension importance weights

        Returns:
            Context-aware trust score (0-1)
        """
        with self.lock:
            # Get dimension scores
            dimensions = self.get_trust_dimensions(source_id, target_id)

            if not dimensions:
                # Fall back to overall score
                return self.get_trust_score(source_id, target_id)

            # Apply context weights or default weights
            total_weight = 0.0
            weighted_score = 0.0

            for dimension, score in dimensions.items():
                # Get weight from context, dimension_weights, or default
                if context and dimension in context:
                    weight = context[dimension]
                elif dimension in self.dimension_weights:
                    weight = self.dimension_weights[dimension]
                else:
                    weight = self.default_dimension_weight

                total_weight += weight
                weighted_score += score * weight

            if total_weight == 0.0:
                return 0.0

            return weighted_score / total_weight

    def detect_sybil_attack(self, node_ids: List[str]) -> Dict[str, float]:
        """
        Detect potential Sybil attacks in a group of nodes.

        Args:
            node_ids: List of node IDs to analyze

        Returns:
            Dictionary mapping node IDs to Sybil probability scores (0-1)
        """
        with self.lock:
            if not node_ids or len(node_ids) < 2:
                return {node_id: 0.0 for node_id in node_ids}

            # Calculate trust correlation matrix
            correlation_matrix = {}

            for i, node1 in enumerate(node_ids):
                for node2 in node_ids[i + 1 :]:
                    # Skip if either node is not in our database
                    if node1 not in self.nodes or node2 not in self.nodes:
                        continue

                    # Calculate correlation of trust opinions
                    correlation = self._calculate_trust_correlation(node1, node2)
                    correlation_matrix[(node1, node2)] = correlation
                    correlation_matrix[(node2, node1)] = correlation

            # Calculate Sybil scores based on correlations
            sybil_scores = {}

            for node_id in node_ids:
                if node_id not in self.nodes:
                    sybil_scores[node_id] = 0.0
                    continue

                # Calculate average correlation with other nodes
                correlations = []
                for other_node in node_ids:
                    if (
                        other_node != node_id
                        and (node_id, other_node) in correlation_matrix
                    ):
                        correlations.append(correlation_matrix[(node_id, other_node)])

                if not correlations:
                    sybil_scores[node_id] = 0.0
                    continue

                # Higher correlation suggests Sybil behavior
                avg_correlation = sum(correlations) / len(correlations)
                sybil_scores[node_id] = avg_correlation

            return sybil_scores

    def _calculate_trust_correlation(self, node1: str, node2: str) -> float:
        """
        Calculate correlation between trust opinions of two nodes.

        Args:
            node1: First node ID
            node2: Second node ID

        Returns:
            Correlation coefficient (-1 to 1)
        """
        # Get all nodes that both node1 and node2 have opinions about
        common_targets = []

        for target in self.nodes:
            if target == node1 or target == node2:
                continue

            if (node1, target) in self.trust_scores and (
                node2,
                target,
            ) in self.trust_scores:
                common_targets.append(target)

        if len(common_targets) < 3:
            return 0.0  # Not enough data for correlation

        # Calculate correlation
        scores1 = [self.trust_scores[(node1, target)] for target in common_targets]
        scores2 = [self.trust_scores[(node2, target)] for target in common_targets]

        # Convert to numpy arrays
        try:
            import numpy as np

            scores1_np = np.array(scores1)
            scores2_np = np.array(scores2)

            # Calculate correlation coefficient
            correlation = np.corrcoef(scores1_np, scores2_np)[0, 1]

            # Handle NaN
            if np.isnan(correlation):
                return 0.0

            return correlation
        except:
            # Fallback if numpy not available
            n = len(scores1)
            sum_x = sum(scores1)
            sum_y = sum(scores2)
            sum_xy = sum(x * y for x, y in zip(scores1, scores2))
            sum_x2 = sum(x * x for x in scores1)
            sum_y2 = sum(y * y for y in scores2)

            numerator = n * sum_xy - sum_x * sum_y
            denominator = (
                (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
            ) ** 0.5

            if denominator == 0:
                return 0.0

            return numerator / denominator

    def get_trust_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current trust state.

        Returns:
            Dictionary with trust statistics
        """
        with self.lock:
            stats = {
                "num_nodes": len(self.nodes),
                "avg_trust_score": 0.0,
                "avg_threshold": 0.0,
                "trust_distribution": {},
                "threshold_distribution": {},
                "network_health": self._calculate_network_health(),
            }

            # Calculate average trust score
            trust_scores = list(self.trust_scores.values())
            if trust_scores:
                stats["avg_trust_score"] = sum(trust_scores) / len(trust_scores)

            # Calculate average threshold
            thresholds = list(self.current_thresholds.values())
            if thresholds:
                stats["avg_threshold"] = sum(thresholds) / len(thresholds)

            # Calculate trust score distribution
            if trust_scores:
                bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                hist, _ = np.histogram(trust_scores, bins=bins)
                stats["trust_distribution"] = {
                    f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(hist[i])
                    for i in range(len(hist))
                }

            # Calculate threshold distribution
            if thresholds:
                bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
                hist, _ = np.histogram(thresholds, bins=bins)
                stats["threshold_distribution"] = {
                    f"{bins[i]:.1f}-{bins[i+1]:.1f}": int(hist[i])
                    for i in range(len(hist))
                }

            return stats
