"""
Mock implementation of MAD-RAPID Router for testing without dependencies.
"""

import random
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)


class MADRAPIDRouter:
    """
    Mock implementation of MAD-RAPID Router for testing.

    This class provides a simplified implementation that mimics the behavior
    of the actual MAD-RAPID Router without requiring dependencies.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the mock MAD-RAPID Router.

        Args:
            config: Configuration parameters
        """
        # Default configuration
        self.config = {
            "optimization_frequency": 50,
            "max_hops": 3,
            "latency_weight": 0.7,
            "throughput_weight": 0.3,
        }

        # Update configuration if provided
        if config:
            self.config.update(config)

        # Initialize routing table
        self.routing_table = {}

        # Performance metrics
        self.metrics = {"average_hops": [], "routing_latency": [], "success_rate": []}

        logger.info("Initialized MockMADRAPIDRouter")

    def find_optimal_route(self, source: str, destination: str) -> List[str]:
        """
        Find optimal route between source and destination shards.

        Args:
            source: Source shard ID
            destination: Destination shard ID

        Returns:
            List of shard IDs representing the route
        """
        # In a real implementation, this would use sophisticated routing algorithms
        # For testing, we just create a simple route

        # Check if route exists in routing table
        route_key = f"{source}_{destination}"
        if route_key in self.routing_table:
            return self.routing_table[route_key]

        # Generate a random route
        num_hops = random.randint(1, self.config["max_hops"])
        route = [source]

        # Add intermediate hops
        for _ in range(num_hops - 1):
            # Generate a random shard ID that's not already in the route
            while True:
                hop = f"shard_{random.randint(0, 15)}"
                if hop not in route and hop != destination:
                    route.append(hop)
                    break

        # Add destination
        route.append(destination)

        # Store in routing table
        self.routing_table[route_key] = route

        # Update metrics
        self.metrics["average_hops"].append(len(route) - 1)
        self.metrics["routing_latency"].append(
            random.uniform(0.1, 0.5) * (len(route) - 1)
        )
        self.metrics["success_rate"].append(1.0)  # Always successful in mock

        return route

    def optimize_routes(self) -> None:
        """Optimize routing table."""
        # In a real implementation, this would use sophisticated optimization
        # For testing, we just add some randomness to the routes

        # Iterate through existing routes
        for route_key, route in list(self.routing_table.items()):
            # 20% chance to optimize a route
            if random.random() < 0.2:
                source, destination = route[0], route[-1]

                # Generate a new route with potentially fewer hops
                num_hops = random.randint(
                    1, min(len(route) - 1, self.config["max_hops"])
                )
                new_route = [source]

                # Add intermediate hops
                for _ in range(num_hops - 1):
                    # Generate a random shard ID that's not already in the new route
                    while True:
                        hop = f"shard_{random.randint(0, 15)}"
                        if hop not in new_route and hop != destination:
                            new_route.append(hop)
                            break

                # Add destination
                new_route.append(destination)

                # Update routing table if new route is better
                if len(new_route) < len(route):
                    self.routing_table[route_key] = new_route
                    logger.info(
                        f"Optimized route from {source} to {destination}: {len(route)} hops -> {len(new_route)} hops"
                    )

    def get_metrics(self) -> Dict[str, List[float]]:
        """
        Get performance metrics.

        Returns:
            Dictionary of metrics
        """
        return self.metrics
