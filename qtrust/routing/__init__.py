"""
Routing Module

This module contains the implementation of the Multi-Agent Dynamic Routing with Adaptive Path
Identification and Decision (MAD-RAPID) system, which provides efficient transaction routing
across shards with congestion awareness and multi-objective optimization.
"""

from .mad_rapid import MADRAPIDRouter, RoutingAgent, CrossShardManager

__all__ = [
    "MADRAPIDRouter",
    "RoutingAgent",
    "CrossShardManager",
    "MultiObjectivePathFinder",
    "CrossShardAwarenessManager",
]
