from typing import Dict, Any
import logging
import time

class QTrustFramework:
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the QTrust framework.

        Args:
            config: Configuration parameters
        """
        # Set up logger
        self.logger = logging
        
        # Default configuration
        self.config = {
            "initial_shards": 4,
            "max_shards": 64,
            "min_shards": 1,
            "target_shard_size": 32,  # nodes per shard
            "rebalance_threshold": 0.3,  # trigger rebalance if imbalance > 30%
            "rebalance_frequency": 100,  # blocks
            "consensus_update_frequency": 50,  # blocks
            "routing_optimization_frequency": 20,  # blocks
            "federated_learning_frequency": 200,  # blocks
            "state_dim": 64,
            "action_dim": 8,
            "use_pytorch": False,  # Use NumPy implementation by default
        } 

    def _init_components(self):
        """Initialize all framework components."""
        # Create shard manager
        self.shard_manager = ShardManager(self.config)
        
        # Initialize core components
        self.consensus_selector = AdaptiveConsensusSelector(self.config)
        self.router = MADRAPIDRouter(self.config)
        self.trust_manager = HTDCM(self.config)
        
        # Initialize federated learning manager if enabled
        if self.config.get("enable_federated_learning", True):
            federated_config = self.config.get("federated_learning", {})
            self.federated_manager = get_privacy_preserving_fl(federated_config)
        else:
            self.federated_manager = None
            
        # Register components with each other
        if hasattr(self.consensus_selector, "register_trust_provider"):
            self.consensus_selector.register_trust_provider(self.trust_manager)
            
        if hasattr(self.router, "register_trust_provider"):
            self.router.register_trust_provider(self.trust_manager)
            
        # Initialized time tracking
        self.last_rebalance = time.time()
        self.last_consensus_update = time.time()
        self.last_routing_optimization = time.time()
        self.last_federated_learning = time.time() 