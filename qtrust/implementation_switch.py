"""
Implementation switch module for QTrust.

This module provides functions to switch between PyTorch and mock implementations.
"""

import os
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Global flag to control whether to use PyTorch or mock implementations
_USE_PYTORCH = True


def set_use_pytorch(use_pytorch: bool) -> None:
    """
    Set whether to use PyTorch or mock implementations.

    Args:
        use_pytorch: Whether to use PyTorch implementations
    """
    global _USE_PYTORCH
    _USE_PYTORCH = use_pytorch
    logger.info(f"Set use_pytorch to {use_pytorch}")


def get_use_pytorch() -> bool:
    """
    Get whether to use PyTorch or mock implementations.

    Returns:
        Whether to use PyTorch implementations
    """
    return _USE_PYTORCH


def get_rainbow_agent(state_dim: int, action_dim: int, config: Dict[str, Any] = None) -> Any:
    """
    Get the appropriate Rainbow agent implementation.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        config: Configuration parameters

    Returns:
        Rainbow agent implementation
    """
    if _USE_PYTORCH:
        try:
            from qtrust.agents.rainbow_agent import RainbowDQNAgent

            return RainbowDQNAgent(state_dim, action_dim, config)
        except ImportError:
            logger.warning("PyTorch not available, falling back to mock implementation")
            from qtrust.mocks.mock_rainbow_agent import MockRainbowDQNAgent

            return MockRainbowDQNAgent(state_dim, action_dim, config)
    else:
        from qtrust.mocks.mock_rainbow_agent import MockRainbowDQNAgent

        return MockRainbowDQNAgent(state_dim, action_dim, config)


def get_adaptive_rainbow_agent(state_dim: int, action_dim: int, config: Dict[str, Any] = None) -> Any:
    """
    Get the appropriate Adaptive Rainbow agent implementation.

    Args:
        state_dim: State dimension
        action_dim: Action dimension
        config: Configuration parameters

    Returns:
        Adaptive Rainbow agent implementation
    """
    if _USE_PYTORCH:
        try:
            from qtrust.agents.adaptive_rainbow import AdaptiveRainbowAgent

            return AdaptiveRainbowAgent(state_dim, action_dim, config)
        except ImportError:
            logger.warning("PyTorch not available, falling back to mock implementation")
            from qtrust.mocks.mock_adaptive_rainbow import MockAdaptiveRainbowAgent

            return MockAdaptiveRainbowAgent(state_dim, action_dim, config)
    else:
        from qtrust.mocks.mock_adaptive_rainbow import MockAdaptiveRainbowAgent

        return MockAdaptiveRainbowAgent(state_dim, action_dim, config)


def get_privacy_preserving_fl(config: Dict[str, Any] = None) -> Any:
    """
    Get the appropriate Privacy-Preserving Federated Learning implementation.

    Args:
        config: Configuration parameters

    Returns:
        Privacy-Preserving Federated Learning implementation
    """
    # Use empty dict if config is None
    actual_config = config if config is not None else {}
    
    if _USE_PYTORCH:
        try:
            from qtrust.federated.privacy_preserving_fl import FederatedLearningManager

            return FederatedLearningManager(actual_config)
        except ImportError:
            logger.warning("PyTorch not available, falling back to mock implementation")
            from qtrust.mocks.mock_privacy_preserving_fl import MockPrivacyPreservingFL

            # Extract num_clients from config, with default value if not specified
            num_clients = actual_config.get("num_clients", 10)
            # Extract dp_params if available
            dp_params = actual_config.get("dp_params", None)
            return MockPrivacyPreservingFL(num_clients, dp_params)
    else:
        from qtrust.mocks.mock_privacy_preserving_fl import MockPrivacyPreservingFL

        # Extract num_clients from config, with default value if not specified
        num_clients = actual_config.get("num_clients", 10)
        # Extract dp_params if available
        dp_params = actual_config.get("dp_params", None)
        return MockPrivacyPreservingFL(num_clients, dp_params)
