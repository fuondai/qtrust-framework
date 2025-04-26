#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - MAD-RAPID Router
This module provides a compatibility wrapper for the MAD-RAPID protocol.
"""

from .mad_rapid import MADRAPID


# Create a compatibility wrapper for MADRAPIDRouter
class MADRAPIDRouter(MADRAPID):
    """
    Compatibility wrapper for MAD-RAPID protocol.
    This class provides backward compatibility with the MADRAPIDRouter interface
    expected by the benchmark scripts.
    """

    def __init__(self, shard_id=None, config=None):
        """
        Initialize MAD-RAPID router.

        Args:
            shard_id: Shard identifier
            config: Configuration dictionary
        """
        super().__init__(shard_id, config)
