"""
Segment Tree implementation for efficient Prioritized Experience Replay in Rainbow DQN.

This module provides a Segment Tree data structure that allows O(log n) updates and queries
for sum and min operations, which are essential for efficient prioritized experience replay.
"""

import numpy as np
from typing import Callable, List, Tuple, Union, Optional


class SegmentTree:
    """
    Segment Tree data structure for efficient sum and min operations with O(log n) complexity.

    This implementation supports:
    - Updating values at specific indices
    - Querying the sum of values in a range
    - Finding the prefix sum index (useful for prioritized sampling)
    - Querying the minimum value in a range
    """

    def __init__(self, capacity: int, operation: Callable, neutral_element: float):
        """
        Initialize a segment tree with given capacity and operation.

        Args:
            capacity: Number of elements in the tree (must be a power of 2)
            operation: Binary operation to perform (e.g., min, sum)
            neutral_element: Neutral element for the operation (e.g., float('inf') for min, 0 for sum)
        """
        # Ensure capacity is a power of 2
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2

        # Initialize tree with neutral elements
        self.tree = [neutral_element for _ in range(2 * self.capacity)]
        self.operation = operation
        self.neutral_element = neutral_element

    def _operate(
        self, start: int, end: int, node: int, node_start: int, node_end: int
    ) -> float:
        """
        Recursively perform the operation on the tree.

        Args:
            start: Start index of query range
            end: End index of query range
            node: Current node index
            node_start: Start index of current node's range
            node_end: End index of current node's range

        Returns:
            Result of the operation on the specified range
        """
        # If the query range is outside the node range, return neutral element
        if start >= node_end or end <= node_start:
            return self.neutral_element

        # If the query range completely contains the node range, return the node value
        if start <= node_start and node_end <= end:
            return self.tree[node]

        # Otherwise, split the range and combine results
        mid = (node_start + node_end) // 2
        return self.operation(
            self._operate(start, end, 2 * node, node_start, mid),
            self._operate(start, end, 2 * node + 1, mid, node_end),
        )

    def query(self, start: int, end: int) -> float:
        """
        Query the result of the operation on the range [start, end).

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)

        Returns:
            Result of the operation on the specified range
        """
        # Ensure valid range
        if end <= start:
            return self.neutral_element

        # Ensure indices are within bounds
        start = max(start, 0)
        end = min(end, self.capacity)

        return self._operate(start, end, 1, 0, self.capacity)

    def update(self, idx: int, val: float) -> None:
        """
        Update the value at the specified index.

        Args:
            idx: Index to update
            val: New value
        """
        # Ensure index is within bounds
        assert 0 <= idx < self.capacity

        # Update leaf node
        idx += self.capacity
        self.tree[idx] = val

        # Update parent nodes
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])


class SumSegmentTree(SegmentTree):
    """
    Segment Tree specialized for sum operations.
    Used for sampling from prioritized experience replay buffer.
    """

    def __init__(self, capacity: int):
        """
        Initialize a sum segment tree with given capacity.

        Args:
            capacity: Number of elements in the tree
        """
        super().__init__(capacity, lambda a, b: a + b, 0.0)

    def sum(self, start: int = 0, end: Optional[int] = None) -> float:
        """
        Calculate the sum of values in the range [start, end).

        Args:
            start: Start index (inclusive)
            end: End index (exclusive), defaults to capacity

        Returns:
            Sum of values in the specified range
        """
        if end is None:
            end = self.capacity
        return self.query(start, end)

    def find_prefixsum_idx(self, prefixsum: float) -> int:
        """
        Find the highest index such that the sum of elements up to that index <= prefixsum.

        Args:
            prefixsum: Target prefix sum

        Returns:
            Highest index meeting the condition
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5

        idx = 1
        while idx < self.capacity:  # While not a leaf node
            if self.tree[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self.tree[2 * idx]
                idx = 2 * idx + 1

        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """
    Segment Tree specialized for min operations.
    Used for calculating the minimum priority in prioritized experience replay.
    """

    def __init__(self, capacity: int):
        """
        Initialize a min segment tree with given capacity.

        Args:
            capacity: Number of elements in the tree
        """
        super().__init__(capacity, min, float("inf"))

    def min(self, start: int = 0, end: Optional[int] = None) -> float:
        """
        Find the minimum value in the range [start, end).

        Args:
            start: Start index (inclusive)
            end: End index (exclusive), defaults to capacity

        Returns:
            Minimum value in the specified range
        """
        if end is None:
            end = self.capacity
        return self.query(start, end)
